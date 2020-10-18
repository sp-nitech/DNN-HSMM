import numpy, torch
from torch import nn

class GaussianLogLikelihood(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean, sdev, o):
        def _log_prob(mean, sdev, o):
            LOGPROBC = - 0.5 * torch.tensor(2.0 * numpy.pi, device=o.device, dtype=o.dtype).log()
            return LOGPROBC - sdev.log() - 0.5 * ((o - mean) / sdev).square()

        ctx.save_for_backward(mean, sdev, o)
        
        mxlen = 100
        o = o.unsqueeze(dim=1)
        if mxlen is None:
            prob = _log_prob(mean, sdev, o).sum(dim=2)
        else:
            prob = torch.cat([_log_prob(mean, sdev, o[t:t+mxlen]).sum(dim=2) for t in range(0, len(o), mxlen)], dim=0)
        return prob

    @staticmethod
    def backward(ctx, gy):
        def _gMean(gy, mean, sdev, o):
            return gy * ((o - mean) / sdev.square())
        def _gSdev(gy, mean, sdev, o):
            return gy * (((o - mean) / sdev).square() - 1.0) / sdev

        mean, sdev, o = ctx.saved_tensors

        mxlen = 100
        o, gy = o.unsqueeze(dim=1), gy.unsqueeze(dim=2)
        if mxlen is None:
            gMean = _gMean(gy, mean, sdev, o).sum(dim=0)
            gSdev = _gSdev(gy, mean, sdev, o).sum(dim=0)
        else:
            gMean = torch.zeros(mean.shape, dtype=mean.dtype, device=mean.device)
            for t in range(0, len(o), mxlen):
                gMean += _gMean(gy[t:t+mxlen], mean, sdev, o[t:t+mxlen]).sum(dim=0)
            gSdev = torch.zeros(sdev.shape, dtype=sdev.dtype, device=sdev.device)
            for t in range(0, len(o), mxlen):
                gSdev += _gSdev(gy[t:t+mxlen], mean, sdev, o[t:t+mxlen]).sum(dim=0)
        return gMean, gSdev, None
_gaussian_log_likelihood = GaussianLogLikelihood.apply

def _set_bd(p, bo, max_dur=500, dmask_dpdf_mergin=None, bmask=None, dmask=None, temperature=1.0, weight_s={'b': [(slice(None), 1.0)], 'd': 1.0}):
    dtype, device = torch.double, bo.device
    LZERO = torch.tensor(-1e+10, dtype=dtype, device=device)
    def _make_dmask_dpdf(mergin):
        mask = []
        for state_min_dur, state_max_dur in [(min(max_dur, max(1, int((mean - mergin * sdev).data.round()))), min(max_dur, max(1, int((mean + mergin * sdev).data.round())))) for mean, sdev in zip(dm[:, 0], dv[:, 0])]:
            mask.append(torch.cat([torch.zeros((state_min_dur - 1, 1), dtype=torch.bool, device=device), torch.ones((state_max_dur - (state_min_dur - 1), 1), dtype=torch.bool, device=device), torch.zeros((max_dur - state_max_dur, 1), dtype=torch.bool, device=device)], dim=0))
        mask = torch.cat(mask, dim=1)
        return mask

    bm, bv, dm, dv = [v.type(dtype) for v in p]
    bo = bo.type(dtype)
    do = torch.arange(1, 1 + max_dur, dtype=dtype, device=device).view(-1, 1)

    b = temperature * torch.stack([w * _gaussian_log_likelihood(bm[:, dim] , bv[:, dim], bo[:, dim]) for dim, w in weight_s['b']]).sum(dim=0)
    if bmask is not None:
        b = b + bmask.logical_not() * LZERO
    d = temperature * weight_s['d'] * _gaussian_log_likelihood(dm, dv, do)
    if dmask is not None:
        d = d + dmask.logical_not() * LZERO
    if dmask_dpdf_mergin is not None:
        d = d + _make_dmask_dpdf(dmask_dpdf_mergin).logical_not() * LZERO

    return b, d

def _padded_generalized_forward_backward_fw(B, D, sizes, max_dur):
    dtype, device = B.dtype, B.device
    LZERO = torch.tensor(-1e+10, dtype=dtype, device=device)
    batch_size, sumT = D.shape[0], len(B)
    mxT, mxN = len(sizes), B.shape[1]

    def _backward(B, D, sizes):
        B = B.unsqueeze(dim=1)

        beta = torch.full((sumT, mxN), LZERO, dtype=dtype, device=device)
        beta_t = torch.full((batch_size, max_dur, mxN), LZERO, dtype=dtype, device=device)

        eidx = sumT
        beta_t[:, :, -1] = D[:, :, -1]
        for t in reversed(range(1, mxT)):
            size = int(sizes[t])
            b = B[eidx-size:eidx]
            beta0 = beta_t[:size] + b
            beta1 = D[:size, :, :-1] + beta0[:size, 0:1, 1:]
            beta_t[:size, -1, -1] = LZERO
            beta_t[:size, -1, :-1] = beta1[:size, -1]
            beta_t[:size, :-1, -1] = beta0[:size, 1:, -1]
            beta_t[:size, :-1, :-1] = torch.stack([beta1[:size, :-1], beta0[:size, 1:, :-1]]).logsumexp(dim=0)
            beta[eidx-size:eidx] = beta0[:, 0]
            eidx -= size
        beta0 = beta_t[:sizes[0]] + B[:sizes[0]]
        beta[:sizes[0]] = beta0[:sizes[0], 0]
        L = beta[:sizes[0], 0]
        return L, beta

    def _forward(B, D, L, beta, sizes):
        B, beta, L = B.unsqueeze(dim=1), beta.unsqueeze(dim=1), L.unsqueeze(dim=1).unsqueeze(dim=2)

        bgamma = torch.zeros((batch_size, mxT + max_dur, mxN), dtype=dtype, device=device)
        dgammaN = torch.zeros((batch_size, max_dur, mxN), dtype=dtype, device=device)
        dgammaT = torch.zeros((batch_size, mxT, max_dur), dtype=dtype, device=device)

        alpha_t = torch.full((batch_size, max_dur, mxN), LZERO, dtype=dtype, device=device)
        alpha_t[:, 0, 0] = B[:sizes[0], 0, 0]
        sidx = int(sizes[0])
        for t in range(1, mxT):
            size = int(sizes[t])
            alpha = alpha_t[:size] + D[:size]

            gamma = (alpha[:size, :, :-1] + beta[sidx:sidx+size, :, 1:] - L[:size]).exp()
            bgamma[:size, t:t+max_dur, :-1] += gamma.flip(dims=(1,)).cumsum(dim=1)
            dgammaN[:size, :, :-1] += gamma
            dgammaT[:size, t, :] += gamma.sum(dim=2)

            alpha = alpha.logsumexp(dim=1)
            alpha_t[:size, 1:] = alpha_t[:size, :-1].clone()
            alpha_t[:size, 0, 1:] = alpha[:size, :-1]
            alpha_t[:size, 0, 0] = LZERO
            alpha_t[:size] += B[sidx:sidx+size]

            sidx += size
        alpha = alpha_t[:sizes[-1]] + D[:sizes[-1]]

        gamma = (alpha[:sizes[-1], :, -1] - L[:sizes[-1]].squeeze(dim=2)).exp()
        bgamma[:sizes[-1], mxT:mxT+max_dur, -1] += gamma.flip(dims=(1,)).cumsum(dim=1)
        dgammaN[:sizes[-1], :, -1] += gamma
        dgammaT[:sizes[-1], mxT-1] += gamma

        bgamma = bgamma[:, max_dur:]

        return bgamma, dgammaN, dgammaT

    L, beta = _backward(B, D, sizes)
    bgamma, dgammaN, dgammaT = _forward(B, D, L, beta, sizes)

    return L, bgamma, dgammaN, dgammaT

class PaddedGeneralizedForwardBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, d, sizes, lengths, max_dur):
        L, bgamma, dgammaN, dgammaT = _padded_generalized_forward_backward_fw(b, d, sizes, max_dur)
        ctx.save_for_backward(bgamma, dgammaN)
        ctx.lengths = lengths
        return L, bgamma, dgammaN, dgammaT

    @staticmethod
    def backward(ctx, gL, gbgamma, gdgammaN, gdgammaT):
        # Error backprop of bgamma, dgammaN and dgammaT is not implemented.
        bgamma, dgammaN = ctx.saved_tensors
        gL = gL.unsqueeze(dim=1).unsqueeze(dim=2)
        gb = nn.utils.rnn.pack_padded_sequence(gL * bgamma, ctx.lengths, batch_first=True, enforce_sorted=False).data
        gd = gL * dgammaN
        return gb, gd, None, None, None
_padded_generalized_forward_backward = PaddedGeneralizedForwardBackward.apply

def _generalized_forward_backward(B, D, max_dur):
    def _padBD(B, D):
        TN = [b.shape for b in B]
        maxN = max([b.shape[1] for b in B]) + 1
        B = [torch.cat([torch.cat([b, torch.full((b.shape[0], maxN - b.shape[1]), -1.0e+10, dtype=b.dtype, device=b.device)], dim=1), torch.full((maxN - b.shape[1], maxN), -1.0e+10, dtype=b.dtype, device=b.device)], dim=0) for b in B]
        for b, (t, n) in zip(B, TN):
            b[t:, n:] = 0.0
        D = [torch.cat([d, torch.zeros((d.shape[0], maxN - d.shape[1]), dtype=d.dtype, device=d.device)], dim=1) for d in D]
        return TN, B, D

    TN, B, D = _padBD(B, D)
    packed = nn.utils.rnn.pack_sequence(B, enforce_sorted=False)
    D = torch.stack([D[i] for i in packed.sorted_indices])
    lengths = torch.tensor([len(B[i]) for i in packed.sorted_indices], dtype=torch.int64, device='cpu')

    L, bgamma, dgammaN, dgammaT = _padded_generalized_forward_backward(packed.data, D, packed.batch_sizes, lengths, max_dur)

    L = [L[i] for i in packed.unsorted_indices]
    bG = [bgamma[i, :t, :n] for i, (t, n) in zip(packed.unsorted_indices, TN)]
    dGN = [dgammaN[i, :, :n] for i, (_, n) in zip(packed.unsorted_indices, TN)]
    dGT = [dgammaT[i, :t, :] for i, (t, _) in zip(packed.unsorted_indices, TN)]
    G = [(bgamma, dgammaN, dgammaT) for bgamma, dgammaN, dgammaT in zip(bG, dGN, dGT)]

    return L, G

def generalized_forward_backward(P, BO, M, max_dur=500, dmask_dpdf_mergin=None, temperature=1.0, weight_s={'b': [(slice(None), 1.0)], 'd': 1.0}, detach=True):
    BD = [_set_bd(p, bo, max_dur, dmask_dpdf_mergin, bmask, dmask, temperature, weight_s) for p, bo, (bmask, dmask) in zip(P, BO, M)]
    B, D = map(lambda i: [bd[i] for bd in BD], range(len(BD[0])))
    L, G = _generalized_forward_backward(B, D, max_dur)
    if detach:
        G = [(bgamma.detach(), dgammaN.detach(), dgammaT.detach()) for bgamma, dgammaN, dgammaT in G]
    L = [l / temperature for l in L]
    L, G = [l.type(torch.float32) for l in L], [(bgamma.type(torch.float32), dgammaN.type(torch.float32), dgammaT.type(torch.float32)) for bgamma, dgammaN, dgammaT in G]
    return L, G

def predict_dur_dpdf(p, max_dur=500):
    _, _, dm, _ = p
    S = [min(max_dur, max(1, int(m.round()))) for m in dm]
    N, T = len(S), sum(S)

    dtype, device = dm.dtype, dm.device
    eye = torch.eye(N, dtype=dtype, device=device)
    bgamma = torch.cat([eye[i].repeat(m, 1) for i, m in enumerate(S)], dim=0)
    dgammaN = torch.zeros((max_dur, N), dtype=dtype, device=device)
    for i in range(len(S)):
        dgammaN[S[i]-1, i] = 1.0
    dgammaT = torch.zeros((T, max_dur), dtype=dtype, device=device)
    for i in range(len(S)):
        dgammaT[sum(S[:i+1])-1, S[i]-1] = 1.0
    return bgamma, dgammaN, dgammaT

class Align(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, d):
        (T, N), max_dur = b.shape, d.shape[0]
        dtype, device = b.dtype, b.device
        LZERO = torch.tensor(-1e+10, dtype=dtype, device=device)

        bp = torch.full((T, N), -1, dtype=torch.int32, device=device)
        alpha_t = torch.full((max_dur, N), LZERO, dtype=dtype, device=device)
        alpha_t[0, 0] = b[0, 0]
        for t in range(1, T):
            alpha = alpha_t + d
            bp[t-1] = alpha.argmax(dim=0) + 1
            alpha = alpha.logsumexp(dim=0)
            alpha_t[1:] = alpha_t[:-1].clone()
            alpha_t[0, 1:] = alpha[:-1]
            alpha_t[0, 0] = LZERO
            alpha_t += b[t]
        alpha = alpha_t + d
        bp[T-1] = alpha.argmax(dim=0) + 1

        S = torch.full((N,), -1, dtype=torch.int32, device=device)
        t = T
        for i in reversed(range(N)):
            S[i] = bp[t-1, i]
            t = t - S[i]

        eye = torch.eye(len(S), dtype=dtype, device=device)
        bgamma = torch.cat([eye[i].repeat(int(m), 1) for i, m in enumerate(S)], dim=0)
        dgammaN = torch.zeros((max_dur, N), dtype=dtype, device=device)
        for i in range(len(S)):
            dgammaN[S[i]-1, i] = 1.0
        dgammaT = torch.zeros((T, max_dur), dtype=dtype, device=device)
        for i in range(len(S)):
            dgammaT[sum(S[:i+1])-1, S[i]-1] = 1.0
        return bgamma, dgammaN, dgammaT

    @staticmethod
    def backward(ctx, gy):
        return None, None, None
_align = Align.apply

def align(P, BO, M, max_dur=500, dmask_dpdf_mergin=None):
    BD = [_set_bd(p, bo, max_dur, dmask_dpdf_mergin, bmask, dmask) for p, bo, (bmask, dmask) in zip(P, BO, M)]
    G = [_align(b, d) for b, d in BD]
    G = [(bgamma.type(torch.float32), dgammaN.type(torch.float32), dgammaT.type(torch.float32)) for bgamma, dgammaN, dgammaT in G]
    return G
