import torch
from torch import nn

from .hsmm import generalized_forward_backward, predict_dur_dpdf, align
from .delta import n_win, append_delta, mlpg
from .utils import magic_number

class Model(nn.Module):
    def __init__(self, dim, normfac, incode_embedding, N, max_state_dur, output_initial_bias, device):
        super(Model, self).__init__()

        wdims = [wdim.stop - wdim.start for wdim, _ in dim['ofeat']]
        odims = [odim.stop - odim.start for _, odim in dim['ofeat']]
        dim['weight'] = sum(wdims)
        dim['output'] = n_win * sum(odims)
        swdim, sodim, dim['order'], dim['stream'] = 0, 0, [], []
        for wdim, odim in zip(wdims, odims):
            dim['order'].append((slice(swdim, swdim + wdim), slice(sodim, sodim + (n_win * odim))))
            dim['stream'].append((slice(swdim, swdim + wdim), slice(swdim + wdim + sodim, swdim + wdim + sodim + (n_win * odim))))
            swdim += wdim
            sodim += n_win * odim

        self.dim = dim
        self.normfac = {'ifeat': {'mean': torch.from_numpy(normfac['ifeat']['mean']).to(device),
                                  'sdev': torch.from_numpy(normfac['ifeat']['sdev']).to(device)},
                        'ofeat': {'mean': torch.from_numpy(normfac['ofeat']['mean']).to(device),
                                  'sdev': torch.from_numpy(normfac['ofeat']['sdev']).to(device)}}
        self.normfac['ifeat']['sdev'] = torch.clamp(self.normfac['ifeat']['sdev'], 1.0E-6)
        self.normfac['ofeat']['sdev'] = torch.clamp(self.normfac['ofeat']['sdev'], 1.0E-6)
        self.N = N
        self.max_state_dur = max_state_dur
        self.mask_x = None
        self.dmask_dpdf_mergin = None
        self.temperature = 1.0
        self.weight_s = {'b': [(slice(None), 1.0)], 'd': 1.0}
        self.lha = incode_embedding
        self.lh1 = nn.Sequential(
            nn.Linear(dim['input'] + incode_embedding.embeddim, 2048),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Tanh())
        self.lbw = nn.Linear(2048, N * dim['weight'])
        self.lbm = nn.Linear(2048, N * dim['output'])
        self.lbv = nn.Linear(2048, N * dim['output'])
        self.ldm = nn.Linear(2048, N * 1)
        self.ldv = nn.Linear(2048, N * 1)

        nn.init.zeros_(self.lbw.weight)
        self.lbw.bias.data = torch.from_numpy(output_initial_bias['bw']).repeat(N)
        nn.init.zeros_(self.lbm.weight)
        self.lbm.bias.data = torch.from_numpy(output_initial_bias['bm']).repeat(N)
        nn.init.zeros_(self.lbv.weight)
        self.lbv.bias.data = torch.from_numpy(output_initial_bias['bv']).repeat(N)
        nn.init.zeros_(self.ldm.weight)
        self.ldm.bias.data = torch.from_numpy(output_initial_bias['dm']).repeat(N)
        nn.init.zeros_(self.ldv.weight)
        self.ldv.bias.data = torch.from_numpy(output_initial_bias['dv']).repeat(N)

        self.to(device)

    def _norm(self, X, O):
        if X is not None:
            X = [(x - self.normfac['ifeat']['mean']) / self.normfac['ifeat']['sdev'] for x in X]
        if O is not None:
            O = [(o - self.normfac['ofeat']['mean']) / self.normfac['ofeat']['sdev'] for o in O]
        return X, O

    def _unnorm(self, P):
        P = [(p[0] * self.normfac['ofeat']['sdev'] + self.normfac['ofeat']['mean'], p[1] * self.normfac['ofeat']['sdev']) for p in P]
        return P

    def _replace_with_magic_number(self, Y):
        def _replace_with_magic_number_ofeat(y):
            def _infunc(dim0, dim1):
                if dim0.stop - dim0.start == 0:
                    return y[:, dim1]
                else:
                    m = ((y[:, dim0] - 0.5).sigmoid() > 0.5).float()
                    return m * y[:, dim1] + (1.0 - m) * magic_number
            return torch.cat([_infunc(dim0, dim1) for dim0, dim1 in self.dim['ofeat']], dim=1)
        Y = [_replace_with_magic_number_ofeat(y) for y in Y]
        return Y

    def _append_delta(self, C):
        def _append_delta_ofeat(c):
            return torch.cat([torch.cat([c[:, dim0], append_delta(c[:, dim1])], dim=1) for dim0, dim1 in self.dim['ofeat']], dim=1)
        O = [_append_delta_ofeat(c) for c in C]
        return O

    def _mlpg(self, PG):
        def _mlpg_stream(p, g):
            return torch.cat([torch.cat([g @ p[0][:, dim0], mlpg(p[0][:, dim1], p[1][:, dim1], g)], dim=1) for dim0, dim1 in self.dim['stream']], dim=1)
        Y = [_mlpg_stream(p, g) for p, g in PG]
        return Y

    def _lossnorm(self, C):
        return 1.0 / sum([c.shape[0] * (self.dim['weight'] + self.dim['output']) for c in C])

    def _loss(self, P, C, M):
        O = self._append_delta(C)
        _, O = self._norm(None, O)

        L, _ = generalized_forward_backward(P, O, M, self.max_state_dur, self.dmask_dpdf_mergin, self.temperature, self.weight_s)
        loss = - torch.stack(L).sum()
        return loss

    def _align(self, P, C, M):
        O = self._append_delta(C)
        _, O = self._norm(None, O)

        G = align(P, O, M, self.max_state_dur)
        return G

    def _forward(self, X, A):
        def _viewN(h):
            return h.view(h.shape[0] * self.N, -1)
        def _to_hsmmparam(h):
            wvfloor = 0.0
            bvfloor = 1.0E-1
            dmfloor, dvfloor = 1.0, 1.0E-1
            wm = h[0].sigmoid()
            wv = h[1].exp() + wvfloor
            bm = h[2]
            bv = h[3].exp() + bvfloor
            dm = h[4].exp() + dmfloor
            dv = h[5].exp() + dvfloor
            return wm, wv, bm, bv, dm, dv
        def _change_order(x0, x1):
            y = torch.cat([torch.cat([x0[:, dim0], x1[:, dim1]], dim=1) for dim0, dim1 in self.dim['order']], dim=1)
            return y

        NX, _ = self._norm(X, None)
        if self.mask_x is not None:
            NX = [self.mask_x * nx for nx in NX]

        HA = A
        HA = [self.lha(ha) for ha in HA]

        H = zip(NX, HA)
        H = [torch.cat([nx, ha.expand(len(nx), len(ha))], dim=1) for nx, ha in H]        
        H = [self.lh1(h) for h in H]
        H = [(_viewN(self.lbw(h)), torch.zeros((h.shape[0] * self.N, self.dim['weight']), dtype=h.dtype, device=h.device), _viewN(self.lbm(h)), _viewN(self.lbv(h)), _viewN(self.ldm(h)), _viewN(self.ldv(h))) for h in H]

        P = H
        P = [_to_hsmmparam(p) for p in P]
        P = [(_change_order(wm, bm), _change_order(wv, bv), dm, dv) for (wm, wv, bm, bv, dm, dv) in P]

        return P

    def _set_mask_x(self, mask_x, device):
        if self.mask_x is not None:
            self.lh1[0].weight.data[:, :self.dim['input']] *= self.mask_x
        self.mask_x = torch.from_numpy(mask_x).to(device) if mask_x is not None else None
        print('mask_x :', self.mask_x)

    def _set_dmask_dpdf_mergin(self, dmask_dpdf_mergin):
        self.dmask_dpdf_mergin = dmask_dpdf_mergin
        print('dmask_dpdf_mergin :', self.dmask_dpdf_mergin)

    def _set_temperature(self, temperature):
        self.temperature = temperature
        print('temperature :', self.temperature)

    def _set_weight_s(self, weight_s):
        self.weight_s = weight_s
        print('weight_s :', self.weight_s)

    def set_hyperparameter(self, hp, device):
        print('-- hyper parameters (dnn-hsmm) --')
        if 'mask_x' in hp:
            self._set_mask_x(hp['mask_x'], device)
        if 'dmask_dpdf_mergin' in hp:
            self._set_dmask_dpdf_mergin(hp['dmask_dpdf_mergin'])
        if 'temperature' in hp:
            self._set_temperature(hp['temperature'])
        if 'weight_s' in hp:
            self._set_weight_s(hp['weight_s'])

    def forward(self, X, A, T, M):
        self.train()
        P = self._forward(X, A)
        loss = self._lossnorm(T) * self._loss(P, T, M)
        return loss

    def align(self, X, A, T, M, train=False):
        self.eval()
        with torch.no_grad():
            P = self._forward(X, A)
            G = self._align(P, T, M)
            return G

    def output(self, X, A):
        self.eval()
        with torch.no_grad():
            P = self._forward(X, A)
            G = [predict_dur_dpdf(p, self.max_state_dur) for p in P]
            Y = self._unnorm(P)
            Y = self._mlpg(zip(Y, [bgamma for (bgamma, _, _) in G]))
            Y = self._replace_with_magic_number(Y)
            return Y
