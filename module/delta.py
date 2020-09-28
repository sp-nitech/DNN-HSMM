import scipy.linalg, torch

delta_window = torch.tensor([[ 0.0,  1.0,  0.0],
                             [-0.5,  0.0,  0.5],
                             [ 1.0, -2.0,  1.0]], dtype=torch.float32)
n_win, width_win = delta_window.shape
assert width_win % 2 == 1

def append_delta(c, pad=True):
    dwin = delta_window.to(c.device)
    T, D = c.shape

    o = c
    if pad and ((width_win-1)//2 > 0):
        o = torch.cat([o[:(width_win-1)//2], o, o[-((width_win-1)//2):]], dim=0)
    else:
        T = T - 2 * ((width_win-1)//2)
    o = dwin @ torch.cat([o.view(1, -1)] + [torch.cat([o[w:], o[-w:]], dim=0).view(1, -1) for w in range(1, width_win)], dim=0)[:, :T*D]
    o = torch.cat([o[n].view(-1, D) for n in range(n_win)], dim=1)

    return o

def mlpg(mean, sdev, gamma=None):
    U = 1.0 / sdev.square()
    MU = mean * U

    if gamma is not None:
        MU = gamma @ MU
        U = gamma @ U

    dtype, device = MU.dtype, MU.device
    dwin = delta_window.to(device)
    T, D = MU.shape[0], MU.shape[1] // n_win

    W = dwin.repeat((T, 1))
    WMU = dwin.T @ torch.cat([MU[:, n*D:(n+1)*D].reshape(1, -1) for n in range(n_win)], dim=0)
    WMU = torch.stack([torch.cat([torch.zeros((w, D), dtype=dtype, device=device), WMU[w].view(-1, D), torch.zeros((width_win-1-w, D), dtype=dtype, device=device)], dim=0) for w in range(width_win)]).sum(dim=0)
    U = U.view(T*n_win, -1)

    C = []
    for d in range(D):
        WU = W * U[:, d:d+1]
        WUW_b = torch.stack([WU * W] + [torch.cat([WU[:, :-w] * W[:, w:], torch.zeros((W.shape[0], w), dtype=dtype, device=device)], dim=1) for w in range(1, width_win)])
        WUW_b = WUW_b.view(width_win, -1, n_win, width_win).sum(dim=2)
        WUW_b = torch.stack([torch.cat([torch.zeros((width_win, w), dtype=dtype, device=device), WUW_b[:, :, w], torch.zeros((width_win, width_win-1-w), dtype=dtype, device=device)], dim=1) for w in range(width_win)]).sum(dim=0)
        WUW_b = WUW_b.detach().cpu().numpy()
        WUW_cb = scipy.linalg.cholesky_banded(WUW_b, lower=True)
        c = scipy.linalg.cho_solve_banded((WUW_cb, True), WMU[:, d:d+1].detach().cpu().numpy())
        c = torch.from_numpy(c).to(device)
        C.append(c)
    C = torch.cat(C, dim=1)
    return C[(width_win-1)//2:-(width_win-1)//2]
