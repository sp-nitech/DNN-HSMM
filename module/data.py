import numpy, torch
from torch.utils.data import Dataset

from .utils import magic_number

class DataSet(Dataset):
    def __init__(self, fnames, N, mergin, keep_data):
        self.fnames = fnames
        self.N = N
        self.mergin = mergin
        self.keep_data = keep_data
        self.D = {}

    def __getitem__(self, idx):
        def _trim(D):
            mn = min([len(d) for d in D])
            return [d[:mn] for d in D]

        def _add_state_idx(x):
            if self.N > 1:
                I = numpy.arange(self.N).astype(numpy.float32)
                I = numpy.tile(numpy.vstack([I, I[::-1]]), x.shape[0]).T
                return numpy.hstack([numpy.tile(x, self.N).reshape(-1, x.shape[1]), I])
            else:
                return x

        def _lf02vuv(x):
            return (x != magic_number).astype(numpy.float32)

        def _lf02iplf0(x):
            T = len(x)
            M = x != magic_number

            idxes = numpy.where(M)[0]
            if len(idxes) == 0:
                return x

            s, sv, e = 0, x[idxes[0]], 0
            while True:
                idxes = numpy.where(M[s:])[0]
                if len(idxes) > 0:
                    e = s + idxes[0]
                    ev = x[e]
                else:
                    e = T
                    ev = sv

                x[s:e] = numpy.linspace(sv, ev, e-s+2)[1:-1].reshape(-1, 1)

                idxes = numpy.where(numpy.logical_not(M[e:]))[0]
                if len(idxes) > 0:
                    s = e + idxes[0]
                    sv = x[s - 1]
                else:
                    return x

        def _make_bmask(dur, N, T, mergin=None):
            if mergin is None:
                return None

            dur = dur.astype(numpy.int32)
            idxes = numpy.cumsum(numpy.concatenate([numpy.array([0], numpy.int32), dur]))
            idxes = [(idxes[i], idxes[i+1]) for i in range(len(idxes)-1)]
            idxes[-1] = (idxes[-1][0], T)
            if mergin['phn'] > 0:
                I = len(idxes)
                idxes = [(idxes[max(0, i-mergin['phn'])][0], idxes[min(I-1, i+mergin['phn'])][1]) for i in range(I)]
            if mergin['frm'] > 0:
                idxes = [(max(0, s-mergin['frm']), min(T, e+mergin['frm'])) for s, e in idxes]
            for i in range(len(idxes)):
                if idxes[i][1] - idxes[i][0] < N:
                    idxes[i] = (max(0, idxes[i][0]-(N-(idxes[i][1]-idxes[i][0]))), min(idxes[i][1]+(N-(idxes[i][1]-idxes[i][0])), T))
            mask = numpy.vstack([numpy.tile(numpy.hstack([numpy.zeros(s, numpy.bool), numpy.ones(e-s, numpy.bool), numpy.zeros(T-e, numpy.bool)]), (N, 1)) for s, e in idxes]).T
            return mask

        def _mask_dmask():
            return None

        fname = self.fnames[idx]
        if fname in self.D:
            X, A, T, M = self.D[fname]
        else:
            D = numpy.load(fname, allow_pickle=True)
            X = X = _add_state_idx(D['lab'])
            A = D['incode'].astype(numpy.int64) if 'incode' in D else numpy.array([0], numpy.int64)
            T = numpy.hstack(_trim([_lf02vuv(D['lf0']), _lf02iplf0(D['lf0']), D['mgc'], D['bap']]))
            M = (_make_bmask(D['dur_phn'], self.N, T.shape[0], self.mergin) if 'dur_phn' in D else None, _mask_dmask())
            if self.keep_data:
                self.D[fname] = (X, A, T)

        X, A, T, M = torch.from_numpy(X), torch.from_numpy(A), torch.from_numpy(T), (torch.from_numpy(M[0]) if M[0] is not None else None, torch.from_numpy(M[1]) if M[1] is not None else None)

        return X, A, T, M

    def __len__(self):
        return len(self.fnames)

class DataCollate(object):
    def __call__(self, batch):
        X, A, T, M = map(lambda i: [b[i] for b in batch], range(len(batch[0])))
        return X, A, T, M
