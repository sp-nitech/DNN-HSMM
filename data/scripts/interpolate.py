import sys, numpy

MAGIC_NUMBER = -1.0E+10

def _lf02iplf0(x):
    T = len(x)
    M = x != MAGIC_NUMBER

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

def main():
    ifname = sys.argv[1]
    x = numpy.load(ifname)['lf0']
    x = _lf02iplf0(x)
    sys.stdout.buffer.write(x)

if __name__ == '__main__':
    main()
