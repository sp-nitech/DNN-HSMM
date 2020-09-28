#!/usr/bin/python3

import sys, numpy

def main():
    assert (len(sys.argv) - 2) % 4 == 0
    ofname  = sys.argv[-1]
    X = {}
    for key, dim, dtype, fname in [sys.argv[1+4*i:1+4*(i+1)] for i in
    range((len(sys.argv)-2)//4)]:
        if dtype == 'i':
            x = numpy.frombuffer(open(fname, 'rb').read(), numpy.int32)
        elif dtype == 'f':
            x = numpy.frombuffer(open(fname, 'rb').read(), numpy.float32)
        else:
            raise NotImplementedError
        if int(dim) > 0:
            x = x.reshape(-1, int(dim))
        X[key] = x
    numpy.savez_compressed(ofname, **X)

if __name__ == '__main__':
    main()
