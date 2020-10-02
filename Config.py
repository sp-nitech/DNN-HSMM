import numpy
from module.delta import n_win

# directory
prjdir = '.'
datadir = {'trn' : prjdir + '/data/trn',
           'test': prjdir + '/data/test'}
normfacdir = prjdir + '/normfac'
ckptdir = prjdir + '/ckpt'
gendir = prjdir + '/gen'

# structure
iN, oN = 1, 5
sdim = 2 if iN > 1 else 0
classnums = [1]
embeddims = [0]

# dim
def _sumdim(keys):
    return sum([featdims[key] for key in keys])
featdims = {'lab': 706, 'vuv': 1, 'lf0': 1, 'mgc': 50, 'bap': 25}
dim = {'ofeat': [(slice(0, _sumdim(('vuv',))), slice(_sumdim(('vuv',)), _sumdim(('vuv','lf0')))), (slice(0, 0), slice(_sumdim(('vuv','lf0')), _sumdim(('vuv','lf0','mgc')))), (slice(0, 0), slice(_sumdim(('vuv','lf0','mgc')), _sumdim(('vuv','lf0','mgc','bap'))))],
       'input': featdims['lab'] + sdim}
ofeatdims = {'lf0': slice(0, _sumdim(('lf0',))),
             'mgc': slice(_sumdim(('lf0',)), _sumdim(('lf0','mgc'))),
             'bap': slice(_sumdim(('lf0','mgc')), _sumdim(('lf0','mgc','bap')))}

# norm factors
def _normfac_sidx():
    if iN > 1:
        feats = numpy.arange(iN).astype(numpy.float32)
        mean = numpy.tile(feats.mean(), sdim)
        sdev = numpy.tile(feats.std(), sdim)
        return mean, sdev
    else:
        return numpy.empty(0, numpy.float32), numpy.empty(0, numpy.float32)
mean_sidx, sdev_sidx = _normfac_sidx()
meansdev = numpy.load(normfacdir+'/meansdev.npz')
keys = {'ifeat': ['lab'],
        'ofeat': ['vuv', 'lf0', 'mgc', 'bap']}
normfac = {'ifeat': {'mean': numpy.concatenate([meansdev[k+'.mean'] for k in keys['ifeat']]+[mean_sidx]),
                     'sdev': numpy.concatenate([meansdev[k+'.sdev'] for k in keys['ifeat']]+[sdev_sidx])},
           'ofeat': {'mean': numpy.concatenate([meansdev[k+'.mean'] for k in keys['ofeat']]),
                     'sdev': numpy.concatenate([meansdev[k+'.sdev'] for k in keys['ofeat']])}}

# training
max_state_dur = 500 // (iN * oN)
output_initial_bias = {'bw': numpy.zeros(sum([wdim.stop - wdim.start for wdim, _ in dim['ofeat']]), numpy.float32),
                       'bm': numpy.zeros(n_win * sum([odim.stop - odim.start for _, odim in dim['ofeat']]), numpy.float32),
                       'bv': numpy.full(n_win * sum([odim.stop - odim.start for _, odim in dim['ofeat']]), numpy.log(numpy.array(1.0, numpy.float32))),
                       'dm': numpy.log(numpy.array([20.0 / (iN * oN)], numpy.float32)),
                       'dv': numpy.log(numpy.array([max_state_dur / 2.0], numpy.float32))}
hp = {}
for i in range(20):
    hp['stepM'+str(i).zfill(2)] = {'data': {'batch_size': 5,
                                            'num_workers': 4,
                                            'bmask_data_mergin': None},
                                   'model' : {'mask_x': None,
                                              'temperature': ((i + 1.0) / 20.0) ** 2},
                                   'optimizer': {'lr': 0.0001},
                                   'trn': {'trigger': {'maxiter': 5000,
                                                       'logger': 100,
                                                       'check': 5000}},
                                   'clean': True}
hp['stepF00'] = {'data': {'batch_size': 5,
                          'num_workers': 4,
                          'bmask_data_mergin': None,
                          'keep_load_data': False},
                 'model' : {'mask_x': None,
                            'temperature': 1.0},
                 'optimizer': {'lr': 0.0001},
                 'trn': {'trigger': {'maxiter': 100000,
                                     'logger': 100,
                                     'check': 10000}},
                 'clean': True}

# gpu
device = 'cuda:0'
