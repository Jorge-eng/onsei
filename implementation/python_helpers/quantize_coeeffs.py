#!/usr/bin/python
import h5py
import sys
import numpy as np

fname = sys.argv[1]

f = h5py.File(fname,'r+')
#nb_layers = f.attrs['nb_layers']

keys = []
for layer_key in f:
    group = f[layer_key]

    for key in group:
        big_key = '%s/%s' % (layer_key,key)
        keys.append(big_key)
        
vals = {}
for key in keys:
    w = f[key]
    print w.value.shape
    val = w.value
    val2 = np.round(val*127) / 127.
    vals[key] = val2
    w[...] = val2
f.close()

f = h5py.File(fname,'r')
for key in keys:
    np.allclose(f[key].value,vals[key])
f.close()

#f = h5py.File(fname,'r')
