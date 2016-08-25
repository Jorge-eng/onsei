import numpy as np
from createTrainingFeatures import load_features, get_conditions
import sys, os
from scipy.io import savemat
import pdb

# Input audio directory
dirName = sys.argv[1]
outName = sys.argv[2]+'.mat'
if len(sys.argv) > 3:
    kws = sys.argv[3].split('+')
else:
    kws = None

pos, neg = get_conditions(kws=kws)
condMatch = pos + neg

idMatch = ['160517_07','160517_08','160606_03','160606_04',
           'VoiceBunny_915641_AFDP9VS','VoiceBunny_916261_97VN760','VoiceBunny_916029_6M3QU0K',  
          ]

identity = []
sampleType = []
features = np.array([])
for i in idMatch:
    for c in condMatch:
        m = '*'+c+'*'+i+'*'
        fea, lab, fn = load_features(dirName, [m])
        if len(fea) == 0:
            continue
        identity.extend([i]*fea.shape[2])
        sampleType.extend([c]*fea.shape[2])
        if features.shape[0] == 0:
            features = fea
        else:
            features = np.append(features, fea, axis=2)

savemat(outName, {'features': features, 'identity': identity, 'sampleType': sampleType,
                  'idMatch': idMatch, 'condMatch': condMatch})

