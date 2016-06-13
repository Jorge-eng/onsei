import numpy as np
from createTrainingFeatures import load_features as load_features_python
from loadEmbeddedFeatures import load_features as load_features_tinyfeats
from createTrainingFeatures import get_conditions
import sys, os
import glob
from scipy.io import savemat
import pdb

# Input audio directory
dirName = sys.argv[1]
outName = sys.argv[2]

pos, neg = get_conditions()
condMatch = pos + neg

idMatch = ['160517_08','160606_03','160606_04']

if len(glob.glob(os.path.join(dirName,'*.wav'))) > 0:
    load_features = load_features_python
    normalizer = lambda x: (x - 7) / 12
elif len(glob.glob(os.path.join(dirName,'*.bin'))) > 0:
    load_features = load_features_tinyfeats
    normalizer = lambda x: (np.float32(x) + 80) / 140 

identity = []
sampleType = []
features = np.array([])
for i in idMatch:
    for c in condMatch:
        fea = load_features(dirName, [c+'_'+i])

        identity.extend([i]*fea.shape[2])
        sampleType.extend([c]*fea.shape[2])
        if features.shape[0] == 0:
            features = fea
        else:
            features = np.append(features, fea, axis=2)

# Normalization (Todo...)
features = normalizer(features)

savemat(outName, {'features': features, 'identity': identity, 'sampleType': sampleType,
                  'idMatch': idMatch, 'condMatch': condMatch})

