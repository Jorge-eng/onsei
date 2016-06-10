import numpy as np
import audioproc as ap
from createTrainingFeatures import load_features, get_conditions
import sys
from scipy.io import savemat
import pickle
import pdb

# Input audio directory
dirName = sys.argv[1]
outName = sys.argv[2]

pos, neg = get_conditions()
condMatch = pos + neg

idMatch = ['160517_08','160606_03','160606_04']

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

savemat(outName, {'features': features, 'identity': identity, 'sampleType': sampleType})

