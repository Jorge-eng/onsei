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
conditions = pos + neg

indivs = ['160517_08','160606_03','160606_04']

features = {}
for indiv in indivs:
    features[indiv] = {}
    for cond in conditions:
        features[indiv][cond] = load_features(dirName, [cond+'_'+indiv])

with open(outName, 'wb') as f:
    pickle.dump({'features': features, 'indivs': indivs, 'conditions': conditions}, f)

