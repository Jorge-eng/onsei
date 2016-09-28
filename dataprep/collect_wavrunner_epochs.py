import sys, os, glob
from scipy.io import loadmat, savemat
import numpy as np
from createTrainingFeatures import get_conditions
import pdb

"""
python collect_wavrunner_epochs.py inDir outFile
"""

inDir = sys.argv[1]
outFile = sys.argv[2]+'.mat'
if len(sys.argv) > 3:
    kws = sys.argv[3].split('+')
else:
    kws = None

allFiles = glob.glob(os.path.join(inDir,'*.mat'))
allFiles.sort()

prob = []
for fn in allFiles:
    data = loadmat(fn)['prob']
    prob.append(data)

inFiles = loadmat(allFiles[0])['allFiles']
pos, neg = get_conditions(kws=kws)
condMatch = pos + neg
sampleType = []
for c in condMatch:
    matched = [c in x for x in inFiles]
    if matched.count(True) > 0:
        sampleType.extend([c]*matched.count(True))
#condMatch = list(set(sampleType))

#prob = np.rollaxis(np.array(prob), 0, 3)
prob = np.array(prob)
savemat(outFile, {'prob':prob,'condMatch':condMatch,'sampleType':sampleType})

