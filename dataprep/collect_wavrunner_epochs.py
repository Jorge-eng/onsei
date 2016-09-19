import sys
import glob
from scipy.io import savemat
import numpy as np
import pdb

"""
python collect_wavrunner_epochs.py inMatcher outFile 
"""

inMatcher = sys.argv[1]
outFile = sys.argv[2]+'.mat'

allFiles = glob.glob(inMatcher+'_ep*.csv')
allFiles.sort()

prob = []
for fn in allFiles:
    data = np.genfromtxt(fn, delimiter=',')
    prob.append(data)

prob = np.rollaxis(np.array(prob), 0, 3)
savemat(outFile, {'prob':prob})

