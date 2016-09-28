import sys, os, glob
from scipy.io import savemat
import numpy as np
import pdb

"""
python collect_wavrunner_files.py inDir outFile 
"""

inDir = sys.argv[1]
outFile = os.path.join(inDir,sys.argv[2]+'.mat')

allFiles = glob.glob(os.path.join(inDir,'*.csv'))
allFiles.sort()

prob = []
for fn in allFiles:
    data = np.genfromtxt(fn, delimiter=',')
    prob.append(data)

#prob = np.rollaxis(np.array(prob), 0, 3)
prob = np.array(prob)
savemat(outFile, {'prob':prob,'allFiles':allFiles})

