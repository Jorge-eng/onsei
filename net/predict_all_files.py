import sys, os, glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import loadmat, savemat
from predict_spec import get_model, get_input
import numpy as np
import pdb

# Usage:
# $ python predict_all_files.py path/to/inputs path/to/outputs model_name [epoch]

inDir = sys.argv[1]
outDir = sys.argv[2]
if not os.path.exists(outDir):
    os.makedirs(outDir)

files = glob.glob(os.path.join(inDir,'*.wav'))
if len(files) > 0:
    inType = 'audio'
else:
    files = glob.glob(os.path.join(inDir,'*.bin'))
    inType = 'tinyfeats'

if len(files)==0:
    raise ValueError('No valid files found in '+inDir)

modelTag = sys.argv[3]
if len(sys.argv) > 4:
    epoch = int(sys.argv[4])
else:
    epoch = None

model, modelType, winLen, offset, scale = get_model(modelTag, epoch=epoch)

for fn in files:

    feaStream = get_input(fn, inType, modelType, offset=offset, scale=scale, winLen=winLen)

    prob = model.predict_proba(feaStream, verbose=1)

    outFile = os.path.join(outDir, os.path.basename(fn)+'.mat')
    print('Saving '+outFile)
    savemat(outFile, {'prob': prob})

