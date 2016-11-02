import sys, os
TOP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TOP_DIR, 'models')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import glob
from scipy.io import loadmat, savemat
from predict_spec import get_arch, get_input
import numpy as np
import pdb

# Usage:
# $ python predict_all_epochs.py audio in.wav out model_name
# $ python predict_all_epochs.py features in.mat out model_name
# $ python predict_all_epochs.py tinyfeats in.bin out model_name

inType = sys.argv[1]
inFile = sys.argv[2]
outFile = sys.argv[3]+'.mat'
modelTag = sys.argv[4]

model, modelType, winLen, offset, scale = get_arch(modelTag)

tag = os.path.join(MODEL_PATH, modelTag)
weightFiles = glob.glob(tag+'_ep*.h5')
weightFiles.sort()

feaStream = get_input(inFile, inType, modelType, offset=offset, scale=scale, winLen=winLen)

prob = []
for wf in weightFiles:
    print(wf)
    model.load_weights(wf)
    p = model.predict_proba(feaStream, batch_size=128, verbose=1)
    prob.append(np.float32(p))

print('Saving '+outFile)
prob = np.array(prob)
savemat(outFile, {'prob': prob})

