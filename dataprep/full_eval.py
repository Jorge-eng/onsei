import sys, os, glob
TOP_DIR = os.path.dirname(os.path.abspath(__file__))
NET_PATH = os.path.join(TOP_DIR, '../net')
MODEL_PATH = os.path.join(NET_PATH, 'models')
sys.path.append(NET_PATH)

from scipy.io import savemat
from predict_spec import get_arch, get_input
import eval_detector
import numpy as np
import pdb

modelTag = sys.argv[1]
nKw = eval(sys.argv[2]) if len(sys.argv) > 2 else 3

model, modelType, winLen, offset, scale = get_arch(modelTag)

tag = os.path.join(MODEL_PATH, modelTag)
weightFiles = glob.glob(tag+'_ep*.h5')
weightFiles.sort()

testDirs, ths, counts = eval_detector.params()

for wf in weightFiles:
    modelTag = wf.replace('.h5','')

    netOutDir = os.path.join(NET_PATH, 'outputs', os.path.basename(modelTag))
    if not os.path.exists(netOutDir):
        os.makedirs(netOutDir)

    print('Loading '+wf)
    model.load_weights(wf)

    for td in testDirs:
        files = glob.glob(os.path.expanduser(os.path.join('~/keyword', td, 'bin', '*.bin')))
        if len(files)==0:
            raise ValueError('No valid files found in '+inDir)

        for fn in files:
            feaStream = get_input(fn, 'tinyfeats', modelType, offset=offset, scale=scale, winLen=winLen)
            prob = model.predict_proba(feaStream, verbose=1)

            testDir = os.path.join(netOutDir, td)
            if not os.path.exists(testDir):
                os.makedirs(testDir)
            outFile = os.path.join(testDir, os.path.basename(fn)+'.mat')
            print('Saving '+outFile)
            savemat(outFile, {'prob': prob})

    eval_detector.eval_all(netOutDir, nKw=nKw, deleteInput=True)

