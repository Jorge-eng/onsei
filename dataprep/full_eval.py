import sys, os, glob
TOP_DIR = os.path.dirname(os.path.abspath(__file__))
NET_PATH = os.path.join(TOP_DIR, '../net')
MODEL_PATH = os.path.join(NET_PATH, 'models')
sys.path.append(NET_PATH)

from keras.models import model_from_json
from scipy.io import loadmat, savemat
from predict_spec import get_arch, get_input
import eval_detector
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import pdb

def file_prob(model, files, modelType, offset, scale, winLen, testDir):
    for fn in files:
        feaStream = get_input(fn, 'tinyfeats', modelType, offset=offset, scale=scale, winLen=winLen)
        prob = model.predict_proba(feaStream, verbose=1)

        outFile = os.path.join(testDir, os.path.basename(fn)+'.mat')
        print('Saving '+outFile)
        savemat(outFile, {'prob': prob})

def chunks(l, n):
    z = int(np.ceil(np.float(len(l)) / n))
    for i in xrange(0, len(l), z):
        yield l[i:i + z]

def get_model_info(modelTag):

    infoFile = os.path.join(MODEL_PATH, modelTag+'.mat')
    info = loadmat(infoFile)

    modelDef = os.path.join(NET_PATH, info['modelDef'][0])
    modelType = info['modelType'][0]
    winLen = int(info['winLen'][0])
    offset = info['offset'][0]
    scale = info['scale'][0]

    return modelDef, modelType, winLen, offset, scale

def compile_model(modelDef):
    return model_from_json(open(modelDef).read())

nWorkers = np.minimum(1, multiprocessing.cpu_count())
sys.setrecursionlimit(10000)

modelTag = sys.argv[1]
nKw = eval(sys.argv[2]) if len(sys.argv) > 2 else 3

modelDef, modelType, winLen, offset, scale = get_model_info(modelTag)

#from multiprocessing import Process
#pids = [Process(target=compile_model, args=(modelDef,)) for _ in range(nWorkers)]
#[p.start() for p in pids]
#[p.join() for p in pids]

models = Parallel(n_jobs=nWorkers)(delayed(compile_model)(modelDef) for _ in range(nWorkers))

tag = os.path.join(MODEL_PATH, modelTag)
weightFiles = glob.glob(tag+'_ep*.h5')
weightFiles.sort()
startEpoch = eval(sys.argv[3]) if len(sys.argv) > 3 else 0
nEpoch = eval(sys.argv[4]) if len(sys.argv) > 4 else len(weightFiles)

weightFiles = weightFiles[startEpoch:startEpoch+nEpoch]

testDirs, ths, counts = eval_detector.params()

for wf in weightFiles:
    modelTag = wf.replace('.h5','')

    netOutDir = os.path.join(NET_PATH, 'outputs', os.path.basename(modelTag))
    if not os.path.exists(netOutDir):
        os.makedirs(netOutDir)

    print('Loading '+wf)
    for model in models:
        model.load_weights(wf)

    for td in testDirs:
        files = glob.glob(os.path.expanduser(os.path.join('~/keyword', td, 'bin', '*.bin')))
        if len(files)==0:
            raise ValueError('No valid files found in '+inDir)

        testDir = os.path.join(netOutDir, td)
        if not os.path.exists(testDir):
            os.makedirs(testDir)

        #pids = [Process(target=file_prob, args=(model, fn, modelType, offset, scale, winLen, testDir)) for model, fn in zip(models, chunks(files, nWorkers))]
        #[p.start() for p in pids]
        #[p.join() for p in pids]

        Parallel(n_jobs=nWorkers)(delayed(file_prob)(model, fn, modelType, offset, scale, winLen, testDir) for model, fn in zip(models, chunks(files, nWorkers)))

    eval_detector.eval_all(netOutDir, nKw=nKw, deleteInput=True)

