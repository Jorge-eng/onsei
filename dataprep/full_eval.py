import sys, os, glob
TOP_DIR = os.path.dirname(os.path.abspath(__file__))
NET_PATH = os.path.join(TOP_DIR, '../net')
MODEL_PATH = os.path.join(NET_PATH, 'models')
sys.path.append(NET_PATH)

from keras.models import model_from_json
from scipy.io import loadmat, savemat
from predict_spec import get_arch, get_input, predict_stateful
import eval_detector
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import pdb

def file_prob(model, files, modelType, offset, scale, winLen, testDir):
    for fn in files:
        feaStream = get_input(fn, 'tinyfeats', modelType, offset=offset, scale=scale, winLen=winLen)

        typeInfo = modelType.split('_')
        if True:#len(typeInfo) > 1 and typeInfo[1] == 'stateful':
            prob = predict_stateful(model, feaStream, reset_states=True)
        else:
            prob = model.predict_proba(feaStream, verbose=0)

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

def eval_epochs(modelTag):

    posDirs, negDirs, ths, counts = eval_detector.params()
    truePosPts = np.arange(0.05, 1.01, 0.01)

    tag = os.path.join(NET_PATH, 'outputs', os.path.basename(modelTag))
    epochDirs = glob.glob(tag+'_ep*')
    epochDirs.sort()

    falseAlarm = []
    truePos = []
    #falseAlarm = np.zeros((len(counts),len(ths),len(posDirs),len(epochDirs)))
    #truePos = np.zeros((len(counts),len(ths),len(posDirs),len(epochDirs)))

    for ep, epochDir in enumerate(epochDirs):
        posRate = []
        for kw, posDir in enumerate(posDirs):
            dn = os.path.join(epochDir, 'eval_'+posDir.split('/')[0]+'.mat')
            data = loadmat(dn)
            posRate.append(data['num'][:,:,:,kw].sum(axis=0) / data['num'].shape[0])
            #posRate[:,:,kw] = data['num'][:,:,:,kw].sum(axis=0) / data['num'].shape[0]

        faNum = []
        faRate = []
        for j, negDir in enumerate(negDirs):
            dn = os.path.join(epochDir, 'eval_'+negDir.split('/')[0]+'.mat')
            data = loadmat(dn)
            faNum.append([])
            faRate.append([])
            for kw in range(0, data['num'].shape[3]):
                faNum[-1].append(data['num'][:,:,:,kw].sum(axis=0))
                faRate[-1].append(faNum[-1][-1] / (data['tot'].sum()*0.015/60/60))
                #faNum[:,:,kw,j] = data['num'][:,:,:,kw].sum(axis=0)
                #faRate[:,:,kw,j] = faNum[:,:,kw,j] / (data['tot'].sum()*0.015/60/60)

        fa, tp = collapse(faRate, posRate, len(negDirs), len(posDirs), truePosPts)
        falseAlarm.append(fa)
        truePos.append(tp)

    falseAlarm = np.rollaxis(np.array(falseAlarm),0,4)
    truePos = np.rollaxis(np.array(truePos),0,4)

    sortIdx, meanFa = rank_epochs(falseAlarm, truePos, truePosPts, tpBar=0.88, kwWeights=[0.6,0.2,0.2])

    return falseAlarm, truePos

def rank_epochs(falseAlarm, truePos, truePosPts, tpBar=0.88, kwWeights=[0.6,0.2,0.2]):

    tpIdx = np.where(truePosPts >= tpBar)[0][0]
    meanFa = falseAlarm[tpIdx,:,:,:].mean(axis=1)

    sortIdx = np.argsort(np.dot(kwWeights, meanFa))
    #sortIdx = np.argsort(meanFa[0])
    meanFa = meanFa[:,sortIdx]

    return sortIdx, meanFa

def collapse(fa, pos, numNegSets, nKw, truePosPts):

    falseAlarm = np.zeros((len(truePosPts),nKw,numNegSets))
    truePos = np.zeros((len(truePosPts),nKw,numNegSets))

    for kw in range(0, nKw):
        thisPos = pos[kw]
        for negSet in range(0, numNegSets):
            thisFa = fa[negSet][kw]
            for ptIdx, pt in enumerate(truePosPts):
                ii = np.where(thisPos >= pt)
                if len(ii[0]) > 0:
                    minIdx = np.argmin(thisFa[ii])
                    falseAlarm[ptIdx, kw, negSet] = thisFa[ii][minIdx]
                    minIdx = np.where(thisFa[ii]==thisFa[ii][minIdx])
                    truePos[ptIdx, kw, negSet] = np.max(thisPos[ii][minIdx])
                else:
                    falseAlarm[ptIdx, kw, negSet] = np.nan
                    truePos[ptIdx, kw, negSet] = pt

    return falseAlarm, truePos


if __name__ == '__main__':

    # Parallelize from command line:
    # for i in `seq 0 10 240`; do THEANO_FLAGS=device=cpu python full_eval.py modelTag nKw $i 10 & done

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

    posDirs, negDirs, ths, counts = eval_detector.params()
    testDirs = posDirs + negDirs

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

