import sys, os
sys.path.append(os.path.abspath('../dataprep'))

import data
from scipy.io import loadmat, savemat
import audioproc
import numpy as np
import pdb

def predict(wavFile, model, winShift=20):

    winLen = model.input_shape[3]

    feaStream, starts = fbank_stream(wavFile, winLen, winShift)
    feaStream = (feaStream-7) / 12

    prob = model.predict_proba(feaStream, batch_size=128, verbose=1)

    return prob, starts

def fbank_stream(wavFile, winLen, winShift):

    logM = audioproc.wav2fbank(wavFile)
    nBands = logM.shape[0]
    nFrames = logM.shape[1]

    starts = np.arange(0, nFrames-winLen, winShift)
    nWindows = len(starts)

    stream = np.zeros((nWindows,1,nBands,winLen),dtype='float32')
    for n, stIdx in enumerate(starts):
        stream[n,0,:,:] = logM[:,stIdx+np.arange(0,winLen)]

    return stream, starts

def detect_events(prob, detWinLen=2, detWait=10, detTh=0.5):

    detect = np.zeros((prob.shape[0],), dtype='float32')

    waiting = False
    waitCount = 0
    for t in range(prob.shape[0]):
        if t < detWinLen-1:
            continue
        if waitCount >= detWait:
            waiting = False
            waitCount = 0
        if waiting:
            waitCount += 1
            continue
        signal = np.sum(prob[t-(detWinLen-1):t,1])
        if signal >= detTh:
            detect[t] = 1
            waiting = True

    return detect


wavFile = sys.argv[1]
infoFile = sys.argv[2]

info = loadmat(infoFile)
modelDef = info['modelDef'][0]
modelWeights = info['modelWeights'][0]

model = data.load_model(modelDef, modelWeights)

prob, startTimes = predict(wavFile, model)

detect = detect_events(prob, detWinLen=2, detWait=10, detTh=0.5)

savemat('prob.mat',{'prob':prob,'startTimes':startTimes,'detect':detect})

