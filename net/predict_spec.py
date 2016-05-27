import sys, os
sys.path.append(os.path.abspath('../dataprep'))

import data
from scipy.io import loadmat, savemat
from scipy.io import wavfile
import audioproc
import numpy as np
import pdb

def predict(wavFile, model, modelType, winLen=None, winShift=20):

    if winLen is None:
        winLen = model.input_shape[3]

    feaStream, starts = fbank_stream(wavFile, winLen, winShift, modelType)
    feaStream = (feaStream-7) / 12

    prob = model.predict_proba(feaStream, batch_size=128, verbose=1)

    return prob, starts

def fbank_stream(wavFile, winLen, winShift, modelType):

    logM = audioproc.wav2fbank(wavFile)
    nBands = logM.shape[0]
    nFrames = logM.shape[1]

    starts = np.arange(0, nFrames-winLen+1, winShift)
    nWindows = len(starts)

    if modelType == 'cnn':
        stream = np.zeros((nWindows,1,nBands,winLen),dtype='float32')
        for n, stIdx in enumerate(starts):
            stream[n,0,:,:] = logM[:,stIdx+np.arange(0,winLen)]

        return stream, starts
    elif modelType == 'rnn':
        logM = np.swapaxes(logM, 0, 1)
        stream = np.zeros((nWindows,winLen,nBands),dtype='float32')
        for n, stIdx in enumerate(starts):
            stream[n,:,:] = logM[stIdx+np.arange(0,winLen)]

        return stream, starts

def detect_events(prob, detWinLen=2, detWait=10, detTh=1.5):

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
        signal = np.sum(prob[t-(detWinLen-1):t+1,1])
        if signal >= detTh:
            detect[t] = 1
            waiting = True

    return detect

def detect_online(wav, prob_prev, model, modelType, winLen, detWait=10, detTh=1.5, waitCount=0, waiting=False):

    prob, starts = predict(wav, model, modelType, winLen)
    prob = prob[0, 1]

    detect = np.float32(0)
    if waitCount >= detWait:
        waiting = False
        waitCount = 0
    if waiting:
        waitCount += 1
    else:
        signal = prob + prob_prev
        if signal >= detTh:
            detect = np.float32(1)
            waiting = True

    return detect, prob, waitCount, waiting

def wav2detect(wavFile, model, modelType, winLen, winLen_s=2.0, winShift_s=0.2, detWait_s=2.0, detTh=1.5):

    (fs, wav) = wavfile.read(wavFile)
    assert fs == 16000

    winSamples = int(winLen_s * fs)
    shiftSamples = int(winShift_s * fs)
    detWait = int(detWait_s / winShift_s)
    waitCount = 0
    waiting = False
    prob_prev = 0

    startIdx = 0
    detect = np.array([],dtype='float32')
    while startIdx <= wav.shape[0]-winSamples:
        wav_buffer = wav[range(startIdx,startIdx+winSamples),:]

        ans, prob_prev, waitCount, waiting = detect_online(
                wav_buffer, prob_prev, model, modelType, winLen, detWait, detTh, waitCount, waiting)

        detect = np.append(detect, ans)
        startIdx += shiftSamples


    return detect


wavFile = sys.argv[1]
infoFile = sys.argv[2]

info = loadmat(infoFile)
modelDef = info['modelDef'][0]
modelWeights = info['modelWeights'][0]
modelType = info['modelType'][0]
winLen = info['winLen'][0]

model = data.load_model(modelDef, modelWeights)

prob, startTimes = predict(wavFile, model, modelType, winLen=winLen)

detect = detect_events(prob, detWinLen=2, detWait=10, detTh=1.5)
#detect = wav2detect(wavFile, model, modelType, winLen, winLen_s=2.0, winShift_s=0.2, detTh=1.5)

savemat('prob.mat',{'prob':prob,'startTimes':startTimes,'detect':detect})

