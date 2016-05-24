import sys, os
sys.path.append(os.path.abspath('../dataprep'))

import data
from scipy.io import loadmat, savemat
import audioproc
import numpy as np
import pdb

def predict(wavFile, model, winShift=20):

    winLen = model.input_shape[3]

    feaStream = fbank_stream(wavFile, winLen, winShift)
    feaStream = (feaStream-7) / 12

    prob = model.predict_proba(feaStream, batch_size=128, verbose=1)

    return prob

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


wavFile = sys.argv[1]
infoFile = sys.argv[2]

info = loadmat(infoFile)
modelDef = info['modelDef'][0]
modelWeights = info['modelWeights'][0]

model = data.load_model(modelDef, modelWeights)

prob, startTimes = predict(wavFile, model)

savemat('prob.mat',{'prob':prob,'startTimes':startTimes})

