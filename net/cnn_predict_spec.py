import data
from scipy.io import loadmat, savemat
import audiproc
import numpy as np
import sys
import pdb

def predict(wavFile, model):

    pdb.set_trace()
    feaStream = fbank_stream(wavFile, 199, 20)
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

    return stream


wavFile = sys.argv[1]
infoFile = sys.argv[2]

info = loadmat(infoFile)
modelDef = info['modelDef'][0]
modelWeights = info['modelWeights'][0]

model = data.load_model(modelDef, modelWeights)

prob = predict(wavFile, model)

savemat('prob.mat',{'prob':prob})

