from keras.models import model_from_json
from scipy.io import loadmat, savemat
import data
import numpy as np
import pdb

def predict(x, model):

    y = fbank_stack(x)

    prob = model.predict_proba(y, batch_size=128, verbose=1)

    return prob

def fbank_stack(wavFile, winLen, winShift):

    logM = data.wav2fbank(wavFile)
    nBands = logM.shape[0]
    nFrames = logM.shape[1]

    starts = np.arange(0, nFrames-winLen, winShift)
    nWindows = len(starts)

    stack = np.zeros((nWindows,1,nBands,winLen),dtype='float32')
    for n, stIdx in enumerate(starts):
        stack[n,0,:,:] = logM[:,stIdx+np.arange(0,winLen)]

    return stack

#wavFile = '160517_04-2_16k.WAV'
#stack = fbank_stack(wavFile, 199, 20)
#savemat('stackpy.mat',{'stack':stack})


