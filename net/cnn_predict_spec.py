from keras.models import model_from_json
from features import fbank # python_speech_features
from scipy.io import loadmat, savemat, wavfile
import numpy as np
import pdb

def load_model(defFile, weightFile):

    model = model_from_json(open(defFile).read())
    model.load_weights(weightFile)

    return model

def predict(x, model):

    y = fbank_stack(x)

    prob = model.predict_proba(y, batch_size=128, verbose=1)

    return prob

def melfilter(N, freq):
    melFreq = 2595*np.log10(1+freq/700)

    maxF = np.max(melFreq)
    minF = np.min(melFreq)
    melBinWidth = (maxF-minF)/(N+1)
    filt = np.zeros((N, len(freq)), dtype='float32')

    for n in range(0, N):
        idx = np.where( melFreq>=(n*melBinWidth+minF) & melFreq<=((n+2)*melBinWidth+minF) )
        binWidth = len(idx)

        filt[n, idx] = np.bartlett(binWidth)

    return filt

def fbank_stack(wavFile, winLen, winShift):

    logM = wav2fbank(wavFile)
    nBands = logM.shape[0]
    nFrames = logM.shape[1]

    starts = np.arange(0, nFrames-winLen, winShift)
    nWindows = len(starts)

    stack = np.zeros((nWindows,1,nBands,winLen),dtype='float32')
    pdb.set_trace()
    for n, stIdx in enumerate(starts):
        stack[n,0,:,:] = logM[:,stIdx+np.arange(0,winLen)]

    return stack

def wav2fbank(wavFile):

    (fs, wav) = wavfile.read(wavFile)
    assert fs == 16000 # requirement for now

    winlen = 0.025
    winstep = 0.010
    nfft = np.int(np.power(2, np.ceil(np.log2(winlen*fs))))
    winfunc = lambda x: np.hanning(x)
    nfilt = 40

    M, E = fbank(wav[:,0], fs, winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft, winfunc=winfunc)

    logM = np.log(M)
    logM = np.swapaxes(logM, 0, 1)

    return logM

#wavFile = '160517_04-2_16k.WAV'
#stack = fbank_stack(wavFile, 199, 20)
#savemat('stackpy.mat',{'stack':stack})

#logM = wav2fbank(wavFile)
#savemat('logMpy.mat',{'logM':logM})

