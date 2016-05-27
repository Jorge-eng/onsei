import os, glob, sndhdr
import numpy as np
from scipy.io import wavfile
from features import fbank # python_speech_features
import pdb

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

def wav2fbank(wavFile, fs=16000):

    if isinstance(wavFile, str):
        (fs, wav) = wavfile.read(wavFile)
        assert fs == 16000 # requirement for now
    elif isinstance(wavFile, np.ndarray):
        wav = wavFile

    winlen = 0.025
    winstep = 0.010
    nfft = np.int(np.power(2, np.ceil(np.log2(winlen*fs))))
    winfunc = lambda x: np.hanning(x)
    nfilt = 40

    if np.ndim(wav) == 2: # Multiple channels; just take left one
        wav = wav[:,0]

    M, E = fbank(wav, fs, winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft, winfunc=winfunc)

    logM = np.log(M)
    logM = np.swapaxes(logM, 0, 1)

    return logM

def find_audio_files(wavDir, matcher=None):

    # Find all audio files in the directory
    allFiles = glob.glob(os.path.join(wavDir,'*.*'))
    audioFiles = []
    for f in allFiles:
        if matcher is not None and matcher not in f:
            continue
        chk = sndhdr.what(f)
        if chk is not None:
            audioFiles.append(f)

    return audioFiles

def wav2fbank_batch(wavDir, matcher=None):

    audioFiles = find_audio_files(wavDir, matcher)
    logM = []
    for f in audioFiles:
        logM.append(wav2fbank(f))

    return logM

