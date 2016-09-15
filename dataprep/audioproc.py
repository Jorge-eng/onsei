import os, glob, sndhdr
import numpy as np
from scipy.io import wavfile
from features import fbank, mfcc # python_speech_features
import fnmatch
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

def wav2fbank(wavFile, fs=16000, maxLen_s=None):

    if isinstance(wavFile, str):
        (fs, wav) = wavfile.read(wavFile)
        assert fs == 16000 # requirement for now
    elif isinstance(wavFile, np.ndarray):
        wav = wavFile

    winlen = 0.025
    winstep = 0.015
    nfft = np.int(np.power(2, np.ceil(np.log2(winlen*fs))))
    winfunc = lambda x: np.hanning(x)
    nfilt = 40
    preemph = 0.97

    if np.ndim(wav) == 2: # Multiple channels; just take left one
        wav = wav[:,0]
    if maxLen_s is not None:
        maxSamp = maxLen_s * fs
        wav = wav[:maxSamp]
  
    if True:
        M, E = fbank(wav, fs, winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft, winfunc=winfunc, preemph=preemph)

        logM = np.log(M)
    else:
        logM = mfcc(wav, fs, numcep=16, winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft, winfunc=winfunc, preemph=preemph)
    logM = np.swapaxes(logM, 0, 1)

    return logM

def find_audio_files(wavDir, matcher=None):

    # Find all audio files in the directory
    allFiles = glob.glob(os.path.join(wavDir,'*.*'))
    audioFiles = []
    for f in allFiles:
        if matcher is not None and not fnmatch.fnmatch(f, matcher):
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

    return logM, audioFiles

def find_bin_files(wavDir, matcher=None):

    # Find all bin files in the directory
    allFiles = glob.glob(os.path.join(wavDir,'*.bin'))
    binFiles = []
    for f in allFiles:
        if matcher is not None and not fnmatch.fnmatch(f, matcher):
            continue
        binFiles.append(f)

    return binFiles

def load_bin(fname, nfilt=40):

    f = open(fname,'r')
    x = np.fromfile(f,dtype=np.int16)
    f.close()

    x = x.reshape(x.shape[0] / nfilt, nfilt)
    x = np.swapaxes(x, 0, 1)

    return x

def bin2fbank_batch(dirName, matcher=None):

    files = find_bin_files(dirName, matcher)
    logM = []
    for f in files:
        logM.append(load_bin(f))

    return logM, files

