import os, glob, sndhdr
import json
import base64
import numpy as np
from scipy.io import wavfile, savemat
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

def append_padded(logM, data, targetSize=None):

    if targetSize is not None:
        if data.shape[1] < targetSize:
            data = np.concatenate((data, np.zeros((data.shape[0], targetSize-data.shape[1]))), axis=1)
        elif data.shape[1] > targetSize:
            data = data[:,:(targetSize-data.shape[1])]

    logM.append(data)

    return logM

def bin2fbank_batch(dirName, matcher=None, targetSize=None):

    files = find_bin_files(dirName, matcher)
    logM = []
    for f in files:
        data = load_bin(f)
        logM = append_padded(logM, data, targetSize=targetSize)

    return logM, files

def bin2mat(inFile, outFile=None):

    if outFile is None:
        outFile = inFile.replace('.bin','.mat')

    features = load_bin(inFile)
    savemat(outFile, {'features': features})

def bin2mat_all(dirName):

    files = find_bin_files(dirName)
    for f in files:
        bin2mat(f)

def bin2mat_batch(dirName, outFile, matcher=None, targetSize=None):

    logM, files = bin2fbank_batch(dirName, matcher=matcher, targetSize=targetSize)
    logM = np.array(logM)
    savemat(outFile, {'logM':logM, 'files':files})

def load_serverfeats(fname):

    f = open(fname, 'r')

    s = f.read()
    f.close()
    datas = s.split("}")

    feats = []
    for data in datas:
        if len(data) <= 2:
            continue

        tmp = data + "}"
        jdata = json.loads(tmp)

        if not '+okay' in jdata['id']:
            continue

        n = jdata['num_cols']
        bindata = base64.b64decode(jdata['payload'])
        x = np.fromstring(bindata,dtype=np.int8)
        istart = x.shape[0] % n
        x2 = x[istart::].reshape((x.shape[0]/n,n))
        feats.append(x2)

    feats = np.array(feats)
    return feats

def serverfeats2mat(inFile, outFile, targetSize=None):

    feats = load_serverfeats(inFile)
    logM = []
    for data in feats:
        logM = append_padded(logM, data.swapaxes(0, 1), targetSize=targetSize)

    logM = np.array(logM)
    savemat(outFile, {'logM':logM})

