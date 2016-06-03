import numpy as np
import audioproc as ap
import sys, os
from scipy.io import savemat
import glob
import pdb

def load_features(dirName, matchers):

    for c, m in enumerate(matchers):
        fea = np.stack(bin2fbank_batch(dirName, m), axis=2)
        if c == 0:
            features = fea
        else:
            features = np.append(features, fea, axis=2)

    return features

def find_bin_files(wavDir, matcher=None):

    # Find all audio files in the directory
    allFiles = glob.glob(os.path.join(wavDir,'*.bin'))
    binFiles = []
    for f in allFiles:
        if matcher is not None and matcher not in f:
            continue
        binFiles.append(f)

    return binFiles

def load_bin(fname, nfilt=40):

    f = open(fname,'r')
    x = np.fromfile(f,dtype=np.int8)
    f.close()

    x = x.reshape(x.shape[0] / nfilt, nfilt)
    x = np.swapaxes(x, 0, 1)

    return x

def bin2fbank_batch(dirName, matcher=None):

    files = find_bin_files(dirName, matcher)
    logM = []
    for f in files:
        logM.append(load_bin(f))

    return logM

# Input audio directory
dirName = sys.argv[1]

# Positive examples
posMatchers = ['kwClip']
features = load_features(dirName, posMatchers)
savemat('spec_pos.mat',{'features': features})

# Negative examples
negMatchers = ['kwRevClip','speechClip','backClip','earlyImplantClip','lateImplantClip','partialEarlyClip','partialLateClip']
features = load_features(dirName, negMatchers)
savemat('spec_neg.mat',{'features': features})

