import numpy as np
import audioproc as ap
from createTrainingFeatures import get_conditions
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

if __name__ == '__main__':

    # Input audio directory
    dirName = sys.argv[1]

    posMatchers, negMatchers = get_conditions()

    # Positive examples
    features = load_features(dirName, posMatchers)
    features = ((np.float32(features) + 80) / 140 * 12) + 7
    savemat('spec_pos.mat',{'features': features})

    # Negative examples
    features = load_features(dirName, negMatchers)
    features = ((np.float32(features) + 80) / 140 * 12) + 7
    savemat('spec_neg.mat',{'features': features})

