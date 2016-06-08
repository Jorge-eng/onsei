import numpy as np
import audioproc as ap
import sys
from scipy.io import savemat
import pdb

""" Usage: 
python createTrainingFeatures.py /path/to/audio/clips
"""

def load_features(dirName, matchers):

    for c, m in enumerate(matchers):
        fea = np.stack(ap.wav2fbank_batch(dirName, m), axis=2)
        if c == 0:
            features = fea
        else:
            features = np.append(features, fea, axis=2)

    return features


# Input audio directory
dirName = sys.argv[1]
outName = sys.argv[2]
matchers = sys.argv[3:]

features = load_features(dirName, matchers)
savemat(outName,{'features': features})

