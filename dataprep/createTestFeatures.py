import numpy as np
import audioproc as ap
from createTrainingFeatures import load_features
import sys
from scipy.io import savemat
import pdb

""" Usage: 
python createTrainingFeatures.py /path/to/audio/clips
"""

if __name__ == '__main__':

    # Input audio directory
    dirName = sys.argv[1]
    outName = sys.argv[2]
    matchers = sys.argv[3:]

    features = load_features(dirName, matchers)
    savemat(outName,{'features': features})

