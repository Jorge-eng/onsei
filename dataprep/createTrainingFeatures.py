import numpy as np
import audioproc as ap
import sys
from scipy.io import savemat
import pdb
import matplotlib.pyplot as plt

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

def get_conditions():

    posMatchers = ['kwClip']
     
    negMatchers = ['kwRevClip',
                   'speechClip',
                   'backClip',
                   'earlyImplantClip','lateImplantClip',
                   'partialEarlyClip','partialLateClip',
                   'shiftEarlyClip','shiftLateClip',
                   ]

    return posMatchers, negMatchers
 
if __name__ == '__main__':

    # Input audio directory
    dirName = sys.argv[1]

    posMatchers, negMatchers = get_conditions()

    # Positive examples
    features = load_features(dirName, posMatchers)

    if False:
        tilt = tilt_compensation(features)
        features = features - tilt[:,np.newaxis,np.newaxis]
        pdb.set_trace()

    savemat('spec_pos.mat',{'features': features})

    # Negative examples
    features = load_features(dirName, negMatchers)
    if False:
        features = features - tilt[:,np.newaxis,np.newaxis]

    savemat('spec_neg.mat',{'features': features})

