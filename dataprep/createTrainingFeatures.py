import numpy as np
import itertools
import audioproc as ap
import sys, os, glob
from scipy.io import savemat
import pdb
import matplotlib.pyplot as plt

""" Usage: 
python createTrainingFeatures.py /path/to/audio/clips
"""

def load_features(dirName, matchers):

    # Auto detect source
    if len(glob.glob(os.path.join(dirName,'*.wav'))) > 0:
        feaReader = ap.wav2fbank_batch
    elif len(glob.glob(os.path.join(dirName,'*.bin'))) > 0:
        feaReader = ap.bin2fbank_batch

    features = np.array([])
    labels = np.array([])
    for c, m in enumerate(matchers):
        fea = feaReader(dirName, '*'+m+'*')
        if len(fea) == 0:
            continue
        fea = np.stack(fea, axis=2)
        lab = c * np.ones((fea.shape[2],), dtype='uint8')
        if len(features) == 0:
            features = fea
            labels = lab
        else:
            features = np.append(features, fea, axis=2)
            labels = np.append(labels, lab, axis=0)

    return features, labels

def get_conditions(kws=None):
        
    posConds = ['kwClip',
               ]
     
    negConds = ['kwRevClip',
                'speechClip',
                'backClip',
                'earlyImplantClip','lateImplantClip',
                'partialEarlyClip','partialLateClip',
                'shiftEarlyClip','shiftLateClip',
                 ]

    if kws is not None:
        posMatchers = [p[0]+'_'+p[1] for p in list(itertools.product(posConds, kws))]
        negMatchers = [p[0]+'_'+p[1] for p in list(itertools.product(negConds, kws))]
    else:
        posMatchers = posConds
        negMatchers = negConds

    return posMatchers, negMatchers
 
if __name__ == '__main__':

    # Input audio directory
    dirName = sys.argv[1]
    outName = sys.argv[2]+'.mat'
    if len(sys.argv) > 3:
        kws = sys.argv[3].split('+')
    else:
        kws = None

    posMatchers, negMatchers = get_conditions(kws=kws)

    # Positive examples
    features_pos, labels_pos = load_features(dirName, posMatchers)
    labels_pos = labels_pos + 1 # labels 1, 2, ...

    # Negative examples
    features_neg, labels_neg = load_features(dirName, negMatchers)
    labels_neg = labels_neg * 0 # labels all 0

    savemat(outName,
            {'features_pos': features_pos, 'labels_pos': labels_pos,
             'features_neg': features_neg, 'labels_neg': labels_neg})

