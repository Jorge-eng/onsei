import numpy as np
import itertools
import audioproc as ap
import sys, os, glob
from scipy.io import savemat
import pdb
import matplotlib.pyplot as plt

""" Usage: 
python createVadTraining.py /path/to/audio/clips
"""

def load_features(dirName, matchers):

    # Auto detect source
    if len(glob.glob(os.path.join(dirName,'*.wav'))) > 0:
        feaReader = ap.wav2fbank_batch
    elif len(glob.glob(os.path.join(dirName,'*.bin'))) > 0:
        feaReader = ap.bin2fbank_batch

    features = np.array([])
    labels = np.array([])
    fileNames = []
    for c, m in enumerate(matchers):
        fea,files = feaReader(dirName, '*'+m+'*')
        if len(fea) == 0:
            continue
        fea = np.stack(fea, axis=2)
        lab = c * np.ones((fea.shape[2],), dtype='uint8')
        if len(features) == 0:
            features = fea
            labels = lab
            fileNames = files
        else:
            features = np.append(features, fea, axis=2)
            labels = np.append(labels, lab, axis=0)
            fileNames.extend(files)

    return features, labels, fileNames

def get_conditions(kws=None):
        
    posConds = ['speechRandomClip']
    negConds = ['backClip']

    if kws is not None:
        posMatchers = [p[0]+'_'+p[1] for p in list(itertools.product(posConds, kws))]
        negMatchers = [p[0]+'_'+p[1] for p in list(itertools.product(negConds, kws))]
    else:
        posMatchers = posConds
        negMatchers = negConds

    return posMatchers, negMatchers
 
def interpret_dist_labels(fileNames, nFr):

    labelsDistributed = np.zeros((len(fileNames),nFr))
    for idx, fn in enumerate(fileNames):
        try:
            # comprised of triplets: class_name,frame_start_pct,frame_end_pct
            anno = np.loadtxt(fn, dtype=str, delimiter=',')
        except:
            print('Could not find '+fn)
            pass
        labVec = np.tile(np.nan,(nFr,))
        
        for k in range(0, len(anno), 3):
            bounds = [np.int(np.float(x)*nFr) for x in anno[k+1:k+3]]
            if anno[k] == 'back':
                labVec[bounds[0]:bounds[1]+1] = 0
            elif anno[k] == 'speech':
                labVec[bounds[0]:bounds[1]+1] = 1
    
        labelsDistributed[idx,:] = labVec        
 
    return labelsDistributed

if __name__ == '__main__':

    # Input audio directory
    dirName = sys.argv[1]
    outName = sys.argv[2]+'.mat'
    timeDistributed = True

    posMatchers, negMatchers = get_conditions()

    # Positive examples
    features_pos, labels_pos, fn_pos = load_features(dirName, posMatchers)
    # Negative examples
    features_neg, labels_neg, fn_neg = load_features(dirName, negMatchers)

    labels_pos = labels_pos + 1 # labels 1, 2, ...
    labels_neg = labels_neg * 0 # labels all 0
    if timeDistributed is True:
        fn_pos = [str.split(x,'.wav')[0]+'.csv' for x in fn_pos]
        fn_neg = [str.split(x,'.wav')[0]+'.csv' for x in fn_neg]
        labels_pos = interpret_dist_labels(fn_pos, features_pos.shape[1])
        labels_neg = interpret_dist_labels(fn_neg, features_neg.shape[1])

    savemat(outName,
            {'features_pos': features_pos, 'labels_pos': labels_pos,
             'features_neg': features_neg, 'labels_neg': labels_neg})

