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
 
def time_distribute_labels(labels_pos, features_pos, labels_neg, nFr, Fs=16000, frameSamples=240):
    
    posMatchers, negMatchers = get_conditions()
    maxLab = np.max(labels_pos)

    # Positive
    labelsPosDistributed = np.zeros((len(labels_pos),nFr))
    for idx in range(features_pos.shape[2]):
        labVec = np.tile(np.nan,(nFr,))
        f = features_pos[:,:,idx] 
        
        fSum = f.sum(axis=0)
        iMask = np.where(fSum > fSum.min() + 0.5*(fSum.max()-fSum.min()))[0]
        iMask = iMask[np.random.permutation(len(iMask))[:np.minimum(5,len(iMask))]]
        labVec[iMask] = 1
        labelsPosDistributed[idx,:] = labVec        
        
    nPosLabels = int(labelsPosDistributed[~np.isnan(labelsPosDistributed)].sum())
    nNegLabelsPer = int(nPosLabels / labels_neg.shape[0])

    pdb.set_trace()
    # Negative 
    labelsNegDistributed = np.zeros((len(labels_neg),nFr))
    for idx in range(labels_neg.shape[0]):
        labVec = np.tile(np.nan,(nFr,))
        randIdx = np.random.permutation(nFr)[:nNegLabelsPer]
        labVec[randIdx] = 0
        labelsNegDistributed[idx,:] = labVec        
        
    return labelsPosDistributed, labelsNegDistributed

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
        labels_pos, labels_neg = time_distribute_labels(labels_pos, features_pos, labels_neg, features_pos.shape[1])

    savemat(outName,
            {'features_pos': features_pos, 'labels_pos': labels_pos,
             'features_neg': features_neg, 'labels_neg': labels_neg})

