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

    posConds = ['kwClip',
               ]

    negConds = ['kwRevClip',
                'speechAlignedClip','speechRandomClip',
                'backClip',
                'earlyImplantClip','lateImplantClip',
                'partialEarlyClip','partialLateClip',
                'shiftEarlyClip','shiftLateClip',
                'falseAlarmClip',
                 ]

    if kws is not None:
        posMatchers = [p[0]+'_'+p[1] for p in list(itertools.product(posConds, kws))]
        negMatchers = [p[0]+'_'+p[1] for p in list(itertools.product(negConds, kws))]
    else:
        posMatchers = posConds
        negMatchers = negConds

    return posMatchers, negMatchers

def time_distribute_labels(labels_pos, fn_pos, labels_neg, fn_neg, nFr, mid_stagger_s=0.075, Fs=16000, frameSamples=240):

    posMatchers, negMatchers = get_conditions()
    maxLab = np.max(labels_pos)
    nFr100 = np.int(np.ceil(0.1*np.float(Fs)/frameSamples))
    nFrMid = np.int(mid_stagger_s*np.float(Fs)/frameSamples)

    # Positive
    labelsPosDistributed = np.zeros((len(labels_pos),nFr))
    for idx, fn in enumerate(fn_pos):
        try:
            anno = np.loadtxt(fn, delimiter=',')
        except:
            print('Could not find '+fn)
            pass
        condition = [x for x in posMatchers if x in fn]
        labVec = np.tile(np.nan,(nFr,))
        if 'kwClip' in condition[0]:
            frStart = np.int(np.floor(anno[0]*nFr))
            frMid = np.int(np.round(anno[1]*nFr))
            frEnd = np.int(np.minimum(nFr, np.round(anno[2]*nFr)))
            labVec[[frStart-nFr100,frStart]] = 0
            labVec[[frMid-nFrMid,frMid]] = labels_pos[idx]+maxLab
            labVec[frEnd-nFr100:frEnd-2] = labels_pos[idx]
            labVec[frEnd-1:frEnd] = 0
        labelsPosDistributed[idx,:] = labVec

    # Negative
    labelsNegDistributed = np.zeros((len(labels_neg),nFr))
    for idx, fn in enumerate(fn_neg):
        try:
            anno = np.loadtxt(fn, delimiter=',')
        except:
            print('Could not find '+fn)
            pass
        condition = [x for x in negMatchers if x in fn]
        labVec = np.tile(np.nan,(nFr,))
        if condition[0] is 'kwRevClip':
            frStart = np.int(np.floor(anno[0]*nFr))
            frMid = np.int(np.round(anno[1]*nFr))
            frEnd = np.int(np.minimum(nFr, np.round(anno[2]*nFr)))
            labVec[[frStart-nFr100,frStart]] = 0
            labVec[[frMid-nFrMid,frMid]] = 0
            labVec[[frEnd-nFr100,frEnd-1]] = 0
        elif condition[0] is 'speechAlignedClip':
            frStart = np.int(np.floor(anno[0]*nFr))
            frMid = np.int(np.round(anno[1]*nFr))
            frEnd = np.int(np.minimum(nFr, np.round(anno[2]*nFr)))
            labVec[[frStart-nFr100,frStart]] = 0
            labVec[[frMid-nFrMid,frMid]] = 0
            labVec[[frEnd-nFr100,frEnd-1]] = 0
        elif condition[0] is 'speechRandomClip':
            randIdx = np.random.permutation(nFr)[:3]
            labVec[randIdx] = 0
            labVec[-nFr100:] = 0
        elif condition[0] is 'falseAlarmClip':
            randIdx = np.random.permutation(nFr)[:3]
            labVec[randIdx] = 0
            labVec[-nFr100:] = 0
        elif condition[0] is 'backClip':
            randIdx = np.random.permutation(nFr)[:3]
            labVec[randIdx] = 0
        elif condition[0] is 'earlyImplantClip':
            frStart = np.int(np.floor(anno[0]*nFr))
            frMid = np.int(np.round(anno[1]*nFr))
            frEnd = np.int(np.minimum(nFr, np.round(anno[2]*nFr)))
            labVec[[frStart-nFr100,frStart]] = 0
            labVec[[frMid-nFrMid,frMid]] = 0
            labVec[[frEnd-nFr100,frEnd-1]] = 0
        elif condition[0] is 'lateImplantClip':
            frStart = np.int(np.floor(anno[0]*nFr))
            frEnd = np.int(np.minimum(nFr, np.round(anno[2]*nFr)))
            labVec[[frStart-nFr100,frStart]] = 0
            labVec[[frEnd-nFr100,frEnd-1]] = 0
        elif condition[0] is 'partialEarlyClip':
            frStart = np.int(np.floor(anno[0]*nFr))
            frEnd = np.int(np.minimum(nFr, np.round(anno[2]*nFr)))
            labVec[[frStart-nFr100,frStart]] = 0
            labVec[[frEnd-nFr100,frEnd-1]] = 0
        elif condition[0] is 'partialLateClip':
            frMid = np.int(np.round(anno[1]*nFr))
            frEnd = np.int(np.minimum(nFr, np.round(anno[2]*nFr)))
            labVec[[frMid-nFrMid,frMid]] = 0
            labVec[[frEnd-nFr100,frEnd-1]] = 0
        elif condition[0] is 'shiftEarlyClip':
            randIdx = np.random.permutation(nFr)[:2]
            frEnd = np.int(np.minimum(nFr, np.round(anno[1]*nFr)))
            labVec[randIdx] = 0
            labVec[[frEnd-nFr100,frEnd-1]] = 0
        elif condition[0] is 'shiftLateClip':
            frStart = np.int(np.floor(anno[0]*nFr))
            labVec[[frStart-nFr100,frStart]] = 0
        labelsNegDistributed[idx,:] = labVec

    return labelsPosDistributed, labelsNegDistributed

if __name__ == '__main__':

    # Input audio directory
    dirName = sys.argv[1]
    outName = sys.argv[2]+'.mat'
    if len(sys.argv) > 3:
        kws = sys.argv[3].split('+')
    else:
        print('Warning: No keywords specified. Mapping all kwClips to single class')
        kws = None

    timeDistributed = True

    posMatchers, negMatchers = get_conditions(kws=kws)

    # Positive examples
    features_pos, labels_pos, fn_pos = load_features(dirName, posMatchers)
    # Negative examples
    features_neg, labels_neg, fn_neg = load_features(dirName, negMatchers)

    labels_pos = labels_pos + 1 # labels 1, 2, ...
    labels_neg = labels_neg * 0 # labels all 0
    if timeDistributed is True:
        fn_pos = [str.split(x,'.wav')[0]+'.csv' for x in fn_pos]
        fn_neg = [str.split(x,'.wav')[0]+'.csv' for x in fn_neg]
        labels_pos, labels_neg = time_distribute_labels(labels_pos, fn_pos, labels_neg, fn_neg, features_pos.shape[1], mid_stagger_s=0.060)

    savemat(outName,
            {'features_pos': features_pos, 'labels_pos': labels_pos,
             'features_neg': features_neg, 'labels_neg': labels_neg})

