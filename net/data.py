import sys, os
TOP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TOP_DIR, '../net/models')

from keras.utils import np_utils
from keras.models import model_from_json
import numpy as np
from scipy.io import loadmat
import pdb

def load_model(defFile, weightFile):
    # as per http://keras.io/faq/#how-can-i-save-a-keras-model

    model = model_from_json(open(defFile).read())
    model.load_weights(weightFile)

    return model

def load_batch(filePath, var='mfc'):
    data = loadmat(filePath)
    data = data[var]
    data = np.rollaxis(data, 2)

    return data

def load_labels(filePath, var='labels'):
    data = loadmat(filePath, variable_names=[var])
    if var in data:
        labels = data[var]
    else:
        labels = []

    if len(labels) == 1:
        labels = labels[0]

    return labels

def reshape_for_model(data, modelType='cnn'):

    if modelType=='cnn':
        data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
    elif modelType=='rnn' or modelType=='rnn_dist':
        data = np.swapaxes(data, 1, 2)

    return data

def permute_pair(X, y):

    perm = np.random.permutation(X.shape[0])
    X = X[perm, ...]
    y = y[perm]

    return X, y

def split_pair(X, y, testSplit):

    splitIdx = np.int(X.shape[0] * (1-testSplit))
    (feaTrain, feaTest) = np.array_split(X, [splitIdx], axis=0)
    (labelTrain, labelTest) = np.array_split(y, [splitIdx], axis=0)

    return feaTrain, labelTrain, feaTest, labelTest

def get_tilt(fea):

    nFr = fea.shape[1]

    # todo: only use VAD frames
    meanSpec = fea.mean(axis=0).mean(axis=1)
    pLin = np.poly1d(np.polyfit(np.arange(nFr), meanSpec, 1))
    tilt = pLin(np.arange(nFr))
    tilt = tilt - tilt.mean()

    return tilt

def get_norm(fea, tilt=None):

    if True:
        fea = np.float64(fea)

        offset = np.mean(fea)
        if tilt is not None:
            offset = offset + tilt
        else:
            offset = np.tile(offset,fea.shape[1])

        scale = np.max(np.abs(np.float32(fea-offset[np.newaxis,:,np.newaxis])))
    else:
        offset = np.float64(7)
        scale = np.float32(12)

    return offset, scale

def apply_norm(fea, offset, scale):

    fea = np.float32(np.float64(fea) - offset[np.newaxis,:,np.newaxis]) / scale

    return fea

def distributed_categorical(labels, numClasses):

    numSamps, numSteps = labels.shape
    labelsTimeDistributed = np.zeros((numSamps,numSteps,numClasses))

    for samp in range(numSamps):
        lab = labels[samp]
        iValid = ~np.isnan(lab)
        labelsTimeDistributed[samp,iValid,:] = np_utils.to_categorical(lab[iValid], numClasses)

    return labelsTimeDistributed

def time_distribute_label(labels, timeSteps, numClasses, labelWindows=None, nullLabel=0):

    # Examples:
    # labelWindows = ((None, range(0,157)),)
    #labelWindows = labelWindows + ((0, range(56,57)),)
    #labelWindows = labelWindows + ((1, range(156,157)),)
    #labelWindows = labelWindows + ((2, range(116,117)),)

    if labelWindows is None:
        labelWindows = ((1,[timeSteps]),)

    labelsTimeDistributed = np_utils.to_categorical(np.tile(nullLabel, (labels.shape[0],1)), numClasses)
    labelsTimeDistributed = np.tile(labelsTimeDistributed[:,None,:], (1,timeSteps,1))

    lab = np_utils.to_categorical(labels, numClasses)
    nullLab = np_utils.to_categorical(nullLabel+0*labels, numClasses)
    midLab = np_utils.to_categorical(2*labels, numClasses)
    for w in labelWindows:
        if w[0] is 0:
            labelWindow = np.tile(nullLab[:,None,:], (1,len(w[1]),1))
        elif w[0] is 2:
            labelWindow = np.tile(midLab[:,None,:], (1,len(w[1]),1))
        else:
            labelWindow = np.tile(lab[:,None,:], (1,len(w[1]),1))
        if w[0] is None:
            labelWindow = labelWindow * 0
        labelsTimeDistributed[:,w[1],:] = labelWindow

    return labelsTimeDistributed

def load_training(inFile, modelType, testSplit=0.1, negRatioTrain=10, negRatioTest=1, normalize=None, permuteBeforeSplit=(True,True)):

    # pos
    feaTrainPos = load_batch(inFile, 'features_pos')
    labelTrainPos = load_labels(inFile, 'labels_pos')
    if len(labelTrainPos) == 0:
        labelTrainPos = np.ones((feaTrainPos.shape[0],), dtype='uint8')

    # neg
    feaTrainNeg = load_batch(inFile, 'features_neg')
    labelTrainNeg = load_labels(inFile, 'labels_neg')
    if len(labelTrainNeg) == 0:
        labelTrainNeg = np.zeros((feaTrainNeg.shape[0],), dtype='uint8')

    # permute before split
    if permuteBeforeSplit[0]: # pos
        (feaTrainPos, labelTrainPos) = permute_pair(feaTrainPos, labelTrainPos)
    if permuteBeforeSplit[1]: # neg
        (feaTrainNeg, labelTrainNeg) = permute_pair(feaTrainNeg, labelTrainNeg)

    # trim negatives to a given multiple of the number of positives
    numNeg = np.min((np.int(feaTrainPos.shape[0] * negRatioTrain), feaTrainNeg.shape[0]))
    feaTrainNeg = feaTrainNeg[:numNeg, ...]
    labelTrainNeg = labelTrainNeg[:numNeg]

    # split
    # . pos
    (feaTrainPos, labelTrainPos, feaTestPos, labelTestPos) = split_pair(feaTrainPos, labelTrainPos, testSplit)
    # . neg
    (feaTrainNeg, labelTrainNeg, feaTestNeg, labelTestNeg) = split_pair(feaTrainNeg, labelTrainNeg, testSplit)

    # rebalance for test set
    numNegTest = np.min((np.int(feaTestPos.shape[0] * negRatioTest), feaTestNeg.shape[0]))
    feaTestNeg = feaTestNeg[:numNegTest, ...]
    labelTestNeg = labelTestNeg[:numNegTest]

    # join pos & neg into train & test sets
    feaTrain = np.concatenate((feaTrainPos, feaTrainNeg), axis=0)
    labelTrain = np.concatenate((labelTrainPos, labelTrainNeg), axis=0)
    feaTest = np.concatenate((feaTestPos, feaTestNeg), axis=0)
    labelTest = np.concatenate((labelTestPos, labelTestNeg), axis=0)

    numClasses = len(np.unique(labelTrain[~np.isnan(labelTrain)]))

    typeInfo = str.split(modelType,'_')
    if len(typeInfo) > 1 and typeInfo[1]=='dist':
        if labelTrain.ndim is not 2:
            raise ValueError('label ndim should be 2 for time distributed')

        labelTrain = distributed_categorical(labelTrain, numClasses)
        labelTest = distributed_categorical(labelTest, numClasses)
    else:
        labelTrain = np.reshape(labelTrain, (len(labelTrain), 1))
        labelTest = np.reshape(labelTest, (len(labelTest), 1))

        # convert class vectors to binary class matrices
        labelTrain = np_utils.to_categorical(labelTrain, numClasses)
        labelTest = np_utils.to_categorical(labelTest, numClasses)

    # normalization
    if normalize is not None:
        if normalize:
            tilt = get_tilt(feaTrainPos[:,:,-100:])
        else:
            tilt = None
        offset, scale = get_norm(feaTrain, tilt)
    else:
        offset = np.tile(np.float64(0),feaTrain.shape[1])
        if feaTrain.dtype.kind == 'i':
            #scale = np.float32(np.power(2, 8*feaTrain.itemsize-1))
            #scale = np.float32(np.power(2, 8-1))
            scale = np.float32(np.power(2, 12))
        else:
            scale = np.float32(1)

    feaTrain = apply_norm(feaTrain, offset, scale)
    feaTest = apply_norm(feaTest, offset, scale)

    feaTrain = reshape_for_model(feaTrain, modelType)
    feaTest = reshape_for_model(feaTest, modelType)

    return (feaTrain, labelTrain), (feaTest, labelTest), (offset, scale)

