import sys, os
TOP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TOP_DIR, 'models')
NET_PATH = os.path.join(TOP_DIR, '../net')
DATA_PATH = os.path.join(TOP_DIR, '../dataprep')
sys.path.append(DATA_PATH)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data
from keras.models import model_from_json, model_from_config
from scipy.io import loadmat, savemat
from scipy.io import wavfile
import audioproc
import numpy as np
import json
import pdb

def get_input(inFile, inType, modelType, offset=0., scale=1., winLen=None, winShift=10):

    if winLen is None:
        winLen = model.input_shape[3]

    typeInfo = modelType.split('_')
    timeDistributed = (len(typeInfo) > 1) and any([x in typeInfo[1] for x in ['dist','stateful']])

    if inType == 'tinyfeats':
        feaStream = audioproc.load_bin(inFile)
        if not timeDistributed:
            feaStream, starts = fbank_stream(feaStream, winLen)
    elif inType == 'audio':
        if os.path.isfile(inFile):
            feaStream = audioproc.wav2fbank(inFile)
            nFiles = 1
            if not timeDistributed:
                feaStream, starts = fbank_stream(feaStream, winLen)
        elif os.path.isdir(inFile): # assumes each clip is winLen
            feaStream, files = audioproc.wav2fbank_batch(inFile)
            nFiles = len(files)
        if timeDistributed:
            feaStream = np.reshape(feaStream, (nFiles, feaStream.shape[0], feaStream.shape[1]))
    elif inType == 'serverfeats':
        feaStream = audioproc.load_serverfeats(inFile)
    elif inType == 'features':
        feaStream = data.load_batch(inFile, var='features')

    feaStream = data.apply_norm(feaStream, offset, scale)
    feaStream = data.reshape_for_model(feaStream, modelType)

    return feaStream

def fbank_stream(logM, winLen, winShift=10):

    nBands = logM.shape[0]
    nFrames = logM.shape[1]

    starts = np.arange(0, nFrames-winLen+1, winShift)
    nWindows = len(starts)

    stream = np.zeros((nWindows,nBands,winLen),dtype='float32')
    for n, stIdx in enumerate(starts):
        stream[n,:,:] = logM[:,stIdx+np.arange(0,winLen)]

    return stream, starts

def detect_online(wav,prob_prev,model,modelType,winLen=None,offset=0.,scale=1.,detWait=10,detTh=1.5,waitCount=0,waiting=False):

    feaStream = get_input(wav, 'audio', modelType, offset=offset, scale=scale, winLen=winLen)
    prob = model.predict_proba(feaStream, batch_size=128, verbose=verbose)

    prob = prob[0, :]

    detect = np.float32(0)
    if waitCount >= detWait:
        waiting = False
        waitCount = 0
    if waiting:
        waitCount += 1
    else:
        signal = prob + prob_prev
        if np.any(signal[1:] >= detTh):
            detect = 1 + np.argmax(signal[1:])
            waiting = True

    return detect, prob, waitCount, waiting

def wav2detect(wavFile,model,modelType,winLen,offset=0.,scale=1.,winLen_s=1.6,winShift_s=0.2,detWait_s=2.0,detTh=1.5):

    (fs, wav) = wavfile.read(wavFile)
    assert fs == 16000
    if len(wav.shape) == 1:
        wav = wav[:,None]

    winSamples = int(winLen_s * fs)
    shiftSamples = int(winShift_s * fs)
    detWait = int(detWait_s / winShift_s)
    waitCount = 0
    waiting = False
    prob_prev = 0

    startIdx = 0
    detect = np.array([],dtype='float32')
    while startIdx <= wav.shape[0]-winSamples:
        wav_buffer = wav[range(startIdx,startIdx+winSamples),:]

        ans, prob_prev, waitCount, waiting = detect_online(
                wav_buffer, prob_prev, model, modelType, winLen, offset, scale, detWait, detTh, waitCount, waiting)

        detect = np.append(detect, ans)
        startIdx += shiftSamples

    return detect

def get_info(modelTag):

    infoFile = os.path.join(MODEL_PATH, modelTag+'.mat')
    info = loadmat(infoFile)

    return info

def get_arch(info, doCompile=True):

    if isinstance(info, str):
        info = get_info(info)

    modelDef = os.path.join(NET_PATH, info['modelDef'][0])
    modelType = info['modelType'][0]
    winLen = int(info['winLen'][0])
    offset = info['offset'][0]
    scale = info['scale'][0]

    model = json.load(open(modelDef, 'r'))
    if doCompile==True:
        model = model_from_config(model)

    return model, modelType, winLen, offset, scale

def get_weights(info, epoch=None):

    modelWeights = os.path.join(NET_PATH, info['modelWeights'][0])
    if 'epoch' in modelWeights:
        if epoch is None:
            val_loss = info['val_loss'][0]
            epoch = np.argmin(val_loss)

        print('Choosing epoch {:03d}'.format(epoch))
        modelWeights = modelWeights.format(**{'epoch':epoch})
        print(modelWeights)

    return modelWeights

def get_model(modelTag, epoch=None, doCompile=True):

    info = get_info(modelTag)
    modelWeights = get_weights(info, epoch)
    model, modelType, winLen, offset, scale = get_arch(info, doCompile=doCompile)
    model.load_weights(modelWeights)

    return model, modelType, winLen, offset, scale

def get_model_and_input(modelTag, inFile, inType, epoch=None):

    info = get_info(modelTag)
    modelWeights = get_weights(info, epoch)
    model, modelType, winLen, offset, scale = get_arch(info, doCompile=False)

    feaStream = get_input(inFile, inType, modelType, offset=offset, scale=scale, winLen=winLen)

    model['layers'][0]['batch_input_shape'] = feaStream.shape
    model = model_from_config(model)
    model.load_weights(modelWeights)

    return model, feaStream

def predict_stateful(model, feaStream, reset_states=True):

    batchSize, chunkSize, nBands = model.input_shape
    nClasses = model.output_shape[2]

    if reset_states:
        model.reset_states()

    numSamples, numFrames, numChannels = feaStream.shape

    # Chop up to chunks of size chunkSize. Pad if needed
    numChunks = np.int(np.ceil(np.float(feaStream.shape[1]) / chunkSize))
    padSize = 0
    if numFrames < numChunks*chunkSize:
        padSize = numChunks*chunkSize - numFrames
        feaStream = np.concatenate((feaStream, np.zeros((numSamples, padSize, nBands))), axis=1)

    feaStream = np.reshape(feaStream, (numChunks*numSamples, chunkSize, nBands))

    numChunks = feaStream.shape[0]

    # Pad to even multiple of batchSize
    numBatches = np.int(np.ceil(numChunks / np.float(batchSize)))
    if numChunks < numBatches*batchSize:
        feaStream = np.concatenate((feaStream, np.zeros((numBatches*batchSize-numChunks,chunkSize,nBands))),axis=0)

    prob = model.predict_proba(feaStream, batch_size=batchSize)
    prob = prob[:numChunks,:,:]

    prob = np.reshape(prob, (numSamples, numFrames+padSize, nClasses))
    prob = prob[:,:numFrames,:]

    return prob


if __name__ == '__main__':
    # Usage:
    # $ python predict_spec.py audio in.wav out model_name [epoch]
    # $ python predict_spec.py features in.mat out model_name [epoch]
    # $ python predict_spec.py tinyfeats in.bin out model_name [epoch]

    inType = sys.argv[1]
    inFile = sys.argv[2]
    outFile = sys.argv[3]+'.mat'
    modelTag = sys.argv[4]
    if len(sys.argv) > 5:
        epoch = int(sys.argv[5])
    else:
        epoch = None

    if not os.path.isfile(inFile) and not os.path.isdir(inFile):
        raise ValueError('Input '+inFile+' does not exist')

    #model, feaStream = get_model_and_input(modelTag, inFile, inType, epoch=epoch)

    model, modelType, winLen, offset, scale = get_model(modelTag, epoch=epoch)

    feaStream = get_input(inFile, inType, modelType, offset=offset, scale=scale, winLen=winLen)

    if 'stateful' in modelType:
        prob = predict_stateful(model, feaStream)
    else:
        prob = model.predict_proba(feaStream, verbose=1)

    print('Saving '+outFile)
    savemat(outFile, {'prob': prob, 'feaStream': feaStream})

