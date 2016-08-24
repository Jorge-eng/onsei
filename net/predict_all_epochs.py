import sys, os
TOP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TOP_DIR, 'models')
NET_PATH = os.path.join(TOP_DIR, '../net')
DATA_PATH = os.path.join(TOP_DIR, '../dataprep')
sys.path.append(DATA_PATH)

import data
import glob
from keras.models import model_from_json
from scipy.io import loadmat, savemat
from scipy.io import wavfile
import audioproc
import numpy as np
import pdb

try: # pyplot throws errors on ec2
    import matplotlib.pyplot as plt
except:
    print('Warning: pyplot failed to import')
    pass

def predict_wav_stream(wavFile, model, modelType, winLen=None, winShift=10, offset=0., scale=1., verbose=0):

    if winLen is None:
        winLen = model.input_shape[3]

    if os.path.isfile(wavFile):
        logM = audioproc.wav2fbank(wavFile)
        feaStream, starts = fbank_stream(logM, winLen, winShift)
    elif os.path.isdir(wavFile): # currently assuming each clip is winLen
        logM,files = audioproc.wav2fbank_batch(wavFile)
        starts = [1]*len(logM)
        feaStream = np.array(logM)

    feaStream = data.apply_norm(feaStream, offset, scale)
    feaStream = data.reshape_for_model(feaStream, modelType)

    prob = model.predict_proba(feaStream, batch_size=128, verbose=verbose)

    pdb.set_trace()
    return prob, starts

def fbank_stream(logM, winLen, winShift=10):

    nBands = logM.shape[0]
    nFrames = logM.shape[1]

    starts = np.arange(0, nFrames-winLen+1, winShift)
    nWindows = len(starts)

    stream = np.zeros((nWindows,nBands,winLen),dtype='float32')
    for n, stIdx in enumerate(starts):
        stream[n,:,:] = logM[:,stIdx+np.arange(0,winLen)]

    return stream, starts

def get_model(modelTag):

    infoFile = os.path.join(MODEL_PATH, modelTag+'.mat')

    info = loadmat(infoFile)
    modelDef = os.path.join(NET_PATH, info['modelDef'][0])
    modelType = info['modelType'][0]
    winLen = int(info['winLen'][0])
    offset = info['offset'][0]
    scale = info['scale'][0]

    model = model_from_json(open(modelDef).read())

    return model, modelType, winLen, offset, scale


if __name__ == '__main__':
    # Usage:
    # $ python predict_spec.py audio in.wav out model_name [epoch]
    # $ python predict_spec.py features in.mat out model_name [epoch]
    # $ python predict_spec.py tinyfeats in.bin out model_name [epoch]

    inType = sys.argv[1]
    inFile = sys.argv[2]
    outFile = sys.argv[3]+'.mat'
    modelTag = sys.argv[4]

    model, modelType, winLen, offset, scale = get_model(modelTag)

    if inType == 'tinyfeats':
        features = audioproc.load_bin(inFile)
        feaStream, starts = fbank_stream(features, winLen)
    elif inType == 'audio':
        features = audioproc.wav2fbank(inFile)
        feaStream, starts = fbank_stream(features, winLen)
    elif inType == 'features':
        feaStream = data.load_batch(inFile, var='features')
     
    feaStream = data.apply_norm(feaStream, offset, scale)
    feaStream = data.reshape_for_model(feaStream, modelType)

    tag = os.path.join(MODEL_PATH, modelTag)
    weightFiles = glob.glob(tag+'_ep*.h5')
    weightFiles.sort()

    prob = []
    for wf in weightFiles:
        print(wf)
        model.load_weights(wf)
        p = model.predict_proba(feaStream, batch_size=128, verbose=1)
        prob.append(p)


    prob = np.array(prob)
    savemat(outFile, {'prob': prob})

