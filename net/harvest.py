import sys, os
TOP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TOP_DIR, 'models')
NET_PATH = os.path.join(TOP_DIR, '../net')
DATA_PATH = os.path.join(TOP_DIR, '../dataprep')
sys.path.append(DATA_PATH)

import data
from scipy.io import loadmat, savemat
from scipy.io import wavfile
import audioproc
import predict_spec
import numpy as np
import pdb

try: # pyplot throws errors on ec2
    import matplotlib.pyplot as plt
except:
    print('Warning: pyplot failed to import')
    pass

def harvest_wav_stream(wavFile, model, modelType, winLen=None, winShift=20, offset=0., scale=1., verbose=0):

    if winLen is None:
        winLen = model.input_shape[3]

    logM = audioproc.wav2fbank(wavFile,maxLen_s=30*60)

    feaStream, starts = predict_spec.fbank_stream(logM, winLen, winShift)
    feaStream = data.apply_norm(feaStream, offset, scale)
    feaStream = data.reshape_for_model(feaStream, modelType)

    prob = model.predict_proba(feaStream, batch_size=128, verbose=verbose)

    #feaStream -= feaStream.mean()
    #feaStream /= feaStream.max()
    #prob2 = model.predict_proba(feaStream, batch_size=128, verbose=verbose)

    #prob = np.maximum(prob, prob2)
 
    return prob, starts

if __name__ == '__main__':
    # Usage:
    # $ python harvest.py inDir outTag model_name [epoch]

    inDir = sys.argv[1]
    outFile = 'prob_'+sys.argv[2]+'.mat'
    modelTag = sys.argv[3]
    if len(sys.argv) > 4:
        epoch = int(sys.argv[4])
    else:
        epoch = None

    model, modelType, winLen, offset, scale = predict_spec.get_model(modelTag, epoch=epoch)

    files = audioproc.find_audio_files(inDir)

    prob = np.array([])
    startTimes = []
    fileIdx = []
    for c, f in enumerate(files):
        print('Processing '+f)
        p, t = harvest_wav_stream(f, model, modelType, winLen=winLen, offset=offset, scale=scale, verbose=1)
        if prob.shape[0] == 0:
            prob = p
        else:
            prob = np.append(prob, p, axis=0)
        startTimes.extend(t)
        fileIdx.extend([c]*len(t))

    savemat(outFile, {'prob':prob, 'startTimes':startTimes, 'fileIdx':fileIdx, 'files':files})

