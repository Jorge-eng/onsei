import sys, os
TOP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TOP_DIR, 'models')
DATA_PATH = os.path.join(TOP_DIR, '../dataprep')
sys.path.append(DATA_PATH)

from scipy.io import loadmat, savemat
from audioproc import wav2fbank
from predict_spec import fbank_stream
import pdb

wavFile = sys.argv[1]
modelTag = sys.argv[2]
outFile = sys.argv[3]

infoFile = os.path.join(MODEL_PATH, modelTag+'.mat')
info = loadmat(infoFile)
winLen = int(info['winLen'])
winShift = 10

logM = wav2fbank(wavFile)
feaStream, starts = fbank_stream(logM, winLen, winShift)

savemat(outFile, {'feaStream':feaStream, 'starts':starts})

