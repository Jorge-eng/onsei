import sys, os, glob
import numpy as np
from scipy.signal import convolve
from scipy.io import loadmat, savemat
import pdb

def detector(data, h, th):

    th = np.tile(th, 2)
    h = np.ones((h,),dtype='float32')

    loc = []
    num = []
    for kw in range(data.shape[1]):
        det = np.float32(data[:,kw] > th[0])
        det = convolve(det, h, mode='same')
        det = np.float32(det == len(h))
        det = np.diff(det) > 0
        num.append(len(np.where(det)[0]))

    return num

#data = np.genfromtxt(sys.argv[1], delimiter=',')
netOutDir = os.path.join('../net/outputs', sys.argv[1])
testDirs = ['testingWavs_kwClip_okay_sense',
            'testingWavs_kwClip_stop',
            'testingWavs_kwClip_snooze',
            'noiseDataset/16k',
            'reverberant_speech/16k',
            'TED']

ths = np.array([0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
counts = np.arange(1, 16, 2)
nKw = 3

for td in testDirs:
    inDir = os.path.join(netOutDir, td)
    allFiles = glob.glob(os.path.join(inDir,'*.mat'))
    allFiles.sort()

    num = np.zeros((len(allFiles), len(counts), len(ths), nKw))
    tot = np.zeros((len(allFiles),))

    for j, fn in enumerate(allFiles):
        print(fn)
        data = np.squeeze(loadmat(fn)['prob'])
        tot[j] = data.shape[0]

        for m, h in enumerate(counts):
            for n, th in enumerate(ths):
                num[j,m,n,:] = detector(data[:,1:1+nKw], h, th)

    #np.savetxt(sys.argv[4], fmt='%0.5f', delimiter=',')
    outFile = os.path.join(netOutDir, 'eval_'+td.split('/')[0]+'.mat')

    savemat(outFile, {'tot':tot,'num':num})

