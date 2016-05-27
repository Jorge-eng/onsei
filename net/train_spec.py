'''Train a simple deep CNN on the audio mfc dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python2 train_cnn_spec.py

'''

from __future__ import print_function
import sys
from keras.callbacks import ModelCheckpoint
import data
import modeldefs
from scipy.io import savemat
import pdb

modelName = sys.argv[1]
modelType = sys.argv[2]

batchSize = 8
numEpoch = 20

# the data, shuffled and split between train and test sets
inFiles = ('spec_pos.mat',
           'spec_neg.mat')

modelDef = 'models/'+modelName+'.json'
modelWeights = 'models/'+modelName+'.h5'
modelInfo = 'models/'+modelName+'.mat'

(feaTrain, labelTrain), (feaTest, labelTest) = data.load_training(
	inFiles, 'features', modelType,
	negRatioTrain=10, negRatioTest=10,
	permuteBeforeSplit=(False,False), testSplit=0.2, normalize=False)

print('feaTrain shape:', feaTrain.shape)
print(feaTrain.shape[0], 'train samples')
print(feaTest.shape[0], 'test samples')

# input dimensions
inputShape = feaTrain.shape[1:]
numClasses = labelTrain.shape[1]

def build_model(inputShape, numClasses):

    modeldef = getattr(modeldefs, modelName)
    model, optimizer = modeldef(inputShape, numClasses)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

model = build_model(inputShape, numClasses)

cbks = [ModelCheckpoint(modelWeights,
        monitor='val_loss', save_best_only=True, mode='auto')]

model.fit(feaTrain, labelTrain, batch_size=batchSize,
          nb_epoch=numEpoch, show_accuracy=True,
          validation_data=(feaTest, labelTest), callbacks=cbks, shuffle=True)

if modelType == 'cnn':
    winLen = inputShape[2]
elif modelType == 'rnn':
    winLen = inputShape[0]

print('Saving to '+modelInfo)
savemat(modelInfo, {'modelDef': modelDef,'modelWeights': modelWeights,
                    'modelType': modelType,'winLen': winLen})

# Write model definition to file
open(modelDef, 'w').write(model.to_json())

