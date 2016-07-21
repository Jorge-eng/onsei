'''
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python2 train_cnn_spec.py
'''

from __future__ import print_function
import sys
from keras.callbacks import ModelCheckpoint
import data
import modeldefs
from scipy.io import savemat
import numpy as np
import ast
import pdb

'''
python train_spec.py inFile modelName modelType modelTag
'''

# .mat file expected to contain features_pos, labels_pos, features_neg, labels_neg
inFile = sys.argv[1]

modelName = sys.argv[2]
modelType = sys.argv[3]
modelTag = sys.argv[4]
if len(sys.argv) > 5:
    normalize = ast.literal_eval(sys.argv[5])
else:
    normalize = True

batchSize = 8
numEpoch = 100

modelDef = 'models/'+modelName+modelTag+'.json'
#modelWeights = 'models/'+modelName+modelTag+'.h5'
modelInfo = 'models/'+modelName+modelTag+'.mat'

(feaTrain, labelTrain), (feaTest, labelTest), (offset, scale) = data.load_training(
	inFile, modelType,
	negRatioTrain=10, negRatioTest=10,
	permuteBeforeSplit=(True,True), testSplit=0.15, normalize=normalize)

print(modelInfo)
print('normalize:', normalize)
print('feaTrain shape:', feaTrain.shape)
print(feaTrain.shape[0], 'train samples')
print(feaTest.shape[0], 'test samples')

# input dimensions
inputShape = feaTrain.shape[1:]
numClasses = labelTrain.shape[1]

def build_model(inputShape, numClasses):

    modeldef_fcn = getattr(modeldefs, modelName)
    model, optimizer, loss = modeldef_fcn(inputShape, numClasses)

    model.compile(loss=loss, optimizer=optimizer)

    return model

model = build_model(inputShape, numClasses)

modelWeights = 'models/'+modelName+modelTag+'_ep{epoch:02d}_{val_loss:.2f}.h5'
cbks = [ModelCheckpoint(modelWeights, monitor='val_loss')]

if True:
    classCount = labelTrain.sum(axis=0)
    w = np.concatenate(([1.], classCount[1:].max() / classCount[1:]))
    classWeight = dict([(i, w[i]) for i in range(numClasses)])
else:
    classWeight = {0:1,1:1,2:10}
print('classWeight:', classWeight)

history = model.fit(feaTrain, labelTrain, batch_size=batchSize,
            nb_epoch=numEpoch, show_accuracy=True,
            validation_data=(feaTest, labelTest), callbacks=cbks, shuffle=True,
            class_weight=classWeight)

if modelType == 'cnn':
    winLen = inputShape[2]
elif modelType == 'rnn':
    winLen = inputShape[0]

print('Saving to '+modelInfo)
savemat(modelInfo, {'modelDef': modelDef,'modelWeights': modelWeights,
                    'modelType': modelType,'winLen': winLen,
                    'offset': offset,'scale': scale,
                    'train_acc':history.history['acc'],
                    'train_loss':history.history['loss'],
                    'val_acc':history.history['val_acc'],
                    'val_loss':history.history['val_loss']})

# Write model definition to file
open(modelDef, 'w').write(model.to_json())

