'''
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python2 train_cnn_spec.py
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys
from keras.callbacks import ModelCheckpoint
import data
import modeldefs
from scipy.io import savemat
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

typeInfo = modelType.split('_')

modelDef = 'models/'+modelName+modelTag+'.json'
modelWeights = 'models/'+modelName+modelTag+'_ep{epoch:03d}.h5'
modelInfo = 'models/'+modelName+modelTag+'.mat'

numEpoch = 252
batchSize = 8
negRatioTrain = 20
negRatioTest = 10
permuteBeforeSplit = (True, True)
testSplit = 0.10
# Load the train and test data
(feaTrain, labelTrain), (feaTest, labelTest), (offset, scale) = data.load_training(
	inFile, modelType,
	negRatioTrain=negRatioTrain, negRatioTest=negRatioTest,
	permuteBeforeSplit=permuteBeforeSplit, testSplit=testSplit, normalize=normalize)


# If stateful, must pare training to a multiple of batchSize
if len(typeInfo) > 1 and typeInfo[1] == 'stateful':
    feaTrain, labelTrain = data.cut_to_batch(feaTrain, labelTrain, batchSize)
    feaTest, labelTest = data.cut_to_batch(feaTest, labelTest, batchSize)

print(modelInfo)
print('normalize:', normalize)
print('feaTrain shape:', feaTrain.shape)
print(feaTrain.shape[0], 'train samples')
print(feaTest.shape[0], 'test samples')

# input dimensions
inputShape = feaTrain.shape[1:]
numClasses = labelTrain.shape[-1]

if typeInfo[0] == 'cnn':
    winLen = inputShape[2]
elif typeInfo[0] == 'rnn':
    winLen = inputShape[0]

# class weights
if len(sys.argv) > 6:
    w = sys.argv[6].split(',')
else:
    classCount = labelTrain.sum(axis=0)
    w = np.concatenate(([1.], classCount[1:].max() / classCount[1:]))

if w[0] == 'None':
    classWeight = None
else:
    classWeight = dict([(i, w[i]) for i in range(numClasses)])
print('classWeight:', classWeight)

def build_model(inputShape, numClasses):

    modeldef_fcn = getattr(modeldefs, modelName)
    model, optimizer, loss = modeldef_fcn(inputShape, numClasses, batchSize=batchSize)

    model.compile(loss=loss, optimizer=optimizer)

    return model

model = build_model(inputShape, numClasses)

modelParams = {'modelDef': modelDef, 'modelWeights': modelWeights,
               'modelType': modelType, 'winLen': winLen,
               'offset': offset, 'scale': scale}
trainParams = {'inFile':inFile, 'batchSize': batchSize,
                'negRatioTrain': negRatioTrain, 'permuteBeforeSplit': permuteBeforeSplit,
                'testSplit': testSplit,'normalize': str(normalize), 'classWeight': w}

# Save info required to run intermediate model during training
print('Saving to '+modelInfo)
savemat(modelInfo, dict(modelParams, **trainParams))

# Write model definition to file
open(modelDef, 'w').write(model.to_json())

cbks = [ModelCheckpoint(modelWeights, monitor='val_loss')]

print(labelTrain.shape)
print(labelTest.shape)

# Train
history = model.fit(feaTrain, labelTrain, batch_size=batchSize,
            nb_epoch=numEpoch, show_accuracy=True,
            validation_data=(feaTest, labelTest), callbacks=cbks, shuffle=True,
            class_weight=classWeight)

# Save all info again plus training history (no way to append)
modelParams = dict(modelParams,
                   **{'train_acc':history.history['acc'],
                      'train_loss':history.history['loss'],
                      'val_acc':history.history['val_acc'],
                      'val_loss':history.history['val_loss']})

print('Saving to '+modelInfo)
savemat(modelInfo, dict(modelParams, **trainParams))

