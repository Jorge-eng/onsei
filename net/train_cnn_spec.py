'''Train a simple deep CNN on the audio mfc dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python2 train_cnn_spec.py

'''

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import data
from scipy.io import savemat
import pdb


batchSize = 8 
numEpoch = 20
# Best so far: rmsprop
#optimizer = 'adadelta'
#optimizer = 'adagrad'
optimizer = 'rmsprop'
#optimizer = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)

# the data, shuffled and split between train and test sets
inFiles = ('spec_pos.mat',
           'spec_neg.mat')

modelDef = 'models/cnn_spec_try.json'
modelWeights = 'models/cnn_spec_try.h5'
modelInfo = 'models/cnn_spec_try.mat'

(feaTrain, labelTrain), (feaTest, labelTest) = data.load_training(
	inFiles, 'features', 'convnet', 
	negRatioTrain=10, negRatioTest=10, 
	permuteBeforeSplit=(False,False), testSplit=0.2, normalize=False)

print('feaTrain shape:', feaTrain.shape)
print(feaTrain.shape[0], 'train samples')
print(feaTest.shape[0], 'test samples')

# input dimensions
inputShape = feaTrain.shape[1:]
numClasses = labelTrain.shape[1]

#pdb.set_trace()
def build_model(inputShape, numClasses, optimizer):
    model = Sequential()

    model.add(Convolution2D(16, 3, 5, border_mode='same', input_shape=inputShape))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, 3, 5))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 1, 3)) # best 64 or 32
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 1, 3)) # best 64 or 32
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Dropout(0.25))

    # New part
    model.add(Convolution2D(64, 1, 3)) # best 64 or 32
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 1, 3)) # best 64 or 32
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Dropout(0.25))
    # end new part

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(numClasses))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

model = build_model(inputShape, numClasses, optimizer)

cbks = [ModelCheckpoint(modelWeights,
        monitor='val_loss', save_best_only=True, mode='auto')]

model.fit(feaTrain, labelTrain, batch_size=batchSize,
          nb_epoch=numEpoch, show_accuracy=True,
          validation_data=(feaTest, labelTest), callbacks=cbks, shuffle=True)

print('Saving to '+modelInfo)
savemat(modelInfo, {'modelDef': modelDef,'modelWeights': modelWeights})

# Write model definition to file
open(modelDef, 'w').write(model.to_json())

