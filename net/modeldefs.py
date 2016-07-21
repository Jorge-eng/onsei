from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.core import Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
import numpy as np
import pdb

def model_may25_lstm_small(inputShape, numClasses):

    optimizer = 'rmsprop'
    #optimizer = SGD(lr=0.1, decay=0.0, momentum=0.9, nesterov=True)
    loss = 'categorical_crossentropy'

    model = Sequential()

    model.add(LSTM(32, return_sequences=True, input_shape=inputShape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))

    return model, optimizer, loss

def model_may25_lstm_large(inputShape, numClasses):

    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'

    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=inputShape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))

    return model, optimizer, loss

def model_may24_large(inputShape, numClasses):

    #optimizer = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'

    model = Sequential()

    model.add(Convolution2D(16, 3, 5, border_mode='same', input_shape=inputShape))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, 3, 5))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 1, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 1, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 1, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 1, 3))
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

    return model, optimizer, loss

def model_may24_small(inputShape, numClasses):

    #optimizer = SGD(lr=0.003, decay=0.0, momentum=0.9, nesterov=True)
    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'

    model = Sequential()

    model.add(Convolution2D(16, 3, 5, border_mode='same', input_shape=inputShape))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 1, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 1, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(numClasses))
    model.add(Activation('softmax'))

    return model, optimizer, loss


def model_may31_small_sigm(inputShape, numClasses):

    #optimizer = SGD(lr=0.003, decay=0.0, momentum=0.9, nesterov=True)
    optimizer = 'rmsprop'
    loss = 'binary_crossentropy'

    model = Sequential()

    model.add(Convolution2D(16, 3, 5, border_mode='valid', input_shape=inputShape))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 1, 3, border_mode='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 1, 3, border_mode='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Dropout(0.25))
    # end new part

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(numClasses))
    model.add(Activation('sigmoid'))

    return model, optimizer, loss

def model_jun17_small_sigm(inputShape, numClasses):

    #optimizer = SGD(lr=0.003, decay=0.0, momentum=0.9, nesterov=True)
    optimizer = 'rmsprop'
    loss = 'binary_crossentropy'

    model = Sequential()

    model.add(Convolution2D(8, 3, 5, border_mode='valid', input_shape=inputShape))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(8, 3, 5, border_mode='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(16, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))
    # end new part

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    #model.add(Dense(256))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(numClasses))
    model.add(Activation('sigmoid'))

    return model, optimizer, loss

def model_jun22_smaller_sigm(inputShape, numClasses):

    optimizer = SGD(lr=0.003, decay=1e-7, momentum=0.9, nesterov=True)
    #optimizer = 'rmsprop'
    #optimizer = 'adagrad'
    #loss = 'categorical_crossentropy'
    loss = 'mse'

    model = Sequential()

    model.add(Convolution2D(8, 3, 5, border_mode='valid', input_shape=inputShape))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(8, 3, 5, border_mode='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(8, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(numClasses))
    model.add(Activation('sigmoid'))

    return model, optimizer, loss

def model_jun30_cnn_lstm(inputShape, numClasses):

    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'

    model = Sequential()

    model.add(Convolution2D(4, 3, 5, border_mode='valid', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    convOutShape = model.layers[-1].output_shape
    model.add(Reshape((np.prod(convOutShape[1:3]), convOutShape[3])))
    model.add(Permute((2, 1)))
    #model.add(LSTM(64, return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))

    return model, optimizer, loss

