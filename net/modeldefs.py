from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

def model_may24(inputShape, numClasses):

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

    return model

