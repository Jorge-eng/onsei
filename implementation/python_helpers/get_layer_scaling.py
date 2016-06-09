#!/usr/bin/python
from keras.optimizers import Adam
from keras.models import Sequential
import numpy as np
import copy
import tensor_utils
from matplotlib.pyplot import *
import sys
f = open('kwClip_43.bin','r')
x = np.fromfile(f,dtype=np.int8)
x = x.reshape(x.shape[0] / 40,40).astype(float)
x += 80.
x /= 140.
x = x.transpose().reshape(1,1,40,199)


#cut_layers = [3,7,11,15,18]
cut_layers = [2,6,10,14,17]
model_name = 'model_may31_small_sigm'

model = tensor_utils.get_model(model_name)

names = [config['name'] for config in model.get_config()['layers']]

for last_layer in cut_layers:
    for i,layer in enumerate(model.layers):
        name = model.get_config()['layers'][i]['name']
        print name

        if i == last_layer:
            break
        
    print '------------'

yesno = raw_input('is this okay?')
if yesno.find('y') < 0:
    sys.exit(0)    

outputs = []
for last_layer in cut_layers:
    model2 = Sequential()
    for i,layer in enumerate(model.layers):
        model2.add(copy.deepcopy(layer))
        
        if i == last_layer:
            break

    model2.compile(Adam(),'mse')

    y = model2.predict(x)
    outputs.append(y)

for y in outputs:
    print np.percentile(y.flatten(),98)
    hist(y.flatten())

show()

         

        
    
