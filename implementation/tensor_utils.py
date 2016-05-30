#!/usr/bin/python
import numpy as np
import copy
from keras.models import model_from_json
from collections import defaultdict


def write_fixed_point_tensor(name,weights,f):
    dims = weights.shape
    vec = (weights.flatten() * (2**7)).astype(int).tolist()
    vecstr = ['%d' % v for v in vec]
    myweights = 'const static Weight_t %s_weights[%d] = {%s};\n' % (name,len(vec),','.join(vecstr))
    mydims = 'const static uint32_t %s_dims[4] = {%d,%d,%d,%d};\n' % (name,dims[0],dims[1],dims[2],dims[3])
    mystruct = 'const static ConstTensor_t %s = {&%s_weights[0],&%s_dims[0]};\n' % (name,name,name)

    f.write(myweights)
    f.write(mydims)
    f.write(mystruct)
    f.write('\n')

def write_conv_weights(name,weights,f):
    w = copy.deepcopy(weights)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
           w[i][j] = w[i][j][::-1,::-1]

    write_fixed_point_tensor(name,w,f) 


class Layer(object):
    def __init__(self,prevlayers,layers,layer_count):
        self.name = '%s_%02d' % (layers[0]['name'],layer_count)
        self.layers = layers
        self.prevlayers = prevlayers
        self.dropout = self.get_prev_dropout()
        self.weights = []

    def get_prev_dropout(self):
        if self.prevlayers == None:
            return 0.0
        
        names = [layer['name'] for layer in self.prevlayers]
        if 'Dropout' not in names:
            return 0.0
        
        idx = names.index('Dropout')
        p = 0.
        if idx != None:
            p = self.prevlayers[idx]['p']

        return p

    def add_weights(self,w):
        self.weights.append(w)
        


class ConvLayer(Layer):
    def write(self,input_shape,f):
        border_mode = self.layers[0]['border_mode']
        w0 = self.weights[0]
        weights_name = self.name + '_conv'
        bias_name = self.name + '_bias'
        write_conv_weights(weights_name,w0,f)
        write_fixed_point_tensor(bias_name,self.weights[1].reshape(self.weights[1].shape[0],1,1,1),f)

        print 'conv weights: %d,%d,%d,%d' % w0.shape + ' border_mode=%s' % border_mode

        if border_mode == 'same':
            s3 = input_shape[3]
            s2 = input_shape[2]
            s1 = w0.shape[0]
            s0 = 1


        elif border_mode == 'valid':
            s3 = input_shape[3] - w0.shape[3] + 1
            s2 = input_shape[2] - w0.shape[2] + 1
            s1 = w0.shape[0]
            s0 = 1
        
        return (s0,s1,s2,s3)
                
        
    def get_num_weights(self):
        return 2

class MaxPoolingLayer(Layer):
    def write(self,input_shape,f):
        pool_size = self.layers[0]['pool_size']
        print 'pool dims: %d,%d' % pool_size
        mystr = 'const static Weight_t %s_pooldims[2] = {%d,%d};\n' % (self.name,pool_size[0],pool_size[1])
        f.write(mystr)

        s3 = input_shape[3] / pool_size[1]
        s2 = input_shape[2] / pool_size[0]
        s1 = input_shape[1]
        s0 = 1

        return (s0,s1,s2,s3)


    def get_num_weights(self):
        return 0
    
class Dense(Layer):
    def write(self,input_shape,f):
        weights_name = self.name + '_conv'
        bias_name = self.name + '_bias'
        w0 = self.weights[0]
        print 'dense weights: %d,%d' % w0.shape

        write_fixed_point_tensor(weights_name,w0.reshape(1,1,w0.shape[0],w0.shape[1]),f)
        write_fixed_point_tensor(bias_name,self.weights[1].reshape(self.weights[1].shape[0],1,1,1),f)

        s3 = w0.shape[1]
        return (1,1,1,s3)
        

    def get_num_weights(self):
        return 2


layer_map = {'Dense' : Dense, 'MaxPooling2D' : MaxPoolingLayer, 'Convolution2D' : ConvLayer}

def create_layer_objects(organized_layers):
    layer_counts = defaultdict(int)
    prev_layers = None
    layerobjs = []
    for layers in organized_layers:
        name = layers[0]['name']
        layer_counts[name] += 1
        layerobjs.append(layer_map[name](prev_layers,layers,layer_counts[name]))
        prev_layers = layers
        
    return layerobjs    
 
def save_model_to_c(model_name):

    with open('%s.json' %(model_name),'r') as f:
        config_json = f.read()

    model = model_from_json(config_json)
    model.load_weights('%s.h5' %(model_name))
    config = model.get_config()

    organized_layers = []

    first = True
    for layer in config['layers']:
        if layer['name'] in layer_map.keys():
            if not first:
                organized_layers.append(copy.deepcopy(this_layer))

            this_layer = []

        this_layer.append(layer)
        first = False

    organized_layers.append(copy.deepcopy(this_layer))

    layerobjs = create_layer_objects(organized_layers)

    weights_idx = 0
    for obj in layerobjs:
        weights_idx_begin = weights_idx
        weights_idx += obj.get_num_weights()

        for idx in range(weights_idx_begin,weights_idx):
            obj.add_weights(model.get_weights()[idx])

    f = open('test.c','w')
    input_shape = layerobjs[0].layers[0]['input_shape']
    input_shape = (1,input_shape[0],input_shape[1],input_shape[2])
    for obj in layerobjs:
        print obj.name,obj.dropout
        print input_shape
        input_shape = obj.write(input_shape,f)

        print input_shape
        print '---------'

if __name__ == '__main__':
    save_model_to_c('cnn_spec_try')
