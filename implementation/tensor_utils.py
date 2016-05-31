#!/usr/bin/python
import numpy as np
import copy
from keras.models import model_from_json
from collections import defaultdict

k_activation_func_map = {'relu' : 'tinytensor_relu', 'softmax' : 'tinytensor_sigmoid', 'linear' :'tinytensor_linear'}

def write_header(f):
    f.write('#include "tinytensor_conv_layer.h"\n')    
    f.write('#include "tinytensor_maxpool_layer.h"\n')    
    f.write('#include "tinytensor_fullyconnected_layer.h"\n')    
    f.write('#include "tinytensor_math.h"\n')
    f.write('#include "tinytensor_net.h"\n')    


def write_fixed_point_tensor(name,weights,f):
    dims = weights.shape
    vec = (weights.flatten() * (2**7)).astype(int).tolist()
    vecstr = ['%d' % v for v in vec]
    weights_name = '%s_x' % name
    dims_name = '%s_dims' % name
    myweights = 'const static Weight_t %s[%d] = {%s};\n' % (weights_name,len(vec),','.join(vecstr))
    mydims = 'const static uint32_t %s[4] = {%d,%d,%d,%d};\n' % (dims_name,dims[0],dims[1],dims[2],dims[3])
    mystruct = 'const static ConstTensor_t %s = {&%s[0],&%s[0]};\n' % (name,weights_name,dims_name)

    f.write(myweights)
    f.write(mydims)
    f.write(mystruct)
    f.write('\n')

    return weights_name,dims_name

def write_conv_weights(name,weights,f):
    w = copy.deepcopy(weights)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
           w[i][j] = w[i][j][::-1,::-1]

    weights_name,dims_name = write_fixed_point_tensor(name,w,f) 

    return weights_name,dims_name

class Layer(object):
    def __init__(self,prevlayers,layers,layer_count):
        self.name = '%s_%02d' % (layers[0]['name'],layer_count)
        self.layer_name = self.name.lower() + '_layer'
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

    def get_activation(self):
        activation = 'linear'
        names = [layer['name'] for layer in self.layers]
        
        if 'Activation' in names:
            idx = names.index('Activation')
            if self.layers[idx].has_key('activation'):
                activation = self.layers[idx]['activation']
                
        return activation
        
    def add_weights(self,w):
        self.weights.append(w)



class ConvLayer(Layer):
    def write(self,input_shape,f):
        border_mode = self.layers[0]['border_mode']
        w0 = self.weights[0]
        weights_name = self.name + '_conv'
        bias_name = self.name + '_bias'
        wn,wd = write_conv_weights(weights_name,w0,f)
        bn,bd  = write_fixed_point_tensor(bias_name,self.weights[1].reshape(self.weights[1].shape[0],1,1,1),f)

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

        output_shape = (s0,s1,s2,s3)

        input_name,output_name = write_dims(input_shape,output_shape,self.name,f)
        
        f.write('const static ConvLayer2D_t %s = {&%s,&%s,%s,%s,TOFIX(%f),%s};\n' % (self.name.lower(),weights_name,bias_name,output_name,input_name,self.dropout,k_activation_func_map[self.get_activation()]))
        f.write('\n\n\n')

        return output_shape

    def write_creation(self,f):
        objname = self.name.lower()
        f.write('tinytensor_create_conv_layer(&%s)' % (objname))
                   
    def get_num_weights(self):
        return 2

class MaxPoolingLayer(Layer):
    def write(self,input_shape,f):
        pool_size = self.layers[0]['pool_size']
        print 'pool dims: %d,%d' % pool_size

        s3 = input_shape[3] / pool_size[1]
        s2 = input_shape[2] / pool_size[0]
        s1 = input_shape[1]
        s0 = 1

        output_shape = (s0,s1,s2,s3)

        pool_dims_name = '%s_pool_dims' % self.name
        write_uint32_array(pool_dims_name,pool_size,f)
        input_name,output_name = write_dims(input_shape,output_shape,self.name,f)

        f.write('const static MaxPoolLayer_t %s = {%s,%s,%s};\n' % (self.name.lower(),pool_dims_name,output_name,input_name))
        f.write('\n\n\n')
        
        return output_shape

    def write_creation(self,f):
        objname = self.name.lower()
        f.write('tinytensor_create_maxpool_layer(&%s)' % (objname))


    def get_num_weights(self):
        return 0
    
class Dense(Layer):
    def write(self,input_shape,f):
        weights_name = self.name + '_full'
        bias_name = self.name + '_bias'
        w0 = self.weights[0]
        print 'dense weights: %d,%d' % w0.shape

        wn,wd = write_fixed_point_tensor(weights_name,w0.reshape(1,1,w0.shape[0],w0.shape[1]),f)
        bn,bd = write_fixed_point_tensor(bias_name,self.weights[1].reshape(self.weights[1].shape[0],1,1,1),f)

        output_shape = (1,1,1,w0.shape[1])

        input_name,output_name = write_dims(input_shape,output_shape,self.name,f)

        activation = self.get_activation()
        hardmax = 0
        if activation == 'softmax':
            hardmax = 1
            
        activation_function = k_activation_func_map[activation]
        f.write('const static FullyConnectedLayer_t %s = {&%s,&%s,%s,%s,TOFIX(%f),%s,%d};\n' % (self.name.lower(),weights_name,bias_name,output_name,input_name,self.dropout,activation_function,hardmax))
        f.write('\n\n\n')
        return output_shape

    def write_creation(self,f):
        objname = self.name.lower()
        f.write('tinytensor_create_fullyconnected_layer(&%s)' % (objname))
                   
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


def write_uint32_array(name,values,f):
    strvalues = ['%d' % v for v in values]
    arrvalue = ','.join(strvalues)
    f.write('const static uint32_t %s[%d] = {%s};\n' % (name,len(values),arrvalue))

def write_dims(input_shape,output_shape,name,f):
    
    input_name = '%s_input_dims' % name
    output_name = '%s_output_dims' % name

    write_uint32_array(input_name,input_shape,f)
    write_uint32_array(output_name,output_shape,f)

    return input_name,output_name

def write_sequential_network(layerobjs,model,f):

    #set weights
    weights_idx = 0
    for obj in layerobjs:
        weights_idx_begin = weights_idx
        weights_idx += obj.get_num_weights()

        for idx in range(weights_idx_begin,weights_idx):
            obj.add_weights(model.get_weights()[idx])


    write_header(f)
    
    input_shape = layerobjs[0].layers[0]['input_shape']
    input_shape = (1,input_shape[0],input_shape[1],input_shape[2])
    for obj in layerobjs:
        print obj.name,obj.dropout
        print input_shape
        input_shape = obj.write(input_shape,f)

        print input_shape
        print '---------'

    f.write('\n\n\n')
    f.write('static ConstLayer_t _layers[%d];\n' % len(layerobjs))
    f.write('static ConstSequentialNetwork_t net = {&_layers[0],%d};\n' % len(layerobjs))

    f.write('ConstSequentialNetwork_t initialize_network(void) {\n\n')

    for idx,obj in enumerate(layerobjs):
        f.write('  _layers[%d] = ' % idx)
        obj.write_creation(f)
        f.write(';\n')

    f.write('  return net;\n')
    f.write('\n}')

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

    f = open('test.c','w')

    write_sequential_network(layerobjs,model,f)

    f.close()

if __name__ == '__main__':
    save_model_to_c('cnn_spec_try')
