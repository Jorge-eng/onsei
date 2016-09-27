#!/usr/bin/python
import sys
import glob
import numpy as np
import copy
from keras.models import model_from_json
from keras.models import Sequential
from collections import defaultdict
from keras.optimizers import Adam
import pdb

QFIXEDPOINT = 12

k_activation_func_map = {'relu' : 'tinytensor_relu',
                         'sigmoid' : 'tinytensor_sigmoid',
                         'linear' :'tinytensor_linear',
                         'softmax' : 'tinytensor_linear',
                         'tanh' : 'tinytensor_tanh'}

def write_header(f):
    f.write('#include "tinytensor_lstm_layer.h"\n')    
    f.write('#include "tinytensor_conv_layer.h"\n')    
    f.write('#include "tinytensor_fullyconnected_layer.h"\n')    
    f.write('#include "tinytensor_math.h"\n')
    f.write('#include "tinytensor_net.h"\n')    


def write_fixed_point_tensor(name,weights,f):
    dims = weights.shape

    the_max = np.max(np.abs(weights))

    scale = 0
    if the_max > 0:
        scale = -int(np.ceil(np.log2(the_max + 1.0/128.0)))

    if scale < -8:
        scale = -8

    if scale > 8:
        scale = 8

    vec = np.round((weights.flatten() * (2**(QFIXEDPOINT+scale)))).astype(int).tolist()
    vecstr = ['%d' % v for v in vec]
    weights_name = '%s_x' % name
    dims_name = '%s_dims' % name
    myweights = 'const static Weight_t %s[%d] = {%s};\n' % (weights_name,len(vec),','.join(vecstr))
    mydims = 'const static uint32_t %s[4] = {%d,%d,%d,%d};\n' % (dims_name,dims[0],dims[1],dims[2],dims[3])
    mystruct = 'const static ConstTensor_t %s = {&%s[0],&%s[0],%d};\n' % (name,weights_name,dims_name,scale)

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
        zero_layer = self.layers[0]

        if zero_layer.has_key('layer'):
            zero_layer = zero_layer['layer']
            
        activation = zero_layer['activation']
        names = [layer['name'] for layer in self.layers]
        
        if 'Activation' in names:
            idx = names.index('Activation')
            if self.layers[idx].has_key('activation'):
                activation = self.layers[idx]['activation']
                
        return activation
        
    def add_weights(self,w):
        self.weights.append(w)



class ConvLayer(Layer):
    def get_max_pool_size(self):
        names = [layer['name'] for layer in self.layers]

        if 'MaxPooling2D' not in names:
            return (1,1)
        
        max_pool_layer = self.layers[names.index('MaxPooling2D')]

        return max_pool_layer['pool_size']
            
    def write(self,input_shape,f):


        pool_size = self.get_max_pool_size()
        poolvar = self.name + '_pool_size'
        write_uint32_array(poolvar,pool_size,f)

        border_mode = self.layers[0]['border_mode']
        w0 = self.weights[0]
        weights_name = self.name + '_conv'
        bias_name = self.name + '_bias'
        wn,wd = write_conv_weights(weights_name,w0,f)

        w1 = self.weights[1].reshape(self.weights[1].shape[0],1,1,1)
        bn,bd  = write_fixed_point_tensor(bias_name,w1,f)

        print 'conv weights: %d,%d,%d,%d' % w0.shape + ' border_mode=%s' % border_mode

        if border_mode == 'same':
            s3 = input_shape[3] / pool_size[1]
            s2 = input_shape[2] / pool_size[0]
            s1 = w0.shape[0]
            s0 = 1


        elif border_mode == 'valid':
            s3 = (input_shape[3] - w0.shape[3] + 1) / pool_size[1]
            s2 = (input_shape[2] - w0.shape[2] + 1) / pool_size[0]
            s1 = w0.shape[0]
            s0 = 1


        
        output_shape = (s0,s1,s2,s3)

        input_name,output_name = write_dims(input_shape,output_shape,self.name,f)
        
        f.write('const static ConvLayer2D_t %s = {&%s,&%s,%s,%s,%s,TOFIX(%f),%s};\n' % (self.name.lower(),weights_name,bias_name,output_name,input_name,poolvar,self.dropout,k_activation_func_map[self.get_activation()]))
        f.write('\n\n\n')

        return output_shape

    def write_creation(self,f):
        objname = self.name.lower()
        f.write('tinytensor_create_conv_layer(&%s)' % (objname))
                   
    def get_num_weights(self):
        return 2


    
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
            use_softmax = 1
            
        activation_function = k_activation_func_map[activation]
        f.write('const static FullyConnectedLayer_t %s = {&%s,&%s,%s,%s,TOFIX(%f),%s,%d};\n' % (self.name.lower(),weights_name,bias_name,output_name,input_name,self.dropout,activation_function,use_softmax))
        f.write('\n\n\n')
        return output_shape

    def write_creation(self,f):
        objname = self.name.lower()
        f.write('tinytensor_create_fullyconnected_layer(&%s)' % (objname))
                   
    def get_num_weights(self):
        return 2

class Lstm(Layer):
    def write(self,input_shape,f):
        nhidden = self.weights[1].shape[1]
        T = input_shape[2]
        output_shape = (1,1,T,nhidden)

        input_name,output_name = write_dims(input_shape,output_shape,self.name,f)


        gates = ['input_gate','cell','forget_gate','output_gate'] #in order of params seen in layer.get_params()

        w = self.weights
        names = []
        for i in range(4):
            gatename = self.name + '_weights_' + gates[i]
            biasname = self.name + '_biases_' + gates[i]

            idx = 3*i
            
            gi = w[idx]
            gr = w[idx + 1]

            g = np.concatenate((gi,gr),axis=0)
            g = g.transpose()
            g = g.reshape((1,1,g.shape[0],g.shape[1]))

                        
            b = w[idx + 2]
            b = b.reshape((1,1,1,b.shape[0]))

            write_fixed_point_tensor(gatename,g,f)
            write_fixed_point_tensor(biasname,b,f)

            names.extend([gatename,biasname])

        objname = self.name.lower()
        activation = self.get_activation()
        
        activation_function = k_activation_func_map[activation]

        gates_names_ptrs = []
        for g in names:
            gates_names_ptrs.append('&' + g)
            
        gates_names = ','.join(gates_names_ptrs)
        f.write('const static LstmLayer_t %s = {%s,%s,%s,TOFIX(%f),%s};\n\n\n' % (objname,gates_names,output_name,input_name,self.dropout,activation_function))
        return output_shape

    def write_creation(self,f):
        objname = self.name.lower()
        f.write('tinytensor_create_lstm_layer(&%s)' % (objname))


    def get_num_weights(self):
        return 12


layer_map = {'TimeDistributed' : Dense, 'Dense' : Dense, 'Convolution2D' : ConvLayer, 'LSTM' : Lstm}

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
            w = copy.deepcopy(model.get_weights()[idx])
                            
            obj.add_weights(w)

        print '-----'

    write_header(f)
    layer_input_shape = layerobjs[0].layers[0]['input_shape']

    if len(layer_input_shape) == 2:
        input_shape = (1,1,1,layer_input_shape[1])
    else:
        input_shape = (1,layer_input_shape[0],layer_input_shape[1],layer_input_shape[2])

        
    original_input_shape = input_shape
    for obj in layerobjs:
        print 'name=%s, dropout=%f' %(obj.name,obj.dropout)
        print input_shape,input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3]
        input_shape = obj.write(input_shape,f)

        print input_shape,input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3]
        print '---------'

    f.write('\n\n\n')
    f.write('static ConstLayer_t _layers[%d];\n' % len(layerobjs))
    f.write('static ConstSequentialNetwork_t net = {&_layers[0],%d};\n' % len(layerobjs))

    f.write('static ConstSequentialNetwork_t initialize_network(void) {\n\n')

    for idx,obj in enumerate(layerobjs):
        f.write('  _layers[%d] = ' % idx)
        obj.write_creation(f)
        f.write(';\n')

    f.write('  return net;\n')
    f.write('\n}')
    return original_input_shape

def save_model_to_c_from_file(json_name,h5file_name=None):

    model = get_model(json_name)
    if type(h5file_name) is str:
        h5file_name = [h5file_name]
    elif h5file_name is None:
        h5file_name = glob.glob(json_name.replace('.json','_ep*.h5'))
        h5file_name.sort()
    
    for weight_file in h5file_name:
        model_name = weight_file.split('.')[0]
        model_name = model_name.replace('+','_')
        print 'model_name %s' % model_name

        model.load_weights(weight_file)    
        save_model_to_c(model,model_name)

def save_all_to_c_from_files(json_name):
    model_name = json_name.split('.')[0]
    model_name = model_name.replace('+','_')
    print 'model_name %s' % model_name

    model = get_model(json_name) 

def get_model(json_name,h5file_name=None):

    with open(json_name,'r') as f:
        config_json = f.read()
        
    print 'read model from %s' % json_name
    print 'compiling...'
    model = model_from_json(config_json)

    if h5file_name is not None:
        weights_filename = h5file_name
        print 'loading weights from %s' % weights_filename
        model.load_weights(weights_filename)

    return model

def get_model_scaling(model,input_shape):
    M = 1000
    N = 1;
    for s in input_shape:
        N *= s

    y = []
    for i in range(M):
        x = 2 * np.random.rand(N).reshape(input_shape) - 1
        y.append(model.predict(x))

    
    ranges = []
    for yy in y:
        ranges.append(np.max(yy) - np.min(yy))

    max_range = np.max(ranges)

    return np.log2(max_range)

def save_model_to_c(model,name):

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

    outname = '%s.c' % name
    print 'writing to %s' % outname
    f = open(outname,'w')

    write_sequential_network(layerobjs,model,f)
    f.close()

if __name__ == '__main__':
    json_name = sys.argv[1]
    pdb.set_trace()
    h5file_name = sys.argv[2] if (len(sys.argv) >= 3) else None

    save_model_to_c_from_file(json_name,h5file_name)
