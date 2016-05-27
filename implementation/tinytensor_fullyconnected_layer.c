#include "tinytensor_fullyconnected_layer.h"
#include "tinytensor_memory.h"
#include "tinytensor_math.h"
#include <assert.h>

static void get_fullyconnectged_output_size(const void * context,uint32_t * dims) {
    const FullyConnectedLayer_t * layer = (const FullyConnectedLayer_t *)context;
    
    uint32_t i;
    for (i = 0; i < TENSOR_DIM; i++) {
        dims[i] = layer->output_dims[i];
    }
    
}

static void eval_fullyconnected(const void * context,Tensor_t * out,const Tensor_t * in) {
    const FullyConnectedLayer_t * layer = (const FullyConnectedLayer_t *)context;
    const uint32_t n_in = in->dims[0]*in->dims[1]*in->dims[2]*in->dims[3];
    const uint32_t n_out = layer->output_dims[3];
    
    const Weight_t * weights = layer->weights->x;
    const Weight_t * bias = layer->biases->x;
    const Weight_t * input = in->x;
    Weight_t * output = out->x;
    
    const int16_t dropout_weight = (1 << QFIXEDPOINT) - layer->incoming_dropout;

    uint32_t iweightrow,iweightcol;
    int32_t accumulator;
    
    assert(layer->activation);

    
    for (iweightrow = 0; iweightrow < n_out; iweightrow++) {
        
        accumulator = 0;
    
        for (iweightcol = 0; iweightcol < n_in; iweightcol++) {
            //TODO OPTIMIZE THIS
            accumulator += weights[iweightcol] * input[iweightcol];
        }
        
        
        accumulator >>= QFIXEDPOINT;
        accumulator *= dropout_weight;
        accumulator >>= QFIXEDPOINT;
        accumulator += bias[iweightrow];
        
        output[iweightrow] = layer->activation(accumulator);
        
        weights += n_in;
    }

}

/*
 does squash(W*x) for each unit (each "unit" is the "slice", 3rd dimension of your data tensor, etc.)
 */

ConstLayer_t tinytensor_create_fullyconnected_layer(const FullyConnectedLayer_t * static_def) {
    ConstLayer_t layer = {eval_fullyconnected,get_fullyconnectged_output_size,static_def};
    return layer;
}
