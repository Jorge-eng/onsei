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
    const uint32_t num_weight_cols = layer->weights->dims[3];
    
    const Weight_t * const weights_start = layer->weights->x;
    const Weight_t * weights;
    const Weight_t * bias = layer->biases->x;
    const Weight_t * const input_start = in->x;
    const Weight_t * input = in->x;
    Weight_t * output = out->x;
    const uint32_t out_len = out->dims[0] * out->dims[1] * out->dims[2] * out->dims[3];
    
    Weight_t temp_weight;
    int8_t temp_scale;
    uint32_t i;
    
    const int16_t dropout_weight = (1 << QFIXEDPOINT) - layer->incoming_dropout;

    uint32_t iweightrow,iweightcol;
    int64_t accumulator;
    int64_t temp64;
    int32_t bias32;
    int8_t bias_scaling_diff;
    int32_t max = 0x80000000; //assumes two complement
    assert(layer->activation);
    
    
    for (iweightcol = 0; iweightcol < n_out; iweightcol++) {
        weights = weights_start + iweightcol;
        input = input_start;
        accumulator = 0;
    
        for (iweightrow = 0; iweightrow < n_in; iweightrow++) {
            //TODO OPTIMIZE THIS
            accumulator += (*weights) * (*input);
            weights += num_weight_cols;
            input++;
        }
        
        

        //dropout
        temp64 = accumulator * dropout_weight;
        temp64 >>= QFIXEDPOINT;
        
        //compensate for weight scaling
        bias_scaling_diff =  layer->weights->scale + in->scale - layer->biases->scale;
        
        bias32 = bias[iweightrow];
        bias32 <<= QFIXEDPOINT;
        
        if (bias_scaling_diff > 0) {
            //bias is bigger!
            bias32 <<= bias_scaling_diff;
        }
        else {
            bias32 >>= -bias_scaling_diff;
        }
        
        
        //add bias
        temp64 += bias32;
        
        temp64 >>= QFIXEDPOINT;
        temp64 >>= layer->weights->scale;
        temp64 >>= in->scale;
        
        if (temp64 > INT32_MAX) {
            temp64 = INT32_MAX;
        }
        
        if (temp64 < INT32_MIN) {
            temp64 = INT32_MIN;
        }
        
        if (temp64 > 127 || temp64 < -127) {
            int foo = 3;
            foo++;
        }

        layer->activation(&temp_weight,&temp_scale,(int32_t)temp64,0);
        assert(temp_scale == 0);
        
        *output = temp_weight;
        output++;
    }
    //printf("\n");
    
    
    max = 0;
    for (i = 0; i < out_len; i++) {
        if (abs(out->x[i]) > abs(max)) {
            max = out->x[i];
        }
    }
    

    out->scale = in->scale;

    
    printf("max=%d\t\ts=%d\n",max,out->scale);


}

/*
 does squash(W*x) for each unit (each "unit" is the "slice", 3rd dimension of your data tensor, etc.)
 */

ConstLayer_t tinytensor_create_fullyconnected_layer(const FullyConnectedLayer_t * static_def) {
    ConstLayer_t layer = {eval_fullyconnected,get_fullyconnectged_output_size,static_def};
    return layer;
}
