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

static void eval_fullyconnected(const void * context,void * layer_state,Tensor_t * out,const Tensor_t * in,ELayer_t prev_layer_type) {
    const FullyConnectedLayer_t * layer = (const FullyConnectedLayer_t *)context;
    const uint32_t n_out = layer->output_dims[3];
    const uint32_t num_weight_cols = layer->weights->dims[3];
    
    const Weight_t * const weights_start = layer->weights->x;
    const Weight_t * weights;
    const Weight_t * bias = layer->biases->x;
    const Weight_t * input;
    const Weight_t * input_start;
    Weight_t * output = out->x;
    const uint32_t out_len = out->dims[0] * out->dims[1] * out->dims[2] * out->dims[3];
    
    Weight_t temp_weight;
    int8_t temp_scale;
    uint32_t i;
    
    const int16_t dropout_weight = (1 << QFIXEDPOINT) - layer->incoming_dropout;
    //const int16_t dropout_weight = 128;
    uint32_t iweightrow,iweightcol;
    int32_t accumulator;
    int32_t temp32;
    int32_t bias32;
    int8_t bias_scaling_diff;
    int8_t delta_descale;
    int8_t descale = 0;
    Weight_t * p;
    int32_t max = 0x80000000; //assumes two complement
    
    uint32_t n_in = 0;
    
    switch (prev_layer_type) {
        case conv_layer:
            //flatten
            n_in = in->dims[0]*in->dims[1]*in->dims[2]*in->dims[3];
            input_start = in->x;
            break;
            
        case lstm_layer:
            //pick off last vector
            n_in = in->dims[3];
            input_start = in->x + (in->dims[2] - 1) * in->dims[3];
            break;
        
        default:
            n_in = in->dims[3];
            input_start = in->x;
    }

    
    assert(layer->activation);
    assert(n_in == layer->weights->dims[2]);
    
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
        temp32 = accumulator * dropout_weight;
        temp32 >>= QFIXEDPOINT;
        
        //compensate for weight scaling
        bias_scaling_diff =  layer->weights->scale + in->scale - layer->biases->scale;
        
        bias32 = bias[iweightcol];
        bias32 <<= QFIXEDPOINT;
        
        if (bias_scaling_diff > 0) {
            bias32 <<= bias_scaling_diff;
        }
        else {
            bias32 >>= -bias_scaling_diff;
        }
        
        
        //add bias
        temp32 += bias32;
        
        //rounding
        temp32 += (1 << (QFIXEDPOINT - 1));
        temp32 >>= QFIXEDPOINT;
        
        if (layer->weights->scale > 0) {
            temp32 >>= layer->weights->scale;
        }
        else if (layer->weights->scale < 0) {
            temp32 <<= -layer->weights->scale;
        }
        
      
        temp32 >>= descale;
        
        //descaling madness
        delta_descale = tiny_tensor_get_descaling(temp32);
        
        if (delta_descale) {
            temp32 >>= delta_descale; //update current temp value
            descale += delta_descale; //update descale
            
            //backtrack -- right shift all previous by delta_scale
            //so fucking inefficient.
            for (p = out->x; p < output; p++) {
                *p >>= delta_descale;
            }
        }

        
        
        layer->activation(&temp_weight,&temp_scale,temp32,in->scale - descale);
        
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
    

    out->scale = temp_scale;

    
    //printf("max=%d\t\ts=%d\n",max,out->scale);
    printf("%d\t",max); fflush(0);


}

/*
 does squash(W*x) for each unit (each "unit" is the "slice", 3rd dimension of your data tensor, etc.)
 */



ConstLayer_t tinytensor_create_fullyconnected_layer(const FullyConnectedLayer_t * static_def) {
    ConstLayer_t layer = {eval_fullyconnected,get_fullyconnectged_output_size,full_layer,static_def,NULL,NULL};
    return layer;
}
