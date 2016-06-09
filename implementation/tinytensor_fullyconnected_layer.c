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
    const uint32_t out_len = out->dims[0] * out->dims[1] * out->dims[2] * out->dims[3];

    Weight_t max_weight = 0;
    int8_t max_scale = 8;
    
    Weight_t temp_weight;
    int8_t temp_scale;
    int8_t delta_scale;
    uint32_t i;
    
    const int16_t dropout_weight = (1 << QFIXEDPOINT) - layer->incoming_dropout;

    uint32_t iweightrow,iweightcol;
    int64_t accumulator;
    int64_t temp64;
    int32_t bias32;
    int8_t bias_scaling_diff;
    int32_t max = 0x80000000; //assumes two complement
    int8_t current_scale;
    int8_t new_scale;
    Weight_t descaled_value;
    assert(layer->activation);
    
    
    for (iweightrow = 0; iweightrow < n_out; iweightrow++) {
        
        accumulator = 0;
    
        for (iweightcol = 0; iweightcol < n_in; iweightcol++) {
            //TODO OPTIMIZE THIS
            accumulator += weights[iweightcol] * input[iweightcol];
        }
        
        

        //dropout
        temp64 = accumulator * dropout_weight;
        temp64 >>= QFIXEDPOINT;
        temp64 >>= layer->weights->scale;
        
        //compensate for weight scaling
        current_scale = in->scale;
        bias_scaling_diff = current_scale - layer->biases->scale;
        
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
        
        if (temp64 > INT32_MAX) {
            temp64 = INT32_MAX;
        }
        
        if (temp64 < INT32_MIN) {
            temp64 = INT32_MIN;
        }
        
        temp64 >>= 2;

        tinytensor_descale(&descaled_value,&new_scale,(int32_t)temp64,current_scale);
        //printf("descaled=%d,s=%d,  input=%d,s=%d\n",descaled_value,new_scale,(int32_t)temp64,current_scale);

        //squash
        //Weight_t * y, int8_t * out_scale, int32_t x,int8_t in_scale
        layer->activation(&temp_weight,&temp_scale,(int32_t)temp64,current_scale);
        //printf("%d,",accumulator);
        
        if (tiny_tensor_compare_scaled_numbers(temp_weight,temp_scale,max_weight,max_scale) > 0) {
            if (temp_scale == 0)
            printf("val=%d,s=%d   squashed=%d,s=%d\n",descaled_value,new_scale,temp_weight,temp_scale);

            max_weight = temp_weight;
            max_scale = temp_scale;
        }

        output[iweightrow] = temp_weight;
        
        weights += n_in;
    }
    //printf("\n");
    
    
    delta_scale = tiny_tensor_get_scaling(max_weight);
    
    //scale output tensor
    //printf("delta_scale=%d\n",delta_scale);
    if (delta_scale > 0) {
        for (i = 0; i < out_len; i++) {
            //if (i!=0) printf(",");
            //printf("%d ",out->x[i]);
            out->x[i] <<= delta_scale;
            //printf("%d",out->x[i]);
            
        }
        //printf("\n");
    }
    
    max = 0;
    for (i = 0; i < out_len; i++) {
        if (abs(out->x[i]) > abs(max)) {
            max = out->x[i];
        }
    }
    
    
    out->scale = max_scale + delta_scale - 2;

    
    printf("max=%d,s=%d   delta=%d\n",max,out->scale,delta_scale);


}

/*
 does squash(W*x) for each unit (each "unit" is the "slice", 3rd dimension of your data tensor, etc.)
 */

ConstLayer_t tinytensor_create_fullyconnected_layer(const FullyConnectedLayer_t * static_def) {
    ConstLayer_t layer = {eval_fullyconnected,get_fullyconnectged_output_size,static_def};
    return layer;
}
