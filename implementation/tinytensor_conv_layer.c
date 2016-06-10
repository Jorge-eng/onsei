#include "tinytensor_conv_layer.h"
#include "tinytensor_memory.h"
#include "tinytensor_math.h"
#include <assert.h>

static void get_conv2d_output_size(const void * context,uint32_t * dims) {
    const ConvLayer2D_t * layer = (const ConvLayer2D_t *)context;
    
    uint32_t i;
    for (i = 0; i < TENSOR_DIM; i++) {
        dims[i] = layer->output_dims[i];
    }
    
}

inline static Weight_t get_max_in_region(const uint32_t num_rows, const uint32_t num_cols, const Weight_t * const startPos, const uint32_t num_image_cols) {
    uint32_t j,i;
    Weight_t max = -MAX_WEIGHT;
    
    const Weight_t * p = startPos;
    for (j = 0; j < num_rows; j++) {
        for (i = 0; i < num_cols; i++) {
            max = p[i] > max ? p[i] : max;
        }
        
        p += num_image_cols;
    }
    
    return max;
}

static void eval_maxpool_single_image(const uint32_t * pool_dims,Weight_t * const output_image_start,const uint32_t num_output_image_rows, const uint32_t num_output_image_cols, const Weight_t * input_image_start, const uint32_t num_input_image_rows, const uint32_t num_input_image_cols) {
    
    
    const uint32_t num_row_regions = num_input_image_rows / pool_dims[0];
    const uint32_t num_col_regions = num_input_image_cols / pool_dims[1];
    
    const uint32_t leftover_rows = num_input_image_rows %  pool_dims[0];
    const uint32_t leftover_cols = num_input_image_cols %  pool_dims[1];
    
    uint32_t iregionrow,iregioncol;
    
    
   // assert (num_row_regions == out->dims[2] || (num_row_regions + 1 == out->dims[2] && leftover_rows));
    //assert (num_col_regions == out->dims[3] || num_col_regions + 1 == out->dims[3] && leftover_cols);
    
    
    {
        const Weight_t * input_image_row = input_image_start;
        Weight_t * output_image_row = output_image_start;
        
        //main part of image
        for (iregionrow = 0; iregionrow < num_row_regions; iregionrow++) {
            
            //start of row
            
            
            const Weight_t * prow = &input_image_row[0];
            
            for (iregioncol = 0; iregioncol < num_col_regions; iregioncol++) {
                output_image_row[iregioncol] = get_max_in_region(pool_dims[0],
                                                                 pool_dims[1],
                                                                 prow,
                                                                 num_input_image_cols);
                
                prow += pool_dims[1];
            }
            
            input_image_row += pool_dims[0] * num_input_image_cols;
            output_image_row += num_output_image_cols;
        }
    }
    
    //handle right edge
    if (leftover_cols > 0) {
        
        //start input/output rows positioned at the very right edge
        const Weight_t * input_image_row = &input_image_start[num_col_regions*pool_dims[1]];
        Weight_t * output_image_row = &output_image_start[num_col_regions];
        
        //go down right edge
        for (iregionrow = 0; iregionrow < num_row_regions; iregionrow++) {
            *output_image_row = get_max_in_region(pool_dims[0],leftover_cols,input_image_row,num_input_image_cols);
            
            input_image_row += pool_dims[0] * num_input_image_cols;
            output_image_row += num_output_image_cols;
            
        }
        
    }
    
    //handle bottom edge
    if (leftover_rows > 0) {
        //start input/output rows position at teh very bottom edge
        const Weight_t * p = input_image_start + num_input_image_cols * pool_dims[0] * num_row_regions;
        Weight_t * output_image_row = output_image_start + num_output_image_cols * num_row_regions;
        
        for (iregioncol = 0; iregioncol < num_col_regions; iregioncol++) {
            output_image_row[iregioncol] = get_max_in_region(leftover_rows, pool_dims[1],p, num_input_image_cols);
            p += pool_dims[1];
        }
    }
    
    //handle lower right corner
    if (leftover_rows > 0 && leftover_cols > 0) {
        const Weight_t * p = input_image_start + num_input_image_cols * pool_dims[0] * num_row_regions + pool_dims[1]*num_col_regions;
        Weight_t * output_image_row = output_image_start + num_output_image_cols * num_row_regions;
        
        output_image_row[num_col_regions] = get_max_in_region(leftover_rows, leftover_cols,p, num_input_image_cols);
    }
    
}


static void eval_conv2d_direct(const void * context,Tensor_t * out,const Tensor_t * in) {
    const ConvLayer2D_t * layer = (ConvLayer2D_t *)context;
    
    uint32_t iout;
    uint32_t i;
    const uint32_t out_len = out->dims[0] * out->dims[1] * out->dims[2] * out->dims[3];

    
    const uint32_t num_out_images = layer->weights->dims[0];
    const uint32_t num_images = layer->weights->dims[1];
    const uint32_t num_weights_rows = layer->weights->dims[2];
    const uint32_t num_weights_cols = layer->weights->dims[3];
    const uint32_t num_image_rows = in->dims[2];
    const uint32_t num_image_cols = in->dims[3];
    
    const Weight_t * weight_start = layer->weights->x;
    const uint32_t weight_filter_size = layer->weights->dims[1] * layer->weights->dims[2] * layer->weights->dims[3];
    
    const Weight_t * const image_start = in->x;    
    const Weight_t * bias = layer->biases->x;
        
    Weight_t * out_start = out->x;
    const uint32_t out_image_size = layer->output_dims[3] * layer->output_dims[2];
    
    assert(layer->weights->dims[1] == in->dims[1]);
    
    for (i = 0; i < TENSOR_DIM; i++) {
        assert(in->dims[i] == layer->input_dims[i]);
    }
    
    //make sure output tensor is ready for this
    for (i = 0; i < TENSOR_DIM; i++) {
        assert(out->dims[i] == layer->output_dims[i]);
    }

    // each of M filters is a 3D tensor (multiple "images") + bias weight
    // each filter has dimensions of 1 x N x P x Q, where P x Q is the filter image size
    // thus the filter, i.e. the weights will have dimensions of M x N x P x Q
    // the biases will have dimensions of M x 1 x 1 x 1
    //
    // there are N images, of size U x V
    // thus dims of the input are 1 x N x U x V
    //
    // and dims of the output are
    // 1 x M x ((U - P + 1)) / pool_y   x    (V - Q + 1)  / pool_x
    //
    // so the idea is to build output images
    // from each filter
    
    for (iout = 0; iout < num_out_images; iout++) {
        
        tinytensor_convolve3d_direct_maxpooling(
                                                out_start,
                                                layer->max_pool_dims,
                                                weight_start,
                                                layer->weights->scale,
                                                image_start,
                                                in->scale,
                                                *bias,
                                                layer->biases->scale,
                                                num_weights_rows,
                                                num_weights_cols,
                                                num_image_rows ,
                                                num_image_cols,
                                                num_images,
                                                layer->incoming_dropout,
                                                layer->activation);
        
        
        bias += 1;
        out_start += out_image_size;
        weight_start += weight_filter_size;
    }
    
    
    
    int32_t max = 0;
    for (i = 0; i < out_len; i++) {
        if (abs(out->x[i]) > abs(max)) {
            max = out->x[i];
        }
    }
    
    out->scale = in->scale;
    
    //printf("max=%d\t\ts=%d\n",max,out->scale);

    
}

ConstLayer_t tinytensor_create_conv_layer(const ConvLayer2D_t * static_def) {
    ConstLayer_t layer = {eval_conv2d_direct,get_conv2d_output_size,static_def};
    return layer;
}



