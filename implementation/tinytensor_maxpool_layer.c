#include "tinytensor_maxpool_layer.h"
#include <assert.h>


static void get_maxpool_output_size(const void * context,uint32_t * dims) {
    const MaxPoolLayer_t * layer = (const MaxPoolLayer_t *)context;
    
    uint32_t i;
    for (i = 0; i < TENSOR_DIM; i++) {
        dims[i] = layer->output_dims[i];
    }
    
}




/*   General idea is this: Given image of 4x4, a 2x4 max pool will produce a 2x1 image
 *   Another example 4x4 input, a 2x2 will produce a 2x2 result
 *   This is per-image
 *
 *   So input tensor will be 1 x N x P x Q
 *   output tensor will be 1 x N x L x M
 *
 *   where L = P / maxpool_cols, M = Q / max_pool_rows
 */

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

static void eval_maxpool(const void * context,Tensor_t * out,const Tensor_t * in) {
    const MaxPoolLayer_t * layer = context;
    
    const Weight_t * input_image_start = in->x;
    const uint32_t input_image_size = in->dims[3] * in->dims[2];
    
    Weight_t * output_image_start = out->x;
    const uint32_t output_image_size = out->dims[3] * out->dims[2];
    
    const uint32_t num_row_regions = in->dims[2] / layer->pool_dims[0];
    const uint32_t num_col_regions = in->dims[3] / layer->pool_dims[1];
    
//    const uint32_t num_pool_rows = layer->pool_dims[0];
//    const uint32_t num_pool_cols = layer->pool_dims[1];
    const uint32_t num_input_image_cols = in->dims[3];
    const uint32_t num_output_image_cols = out->dims[3];

    const uint32_t leftover_rows = in->dims[2] %  layer->pool_dims[0];
    const uint32_t leftover_cols = in->dims[3] %  layer->pool_dims[1];

    const uint32_t num_images = in->dims[1];
    uint32_t iimage,iregionrow,iregioncol;
    
    
    assert (num_row_regions == out->dims[2] || (num_row_regions + 1 == out->dims[2] && leftover_rows));
    assert (num_col_regions == out->dims[3] || num_col_regions + 1 == out->dims[3] && leftover_cols);
    
   // assert( in->dims[3] %  layer->pool_dims[1] == 0);
   // assert( in->dims[2] %  layer->pool_dims[0] == 0);

    for (iimage = 0; iimage < num_images; iimage++) {
        {
            const Weight_t * input_image_row = input_image_start;
            Weight_t * output_image_row = output_image_start;
            
            //main part of image
            for (iregionrow = 0; iregionrow < num_row_regions; iregionrow++) {
                
                //start of row
                
                
                const Weight_t * prow = &input_image_row[0];
                
                for (iregioncol = 0; iregioncol < num_col_regions; iregioncol++) {
                    output_image_row[iregioncol] = get_max_in_region(layer->pool_dims[0],
                                                                     layer->pool_dims[1],
                                                                     prow,
                                                                     num_input_image_cols);
                    
                    prow += layer->pool_dims[1];
                }
                
                input_image_row += layer->pool_dims[0] * num_input_image_cols;
                output_image_row += num_output_image_cols;
            }
        }
        
        //handle right edge
        if (leftover_cols > 0) {
            
            //start input/output rows positioned at the very right edge
            const Weight_t * input_image_row = &input_image_start[num_col_regions*layer->pool_dims[1]];
            Weight_t * output_image_row = &output_image_start[num_col_regions];
            
            //go down right edge
            for (iregionrow = 0; iregionrow < num_row_regions; iregionrow++) {
                *output_image_row = get_max_in_region(layer->pool_dims[0],leftover_cols,input_image_row,num_input_image_cols);
                
                input_image_row += layer->pool_dims[0] * num_input_image_cols;
                output_image_row += num_output_image_cols;

            }
            
        }
        
        //handle bottom edge
        if (leftover_rows > 0) {
            //start input/output rows position at teh very bottom edge
            const Weight_t * p = input_image_start + num_input_image_cols * layer->pool_dims[0] * num_row_regions;
            Weight_t * output_image_row = output_image_start + num_output_image_cols * num_row_regions;
            
            for (iregioncol = 0; iregioncol < num_col_regions; iregioncol++) {
                output_image_row[iregioncol] = get_max_in_region(leftover_rows, layer->pool_dims[1],p, num_input_image_cols);
                p += layer->pool_dims[1];
            }
        }
        
        //handle lower right corner
        if (leftover_rows > 0 && leftover_cols > 0) {
            const Weight_t * p = input_image_start + num_input_image_cols * layer->pool_dims[0] * num_row_regions + layer->pool_dims[1]*num_col_regions;
            Weight_t * output_image_row = output_image_start + num_output_image_cols * num_row_regions;

            output_image_row[num_col_regions] = get_max_in_region(leftover_rows, leftover_cols,p, num_input_image_cols);
        }
        
        
        //next image
        input_image_start += input_image_size;
        output_image_start += output_image_size;
    }
    
}

ConstLayer_t tinytensor_create_maxpool_layer(const MaxPoolLayer_t * static_def) {
    ConstLayer_t layer = {eval_maxpool,get_maxpool_output_size,static_def};
    return layer;
}




