#include "tinytensor_math.h"
#include <assert.h>

#define MAX_MAX_POOL_SIZE (8)

const static int8_t tanh_table[356] = {0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,36,37,38,39,40,41,42,43,44,44,45,46,47,48,49,50,51,51,52,53,54,55,55,56,57,58,59,59,60,61,62,63,63,64,65,65,66,67,68,68,69,70,70,71,72,73,73,74,75,75,76,76,77,78,78,79,80,80,81,81,82,83,83,84,84,85,85,86,86,87,88,88,89,89,90,90,91,91,92,92,93,93,93,94,94,95,95,96,96,97,97,97,98,98,99,99,99,100,100,101,101,101,102,102,102,103,103,103,104,104,104,105,105,105,106,106,106,107,107,107,108,108,108,108,109,109,109,109,110,110,110,110,111,111,111,111,112,112,112,112,113,113,113,113,113,114,114,114,114,114,115,115,115,115,115,116,116,116,116,116,116,117,117,117,117,117,117,118,118,118,118,118,118,118,119,119,119,119,119,119,119,119,120,120,120,120,120,120,120,120,120,121,121,121,121,121,121,121,121,121,121,122,122,122,122,122,122,122,122,122,122,122,122,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,124,124,124,124,124,124,124,124,124,124,124,124,124,124,124,124,124,124,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,125,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,127};


int32_t tinymath_abs_int8(int8_t x) {
    if (((uint8_t)x) == 0x80) {
        x++;
    }
    
    return x >= 0 ? x : -x;
}

int32_t tinymath_abs_int32(int32_t x) {
    if (((uint32_t)x) == 0x80000000) {
        x++;
    }

    return x >= 0 ? x : -x;
}

void tinytensor_descale(Weight_t * y, int8_t * out_scale, int32_t x, int8_t in_scale) {
    //make it fit in 8 bits
    uint8_t i;
    int32_t ux = tinymath_abs_int32(x);
    
    for (i = 0; i < 24; i++) {
        if (  (ux >> i) <= 127 ) {
            break;
        }
    }
    
    *out_scale = in_scale - i;
    *y = x >> i;
    
}

void tinytensor_tanh(Weight_t * y, int8_t * out_scale, int32_t x,int8_t in_scale) {
    const static uint32_t k_max_len = sizeof(tanh_table) / sizeof(tanh_table[0]);
    const uint8_t sign = x < 0;
    Weight_t yy = 0x7F;
    
    *out_scale = 0;
    
    x = tinymath_abs_int32(x);
    
    if (in_scale > 0) {
        x >>= in_scale;
    }
    
    if (x < k_max_len) {
        yy = tanh_table[x];
    }


    if (sign) {
        *y = -yy;
        return;
    }

    *y = yy;
}

//(tanh + 1) / 2 == sigmoid
//crud, but should work okay
void tinytensor_sigmoid(Weight_t * y, int8_t * out_scale, int32_t x,int8_t in_scale) {
    Weight_t tanh;
    int16_t temp16;
    
    tinytensor_tanh(&tanh,out_scale,x,in_scale);
    temp16 = tanh;
    
    //add one
    temp16 += (1 << QFIXEDPOINT);
    
    //divide by two
    temp16 >>= 1;
    
    if (temp16 > MAX_WEIGHT) {
        temp16 = MAX_WEIGHT;
    }
    
    *y = (Weight_t)temp16;
    *out_scale = 0;
}

void tinytensor_linear(Weight_t * y, int8_t * out_scale, int32_t x,int8_t in_scale) {
    
    if (x > MAX_WEIGHT) {
        x = MAX_WEIGHT;
    }
    
    if (x < -MAX_WEIGHT) {
        x = -MAX_WEIGHT;
    }
    
    *y = x;
    *out_scale = in_scale;
}


void tinytensor_relu(Weight_t * y, int8_t * out_scale, int32_t x,int8_t in_scale) {
    x =  x < 0 ? 0 : x;
    
    if (x > MAX_WEIGHT) {
        x = MAX_WEIGHT;
    }
    
    *y = x;
    *out_scale = in_scale;

}




int8_t tiny_tensor_get_scaling(Weight_t max_weight) {
    int8_t i;
    
    if (max_weight == 0) {
        return 0;
    }

    //find max scaling
    max_weight = tinymath_abs_int8(max_weight);
    
    for (i = 0; i < 8; i++) {
        if (( ((int16_t)max_weight) << i) > MAX_WEIGHT/2) {
            break;
        }
    }
    
    return i;
    
}

int8_t tiny_tensor_compare_scaled_numbers(const Weight_t x1, const int8_t scale1, const Weight_t x2, const int8_t scale2) {
    int32_t xx1 = x1 << 16;
    int32_t xx2 = x2 << 16;
    
    xx1 = tinymath_abs_int32(xx1);
    xx2 = tinymath_abs_int32(xx2);
    
    if (scale1 > 0) {
        xx1 >>= scale1;
    }
    else {
        xx1 <<= -scale1;
    }
    
    if (scale2 > 0) {
        xx2 >>= scale2;
    }
    else {
        xx2 <<= -scale2;
    }
  
    
    if (xx1 > xx2) {
        return 1;
    }
    
    if (xx2 > xx1) {
        return -1;
    }
    
    return 0;
    
}

//takes two 3 dimensional tensors (a bunch of images, and a bunch of convolutional filters),
//applies them accross each of the innermost tensor dimension (i.e. A1 x B1 x C1, A2 x B2 x C2, we're talking about A1 and A2, and A1 == A2)
void tinytensor_convolve3d_direct_maxpooling(
                                             Weight_t * out,
                                             const uint32_t * pool_dims,
                                             const Weight_t * weights,
                                             int8_t weight_scaling,
                                             const Weight_t * image,
                                             int8_t input_scaling,
                                             const Weight_t bias,
                                             int8_t bias_scaling,
                                             const uint32_t num_weights_rows,
                                             const uint32_t num_weights_cols,
                                             const uint32_t num_image_rows,
                                             const uint32_t num_image_cols,
                                             const uint32_t num_images,
                                             const Weight_t incoming_dropout,
                                             SquashFunc_t activation) {
    

    Weight_t temp_weight;
    int8_t temp_scale;
    
    const uint32_t num_rows_out = num_image_rows - num_weights_rows + 1;
    const uint32_t num_cols_out = num_image_cols - num_weights_cols + 1;
    const uint32_t weight_size = num_weights_rows * num_weights_cols;
    const uint32_t image_size = num_image_rows * num_image_cols;
    
    const uint32_t num_pool_rows = num_rows_out / pool_dims[0];
    const uint32_t num_pool_cols = num_cols_out / pool_dims[1];
    

    
    uint32_t ioutrow,ioutcol;
    uint32_t iimage;
    uint32_t j,i;
    uint32_t ipool_row,ipool_col;

    
    int32_t max_pool[MAX_MAX_POOL_SIZE][MAX_MAX_POOL_SIZE];
    int32_t temp32;
    int64_t temp64;
    int32_t bias32;
    int8_t bias_scaling_diff;
   // const int16_t dropout_weight = (1 << QFIXEDPOINT) - incoming_dropout;
    //const int16_t dropout_weight = incoming_dropout == 0 ? 128 : incoming_dropout;
    const int16_t dropout_weight = 128;
    Weight_t * out_row = out;
    
    assert(activation);
    if (num_images > 1) {
        bias32 = 0;
    }
    
    for (ipool_row = 0; ipool_row < num_pool_rows; ipool_row++) {
        for (ipool_col = 0; ipool_col < num_pool_cols; ipool_col++)
        {
            const uint32_t outrow_start = ipool_row * pool_dims[0];
            const uint32_t outcol_start = ipool_col * pool_dims[1];
            const uint32_t outrow_end = outrow_start + pool_dims[0];
            const uint32_t outcol_end = outcol_start + pool_dims[1];
            
            
            for (ioutrow = outrow_start; ioutrow < outrow_end; ioutrow++)
            {
                for (ioutcol = outcol_start; ioutcol < outcol_end; ioutcol++)
                {
                    
                    int32_t accumulator = 0;
                    const Weight_t * weight_start = weights;
                    const Weight_t * image_start = image +  (ioutrow * num_image_cols) + ioutcol;
                    
                    for (iimage = 0; iimage < num_images; iimage++) {
                        
                        //element by element multiply of one matrix against another
                        //summing as you go.
                        
                        const Weight_t * image_row = image_start;
                        const Weight_t * weight_row = weight_start;
                        
                        for (j = 0; j < num_weights_rows; j++) {
                            
                            
                            
                            // ***** TODO optimize this right here *****
                            for (i = 0; i < num_weights_cols; i++) {
//                                if (num_images > 1) if (i != 0) printf(",  ");
//                                if (num_images > 1) printf("%d,%d",weight_row[i],image_row[i]);
//                                if (num_images > 1) fflush(0);
                                accumulator += image_row[i] * weight_row[i];
                            }
//                            if (num_images > 1) printf("\n");
                            weight_row += num_weights_cols;
                            image_row += num_image_cols;
                        }
                        
                        //traverse to next slice for weights and image
                        weight_start += weight_size;
                        image_start += image_size;
                    }
                    
//                    if (num_images > 1) printf("----------\n");

                    max_pool[ioutrow % pool_dims[0]][ioutcol % pool_dims[1]] = accumulator;
                }
                
            }

            
            //find max in pool
            temp32 = INT32_MIN;
            for (j = 0; j < pool_dims[0]; j++) {
                for (i = 0; i < pool_dims[1]; i++) {
                    if (temp32 < max_pool[j][i]) {
                        temp32 = max_pool[j][i];
                    }
                }
            }
            
           
            //dropout
            temp64 = temp32 * dropout_weight;
            temp64 >>= QFIXEDPOINT;
            
            //compensate for weight scaling
            bias_scaling_diff = weight_scaling + input_scaling - bias_scaling;
            
            bias32 = bias;
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
            
            temp64 >>= weight_scaling;
            temp64 >>= input_scaling;
            
            temp64 += (1 << (QFIXEDPOINT - 1));
            temp64 >>= QFIXEDPOINT;


            if (temp64 > INT32_MAX) {
                temp64 = INT32_MAX;
            }
            
            if (temp64 < INT32_MIN) {
                temp64 = INT32_MIN;
            }
            
            activation(&temp_weight,&temp_scale,(int32_t)temp64,input_scaling);
            assert(temp_scale == 0);
            out_row[ipool_col] = temp_weight;
        }
        //printf("-----\n");
        out_row += num_pool_cols;
    }
    
}



