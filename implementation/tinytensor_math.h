#ifndef _TINYTENSOR_MATH_H_
#define _TINYTENSOR_MATH_H_

#include "tinytensor_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QFIXEDPOINT (7)
#define QFIXEDPOINT_INT16 (15)
#define TOFIX(x)\
        (Weight_t)(x * (1 << QFIXEDPOINT))
    
#define TOFLT(x)\
        ( ((float)x) / (float)(1 << QFIXEDPOINT))
    
#define MUL16(a,b)\
    ((int16_t)(((int32_t)(a * b)) >> QFIXEDPOINT_INT16))

/* INPUTS ARE EXPECTED TO BE IN Q7, JUST POTENTIALLY VERY LARGE IN MAGNITUDE */
void tinytensor_tanh(Weight_t * y, int8_t * out_scale, int32_t x,int8_t in_scale);
void tinytensor_sigmoid(Weight_t * y, int8_t * out_scale, int32_t x,int8_t in_scale);
void tinytensor_linear(Weight_t * y, int8_t * out_scale, int32_t x,int8_t in_scale);
void tinytensor_relu(Weight_t * y, int8_t * out_scale, int32_t x,int8_t in_scale);

void tinytensor_descale(Weight_t * y, int8_t * out_scale, int32_t x, int8_t in_scale);
int8_t tiny_tensor_compare_scaled_numbers(const Weight_t x1, const int8_t scale1, const Weight_t x2, const int8_t scale2);
int8_t tiny_tensor_get_scaling(Weight_t maxWeight);



void tinytensor_convolve3d_direct_maxpooling(
                                             ImagePixel_t * out,
                                             const uint32_t * pool_dims,
                                             const Weight_t * weights,
                                             int8_t weight_scaling,
                                             const ImagePixel_t * image,
                                             int8_t incoming_scaling,
                                             const Weight_t bias,
                                             int8_t bias_scaling,
                                             const uint32_t num_weights_rows,
                                             const uint32_t num_weights_cols,
                                             const uint32_t num_image_rows,
                                             const uint32_t num_image_cols,
                                             const uint32_t num_images,
                                             const Weight_t incoming_dropout,
                                             SquashFunc_t activation);
    
#ifdef __cplusplus
}
#endif

#endif
