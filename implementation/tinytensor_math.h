#ifndef _TINYTENSOR_MATH_H_
#define _TINYTENSOR_MATH_H_

#include "tinytensor_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QFIXEDPOINT (7)
    
#define TOFIX(x)\
        (Weight_t)(x * (1 << QFIXEDPOINT))
    
#define TOFLT(x)\
        ( ((float)x) / (float)(1 << QFIXEDPOINT))

/* INPUTS ARE EXPECTED TO BE IN Q7, JUST POTENTIALLY VERY LARGE IN MAGNITUDE */
Weight_t tinytensor_tanh(int32_t x);
Weight_t tinytensor_sigmoid(int32_t x);
Weight_t tinytensor_linear(int32_t x);
Weight_t tinytensor_relu(int32_t x);

void tinytensor_convolve3d_direct(Weight_t * out, const Weight_t * weights,const Weight_t * image, const Weight_t bias,const uint32_t num_weights_rows,const uint32_t num_weights_cols, const uint32_t num_image_rows, const uint32_t num_image_cols,const uint32_t num_images,const Weight_t incoming_dropout,SquashFunc_t activation);

    
#ifdef __cplusplus
}
#endif

#endif
