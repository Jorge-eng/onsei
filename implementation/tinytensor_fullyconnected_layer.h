#ifndef _TINYTENSOR_FULLYCONNECTED_LAYER_H_
#define _TINYTENSOR_FULLYCONNECTED_LAYER_H_

#include "tinytensor_types.h"

#ifdef __cplusplus
extern "C" {
#endif
    
typedef struct {
    const ConstTensor_t * weights;
    const ConstTensor_t * biases;
    const uint32_t * output_dims;
    const uint32_t * input_dims;
    SquashFunc_t squash;
} FullyConnectedLayer_t;

ConstLayer_t tinytensor_create_fullyconnected_layer(const FullyConnectedLayer_t * static_def);

#ifdef __cplusplus
}
#endif

#endif //_TINYTENSOR_FULLYCONNECTED_LAYER_H_
