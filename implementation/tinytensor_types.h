#ifndef _TINYTENSOR_TYPES_H_
#define _TINYTENSOR_TYPES_H_
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_WEIGHT (0x7F)
#define MAX_LONG_WEIGHT (0x7FFF)

typedef int8_t Weight_t;
typedef Weight_t Data_t; //data and weight are the same for now
    
//callback defs
typedef void (*SquashFunc_t)(Weight_t * y, int8_t * out_scale, int32_t x,int8_t in_scale);

#define TENSOR_DIM (4)

/*
      TENSOR DEFS
 */
typedef void(*TensorDelete_t)(void * context);

typedef struct{
    Weight_t * x;
    uint32_t dims[TENSOR_DIM];
    TensorDelete_t delete_me;
    int8_t scale;
} Tensor_t;
    
typedef struct{
    const Weight_t * x;
    const uint32_t * dims;
    int8_t scale;
} ConstTensor_t;

/*
      LAYER DEFS
 */
    
typedef enum {
    input_layer,
    lstm_layer,
    full_layer,
    conv_layer
} ELayer_t;
    
typedef void (*ConstLayerEval_t)(const void * context,Tensor_t * out,const Tensor_t * in, ELayer_t prev_layer_type);

typedef void (*ConstLayerDims_t)(const void * context,uint32_t * dims);

typedef struct {
    ConstLayerEval_t eval;
    ConstLayerDims_t get_output_dims;
    ELayer_t layer_type;
    const void * context;
} ConstLayer_t;

    
#ifdef __cplusplus
}
#endif



#endif //_TINYTENSOR_TYPES_H_
