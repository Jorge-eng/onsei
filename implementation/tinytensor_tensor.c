#include "tinytensor_tensor.h"
#include "tinytensor_memory.h"

void delete_tensor(void * context) {
    Tensor_t * p = (Tensor_t *) context;
    
    FREE(p->x);
    FREE(p);
}

void tinytensor_zero_out_tensor(Tensor_t * tensor) {
    uint32_t i;
    uint32_t num_elements= tensor->dims[0];
    for (i = 1; i < TENSOR_DIM; i++) {
        num_elements *= tensor->dims[i];
    }
    
    memset(tensor->x,0,num_elements*sizeof(Weight_t));
}


Tensor_t * tinytensor_create_new_tensor(const uint32_t dims[TENSOR_DIM]) {
    uint32_t i;
    uint32_t num_elements= dims[0];
    for (i = 1; i < TENSOR_DIM; i++) {
        num_elements *= dims[i];
    }
    Tensor_t * tensor = (Tensor_t *)MALLLOC(sizeof(Tensor_t));
    MEMSET(tensor,0,sizeof(Tensor_t));
    MEMCPY(tensor->dims, dims, sizeof(tensor->dims));
    tensor->x = MALLLOC(num_elements * sizeof(Weight_t));
    tensor->delete_me = delete_tensor;
    
    return tensor;
}

Tensor_t * tinytensor_clone_new_tensor(const ConstTensor_t * in) {
    uint32_t num_elements= in->dims[0];
    uint32_t i;
    for (i = 1; i < TENSOR_DIM; i++) {
        num_elements *= in->dims[i];
    }
    
    Tensor_t * tensor = tinytensor_create_new_tensor(in->dims);
    memcpy(tensor->x,in->x,sizeof(Weight_t) * num_elements);
    tensor->scale = in->scale;
    return tensor;
}

