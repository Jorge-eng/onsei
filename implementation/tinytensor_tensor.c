#include "tinytensor_tensor.h"
#include "tinytensor_memory.h"

void delete_tensor(void * context) {
    ImageTensor_t * p = (ImageTensor_t *) context;
    
    FREE(p->x);
    FREE(p);
}

void tinytensor_zero_out_tensor(ImageTensor_t * tensor) {
    uint32_t i;
    uint32_t num_elements= tensor->dims[0];
    for (i = 1; i < TENSOR_DIM; i++) {
        num_elements *= tensor->dims[i];
    }
    
    memset(tensor->x,0,num_elements*sizeof(Weight_t));
}


ImageTensor_t * tinytensor_create_new_image_tensor(const uint32_t dims[TENSOR_DIM]) {
    uint32_t i;
    uint32_t num_elements= dims[0];
    for (i = 1; i < TENSOR_DIM; i++) {
        num_elements *= dims[i];
    }
    ImageTensor_t * tensor = (ImageTensor_t *)MALLLOC(sizeof(ImageTensor_t));
    MEMSET(tensor,0,sizeof(ImageTensor_t));
    MEMCPY(tensor->dims, dims, sizeof(tensor->dims));
    tensor->x = MALLLOC(num_elements * sizeof(Weight_t));
    tensor->delete_me = delete_tensor;
    
    return tensor;
}

ImageTensor_t * tinytensor_clone_new_image_tensor(const ConstImageTensor_t * in) {
    uint32_t num_elements= in->dims[0];
    uint32_t i;
    for (i = 1; i < TENSOR_DIM; i++) {
        num_elements *= in->dims[i];
    }
    
    ImageTensor_t * tensor = tinytensor_create_new_image_tensor(in->dims);
    memcpy(tensor->x,in->x,sizeof(Weight_t) * num_elements);
    tensor->scale = in->scale;
    return tensor;
}

