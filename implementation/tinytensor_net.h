#ifndef _TINYTENSOR_NET_H_
#define _TINYTENSOR_NET_H_

#include "tinytensor_types.h"
#include "tinytensor_tensor.h"

#define MAX_OUTPUT_SIZE (256)

#ifdef __cplusplus
extern "C" {
#endif

    
typedef struct {
    ConstLayer_t * layers;
    uint32_t num_layers;
} ConstSequentialNetwork_t;
    
ImageTensor_t * eval_net(const ConstSequentialNetwork_t * net,ImageTensor_t * input);
    


#ifdef __cplusplus
}
#endif



#endif //_TINYTENSOR_NET_H_
