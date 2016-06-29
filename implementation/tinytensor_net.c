#include "tinytensor_net.h"
#include "tinytensor_math.h"
#include "tinytensor_tensor.h"
#include <assert.h>




Tensor_t * eval_net(const ConstSequentialNetwork_t * net,Tensor_t * input) {

    Tensor_t * current_input = input;
    Tensor_t * current_output = 0;
    uint32_t ilayer;
    ELayer_t prev_layer = input_layer;

    for (ilayer = 0; ilayer < net->num_layers; ilayer++) {
        const ConstLayer_t * const layer = &net->layers[ilayer];
        
        uint32_t output_dims[TENSOR_DIM];
        layer->get_output_dims(layer->context,&output_dims[0]);
        
        //allocate output
        current_output = tinytensor_create_new_tensor(output_dims);

        //perform evaluation
        layer->eval(layer->context,current_output,current_input,prev_layer);
        
        //output becomes new input --- so delete input if we can
        if (current_input->delete_me && current_input != input) {
            current_input->delete_me(current_input);
        }

        current_input = current_output;
        prev_layer = layer->layer_type;

    }
    
    //whomever received this object must delete it
    return current_output;
    
}


Tensor_t * eval_partial_net(const ConstSequentialNetwork_t * net,Tensor_t * input,const uint32_t stop_layer) {
    
    Tensor_t * current_input = input;
    Tensor_t * current_output = 0;
    uint32_t ilayer;
    uint32_t end = net->num_layers < stop_layer ? net->num_layers : stop_layer;
    ELayer_t prev_layer = input_layer;
    
    for (ilayer = 0; ilayer < end; ilayer++) {
        const ConstLayer_t * const layer = &net->layers[ilayer];
        
        uint32_t output_dims[TENSOR_DIM];
        layer->get_output_dims(layer->context,&output_dims[0]);
        
        //allocate output
        current_output = tinytensor_create_new_tensor(output_dims);
        
        //perform evaluation
        layer->eval(layer->context,current_output,current_input,prev_layer);
        
        //output becomes new input --- so delete input if we can
        if (current_input->delete_me && current_input != input) {
            current_input->delete_me(current_input);
        }
        
        current_input = current_output;
        prev_layer = layer->layer_type;
    }
    
    //whomever received this object must delete it
    return current_output;
    
}