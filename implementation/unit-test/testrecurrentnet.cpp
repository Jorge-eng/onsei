#include <iostream>
#include "gtest/gtest.h"
#include "../tinytensor_types.h"
#include "../tinytensor_math.h"
#include "../tinytensor_conv_layer.h"
#include "../tinytensor_tensor.h"

#include "data/lstm1.c"
#include "data/lstm1_input.c"
#include "data/lstm1_ref.c"
#include "data/lstm3.c"
#include "data/lstm3_ref.c"

//#include "data/model_may25_lstm_large.c"
//#include "data/kwClip_160517_02_1_lstm.c"

class TestRecurrentNet : public ::testing::Test {
protected:
    int idx;
    
    virtual void SetUp() {
        tensor_in = NULL;
        tensor_out = NULL;
        idx = -1;
    }
    
    virtual void TearDown() {
        if (tensor_in) {
            tensor_in->delete_me(tensor_in);
        }
        
        if (tensor_out) {
            tensor_out->delete_me(tensor_out);
        }
        
        if (idx >= 0) {
            std::cout << "idx=" << idx << std::endl;
        }
        
    }
    
    Tensor_t * tensor_in;
    Tensor_t * tensor_out;
    
};

class DISABLED_TestRecurrentNet : public TestRecurrentNet {};


TEST_F(TestRecurrentNet, TestRandInput) {
    
    ConstLayer_t lstm_layer = tinytensor_create_lstm_layer(&LSTM2_01);
    
    LstmLayerState_t * state = (LstmLayerState_t *) lstm_layer.alloc_state(lstm_layer.context);
    
    tensor_in = tinytensor_clone_new_tensor(&lstm1_input);
    
    uint32_t dims[4];
    lstm_layer.get_output_dims(lstm_layer.context,dims,tensor_in->dims);
    dims[2] = 1;
    
    const uint32_t * d = lstm1_ref.dims;
    int n = d[0] * d[1] * d[2] * d[3];
    
    tensor_out = tinytensor_create_new_tensor(dims);
    Weight_t out[n];
    
    
    Weight_t * p = &out[0];
    
    for (uint32_t t = 0; t < tensor_in->dims[2]; t++) {
        Tensor_t temp_tensor;

        temp_tensor.x = tensor_in->x + t * tensor_in->dims[3];
        temp_tensor.dims[0] = tensor_in->dims[0];
        temp_tensor.dims[1] = tensor_in->dims[1];
        temp_tensor.dims[2] = 1;
        temp_tensor.dims[3] = tensor_in->dims[3];
        temp_tensor.scale = tensor_in->scale;
        temp_tensor.delete_me = NULL;
        
        lstm_layer.eval(lstm_layer.context,state,tensor_out,&temp_tensor,input_layer);
        
        
        for (int i = 0; i < d[3]; i++) {
            *p = tensor_out->x[i] >> tensor_out->scale;
            p++;
        }
    }
    
    p = &out[0];

    for (int i = 0; i < n; i++) {
        int x1 = *p;
        int x2 = lstm1_ref_x[i] >> lstm1_ref.scale;
        p++;
        idx = i;
        ASSERT_NEAR(x1,x2,2);
    }
    
    idx = -1;

    

}

TEST_F(TestRecurrentNet, TwoLayers) {
    tensor_in = tinytensor_clone_new_tensor(&lstm1_input);

    const uint32_t * d = lstm3_ref.dims;
    int n = d[0] * d[1] * d[2] * d[3];
    Weight_t out[n];

    ConstSequentialNetwork_t net = initialize_network03();
    SequentialNetworkStates_t states;

    tinytensor_allocate_states(&states, &net);
    
    Weight_t * p = &out[0];
    for (uint32_t t = 0; t < tensor_in->dims[2]; t++) {
        Tensor_t temp_tensor;
        
        temp_tensor.x = tensor_in->x + t * tensor_in->dims[3];
        temp_tensor.dims[0] = tensor_in->dims[0];
        temp_tensor.dims[1] = tensor_in->dims[1];
        temp_tensor.dims[2] = 1;
        temp_tensor.dims[3] = tensor_in->dims[3];
        temp_tensor.scale = tensor_in->scale;
        temp_tensor.delete_me = NULL;
        
        Tensor_t * output = tinytensor_eval_stateful_net(&net, &states, &temp_tensor);
        
        for (int i = 0; i < output->dims[3]; i++) {
            *p = output->x[i] >> output->scale;
            p++;
        }
        
        output->delete_me(output);
    }
    
    

    p = &out[0];
    for (int i = 0; i < n; i++) {
        int x1 = *p;
        int x2 = lstm3_ref_x[i] >> lstm3_ref.scale;
        
        ASSERT_NEAR(x1,x2,2);
        p++;
    }
    
    
    tinytensor_free_states(&states, &net);

}






