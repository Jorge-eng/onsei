#include "gtest/gtest.h"
//#include "../test.c"
#include "../tinytensor_net.h"

class TestNet : public ::testing::Test {
protected:
    
    
    virtual void SetUp() {
        tensor_in = NULL;
        tensor_out = NULL;
    }
    
    virtual void TearDown() {
        if (tensor_in) {
            tensor_in->delete_me(tensor_in);
        }
        
        if (tensor_out) {
            tensor_out->delete_me(tensor_out);
        }
    }
    
    Tensor_t * tensor_in;
    Tensor_t * tensor_out;
    
};

class DISABLED_TestNet : public TestNet {};


TEST_F(DISABLED_TestNet, Realistic) {
    //(1, 1, 40, 199)
    const uint32_t dims[4] = {1,1,40,199};
    
    tensor_in = tinytensor_create_new_tensor(dims);
    
    tinytensor_zero_out_tensor(tensor_in);
    
//    ConstSequentialNetwork_t net = initialize_network();
//    eval_net(&net,tensor_in);
    
    
}



