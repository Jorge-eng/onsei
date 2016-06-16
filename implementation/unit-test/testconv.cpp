#include "gtest/gtest.h"
#include "../tinytensor_types.h"
#include "../tinytensor_math.h"
#include "../tinytensor_conv_layer.h"
#include "../tinytensor_tensor.h"



class TestConv : public ::testing::Test {
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
    
    ImageTensor_t * tensor_in;
    ImageTensor_t * tensor_out;
    
};

class DISABLED_TestConv : public TestConv {};




