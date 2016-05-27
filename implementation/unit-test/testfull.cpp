#include "gtest/gtest.h"
#include "../tinytensor_types.h"
#include "../tinytensor_math.h"
#include "../tinytensor_fullyconnected_layer.h"
#include "../tinytensor_tensor.h"

#include "data/fullweights.c"
#include "data/fullbiases.c"
#include "data/small_model_ref.c"

static const uint32_t output_ref_dims[] = {1,1,1,1};
const static FullyConnectedLayer_t small_layer = {&fullweights,&fullbiases,output_ref_dims,small_model_ref_dims,tinytensor_sigmoid};


class TestFull : public ::testing::Test {
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

class DISABLED_Test1 : public TestFull {};


TEST_F(TestFull, TestSmall) {
    ConstLayer_t layer = tinytensor_create_fullyconnected_layer(&small_layer);
    
    tensor_in = tinytensor_clone_new_tensor(&small_model_ref);
    tensor_out = tinytensor_create_new_tensor(output_ref_dims);
    
    layer.eval(layer.context,tensor_out,tensor_in);
    
    ASSERT_NEAR(tensor_out->x[0],19,5);
}

