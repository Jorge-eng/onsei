#include "gtest/gtest.h"
#include "../tinytensor_types.h"
#include "../tinytensor_math.h"
#include "../tinytensor_fullyconnected_layer.h"
#include "../tinytensor_tensor.h"

#include "data/fullweights.c"
#include "data/fullbiases.c"
#include "data/small_model_ref.c"

#include "data/big_full_weights.c"
#include "data/big_full_bias.c"
#include "data/big_full_input.c"
#include "data/big_full_output.c"

#define NO_DROPOUT (0)
#define NO_HARD_MAX (0)
static const uint32_t output_ref_dims[] = {1,1,1,1};
const static FullyConnectedLayer_t small_layer = {&fullweights,&fullbiases,output_ref_dims,small_model_ref_dims,NO_DROPOUT,tinytensor_linear,NO_HARD_MAX};

const static uint32_t big_input_ref_dims[4] = {1,1,1,3648};
const static uint32_t big_output_ref_dims[4] = {1,1,1,512};
const static FullyConnectedLayer_t big_layer = {&big_full_weights,&big_full_bias,big_output_ref_dims,big_input_ref_dims,NO_DROPOUT,tinytensor_relu,NO_HARD_MAX};

class TestFull : public ::testing::Test {
protected:
    
    
    virtual void SetUp() {
        tensor_in = NULL;
        tensor_out = NULL;
        ref = NULL;
        idx = -1;
    }
    
    virtual void TearDown() {
        if (tensor_in) {
            tensor_in->delete_me(tensor_in);
        }
        
        if (tensor_out) {
            tensor_out->delete_me(tensor_out);
        }
        
        if (ref) {
            ref->delete_me(ref);
        }
        
        if (idx >= 0) {
            std::cout << "index was " << idx << std::endl;
        }
        
    }
    
    /*
    virtual void OnTestEnd( const ::testing::TestInfo& test_info ) {
        if ( test_info.result()->Failed() ) {
            if (idx >= 0) {
                std::cout << "index was " << idx << std::endl;
            }
            //std::cout << test_info.test_case_name() << "   failed   " << test_info.name() << std::endl;
        }
    }
     */
    
    Tensor_t * tensor_in;
    Tensor_t * tensor_out;
    Tensor_t * ref;
    int idx;
    
};

class DISABLED_TestFull : public TestFull {};


TEST_F(TestFull, TestSmall) {
    ConstLayer_t layer = tinytensor_create_fullyconnected_layer(&small_layer);
    
    tensor_in = tinytensor_clone_new_tensor(&small_model_ref);
    tensor_out = tinytensor_create_new_tensor(output_ref_dims);
    
    layer.eval(layer.context,tensor_out,tensor_in,input_layer);
    
    ASSERT_NEAR(tensor_out->x[0],19,5);
}

TEST_F(TestFull, TestBig) {
    
    ConstLayer_t layer = tinytensor_create_fullyconnected_layer(&big_layer);
    
    tensor_in = tinytensor_clone_new_tensor(&big_full_input);
    tensor_out = tinytensor_create_new_tensor(big_output_ref_dims);
    
    layer.eval(layer.context,tensor_out,tensor_in,input_layer);
    
    
    for (int i = 0; i < 512; i++) {
        
        int val = tensor_out->x[i];
        int refval = big_full_output_x[i];
        
        if (tensor_out->scale > 0) {
            val >>= tensor_out->scale;
        }
        else {
            val <<= -tensor_out->scale;
        }
        
        
        idx = i;
        
        int tol = refval * 0.1;
        tol = tol < 8 ? 8 : tol;
        
        ASSERT_NEAR( refval , val,tol);
    }
    
    idx = -1;
    
    
}

