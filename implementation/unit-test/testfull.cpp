#include "gtest/gtest.h"
#include "../tinytensor_types.h"
#include "../tinytensor_math.h"
#include "../tinytensor_fullyconnected_layer.h"
#include "../tinytensor_tensor.h"

#include "data/big_full_weights.c"
#include "data/big_full_bias.c"
#include "data/big_full_input.c"
#include "data/big_full_output.c"

#define NO_DROPOUT (0)
#define NO_HARD_MAX (0)


const static uint32_t big_input_ref_dims[4] = {1,1,1,3648};
const static uint32_t big_output_ref_dims[4] = {1,1,1,512};
const static FullyConnectedLayer_t big_layer = {&big_full_weights,&big_full_bias,big_output_ref_dims,big_input_ref_dims,NO_DROPOUT,tinytensor_linear,NO_HARD_MAX};

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
    
    ImageTensor_t * tensor_in;
    ImageTensor_t * tensor_out;
    ImageTensor_t * ref;
    int idx;
    
};

class DISABLED_TestFull : public TestFull {};



TEST_F(TestFull, TestBig) {
    
    ConstLayer_t layer = tinytensor_create_fullyconnected_layer(&big_layer);
    
    tensor_in = tinytensor_clone_new_image_tensor(&big_full_input);
    tensor_out = tinytensor_create_new_image_tensor(big_output_ref_dims);
    
    layer.eval(layer.context,tensor_out,tensor_in);
    
    
    for (int i = 0; i < 512; i++) {
        int val = tensor_out->x[i] >> tensor_out->scale;
        int refval = big_full_output_x[i];
        
        refval = refval > 127 ? 127 : refval;
        refval = refval < -127 ? -127 : refval;
        
        idx = i;
        
        ASSERT_NEAR( refval , val,20);
    }
    
    idx = -1;
    
    
}

