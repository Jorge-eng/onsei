#include "gtest/gtest.h"
#include "../tinytensor_types.h"
#include "../tinytensor_math.h"
#include "../tinytensor_maxpool_layer.h"
#include "../tinytensor_tensor.h"


class TestMaxPool : public ::testing::Test {
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

class DISABLED_Test1 : public TestMaxPool {};


TEST_F(TestMaxPool, OneImageSquare) {
    //one layer
    const uint32_t input_dims[] = {1,1,4,4};
    tensor_in = tinytensor_create_new_tensor(input_dims);
    
    const uint32_t pool_dims[] = {2,2};
    const uint32_t output_dims[] = {1,1,2,2};
    
    //  0  1  2  3
    //  4  5  6  7
    //  8  9  10 11
    //  12 13 14 15
    
    //  5  7
    //  13 15
    
    const Weight_t ref[] = {5,7,13,15};
    
    for (int i = 0; i < 16; i++) {
        tensor_in->x[i] = i;
    }
    
    const MaxPoolLayer_t max_pool_layer_def = {pool_dims,output_dims,input_dims};
    
    ConstLayer_t layer = tinytensor_create_maxpool_layer(&max_pool_layer_def);
    
    tensor_out = tinytensor_create_new_tensor(output_dims);
    
    layer.eval(layer.context,tensor_out,tensor_in);
    
    
    for (int j = 0; j < 4; j++) {
        ASSERT_EQ(ref[j], tensor_out->x[j]);
    }
    
    
}


TEST_F(TestMaxPool, TwoImageSquare) {
    //one layer
    const uint32_t input_dims[] = {1,2,4,4};
    tensor_in = tinytensor_create_new_tensor(input_dims);
    
    const uint32_t pool_dims[] = {2,2};
    const uint32_t output_dims[] = {1,2,2,2};
    
    //  0  1  2  3
    //  4  5  6  7
    //  8  9  10 11
    //  12 13 14 15
    
    //  16 17 18 19
    //  20 21 22 23
    //  24 25 26 27
    //  28 29 30 31
    
    
    const Weight_t ref[] = {5,7,13,15,21,23,29,31};
    
    for (int i = 0; i < 32; i++) {
        tensor_in->x[i] = i;
    }
    
    const MaxPoolLayer_t max_pool_layer_def = {pool_dims,output_dims,input_dims};
    
    ConstLayer_t layer = tinytensor_create_maxpool_layer(&max_pool_layer_def);
    
    tensor_out = tinytensor_create_new_tensor(output_dims);
    
    layer.eval(layer.context,tensor_out,tensor_in);
    
    
    for (int j = 0; j < 8; j++) {
        ASSERT_EQ(ref[j], tensor_out->x[j]);
    }
    
    
    
    
    
    
}

