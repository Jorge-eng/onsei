#include "gtest/gtest.h"
#include "../tinytensor_types.h"
#include "../tinytensor_math.h"
#include "../tinytensor_conv_layer.h"
#include "../tinytensor_tensor.h"

#include "data/weights1.c"
#include "data/biases1.c"
#include "data/image.c"
#include "data/ref1.c"

#include "data/weights2.c"
#include "data/biases2.c"
#include "data/image2.c"
#include "data/ref2.c"


//#include "data/small_conv_network.c"
//#include "data/testinput_for_small_network.c"
//#include "data/testoutput_for_small_network.c"

#include "data/small_conv2.c"
#include "data/testoutput2_for_small_network.c"
#include "data/testinput2_for_small_network.c"

#define NO_DROPOUT (0)
const static uint32_t max_pool_dims_no_pooling[2] = {1,1};
const static ConvLayer2D_t conv_layer_def2 = {&weights2,&biases2,ref2_dims,image2_dims,max_pool_dims_no_pooling,NO_DROPOUT,tinytensor_linear};
const static ConvLayer2D_t conv_layer_def1 = {&weights1,&biases1,ref1_dims,image_dims,max_pool_dims_no_pooling,NO_DROPOUT,tinytensor_linear};

/*
const static Weight_t zero_biases[3] = {0,0,0};
ConstT
const static uint32_t final_out_dims[4] = {1, 2, 5, 4}
const static ConvLayer2D_t conv_layer_def4 = {&weights2,&zero_biases,ref2_dims,image2_dims,NO_DROPOUT,tinytensor_linear};
const static ConvLayer2D_t conv_layer_def3 = {&weights1,&zero_biases,ref1_dims,image_dims,NO_DROPOUT,tinytensor_linear};
*/

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
    
    Tensor_t * tensor_in;
    Tensor_t * tensor_out;
    
};

class DISABLED_Test1 : public TestConv {};


TEST_F(TestConv, Test3DConv) {
  //  tinytensor_convolve3d_direct(
}

TEST_F(TestConv,TestSmallConvLayerSingleImage) {

    tensor_in = tinytensor_clone_new_tensor(&image2);

    ConstLayer_t layer = tinytensor_create_conv_layer(&conv_layer_def2);
    
    uint32_t output_dims[TENSOR_DIM];
    layer.get_output_dims(layer.context,output_dims,tensor_in->dims);
    
    
    tensor_out = tinytensor_create_new_tensor(output_dims);

    layer.eval(layer.context,NULL,tensor_out,tensor_in,input_layer);

    uint32_t out_size = tensor_out->dims[0] * tensor_out->dims[1] * tensor_out->dims[2] * tensor_out->dims[3];
    ASSERT_TRUE(out_size == ref2_dims[0]*ref2_dims[1]*ref2_dims[2] * ref2_dims[3]);
    
    for (uint32_t i = 0; i < out_size; i++) {
        int val1 = ref2_weights[i];
        
        if (val1 > 127) {
            val1 = 127;
        }
        
        if (val1 < -127) {
            val1 = -127;
        }
        
        int val2 = tensor_out->x[i];
        
        if (tensor_out->scale > 0) {
            val2 >>= tensor_out->scale;
        }
        
        int diff = val2 - val1;

        if (abs(diff) > 5) {
            std::cout << "ref=" << val1 << " output=" << val2 << std::endl;
            ASSERT_TRUE(false);
        }
    }

}


TEST_F(TestConv, TestVerySimpleTwoImageInput) {
    uint32_t input_dims[4] = {1,2,2,3};
    
    Weight_t input_x[12] = {10,20,30,
                           40,50,60,
        
                           54,28,39,
                           10,25,85};
    
    ConstTensor_t input = {input_x,input_dims};
    
    Weight_t x[16] = {21,22,
                      23,24,
                      41,42,
                      43,45,
        
                      33,34,
                      35,36,
                      23,42,
                      43,59};
    
    uint32_t xdims[4] = {2,2,2,2};
    
    Weight_t biasx[2] = {0,0};
    uint32_t biasdim[4] = {2,1,1,1};
    
    ConstTensor_t conv_x = {x,xdims};
    ConstTensor_t conv_bias = {biasx,biasdim};
    
    uint32_t output_dims[4] = {2,1,1,2};
    
    ConvLayer2D_t conv_context = {&conv_x,&conv_bias,output_dims,input_dims,max_pool_dims_no_pooling,0,tinytensor_linear};
    
    ConstLayer_t layer = tinytensor_create_conv_layer(&conv_context);
    
    tensor_in = tinytensor_clone_new_tensor(&input);
    tensor_out = tinytensor_create_new_tensor(output_dims);
    
    layer.eval(layer.context,NULL,tensor_out,tensor_in,input_layer);
    
    int32_t accumulator = 0;
    accumulator += input_x[0] * x[0] + input_x[1]*x[1] + input_x[3] * x[2] + input_x[4] *x[3];
    accumulator += input_x[6] * x[4] + input_x[7]*x[5] + input_x[9] * x[6]+ input_x[10] * x[7];
    accumulator >>= QFIXEDPOINT;
    
    ASSERT_NEAR(tensor_out->x[0] >> tensor_out->scale,accumulator,5);
    
    accumulator = 0;
    accumulator += input_x[1] * x[0] + input_x[2]*x[1] + input_x[4] * x[2] + input_x[5] * x[3];
    accumulator += input_x[7] * x[4] + input_x[8]*x[5] + input_x[10] * x[6]+ input_x[11] * x[7];
    accumulator >>= QFIXEDPOINT;
    
    ASSERT_NEAR(tensor_out->x[1] >> tensor_out->scale,accumulator,5);

    accumulator = 0;
    accumulator += input_x[0] * x[8] + input_x[1]*x[9] + input_x[3] * x[10] + input_x[4] *x[11];
    accumulator += input_x[6] * x[12] + input_x[7]*x[13] + input_x[9] * x[14]+ input_x[10] * x[15];
    accumulator >>= QFIXEDPOINT;

    ASSERT_NEAR(tensor_out->x[2] >> tensor_out->scale,accumulator,5);

    accumulator = 0;
    accumulator += input_x[1] * x[8] + input_x[2]*x[9] + input_x[4] * x[10] + input_x[5] * x[11];
    accumulator += input_x[7] * x[12] + input_x[8]*x[13] + input_x[10] * x[14]+ input_x[11] * x[15];
    accumulator >>= QFIXEDPOINT;

    ASSERT_NEAR(tensor_out->x[3] >> tensor_out->scale,accumulator,5);



}

TEST_F(TestConv,TestLargeConvLayerSingleImage) {
    
    tensor_in = tinytensor_clone_new_tensor(&image);
    
    ConstLayer_t layer = tinytensor_create_conv_layer(&conv_layer_def1);
    
    uint32_t output_dims[TENSOR_DIM];
    layer.get_output_dims(layer.context,output_dims,tensor_in->dims);
    
    
    tensor_out = tinytensor_create_new_tensor(output_dims);
    
    layer.eval(layer.context,NULL,tensor_out,tensor_in,input_layer);
    
    uint32_t out_size = tensor_out->dims[0] * tensor_out->dims[1] * tensor_out->dims[2] * tensor_out->dims[3];
    ASSERT_TRUE(out_size == ref1_dims[0]*ref1_dims[1]*ref1_dims[2] * ref1_dims[3]);
    int out_scale = tensor_out->scale;
    for (uint32_t i = 0; i < out_size; i++) {
        int val1 = ref1_weights[i];
        
        /*
        if (val1 > 127) {
            val1 = 127;
        }
        
        if (val1 < -127) {
            val1 = -127;
        }
*/
        
        int val2 = tensor_out->x[i];
        
        if (out_scale < 0) {
            val2 <<= -out_scale;
        }
        
        if (out_scale> 0) {
            val2 >>= out_scale;
        }
        
        int diff = val2 - val1;
        
        if (abs(diff) > 8) {
            std::cout << "ref=" << val1 << " output=" << val2 << " at index " << i << std::endl;
            ASSERT_TRUE(false);
        }
    }
    
}

TEST_F(TestConv,TestSmallConvNetwork) {
    tensor_in = tinytensor_clone_new_tensor(&testinput2_smallnet);
    
    ConstSequentialNetwork_t net = initialize_network();
    
    tensor_out = tinytensor_eval_net(&net,tensor_in);
    
    const uint32_t len = testoutput2_smallnet_dims[0]*testoutput2_smallnet_dims[1] * testoutput2_smallnet_dims[2] * testoutput2_smallnet_dims[3];
    
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(testoutput2_smallnet_dims[i],tensor_out->dims[i]);
    }
    
    for (int i = 0; i < len; i++) {
        int val1 = testoutput2_smallnet.x[i];
        if (val1 > 127) {
            val1 = 127;
        }
        
        if (val1 < -127) {
            val1 = -127;
        }
        
        ASSERT_NEAR(val1,tensor_out->x[i] >> tensor_out->scale,10);
    }
    
    
}


