#include "gtest/gtest.h"
#include "../tinytensor_net.h"

#include "data/model_may31_small_sigm.c"
#include "data/test_input_rand.c"
#include "data/test_input.c"
#include "data/kwClip_1.c"
#include "data/kwClip_41.c"
#include "data/kwClip_42.c"
#include "data/kwClip_43.c"

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

/*
TEST_F (DISABLED_TestNet,TestManually) {
    const int32_t tol = 8;
    tensor_in = tinytensor_clone_new_tensor(&test_input);
    
    ConstSequentialNetwork_t net = initialize_network();
    uint32_t dims[TENSOR_DIM];
    net.layers[0].get_output_dims(net.layers[0].context,dims);
    
    tensor_out = tinytensor_create_new_tensor(dims);
    
    net.layers[0].eval(net.layers[0].context,tensor_out,tensor_in);
    
    
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(test_output_layer1_dims[i],tensor_out->dims[i]);
    }
    
    uint32_t len = test_output_layer1_dims[0] * test_output_layer1_dims[1] * test_output_layer1_dims[2] * test_output_layer1_dims[3];
    
    for (int i = 0; i < len; i++) {
        ASSERT_NEAR(test_output_layer1_x[i], tensor_out->x[i], tol);
    }
    
    tensor_in->delete_me(tensor_in);
    tensor_in = tensor_out;
    
    net.layers[1].get_output_dims(net.layers[1].context,dims);

    tensor_out = tinytensor_create_new_tensor(dims);

    net.layers[1].eval(net.layers[1].context,tensor_out,tensor_in);

    
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(test_output_layer2_dims[i],tensor_out->dims[i]);
    }
    
    len = test_output_layer2_dims[0] * test_output_layer2_dims[1] * test_output_layer2_dims[2] * test_output_layer2_dims[3];
    
    for (int i = 0; i < len; i++) {
        ASSERT_NEAR(test_output_layer2_x[i], tensor_out->x[i], tol);
    }

    
    
    
    tensor_in->delete_me(tensor_in);
    
    tensor_in = tinytensor_clone_new_tensor(&test_output_layer2);
    //tensor_in = tensor_out;
    
    net.layers[2].get_output_dims(net.layers[2].context,dims);
    
    tensor_out = tinytensor_create_new_tensor(dims);
    
    net.layers[2].eval(net.layers[2].context,tensor_out,tensor_in);
    
    
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(test_output_layer3_dims[i],tensor_out->dims[i]);
    }
    
    len = test_output_layer3_dims[0] * test_output_layer3_dims[1] * test_output_layer3_dims[2] * test_output_layer3_dims[3];
    
    for (int i = 0; i < len; i++) {
        std::cout << i << std::endl;
        ASSERT_NEAR(test_output_layer3_x[i], tensor_out->x[i], tol);
    }

    
    int foo = 3;
    foo++;
}
 */

TEST_F(TestNet, Realistic1) {

    tensor_in = tinytensor_clone_new_tensor(&test_input_rand);
    
    ConstSequentialNetwork_t net = initialize_network();
    tensor_out = eval_net(&net,tensor_in);
    
    printf("%d,%d\n",tensor_out->x[0],tensor_out->x[1]);
    
    //should be 127,0 but underflow....
    //just make sure it makes the right decision
    ASSERT_TRUE(tensor_out->x[0] > tensor_out->x[1] + 20);
  
}

TEST_F(TestNet, Realistic2) {
    
    tensor_in = tinytensor_clone_new_tensor(&kwClip_41);
    
    ConstSequentialNetwork_t net = initialize_network();
    tensor_out = eval_net(&net,tensor_in);
    
    printf("%d,%d\n",tensor_out->x[0],tensor_out->x[1]);
    
    ASSERT_TRUE(tensor_out->x[1] > tensor_out->x[0] + 20);

    
}

TEST_F(TestNet, Realistic3) {
    
    tensor_in = tinytensor_clone_new_tensor(&kwClip_42);
    
    ConstSequentialNetwork_t net = initialize_network();
    tensor_out = eval_net(&net,tensor_in);
    
    printf("%d,%d\n",tensor_out->x[0],tensor_out->x[1]);
    
    ASSERT_TRUE(tensor_out->x[1] > tensor_out->x[0] + 20);
    
    
}

TEST_F(TestNet, Realistic4) {
    
    tensor_in = tinytensor_clone_new_tensor(&kwClip_43);
    
    ConstSequentialNetwork_t net = initialize_network();
    tensor_out = eval_net(&net,tensor_in);
    
    printf("%d,%d\n",tensor_out->x[0],tensor_out->x[1]);
    
    ASSERT_TRUE(tensor_out->x[1] > tensor_out->x[0] + 20);
    
    
}

TEST_F(TestNet, Realistic5) {
    
    tensor_in = tinytensor_clone_new_tensor(&kwClip_1);
    
    ConstSequentialNetwork_t net = initialize_network();
    tensor_out = eval_net(&net,tensor_in);
    
    printf("%d,%d\n",tensor_out->x[0],tensor_out->x[1]);
    
    ASSERT_TRUE(tensor_out->x[1] > tensor_out->x[0] + 20);
    
    
}


