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

class TestLstm : public ::testing::Test {
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

class DISABLED_TestLstm : public TestLstm {};


TEST_F(TestLstm, TestZeros) {
    
    ConstLayer_t lstm_layer = tinytensor_create_lstm_layer(&LSTM2_01);
    
    tensor_in = tinytensor_clone_new_tensor(&lstm1_input_zeros);
    
    uint32_t dims[4];
    lstm_layer.get_output_dims(lstm_layer.context,dims);
    
    tensor_out = tinytensor_create_new_tensor(dims);
    
    lstm_layer.eval(lstm_layer.context,NULL,tensor_out,tensor_in,input_layer);
    
   
    uint32_t * d = tensor_out->dims;
    int n = d[0] * d[1] * d[2] * d[3];
    
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(tensor_out->x[i],0,1);
    }
    
}

TEST_F(TestLstm, TestRandInput) {
    
    ConstLayer_t lstm_layer = tinytensor_create_lstm_layer(&LSTM2_01);
    
    tensor_in = tinytensor_clone_new_tensor(&lstm1_input);
    
    uint32_t dims[4];
    lstm_layer.get_output_dims(lstm_layer.context,dims);
    
    tensor_out = tinytensor_create_new_tensor(dims);
    
    lstm_layer.eval(lstm_layer.context,NULL,tensor_out,tensor_in,input_layer);
    
    
    const uint32_t * d = lstm1_ref.dims;
    int n = d[0] * d[1] * d[2] * d[3];
    
    for (int i = 0; i < n; i++) {
        int x1 = tensor_out->x[i] >> tensor_out->scale;
        int x2 = lstm1_ref_x[i] >> lstm1_ref.scale;
        
        ASSERT_NEAR(x1,x2,2);
    }

    
    /*
    Weight_t * p = tensor_out->x;
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < tensor_out->dims[3]; i++) {
            if (i!=0) std::cout << ",";
            std::cout << (int)(*p++);
        }
        std::cout << std::endl;
    }
    
    int foo = 3;
    foo++;
     */
}

TEST_F(TestLstm, TwoLayers) {
    tensor_in = tinytensor_clone_new_tensor(&lstm1_input);
    
    ConstSequentialNetwork_t net = initialize_network03();
    tensor_out = tinytensor_eval_partial_net(&net,tensor_in,3);

    const uint32_t * d = lstm3_ref.dims;
    int n = d[0] * d[1] * d[2] * d[3];
    
    for (int i = 0; i < n; i++) {
        int x1 = tensor_out->x[i] >> tensor_out->scale;
        int x2 = lstm3_ref_x[i] >> lstm3_ref.scale;
        
        ASSERT_NEAR(x1,x2,2);
    }

}


/*
TEST_F(TestLstm, kwClip_160517_02_1) {
    
    tensor_in = tinytensor_clone_new_tensor(&kwClip_160517_02_1_lstm);
    
    ConstSequentialNetwork_t net = initialize_network();
    tensor_out = eval_partial_net(&net,tensor_in,3);
    
    Weight_t * p = tensor_out->x;
    for (int j = 0; j < tensor_out->dims[2]; j++) {
        for (int i = 0; i < tensor_out->dims[3]; i++) {
            if (i!=0) std::cout << ",";
            std::cout << (int)(*p++);
        }
        std::cout << std::endl;
    }
    
    int foo = 3;
    foo++;

    
  //  printf("%f,%f\n",tensor_out->x[0]/128.,tensor_out->x[1]/128.);
    
  //  ASSERT_NEAR(tensor_out->x[1],90,20); //relaxing our standards mightily here.  Why? agh.
    
    
}
 */

