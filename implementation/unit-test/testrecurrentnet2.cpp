#include <iostream>
#include <fstream>
#include <istream>
#include <vector>
#include "gtest/gtest.h"
#include "../tinytensor_net.h"
#include "../tinytensor_features.h"

#include "unit-test/data/cryb1weights.c"
static const char * INPUT_FILE_1 = "cry_feats.bin";


class TestRecurrentNet2 : public ::testing::Test {
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

class DISABLED_TestRecurrentNet2 : public TestRecurrentNet2 {};




TEST_F(TestRecurrentNet2,TestTheWholeDeal) {
    
    ConstSequentialNetwork_t net = initialize_network();
    
    SequentialNetworkStates_t states;
    tinytensor_allocate_states(&states, &net);
    
    const uint32_t input_dims[4] = {1,1,1,NUM_MEL_BINS};

    tensor_in = tinytensor_create_new_tensor(input_dims);

    std::ifstream input( INPUT_FILE_1, std::ios::binary );

    ASSERT_TRUE(input.is_open());
    
    
    std::vector<char> buffer((
                              std::istreambuf_iterator<char>(input)),
                             (std::istreambuf_iterator<char>()));

    
    const uint32_t vec_size_bytes = NUM_MEL_BINS * sizeof(int16_t);
    const uint32_t out_vec_size_bytes = 3 * sizeof(int16_t);

    const uint32_t len = buffer.size() / vec_size_bytes;

    
    for (uint32_t i = 0; i < len; i++) {
        int16_t * feat = (int16_t *) (buffer.data() + vec_size_bytes * i);
        
       
        memcpy(tensor_in->x,feat,vec_size_bytes);

        
        tensor_out = tinytensor_eval_stateful_net(&net, &states, tensor_in,NET_FLAGS_NONE);

        if (i == len - 1) {
            ASSERT_NEAR(tensor_out->x[0],TOFIX(0.95),TOFIX(0.05));
        }
        
        tensor_out->delete_me(tensor_out);
        tensor_out = NULL;


    }
    
    
    
    tinytensor_free_states(&states, &net);

    
    
}




