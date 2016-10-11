#include <iostream>
#include <fstream>
#include <istream>
#include <vector>
#include "gtest/gtest.h"
#include "../tinytensor_net.h"
#include "../tinytensor_features.h"

#include "unit-test/data/model_oct09_lstm_24x24_dist_okay_sense_tiny_95_1009_ep044.c"
const char * INPUT_FILE_1 = "reference_wav_7s.bin";
const char * REF_OUT_FILE_1 = "reference_wav_7s_lstm_out.bin";



class TestRecurrentNet : public ::testing::Test {
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

class DISABLED_TestRecurrentNet : public TestRecurrentNet {};




TEST_F(TestRecurrentNet,TestTheWholeDeal) {
    
    ConstSequentialNetwork_t net = initialize_network();
    
    SequentialNetworkStates_t states;
    tinytensor_allocate_states(&states, &net);
    
    const uint32_t input_dims[4] = {1,1,1,NUM_MEL_BINS};

    tensor_in = tinytensor_create_new_tensor(input_dims);

    
    std::ifstream input( INPUT_FILE_1, std::ios::binary );
    std::ifstream refoutput( REF_OUT_FILE_1, std::ios::binary );

    ASSERT_TRUE(input.is_open());
    
    
    std::vector<char> buffer((
                              std::istreambuf_iterator<char>(input)),
                             (std::istreambuf_iterator<char>()));
    
    ASSERT_TRUE(refoutput.is_open());

    std::vector<char> refbuffer((
                              std::istreambuf_iterator<char>(refoutput)),
                             (std::istreambuf_iterator<char>()));
    
    const uint32_t vec_size_bytes = NUM_MEL_BINS * sizeof(int16_t);
    const uint32_t out_vec_size_bytes = 3 * sizeof(int16_t);

    const uint32_t len = buffer.size() / vec_size_bytes;
    const uint32_t len2 = refbuffer.size() / out_vec_size_bytes;

    ASSERT_TRUE(len == len2);
    
    for (uint32_t i = 0; i < len; i++) {
        int16_t * feat = (int16_t *) (buffer.data() + vec_size_bytes * i);
        int16_t * ref = (int16_t *) (refbuffer.data() + out_vec_size_bytes * i);

        memcpy(tensor_in->x,feat,vec_size_bytes);

        
        tensor_out = tinytensor_eval_stateful_net(&net, &states, tensor_in,NET_FLAGS_NONE);

        int err = 0;
        
        for (uint32_t j = 0; j < tensor_out->dims[3]; j++) {
            err += abs(tensor_out->x[j] - ref[j]);
        }
        
        if (i > 2) {
            ASSERT_LT(err,5);
        }
        
        /*
        for (uint32_t j = 0; j < tensor_out->dims[3]; j++) {
            if (j != 0) std::cout << ",";
            std::cout << tensor_out->x[j];
        }
        for (uint32_t j = 0; j < tensor_out->dims[3]; j++) {
            std::cout << ",";
            std::cout << ref[j];
        }
         std::cout << std::endl;

        */
        
        
        tensor_out->delete_me(tensor_out);
        tensor_out = NULL;


    }
    
    
    
    tinytensor_free_states(&states, &net);

    
    
}




