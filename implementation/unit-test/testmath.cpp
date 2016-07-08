#include "gtest/gtest.h"
#include "../tinytensor_math.h"


extern "C" {
    void tiny_tensor_features_add_to_buffer(const int16_t * samples, const uint32_t num_samples);
    void tiny_tensor_features_get_latest_samples(int16_t * outbuffer, const uint32_t num_samples);
    void tinytensor_features_get_mel_bank(int16_t * melbank,const int16_t * fr, const int16_t * fi);
}

class TestMath : public ::testing::Test {
protected:
    
    
    virtual void SetUp() {
    }
    
    virtual void TearDown() {
    }
    
};

class DISABLED_TestMath : public TestMath {};


TEST_F(TestMath, SoftMax) {
  
    const Weight_t x[8] = {123, -86,  57,  45,   7, 105, -59,  -4};
    
    
    Weight_t x1[8];
    Weight_t x2[8];
    
    memcpy(x1,x,sizeof(x1));
    memcpy(x2,x,sizeof(x2));

    const Weight_t y1[8] = {30,  6, 18, 17, 12, 26,  7, 11};
    const Weight_t y2[8] = {49,  0, 15, 10,  3, 49,  0,  2};
    
    tinytensor_vec_softmax_in_place(x1, 8, 0);
    tinytensor_vec_softmax_in_place(x2, 8, -2);
    
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(x1[i],y1[i],3);
    }
    
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(x2[i],y2[i],3);
    }

}

