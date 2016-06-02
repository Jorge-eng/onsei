#include "gtest/gtest.h"
#include "../tinytensor_features.h"


extern "C" {
    void tiny_tensor_features_add_to_buffer(const int16_t * samples, const uint32_t num_samples);
    void tiny_tensor_features_get_latest_samples(int16_t * outbuffer, const uint32_t num_samples);
    void tinytensor_features_get_mel_bank(int16_t * melbank,const int16_t * fr, const int16_t * fi);
}

class TestFeatures : public ::testing::Test {
protected:
    
    
    virtual void SetUp() {
        tinytensor_features_initialize();
    }
    
    virtual void TearDown() {
        tinytensor_features_deinitialize();
    }
    
};

class DISABLED_TestFeatures : public TestFeatures {};


TEST_F(TestFeatures, TestOneByOne) {
  
    int16_t val[BUF_SIZE_IN_SAMPLES];
    
    for (int i = 0; i < BUF_SIZE_IN_SAMPLES; i++) {
        val[0] = i;
        tiny_tensor_features_add_to_buffer(val,1);
    }
    
    tiny_tensor_features_get_latest_samples(val,BUF_SIZE_IN_SAMPLES);
    
    for (int i = 0; i < BUF_SIZE_IN_SAMPLES; i++) {
        ASSERT_EQ(val[i],i);
    }
    
}

TEST_F(TestFeatures, TestBatched) {
    
    int16_t val[BUF_SIZE_IN_SAMPLES];
    
    for (int i = 0; i < BUF_SIZE_IN_SAMPLES; i++) {
        val[i] = i;
    }
    
    tiny_tensor_features_add_to_buffer(val,100);
    tiny_tensor_features_add_to_buffer(&val[100],120);

    int16_t val2[BUF_SIZE_IN_SAMPLES] = {0};

    tiny_tensor_features_get_latest_samples(val2,140);
    
    for (int i = 0; i < 140; i++) {
        ASSERT_EQ(val2[i],i + 80);
    }
    
}

