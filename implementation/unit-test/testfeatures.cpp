#include "gtest/gtest.h"
#include "../tinytensor_features.h"


extern "C" {
    void tiny_tensor_features_add_to_buffer(const int16_t * samples, const uint32_t num_samples);
    uint8_t tiny_tensor_features_consume_oldest_samples(int16_t * outbuffer, const uint32_t num_samples_to_read,const uint32_t num_samples_to_consume );
    void tinytensor_features_get_mel_bank(int16_t * melbank,const int16_t * fr, const int16_t * fi);
}

class TestFeatures : public ::testing::Test {
protected:
    
    
    virtual void SetUp() {
        tinytensor_features_initialize(NULL,NULL,NULL);
        
        for (int i = 0; i < 2*BUF_SIZE_IN_SAMPLES; i++) {
            val[i] = i;
        }
        
        idx = -1;
    }
    
    virtual void TearDown() {
        tinytensor_features_deinitialize();
        
        if (idx >= 0) {
            std::cout << "INDEX OF FAILURE IS " << idx << std::endl;
        }
    }
    
    int16_t val[2*BUF_SIZE_IN_SAMPLES];

    int32_t idx;
};

class DISABLED_TestFeatures : public TestFeatures {};



TEST_F(TestFeatures, TestBatched) {
    const int16_t buf_samples_to_get = 360;
    const int16_t num_buf_samples_to_consume = 160;
  
    tinytensor_features_deinitialize();

    for (int i = 1; i < BUF_SIZE_IN_SAMPLES/2; i++) {
    
        const uint32_t batch_size = i;
        idx = i;
        
        tinytensor_features_initialize(NULL,NULL,NULL);


        for (const int16_t * p = &val[0]; p < &val[BUF_SIZE_IN_SAMPLES] - batch_size; p += batch_size) {
            tiny_tensor_features_add_to_buffer(p,batch_size);
        }
        
        int16_t val2[BUF_SIZE_IN_SAMPLES] = {0};
        
        tiny_tensor_features_consume_oldest_samples(val2,buf_samples_to_get,num_buf_samples_to_consume);
        
        for (int j = 0; j < buf_samples_to_get; j++) {
            const uint32_t expected_val = j;
            ASSERT_EQ(val2[j],expected_val);
        }
        //std::cout << "FINISHED ONE, batch size is " << batch_size << std::endl;
    }
    idx = -1;
}

TEST_F(TestFeatures, TestTypicalAddAndConsume) {
    
    const uint32_t batch_size = 7;
    const uint32_t num_buf_samples_to_get = 400;
    const uint32_t num_buf_samples_to_consume = 160;
    int16_t val_out[BUF_SIZE_IN_SAMPLES];
    int i;
    uint32_t ngets = 0 ;
    uint32_t target_val = 0;
    
    //add samples
    for (const int16_t * p = &val[0]; p < &val[BUF_SIZE_IN_SAMPLES]; p += batch_size) {
        
        tiny_tensor_features_add_to_buffer(p,batch_size);
        
        if (tiny_tensor_features_consume_oldest_samples(val_out,num_buf_samples_to_get,num_buf_samples_to_consume)) {
            const uint32_t pointer_pos = p + batch_size - &val[0];
            
            ngets++;
            
            for (i = 0; i < num_buf_samples_to_get; i++) {
                target_val = i + (ngets - 1) * num_buf_samples_to_consume;

                ASSERT_EQ(target_val,val_out[i]);
            }
            
            
        }

    }
    
    
    idx = -1;

    

}

