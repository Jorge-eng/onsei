#include "gtest/gtest.h"
#include "../tinytensor_math.h"
#include <math.h>

class TestMath : public ::testing::Test {
protected:
    
    int idx;
    virtual void SetUp() {
        idx = INT32_MIN;
    }
    
    virtual void TearDown() {
        if (idx != INT32_MIN) {
            std::cout << "idx = " << idx << std::endl;
        }
    }
    
};

class DISABLED_TestMath : public TestMath {};


TEST_F(TestMath, SoftMax) {
  
    
    int i;
    const float fx[8] = {1.921875, -1.34375 ,  0.890625,  0.703125,  0.109375,  1.640625,
        -0.921875, -0.0625};
    
    
    float expfx[8];
    
    for (i = 0; i < 8; i++) {
        expfx[i] = exp(fx[i]);
    }
    
    float sum = 0.0;
    
    for (i = 0; i < 8; i++) {
        sum += expfx[i];
    }
    for (i = 0; i < 8; i++) {
        expfx[i] /= sum;
    }
    
    Weight_t x1[8];

    for (i = 0; i < 8; i++) {
        x1[i] = fx[i] * (1 << QFIXEDPOINT);
    }
    
    tinytensor_vec_softmax_in_place(x1, 8, 0);
    
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(x1[i],expfx[i] * (1 << QFIXEDPOINT),14);
    }
    
   

}

TEST_F(TestMath,SoftMax2) {
    const Weight_t x[3] = {-52,-80,-109};
    
    Weight_t x1[3];
    memcpy(x1,x,sizeof(x1));
    tinytensor_vec_softmax_in_place(x1, 3, -3);
    
    int foo = 3;
    foo++;

    
    
}


TEST_F(TestMath,TestTanh) {
    
    const float qfixedpoint = (1 << QFIXEDPOINT);
    for (int i = -500; i < 500; i++) {
        idx = i;
        float f = i / 100.0;
    
        float yref = tanh(f);
        
        int32_t x = (int)(f * qfixedpoint);
        Weight_t y;
        int8_t out_scale,in_scale = 0;
        
        tinytensor_tanh(&y, &out_scale, x, in_scale);
        
        float yy = (float)y;
        yy /= qfixedpoint;
        ASSERT_NEAR(yref,yy,0.001);
        
    }
    
    idx = INT32_MIN;
    
    
}

