#include "gtest/gtest.h"
#include "../tinytensor_math.h"

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

    const Weight_t y1[8] = {30,  5, 18, 16, 12, 26,  7, 11};
    const Weight_t y2[8] = {69,  0,  8,  6,  1, 39,  0,  1};
    
    tinytensor_vec_softmax_in_place(x1, 8, 0);
    tinytensor_vec_softmax_in_place(x2, 8, -2);
    
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(x1[i],y1[i],14);
    }
    
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(x2[i],y2[i],14);
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

