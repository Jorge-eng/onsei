#include "gtest/gtest.h"
#include "../tinytensor_net.h"

#include "data/model_may31_small_sigm.c"
//#include "data/test_input_rand.c"
//#include "data/test_input.c"
//#include "data/kwClip_160517_02_1.c"
//#include "data/kwClip_160517_03_1.c"
//#include "data/kwClip_160517_04.c"
//#include "data/kwClip_160517_05_1.c"


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
    
    ImageTensor_t * tensor_in;
    ImageTensor_t * tensor_out;
    
};

class DISABLED_TestNet : public TestNet {};


#define NNTEST(testname,inputname,expectedoutput)\
TEST_F(TestNet, testname) {\
\
tensor_in = tinytensor_clone_new_tensor(&inputname);\
\
ConstSequentialNetwork_t net = initialize_network();\
tensor_out = eval_net(&net,tensor_in);\
\
printf("%d,%d\n",tensor_out->x[0],tensor_out->x[1]);\
\
ASSERT_NEAR(tensor_out->x[1],expectedoutput,10);\
\
}


/*
NNTEST(TestRandInput,test_input_rand,10)

NNTEST(Realistic2,kwClip_160517_02_1,90)

NNTEST(Realistic3,kwClip_160517_03_1,127)

NNTEST(Realistic4,kwClip_160517_04,127)

NNTEST(Realistic5,kwClip_160517_05_1,127)

*/


