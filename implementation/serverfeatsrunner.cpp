#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <functional>
#include "tinytensor_features.h"
#include "tinytensor_net.h"
#include "tinytensor_tensor.h"
#include "tinytensor_math.h"

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

#include "model_def.c"



using namespace std;
using namespace rapidjson;

#define BUF_SIZE (1<<10)

#define OPTIONAL_PRINT_THRESHOLD (TOFIX(0.1))
extern "C" {
    void results_callback(void * context, int8_t * melbins);
}

static bool _is_printing_only_if_activity = false;
typedef struct  {
    int8_t buf[NUM_MEL_BINS][MEL_FEAT_BUF_TIME_LEN];
    uint32_t bufidx;
    ConstSequentialNetwork_t net;
    SequentialNetworkStates_t state;
} CallbackContext ;

void results_callback(void * context, int16_t * melbins) {
    static uint32_t counter = 0;
    CallbackContext * p = static_cast<CallbackContext *>(context);
    
    static bool last_is_printing = false;
    
    Tensor_t temp_tensor;
    temp_tensor.dims[0] = 1;
    temp_tensor.dims[1] = 1;
    temp_tensor.dims[2] = 1;
    temp_tensor.dims[3] = NUM_MEL_BINS;
    
    temp_tensor.x = melbins;
    temp_tensor.scale = 0;
    temp_tensor.delete_me = 0;
   
    
    Tensor_t * out = tinytensor_eval_stateful_net(&p->net, &p->state, &temp_tensor,NET_FLAGS_NONE);
    
    
        
    out->delete_me(out);
    counter++;
    
    
    
    
}

static void do_something_with_the_text(const std::string & str) {
    Document d;
    d.Parse(str.c_str());
    
    std::cout << d.IsObject() << std::endl;

}


int main(int argc, char * argv[]) {
    
    if (argc < 2) {
        std::cout << "need to have input file specified" << std::endl;
        return 0;
    }
    

    std::ifstream file(argv[1]);
    
    if (!file.is_open()) {
        std::cout << "unable to open file" << std::endl;
        return 0;
    }
    
    
    
    
    
    
    //Value::ConstValueIterator
    
    file.seekg(0,std::ios::end);
    size_t pos = file.tellg();
    file.seekg(0,std::ios::beg);
    
    
    char buf[pos];
    std::string text;

    text.resize(pos );
    
    file.read(buf,sizeof(buf));
    text.assign(buf,sizeof(buf));
    
    size_t prevpos = 0;
    
    while(1) {
        pos = text.find("}",prevpos);

        if (pos == std::string::npos) {
            break;
        }
        
        std::string objecttext = text.substr(prevpos,pos-prevpos + 1);
        
        do_something_with_the_text(objecttext);
        
        prevpos = pos + 1;
    }
    
    
    
    
    
    /*
    CallbackContext context;
    memset(&context,0,sizeof(context));
    context.net = initialize_network();
    tinytensor_allocate_states(&context.state, &context.net);
    
    tinytensor_features_initialize(&context,results_callback,NULL);
 
    tinytensor_features_deinitialize();
    tinytensor_free_states(&context.state, &context.net);
    */
    
    return 0;
}
