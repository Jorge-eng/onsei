#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <sndfile.hh>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <functional>
#include "tinytensor_features.h"
#include "tinytensor_net.h"
#include "tinytensor_tensor.h"
#include "tinytensor_math.h"


//#include "model_aug15_lstm_small_dist_okay_sense_tiny_825.c"

#if 0

#if QFIXEDPOINT == 15
#include "model_aug30_lstm_med_dist_okay_sense_stop_snooze_tiny_912_q15.c"
#elif QFIXEDPOINT == 12
#include "model_aug30_lstm_med_dist_okay_sense_stop_snooze_tiny_912_q12.c"
#elif QFIXEDPOINT == 10
#include "model_aug30_lstm_med_dist_okay_sense_stop_snooze_tiny_912_q10.c"
#elif QFIXEDPOINT == 9
#include "model_aug30_lstm_med_dist_okay_sense_stop_snooze_tiny_912_q9.c"
#elif QFIXEDPOINT == 7
#include "model_aug30_lstm_med_dist_okay_sense_stop_snooze_tiny_912_q7.c"
#else
#error "unsupported fixed point format"
#endif

#else
#include "model_def.c"
#endif



using namespace std;


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
    
    bool is_printing = !_is_printing_only_if_activity;
    
    for (int i = 1; i < out->dims[3]; i++) {
        if (out->x[i] > OPTIONAL_PRINT_THRESHOLD) {
            is_printing = true;
            break;
        }
    }
    
    if (is_printing) {
    
        if (!last_is_printing) {
            last_is_printing = true;
    }
    
    
    
        if (_is_printing_only_if_activity) {
            printf("%d,",counter);
    }
    
        for (int i = 0; i < out->dims[3]; i++) {
            if (i!=0)printf(",");
            printf("%d",out->x[i]);
    }
   
     
    
        printf("\n");
            }
            
    else {
        if (last_is_printing) {
            last_is_printing = false;
        }
    }
        
    out->delete_me(out);
    counter++;
    
    
    
    
}


int main(int argc, char * argv[]) {
    
    if (argc < 2) {
        std::cout << "need to have input file specified" << std::endl;
        return 0;
    }
    
    CallbackContext context;
    memset(&context,0,sizeof(context));
    context.net = initialize_network();
    tinytensor_allocate_states(&context.state, &context.net);
    
    tinytensor_features_initialize(&context,results_callback,NULL);
    const std::string inFile = argv[1];
    
    if (inFile == "white") {
        _is_printing_only_if_activity = true;
        // First create an instance of an engine.
#define MAGNITUDE (100)
#define LEN (NUM_SAMPLES_TO_RUN_FFT)
        random_device rnd_device;
        // Specify the engine and distribution.
        mt19937 mersenne_engine(rnd_device());
        uniform_int_distribution<int16_t> dist(-MAGNITUDE, MAGNITUDE);
        
        auto gen = std::bind(dist, mersenne_engine);
        
        for (int i = 0; i < 66 * 3600 * 10; i++) {
            vector<int16_t> vec(NUM_SAMPLES_TO_RUN_FFT);
            generate(begin(vec), end(vec), gen);
            
            tinytensor_features_add_samples(vec.data(), NUM_SAMPLES_TO_RUN_FFT);
        }
    }
    else {
    SndfileHandle file = SndfileHandle (inFile) ;
   /*
    printf ("Opened file '%s'\n", inFile.c_str()) ;
    printf ("    Sample rate : %d\n", file.samplerate ()) ;
    printf ("    Channels    : %d\n", file.channels ()) ;
    printf ("    Frames      : %d\n", file.frames ()) ;
    */
    std::vector<int16_t> mono_samples;
    mono_samples.reserve(file.frames());
    int16_t buf[BUF_SIZE];
    
    if (file.samplerate () != 16000) {
        std::cout << "only accepts 16khz inputs" << std::endl;
        return 0;
    }
    
    while (true) {
        int count = file.read(buf, BUF_SIZE);
        
        if (count <= 0) {
            break;
        }
                
        for (int i = 0; i < BUF_SIZE/file.channels(); i ++) {
            mono_samples.push_back(buf[file.channels() * i]);
        }
    }
    
    
    for (int i = 0; i < mono_samples.size() - NUM_SAMPLES_TO_RUN_FFT; i++) {
        if (i % NUM_SAMPLES_TO_RUN_FFT == 0) {
            
            int16_t tempbuf[NUM_SAMPLES_TO_RUN_FFT];
            memset(tempbuf,0xFF,sizeof(tempbuf));
            for (int t= 0; t < NUM_SAMPLES_TO_RUN_FFT; t++) {
                tempbuf[t] = (int) (1.0 * mono_samples[i + t]);
            }
            
            tinytensor_features_add_samples(tempbuf, NUM_SAMPLES_TO_RUN_FFT);
        }
        
    }
    
    }
    tinytensor_features_deinitialize();
    tinytensor_free_states(&context.state, &context.net);
    
    return 0;
}
