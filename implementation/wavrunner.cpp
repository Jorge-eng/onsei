#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <sndfile.hh>
#include <vector>
#include "tinytensor_features.h"
#include "tinytensor_net.h"
#include "tinytensor_tensor.h"

#include "unit-test/data/model_jun22_smaller_sigm.c"
//#include "unit-test/data/model_may25_lstm_large.c"

using namespace std;


#define BUF_SIZE (1<<10)

extern "C" {
    void results_callback(void * context, int8_t * melbins);
}

typedef struct  {
    int8_t buf[NUM_MEL_BINS][MEL_FEAT_BUF_TIME_LEN];
    uint32_t bufidx;
    ConstSequentialNetwork_t net;
} CallbackContext ;

void results_callback(void * context, int8_t * melbins) {
    static uint32_t counter = 0;
    CallbackContext * p = static_cast<CallbackContext *>(context);

    for (uint32_t i = 0; i < NUM_MEL_BINS; i++) {
        int32_t temp32 = melbins[i] + 0;
        if (temp32 > MAX_WEIGHT) {
            temp32 = MAX_WEIGHT;
        }
        p->buf[i][p->bufidx] = (Weight_t)temp32;
    }
    
    
    if (++(p->bufidx) >= MEL_FEAT_BUF_TIME_LEN) {
        p->bufidx = 0;
    }
    
    
    
    if (++counter % 10) {
        return;
    }
    
    if (counter < MEL_FEAT_BUF_TIME_LEN) {
        return;
    }
   
     
    
    const uint32_t dims[4] = {1,1,NUM_MEL_BINS,MEL_FEAT_BUF_TIME_LEN};
    
    Tensor_t * tensor_in = tinytensor_create_new_tensor(dims);
    
    Weight_t * px = tensor_in->x;
    for (uint32_t i = 0; i < NUM_MEL_BINS; i++ ) {
        uint32_t bufidx = p->bufidx;

        for (uint32_t t = 0; t < MEL_FEAT_BUF_TIME_LEN; t++) {
            *px = p->buf[i][bufidx];
            
            if (++bufidx >= MEL_FEAT_BUF_TIME_LEN) {
                bufidx = 0;
            }
            
            px++;
        }
    }
        
    
    
    /*
     Tensor_t * transposed = tinytensor_create_new_transposed_tensor(tensor_in);
     
     if (tensor_in) {
        tensor_in->delete_me(tensor_in);
     }
     */

    Tensor_t * tensor_out = tinytensor_eval_net(&(p->net),tensor_in);
    
    std::cout << (int32_t)tensor_out->x[0] << "," << (int32_t)tensor_out->x[1] << std::endl;
    
  
    if (tensor_out) {
        tensor_out->delete_me(tensor_out);
    }
    
    
}


int main(int argc, char * argv[]) {
    
    if (argc < 2) {
        std::cout << "need to have input file specified" << std::endl;
        return 0;
    }
    
    CallbackContext context;
    memset(&context,0,sizeof(context));
    context.net = initialize_network();
    
    tinytensor_features_initialize(&context,results_callback);
    const std::string inFile = argv[1];
    
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
    
    tinytensor_features_deinitialize();
    
    
    return 0;
}
