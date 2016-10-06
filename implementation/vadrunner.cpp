#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <sndfile.hh>
#include <vector>
#include "tinytensor_features.h"
#include "tinytensor_types.h"
using namespace std;


#define BUF_SIZE (1<<10)

extern "C" {
    void results_callback(void * context, int16_t * melbins);
}

typedef struct  {
    bool isSpeech;
} CallbackContext ;

void results_callback(void * context, SpeechTransition_t transition) {
    CallbackContext * p = static_cast<CallbackContext *>(context);
    
    if (transition == start_speech && !p->isSpeech) {
        p->isSpeech = true;
    }
    
    if (transition == stop_speech && p->isSpeech) {
        p->isSpeech = false;
    }
    
}


int main(int argc, char * argv[]) {
    
    if (argc < 2) {
        std::cout << "need to have input file specified" << std::endl;
        return 0;
    }
    
    CallbackContext context;
    
    context.isSpeech = false;
    
    tinytensor_features_initialize(&context,NULL,results_callback);
    const std::string inFile = argv[1];
    
    SndfileHandle file = SndfileHandle (inFile) ;
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
            
            if (context.isSpeech) {
                std::cout << "0" << std::endl;
            }
            else {
                std::cout << "1" << std::endl;
            }
            

        }
        
    }
    
    tinytensor_features_deinitialize();
    
    
    return 0;
}
