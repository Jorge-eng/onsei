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
    void results_callback(void * context, int8_t * melbins);
}

typedef struct  {
    FILE * file;
} CallbackContext ;

void results_callback(void * context, int16_t * melbins,uint32_t flags) {
    CallbackContext * p = static_cast<CallbackContext *>(context);
    if (p->file) {
        fwrite(melbins,sizeof(Weight_t),NUM_MEL_BINS,p->file);
    }
}


int main(int argc, char * argv[]) {
    
    if (argc < 3) {
        std::cout << "need to have input file and output file specified" << std::endl;
        return 0;
    }
    
    CallbackContext context;
    context.file = fopen(argv[2], "w");
    
    tinytensor_features_initialize(&context,results_callback,NULL);
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
    
    fclose(context.file);
    tinytensor_features_deinitialize();
    
    
    return 0;
}
