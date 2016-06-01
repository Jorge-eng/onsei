#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <sndfile.hh>
#include <vector>
#include "tinytensor_features.h"

using namespace std;


#define BUF_SIZE (1<<10)


int main(int argc, char * argv[]) {
    
    tinytensor_features_initialize();
    const std::string inFile = argv[1];
    
    SndfileHandle file = SndfileHandle (inFile) ;
    
    printf ("Opened file '%s'\n", inFile.c_str()) ;
    printf ("    Sample rate : %d\n", file.samplerate ()) ;
    printf ("    Channels    : %d\n", file.channels ()) ;
    printf ("    Frames      : %d\n", file.frames ()) ;
    uint32_t n = file.frames();
    std::vector<int16_t> mono_samples;
    mono_samples.reserve(file.frames());
    int16_t buf[BUF_SIZE];
    
    while (true) {
        int count = file.read(buf, BUF_SIZE);
        
        if (count <= 0) {
            break;
        }
                
        for (int i = 0; i < BUF_SIZE/file.channels(); i ++) {
            mono_samples.push_back(buf[file.channels() * i]);
        }
    }
    
    for (int i = 0; i < n; i++) {
        if (i % NUM_SAMPLES_TO_RUN_FFT == 0) {
            tinytensor_features_add_samples(&mono_samples.data()[i], NUM_SAMPLES_TO_RUN_FFT);
        }
    }
    
    
    tinytensor_features_deinitialize();
    
    
    
    return 0;
}
