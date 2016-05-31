#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <sndfile.hh>

#include "tinytensor_conv_layer.h"
#include "tinytensor_math.h"

using namespace std;


#define BUF_SIZE (1<<20)


int main(int argc, char * argv[]) {
    
    const std::string inFile = argv[1];
    
    SndfileHandle file = SndfileHandle (inFile) ;
    
    printf ("Opened file '%s'\n", inFile.c_str()) ;
    printf ("    Sample rate : %d\n", file.samplerate ()) ;
    printf ("    Channels    : %d\n", file.channels ()) ;
    printf ("    Frames      : %d\n", file.frames ()) ;
    
    int16_t mono_samples[file.frames()];
    int16_t buf[BUF_SIZE];
    
    while (true) {
        int count = file.read(buf, BUF_SIZE * file.channels());
        
        if (count <= 0) {
            break;
        }
        
        memset(buf,0,sizeof(buf));
        
        for (int i = 0; i < BUF_SIZE; i ++) {
            mono_samples[i] = buf[file.channels() * i];
        }
    }
    
    

    
    return 0;
}
