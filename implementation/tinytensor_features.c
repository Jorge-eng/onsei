#include "tinytensor_features.h"
#include "tinytensor_memory.h"
#include "hellomath/fft.h"

#define BUF_SIZE (2048)
#define FFT_SIZE_2N (9)
#define FFT_SIZE (1 << FFT_SIZE_2N)
#define FFT_UNPADDED_SIZE (400) 
#define NUM_SAMPLES_TO_RUN_FFT (160)
typedef struct {
    int16_t * buf;
    uint32_t num_samples;
    
} TinyTensorFeatures_t;

static TinyTensorFeatures_t _this;

void tinytensor_features_initialize(void) {
    _this.buf = MALLOC(BUF_SIZE);
    _this.num_samples = BUF_SIZE / 2;
}

void tinytensor_features_deinitialize(void) {
    FREE(_this.buf);
}


void tinytensor_features_add_samples(const int16_t * samples, const uint32_t num_samples) {
    
    /* add samples to circular buffer
       
        while current pointer is NUM_SAMPLES_TO_RUN_FFT behind the buffer pointer
        then we copy the last FFT_UNPADDED_SIZE samples to the FFT buf, zero pad it up to FFT_SIZE
     
     
     */
}

