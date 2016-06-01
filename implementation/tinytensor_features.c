#include "tinytensor_features.h"
#include "tinytensor_memory.h"
#include "hellomath/fft.h"

#define BUF_SIZE_IN_SAMPLES (512)
#define FFT_SIZE_2N (9)
#define FFT_SIZE (1 << FFT_SIZE_2N)
#define FFT_UNPADDED_SIZE (400) 



typedef struct {
    int16_t * buf;
    int16_t * pbuf;
    int16_t * end;
    uint32_t num_samples_left_to_process;
    
} TinyTensorFeatures_t;

static TinyTensorFeatures_t _this;

void tinytensor_features_initialize(void) {
    memset(&_this,0,sizeof(TinyTensorFeatures_t));
    _this.buf = malloc(BUF_SIZE_IN_SAMPLES*sizeof(int16_t));
    _this.pbuf = _this.buf;
    _this.end = _this.buf + BUF_SIZE_IN_SAMPLES;
}

void tinytensor_features_deinitialize(void) {
    FREE(_this.buf);
}

static void add_to_buffer(const int16_t * samples, const uint32_t num_samples) {
    
    const int32_t memleft = _this.end - _this.pbuf;
    int32_t chunk_size = memleft > num_samples ? num_samples : memleft;

    MEMCPY(_this.pbuf,samples,chunk_size*sizeof(int16_t));
        
    if (chunk_size < num_samples) {
        const int32_t chunk_size2 = num_samples - chunk_size;
        _this.pbuf = _this.buf;
        MEMCPY(_this.pbuf,&samples[chunk_size],chunk_size2 * sizeof(int16_t));
        _this.pbuf += chunk_size;
    }
    else {
        _this.pbuf += num_samples;
    }
    
    _this.num_samples_left_to_process += num_samples;
}

static void read_out_buffer(int16_t * outbuffer, const uint32_t num_samples) {
    //find start
    
    int16_t * pstart = _this.pbuf - num_samples < _this.buf ? _this.buf : _this.pbuf - num_samples;
    
    uint32_t num_to_read_out = _this.pbuf - pstart;
    
    memcpy(outbuffer,pstart,num_to_read_out * sizeof(int16_t));
    
    const uint32_t num_to_read_out2 = num_samples - num_to_read_out;

    if (num_to_read_out2 > 0) {
        memcpy(outbuffer + num_to_read_out,pstart + num_to_read_out,num_to_read_out2*sizeof(int16_t));
    }
    
    _this.num_samples_left_to_process -= num_samples;
}

void tinytensor_features_add_samples(const int16_t * samples, const uint32_t num_samples) {
    int16_t fr[FFT_SIZE] = {0};
    int16_t fi[FFT_SIZE] = {0};
    
    /* add samples to circular buffer
       
        while current pointer is NUM_SAMPLES_TO_RUN_FFT behind the buffer pointer
        then we copy the last FFT_UNPADDED_SIZE samples to the FFT buf, zero pad it up to FFT_SIZE
    
     */
    
    add_to_buffer(samples,num_samples);

    

    if (_this.num_samples_left_to_process < FFT_UNPADDED_SIZE) {
        return;
    }
    
    read_out_buffer(fr,FFT_UNPADDED_SIZE);
    
    if (fr[0] != 0) {
        int foo = 3;
        foo++;
    }
    
    fft(fr,fi,FFT_SIZE_2N);

    //get mel bank feats
    
    int foo = 3;
    foo++;

    
}

