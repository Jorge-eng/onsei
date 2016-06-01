#include "tinytensor_features.h"
#include "tinytensor_memory.h"
#include "tinytensor_math.h"

#include "hellomath/fft.h"
#include "hellomath/hellomath.h"

#define BUF_SIZE_IN_SAMPLES (512)
#define FFT_SIZE_2N (9)
#define FFT_SIZE (1 << FFT_SIZE_2N)
#define FFT_UNPADDED_SIZE (400) 
#define NUM_MEL_BINS (40)
#define PREEMPHASIS (TOFIX(0.95,15))

const static int16_t k_hanning[FFT_UNPADDED_SIZE] = {0,2,8,18,32,51,73,99,130,164,203,245,292,342,397,455,517,584,654,728,806,888,973,1063,1156,1253,1354,1459,1567,1679,1794,1914,2036,2163,2293,2426,2563,2703,2847,2994,3144,3298,3455,3615,3778,3944,4114,4286,4462,4640,4821,5006,5193,5383,5575,5770,5968,6169,6372,6577,6785,6995,7208,7423,7640,7859,8080,8304,8529,8757,8986,9217,9450,9684,9921,10159,10398,10639,10881,11125,11370,11616,11863,12112,12362,12612,12864,13116,13369,13623,13878,14133,14389,14645,14902,15159,15417,15674,15932,16190,16448,16706,16964,17222,17479,17736,17993,18250,18506,18762,19016,19271,19524,19777,20029,20280,20530,20779,21027,21274,21520,21764,22007,22249,22489,22728,22965,23200,23434,23666,23896,24124,24351,24575,24798,25018,25236,25452,25666,25877,26086,26293,26497,26699,26898,27095,27289,27480,27668,27854,28037,28216,28393,28567,28738,28906,29071,29233,29391,29546,29698,29847,29992,30134,30273,30408,30540,30668,30792,30913,31031,31145,31255,31361,31464,31563,31658,31749,31837,31921,32001,32077,32149,32217,32281,32342,32398,32451,32499,32544,32584,32620,32653,32681,32706,32726,32742,32754,32762,32766,32766,32762,32754,32742,32726,32706,32681,32653,32620,32584,32544,32499,32451,32398,32342,32281,32217,32149,32077,32001,31921,31837,31749,31658,31563,31464,31361,31255,31145,31031,30913,30792,30668,30540,30408,30273,30134,29992,29847,29698,29546,29391,29233,29071,28906,28738,28567,28393,28216,28037,27854,27668,27480,27289,27095,26898,26699,26497,26293,26086,25877,25666,25452,25236,25018,24798,24575,24351,24124,23896,23666,23434,23200,22965,22728,22489,22249,22007,21764,21520,21274,21027,20779,20530,20280,20029,19777,19524,19271,19016,18762,18506,18250,17993,17736,17479,17222,16964,16706,16448,16190,15932,15674,15417,15159,14902,14645,14389,14133,13878,13623,13369,13116,12864,12612,12362,12112,11863,11616,11370,11125,10881,10639,10398,10159,9921,9684,9450,9217,8986,8757,8529,8304,8080,7859,7640,7423,7208,6995,6785,6577,6372,6169,5968,5770,5575,5383,5193,5006,4821,4640,4462,4286,4114,3944,3778,3615,3455,3298,3144,2994,2847,2703,2563,2426,2293,2163,2036,1914,1794,1679,1567,1459,1354,1253,1156,1063,973,888,806,728,654,584,517,455,397,342,292,245,203,164,130,99,73,51,32,18,8,2,0};


const static uint8_t k_coeffs[454] = {255,255,127,127,255,127,127,255,127,127,255,127,127,255,127,127,255,127,127,255,127,127,255,170,85,85,170,255,127,127,255,170,85,85,170,255,170,85,85,170,255,170,85,85,170,255,170,85,85,170,255,191,127,63,63,127,191,255,191,127,63,63,127,191,255,191,127,63,63,127,191,255,191,127,63,63,127,191,255,204,153,102,51,51,102,153,204,255,204,153,102,51,51,102,153,204,255,204,153,102,51,51,102,153,204,255,204,153,102,51,51,102,153,204,255,212,170,127,85,42,42,85,127,170,212,255,212,170,127,85,42,42,85,127,170,212,255,218,182,145,109,72,36,36,72,109,145,182,218,255,218,182,145,109,72,36,36,72,109,145,182,218,255,223,191,159,127,95,63,31,31,63,95,127,159,191,223,255,218,182,145,109,72,36,36,72,109,145,182,218,255,226,198,170,141,113,85,56,28,28,56,85,113,141,170,198,226,255,226,198,170,141,113,85,56,28,28,56,85,113,141,170,198,226,255,226,198,170,141,113,85,56,28,28,56,85,113,141,170,198,226,255,231,208,185,162,139,115,92,69,46,23,23,46,69,92,115,139,162,185,208,231,255,229,204,178,153,127,102,76,51,25,25,51,76,102,127,153,178,204,229,255,233,212,191,170,148,127,106,85,63,42,21,21,42,63,85,106,127,148,170,191,212,233,255,233,212,191,170,148,127,106,85,63,42,21,21,42,63,85,106,127,148,170,191,212,233,255,235,215,196,176,156,137,117,98,78,58,39,19,19,39,58,78,98,117,137,156,176,196,215,235,255,236,218,200,182,163,145,127,109,91,72,54,36,18,18,36,54,72,91,109,127,145,163,182,200,218,236,255,238,221,204,187,170,153,136,119,102,85,68,51,34,17,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255,238,221,204,187,170,153,136,119,102,85,68,51,34,17,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255,240,225,210,195,180,165,150,135,120,105,90,75,60,45,30,15};

const static uint8_t k_fft_index_pairs[40][2] = {{1,1},{2,3},{3,5},{5,7},{7,9},{9,11},{11,13},{13,15},{15,18},{17,20},{20,23},{22,26},{25,29},{28,32},{31,36},{34,40},{38,44},{42,48},{46,53},{50,58},{55,63},{60,68},{65,74},{70,80},{76,87},{82,94},{89,102},{96,109},{104,118},{111,127},{120,136},{129,147},{138,157},{149,169},{159,181},{171,194},{183,208},{196,223},{210,238},{225,255}};

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

static uint8_t add_samples_and_get_mel(int16_t * melbank, const int16_t * samples, const uint32_t num_samples) {
    const int16_t preemphasis_coeff = PREEMPHASIS;

    int16_t fr[FFT_SIZE] = {0};
    int16_t fi[FFT_SIZE] = {0};
    
    uint32_t i;
    uint32_t ifft;
    uint32_t imel;
    uint32_t idx;
    uint32_t utemp32;
    uint64_t accumulator;
    int16_t temp16;
    /* add samples to circular buffer
       
        while current pointer is NUM_SAMPLES_TO_RUN_FFT behind the buffer pointer
        then we copy the last FFT_UNPADDED_SIZE samples to the FFT buf, zero pad it up to FFT_SIZE
    
     */
    
    add_to_buffer(samples,num_samples);

    if (_this.num_samples_left_to_process < FFT_UNPADDED_SIZE) {
        return 0;
    }
    
    read_out_buffer(fr,FFT_UNPADDED_SIZE);
    
    return 0;

    
    //"preemphasis"
    for (i = 1; i < FFT_UNPADDED_SIZE; i++) {
        temp16 = MUL16(fr[i-1], preemphasis_coeff);
        fr[i] = fr[i] - temp16;
    }
    
    //APPLY WINDOW
    for (i = 0; i < FFT_UNPADDED_SIZE; i++) {
        fr[i] = MUL16(k_hanning[i],fr[i]);
    }
   
    //PERFORM FFT
    fft(fr,fi,FFT_SIZE_2N);
    
    //get mel bank feats
    idx = 0;
    for (imel = 0; imel < NUM_MEL_BINS; imel++) {
        accumulator = 0;
        for (ifft = k_fft_index_pairs[imel][0]; ifft <= k_fft_index_pairs[imel][1]; ifft++) {
            utemp32 = 1;
            utemp32 += ((uint32_t)fr[ifft]*fr[ifft]) + ((uint32_t)fi[ifft]*fi[ifft]); //q15 + q15 = q30, q30 * 2 --> q31, unsigned 32 is safe
            utemp32 = (uint32_t)((((uint64_t) utemp32) * k_coeffs[idx]) >> 8);
            accumulator += utemp32;
            idx++;
        }
        
        melbank[imel] = FixedPointLog2Q10(accumulator);
    }
    
    return 1;
}

void tinytensor_features_add_samples(const int16_t * samples, const uint32_t num_samples) {
    int16_t melbank[NUM_MEL_BINS];
    int16_t scaled_melbank[NUM_MEL_BINS];
    uint32_t i;
    if (add_samples_and_get_mel(melbank,samples,num_samples)) {
        
        for (i = 0; i < NUM_MEL_BINS; i++) {
            melbank[i] >>= 5;
            if (melbank[i] < -MAX_WEIGHT) {
                melbank[i] = -MAX_WEIGHT;
            }
           
            if (melbank[i] > MAX_WEIGHT) {
                melbank[i] = MAX_WEIGHT;
            }
        
            scaled_melbank[i] = melbank[i];
            
        }
        
        for (i = 5; i < NUM_MEL_BINS; i++) {
            if (scaled_melbank[i] > 0) {
                int foo = 3;
                foo++;
            }

        }
        
        //TODO add to circular buffer for mel bins
        int foo = 3;
        foo++;
    }
    
}
