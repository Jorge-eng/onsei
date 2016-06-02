#include "tinytensor_features.h"
#include "tinytensor_memory.h"

#include "hellomath/fft.h"
#include "hellomath/hellomath.h"

#define FFT_SIZE_2N (9)
#define FFT_SIZE (1 << FFT_SIZE_2N)

//0.95 in Q15
#define PREEMPHASIS (31129)

//0.01031 in Q15
#define SCALE_TO_8_BITS (338)

#define QFIXEDPOINT_INT16 (15)

#define MUL16(a,b)\
((int16_t)(((int32_t)(a * b)) >> QFIXEDPOINT_INT16))

#define LOGOFFSET (5400)
#define NOISE_FLOOR (1)

//hanning * 0.84
const static int16_t k_hanning[FFT_UNPADDED_SIZE] = {0,2,7,15,27,43,61,84,109,138,170,206,245,287,333,382,435,490,549,611,677,746,818,893,971,1053,1137,1225,1316,1410,1507,1608,1711,1817,1926,2038,2153,2271,2391,2515,2641,2770,2902,3036,3173,3313,3456,3600,3748,3898,4050,4205,4362,4521,4683,4847,5013,5182,5352,5525,5699,5876,6054,6235,6417,6601,6787,6975,7164,7355,7548,7742,7938,8135,8333,8533,8734,8937,9140,9345,9551,9757,9965,10174,10384,10594,10806,11018,11230,11444,11657,11872,12087,12302,12518,12734,12950,13166,13383,13600,13816,14033,14250,14466,14682,14899,15114,15330,15545,15760,15974,16187,16400,16613,16825,17035,17245,17455,17663,17870,18077,18282,18486,18689,18891,19091,19290,19488,19684,19879,20073,20264,20455,20643,20830,21015,21198,21380,21559,21737,21913,22086,22258,22427,22594,22760,22922,23083,23241,23397,23551,23702,23851,23997,24140,24281,24420,24556,24689,24819,24947,25072,25194,25313,25429,25543,25653,25761,25866,25967,26066,26161,26254,26343,26430,26513,26593,26669,26743,26813,26881,26944,27005,27062,27116,27167,27214,27259,27299,27337,27371,27401,27428,27452,27473,27490,27503,27514,27520,27524,27524,27520,27514,27503,27490,27473,27452,27428,27401,27371,27337,27299,27259,27214,27167,27116,27062,27005,26944,26881,26813,26743,26669,26593,26513,26430,26343,26254,26161,26066,25967,25866,25761,25653,25543,25429,25313,25194,25072,24947,24819,24689,24556,24420,24281,24140,23997,23851,23702,23551,23397,23241,23083,22922,22760,22594,22427,22258,22086,21913,21737,21559,21380,21198,21015,20830,20643,20455,20264,20073,19879,19684,19488,19290,19091,18891,18689,18486,18282,18077,17870,17663,17455,17245,17035,16825,16613,16400,16187,15974,15760,15545,15330,15114,14899,14682,14466,14250,14033,13816,13600,13383,13166,12950,12734,12518,12302,12087,11872,11657,11444,11230,11018,10806,10594,10384,10174,9965,9757,9551,9345,9140,8937,8734,8533,8333,8135,7938,7742,7548,7355,7164,6975,6787,6601,6417,6235,6054,5876,5699,5525,5352,5182,5013,4847,4683,4521,4362,4205,4050,3898,3748,3600,3456,3313,3173,3036,2902,2770,2641,2515,2391,2271,2153,2038,1926,1817,1711,1608,1507,1410,1316,1225,1137,1053,971,893,818,746,677,611,549,490,435,382,333,287,245,206,170,138,109,84,61,43,27,15,7,2,0};


const static uint8_t k_coeffs[454] = {255,255,127,127,255,127,127,255,127,127,255,127,127,255,127,127,255,127,127,255,127,127,255,170,85,85,170,255,127,127,255,170,85,85,170,255,170,85,85,170,255,170,85,85,170,255,170,85,85,170,255,191,127,63,63,127,191,255,191,127,63,63,127,191,255,191,127,63,63,127,191,255,191,127,63,63,127,191,255,204,153,102,51,51,102,153,204,255,204,153,102,51,51,102,153,204,255,204,153,102,51,51,102,153,204,255,204,153,102,51,51,102,153,204,255,212,170,127,85,42,42,85,127,170,212,255,212,170,127,85,42,42,85,127,170,212,255,218,182,145,109,72,36,36,72,109,145,182,218,255,218,182,145,109,72,36,36,72,109,145,182,218,255,223,191,159,127,95,63,31,31,63,95,127,159,191,223,255,218,182,145,109,72,36,36,72,109,145,182,218,255,226,198,170,141,113,85,56,28,28,56,85,113,141,170,198,226,255,226,198,170,141,113,85,56,28,28,56,85,113,141,170,198,226,255,226,198,170,141,113,85,56,28,28,56,85,113,141,170,198,226,255,231,208,185,162,139,115,92,69,46,23,23,46,69,92,115,139,162,185,208,231,255,229,204,178,153,127,102,76,51,25,25,51,76,102,127,153,178,204,229,255,233,212,191,170,148,127,106,85,63,42,21,21,42,63,85,106,127,148,170,191,212,233,255,233,212,191,170,148,127,106,85,63,42,21,21,42,63,85,106,127,148,170,191,212,233,255,235,215,196,176,156,137,117,98,78,58,39,19,19,39,58,78,98,117,137,156,176,196,215,235,255,236,218,200,182,163,145,127,109,91,72,54,36,18,18,36,54,72,91,109,127,145,163,182,200,218,236,255,238,221,204,187,170,153,136,119,102,85,68,51,34,17,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255,238,221,204,187,170,153,136,119,102,85,68,51,34,17,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255,240,225,210,195,180,165,150,135,120,105,90,75,60,45,30,15};

const static uint8_t k_fft_index_pairs[40][2] = {{1,1},{2,3},{3,5},{5,7},{7,9},{9,11},{11,13},{13,15},{15,18},{17,20},{20,23},{22,26},{25,29},{28,32},{31,36},{34,40},{38,44},{42,48},{46,53},{50,58},{55,63},{60,68},{65,74},{70,80},{76,87},{82,94},{89,102},{96,109},{104,118},{111,127},{120,136},{129,147},{138,157},{149,169},{159,181},{171,194},{183,208},{196,223},{210,238},{225,255}};

typedef struct {
    int16_t * buf;
    int16_t * pbuf_write;
    int16_t * end;
    uint32_t num_samples_in_buffer;
    
} TinyTensorFeatures_t;

static TinyTensorFeatures_t _this;

void tinytensor_features_initialize(void) {
    memset(&_this,0,sizeof(TinyTensorFeatures_t));
    _this.buf = malloc(BUF_SIZE_IN_SAMPLES*sizeof(int16_t));
    memset(_this.buf,0,BUF_SIZE_IN_SAMPLES*sizeof(int16_t));
    _this.pbuf_write = _this.buf;
    _this.end = _this.buf + BUF_SIZE_IN_SAMPLES;
}

void tinytensor_features_deinitialize(void) {
    FREE(_this.buf);
}


void tiny_tensor_features_add_to_buffer(const int16_t * samples, const uint32_t num_samples) {
   
    int32_t ibuf;
    
    for (ibuf = 0; ibuf < num_samples; ibuf++) {
        *(_this.pbuf_write) = samples[ibuf];
        
        if (++_this.pbuf_write >= _this.end) {
            _this.pbuf_write = _this.buf;
        }
    }
    
    _this.num_samples_in_buffer += num_samples;
    
    if (_this.num_samples_in_buffer > BUF_SIZE_IN_SAMPLES) {
        _this.num_samples_in_buffer = BUF_SIZE_IN_SAMPLES;
    }
}

void tiny_tensor_features_get_latest_samples(int16_t * outbuffer, const uint32_t num_samples) {
    int32_t ibuf;
    int16_t * pstart = _this.pbuf_write - num_samples;
    
    if (pstart < _this.buf) {
        pstart += BUF_SIZE_IN_SAMPLES;
    }
    
    
    for (ibuf = 0; ibuf < num_samples; ibuf++) {
        outbuffer[ibuf] = *pstart;
        
        if (++pstart >= _this.end) {
            pstart = _this.buf;
        }
    }
}

void tinytensor_features_get_mel_bank(int16_t * melbank,const int16_t * fr, const int16_t * fi) {
    uint32_t ifft;
    uint32_t imel;
    uint32_t idx;
    uint32_t utemp32;
    uint64_t accumulator;
    
    //get mel bank feats
    idx = 0;
    for (imel = 0; imel < NUM_MEL_BINS; imel++) {
        accumulator = 0;
        for (ifft = k_fft_index_pairs[imel][0]; ifft <= k_fft_index_pairs[imel][1]; ifft++) {
            utemp32 = NOISE_FLOOR;
            utemp32 += ((uint32_t)fr[ifft]*fr[ifft]) + ((uint32_t)fi[ifft]*fi[ifft]); //q15 + q15 = q30, q30 * 2 --> q31, unsigned 32 is safe
            utemp32 = (uint32_t)((((uint64_t) utemp32) * k_coeffs[idx]) >> 8);
            accumulator += utemp32;
            idx++;
        }
        
        melbank[imel] = FixedPointLog2Q10(accumulator);
    }

}

static uint8_t add_samples_and_get_mel(int16_t * melbank, const int16_t * samples, const uint32_t num_samples) {
    int16_t fr[FFT_SIZE] = {0};
    int16_t fi[FFT_SIZE] = {0};
    const int16_t preemphasis_coeff = PREEMPHASIS;
    uint32_t i;

    int32_t temp32;
    /* add samples to circular buffer
       
        while current pointer is NUM_SAMPLES_TO_RUN_FFT behind the buffer pointer
        then we copy the last FFT_UNPADDED_SIZE samples to the FFT buf, zero pad it up to FFT_SIZE
    
     */
    
    tiny_tensor_features_add_to_buffer(samples,num_samples);

    if (_this.num_samples_in_buffer < FFT_UNPADDED_SIZE) {
        return 0;
    }
    
    tiny_tensor_features_get_latest_samples(fr,FFT_UNPADDED_SIZE);
    
  
    //"preemphasis", and apply window as you go
    memcpy(fi,fr,sizeof(fi));
    for (i = 1; i < FFT_UNPADDED_SIZE; i++) {

        temp32 = fr[i] - MUL16(preemphasis_coeff,fi[i-1]);
        
        //APPLY WINDOW
        temp32 *= k_hanning[i];
        temp32 >>= QFIXEDPOINT_INT16;
        
        if (temp32 > MAX_INT_16) {
            temp32 = MAX_INT_16;
        }
        
        if (temp32 < -MAX_INT_16) {
            temp32 = -MAX_INT_16;
        }
        
        fr[i] = (int16_t)temp32;
    }
    
    fr[0] = MUL16(k_hanning[0],fr[0]);

    memset(fi,0,sizeof(fi));
   
    //PERFORM FFT
    fft(fr,fi,FFT_SIZE_2N);
    
    tinytensor_features_get_mel_bank(melbank,fr,fi);
    
    return 1;
}

void tinytensor_features_add_samples(const int16_t * samples, const uint32_t num_samples) {
    int16_t melbank[NUM_MEL_BINS];
    int8_t melbank8[NUM_MEL_BINS];
    int16_t temp16;
    uint32_t i;
    if (add_samples_and_get_mel(melbank,samples,num_samples)) {
        
        for (i = 0; i <NUM_MEL_BINS; i++) {
            temp16 = melbank[i];
            
            temp16 -= LOGOFFSET;  //the Q10 logarithm will typically appear between -6144 and 18432, so we subtract the midpoint
            

            temp16 = MUL16(temp16, SCALE_TO_8_BITS);
            
            if (temp16 > INT8_MAX) {
                temp16 = INT8_MAX;
            }
            
            if (temp16 < -INT8_MAX) {
                temp16 = -INT8_MAX;
            }
            
            melbank8[i] = (int8_t)temp16;
        }
        
        for (i = 0; i < 40; i++) {
            if (i!= 0) {
                printf (",");
            }
            printf("%d",melbank8[i]);
        }
        
        printf("\n");
        
        
        //TODO add to circular buffer for mel bins
        int foo = 3;
        foo++;
    }
    
}
