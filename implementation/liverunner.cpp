#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "portaudio.h"
#include "tinytensor_features.h"
#include "tinytensor_net.h"
#include "tinytensor_tensor.h"

#include <iostream>
#include <sys/time.h>
#define HAVE_NET (1)
#define PRINT_TIMING (0)



#if HAVE_NET
#include "unit-test/data/model_may31_small_sigm.c"
#endif 

#define MIN_PROB_TO_USE_FOR_SUM (40)
#define OUTBUF_SUM_LEN (2)
#define NET_RUN_PERIOD (20)
#define MIN_FEAT (-40)
#define FEAT_OFFSET (80)
#define MIN_FEAT_COUNT (200)
#define SAMPLE_RATE  (16000)
#define THRESHOLD (150)
#define FRAMES_PER_BUFFER (128)
#define NUM_SECONDS     (3600)
#define NUM_CHANNELS    (1)
/* #define DITHER_FLAG     (paDitherOff) */
#define DITHER_FLAG     (0) /**/

/* Select sample format. */
#define PA_SAMPLE_TYPE  paInt16
typedef int16_t SAMPLE;
#define SAMPLE_SILENCE  (0)
#define PRINTF_S_FORMAT "%d"

typedef struct
{
    int          frameIndex;  /* Index into sample array. */
    int          maxFrameIndex;
    SAMPLE      *recordedSamples;
}
paTestData;

/* This routine will be called by the PortAudio engine when audio is needed.
 ** It may be called at interrupt level on some machines so don't do anything
 ** that could mess up the system like calling malloc() or free().
 */
static struct timeval tp;
static int64_t ms;

static int recordCallback(const void *inputBuffer, void *outputBuffer,
                          unsigned long framesPerBuffer,
                          const PaStreamCallbackTimeInfo* timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void *userData )
{
    paTestData *data = (paTestData*)userData;
    const SAMPLE *rptr = (const SAMPLE*)inputBuffer;
    SAMPLE localbuf[FRAMES_PER_BUFFER * 2];
    SAMPLE *wptr = &localbuf[0];
    long framesToCalc;
    long i;
    int finished;
    unsigned long framesLeft = data->maxFrameIndex - data->frameIndex;
    
    (void) outputBuffer; /* Prevent unused variable warnings. */
    (void) timeInfo;
    (void) statusFlags;
    (void) userData;
    
    if( framesLeft < framesPerBuffer ) {
        framesToCalc = framesLeft;
        finished = paComplete;
    }
    else {
        framesToCalc = framesPerBuffer;
        finished = paContinue;
    }
    
    if( inputBuffer == NULL ) {
        return finished;
    }
    
    for( i=0; i<framesToCalc; i++ ) {
        *wptr++ = *rptr++;  /* left */
        //if( NUM_CHANNELS == 2 ) *wptr++ = *rptr++;  /* right */
    }
    
    tinytensor_features_add_samples(localbuf, framesToCalc);
    
#if PRINT_TIMING
    gettimeofday(&tp, NULL);
    int64_t ms2 = tp.tv_sec * 1000 + tp.tv_usec / 1000; //get current timestamp in milliseconds
    
    std::cout << (float)(ms2 - ms) << std::endl;
#endif
    
    data->frameIndex += framesToCalc;
    return finished;
}

#define NUM_TIME_ELEMENTS (199)
typedef struct {
    ConstSequentialNetwork_t net;
    int8_t buf[NUM_MEL_BINS][NUM_TIME_ELEMENTS];
    uint32_t bufidx;
} FeatsCallbackContext;

static void feats_callback(void * context, int8_t * feats) {
    FeatsCallbackContext * p = (FeatsCallbackContext *) context;
    static uint32_t counter = 0;
    static ImagePixel_t outbuf[OUTBUF_SUM_LEN] = {0};
    static uint32_t ioutbuf = 0;
    //desire to have the dims as 40 x 199
    //data comes in as 40 x 1 vectors, soo
    
    const static uint32_t dims[4] = {1,1,NUM_MEL_BINS,NUM_TIME_ELEMENTS};
    int32_t temp32;
    //get feats

    for (uint32_t i = 0; i < NUM_MEL_BINS; i++) {
        p->buf[i][p->bufidx] = feats[i] + FEAT_OFFSET;
    }
    
    
    if (++(p->bufidx) >= NUM_TIME_ELEMENTS) {
        p->bufidx = 0;
    }
    
    
    if (++counter % NET_RUN_PERIOD) {
        return;
    }
    
    if (counter < NUM_TIME_ELEMENTS) {
        return;
    }
    
#if HAVE_NET
    
    ImageTensor_t * tensor_in = tinytensor_create_new_image_tensor(dims);
    
    ImagePixel_t * px = tensor_in->x;
    for (uint32_t i = 0; i < NUM_MEL_BINS; i++ ) {
        uint32_t bufidx = p->bufidx;
        
        for (uint32_t t = 0; t < NUM_TIME_ELEMENTS; t++) {
            *px = p->buf[i][bufidx];
            
            if (++bufidx >= NUM_TIME_ELEMENTS) {
                bufidx = 0;
            }
            
            px++;
        }
    }
    
    bool okay_to_run = false;
    uint32_t feat_count = 0;
    const uint32_t len = tensor_in->dims[0]*tensor_in->dims[1]*tensor_in->dims[2]*tensor_in->dims[3];
    for (uint32_t i = 0; i < len; i++) {
        if (tensor_in->x[i] > MIN_FEAT + FEAT_OFFSET) {
            feat_count++;
        }
    }
    
    if (feat_count > MIN_FEAT_COUNT) {
        okay_to_run = true;
    }
    
    if (!okay_to_run) {
        memset(outbuf,0,sizeof(outbuf));
        tensor_in->delete_me(tensor_in);
        return;
    }

    
    ImageTensor_t * tensor_out = eval_net(&(p->net),tensor_in);

    outbuf[ioutbuf++ % OUTBUF_SUM_LEN] = tensor_out->x[1];
    
    int outsum = 0;
    for (uint32_t i = 0; i < OUTBUF_SUM_LEN; i++) {
        if (outbuf[i] > MIN_PROB_TO_USE_FOR_SUM) {
            outsum += outbuf[i];
        }
    }

    

    printf("%4.2f,%4.2f,%d\n",tensor_out->x[0] / 128.0,tensor_out->x[1] / 128.0,outsum);
    tensor_out->delete_me(tensor_out);

    if (outsum > THRESHOLD) {
        printf("THRESHOLD!\n");
        putchar('\a');
    }

#endif
}




/*******************************************************************/
int main(void) {
    FeatsCallbackContext featsContext;
    memset(&featsContext,0,sizeof(featsContext));

    gettimeofday(&tp, NULL);
    ms = tp.tv_sec * 1000 + tp.tv_usec / 1000; //get current timestamp in milliseconds

#if HAVE_NET
    featsContext.net = initialize_network();
#endif

    
    PaStreamParameters  inputParameters;
    PaStream*           stream;
    PaError             err = paNoError;
    paTestData          data;
    int                 i;
    int                 totalFrames;
    int                 numSamples;
    int                 numBytes;
    SAMPLE              max, val;
    double              average;
    
    
    tinytensor_features_initialize(&featsContext, feats_callback);
    
    data.maxFrameIndex = totalFrames = NUM_SECONDS * SAMPLE_RATE; /* Record for a few seconds. */
    data.frameIndex = 0;
    numSamples = totalFrames * NUM_CHANNELS;
    numBytes = numSamples * sizeof(SAMPLE);
    data.recordedSamples = (SAMPLE *) malloc( numBytes ); /* From now on, recordedSamples is initialised. */
    if( data.recordedSamples == NULL )
    {
        printf("Could not allocate record array.\n");
        goto done;
    }
    for( i=0; i<numSamples; i++ ) data.recordedSamples[i] = 0;
    
    err = Pa_Initialize();
    if( err != paNoError ) goto done;
    
    inputParameters.device = Pa_GetDefaultInputDevice(); /* default input device */
    if (inputParameters.device == paNoDevice) {
        fprintf(stderr,"Error: No default input device.\n");
        goto done;
    }
    inputParameters.channelCount = 1;                    /* mono input */
    inputParameters.sampleFormat = PA_SAMPLE_TYPE;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo( inputParameters.device )->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;
    
    /* Record some audio. -------------------------------------------- */
    err = Pa_OpenStream(&stream,
                        &inputParameters,
                        NULL,                  /* &outputParameters, */
                        SAMPLE_RATE,
                        FRAMES_PER_BUFFER,
                        paClipOff,      /* we won't output out of range samples so don't bother clipping them */
                        recordCallback,
                        &data );
    if( err != paNoError ) goto done;
    
    err = Pa_StartStream( stream );
    if( err != paNoError ) goto done;
    printf("\n=== Now recording!! Please speak into the microphone. ===\n"); fflush(stdout);
    
    while( ( err = Pa_IsStreamActive( stream ) ) == 1 )
    {
        Pa_Sleep(1000);
       // printf("index = %d\n", data.frameIndex ); fflush(stdout);
    }
    if( err < 0 ) goto done;
    
    err = Pa_CloseStream( stream );
    if( err != paNoError ) goto done;
    
    /* Measure maximum peak amplitude. */
    max = 0;
    average = 0.0;
    for( i=0; i<numSamples; i++ )
    {
        val = data.recordedSamples[i];
        if( val < 0 ) val = -val; /* ABS */
        if( val > max )
        {
            max = val;
        }
        average += val;
    }
    
    average = average / (double)numSamples;
    
    printf("sample max amplitude = "PRINTF_S_FORMAT"\n", max );
    printf("sample average = %lf\n", average );
    

    
done:
    Pa_Terminate();
    if( data.recordedSamples )       /* Sure it is NULL or valid. */
        free( data.recordedSamples );
    if( err != paNoError )
    {
        fprintf( stderr, "An error occured while using the portaudio stream\n" );
        fprintf( stderr, "Error number: %d\n", err );
        fprintf( stderr, "Error message: %s\n", Pa_GetErrorText( err ) );
        err = 1;          /* Always return 0 or 1, but no other return codes. */
    }
    
    tinytensor_features_deinitialize();
    
    return err;
}

