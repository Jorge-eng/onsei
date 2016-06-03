#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "portaudio.h"
#include "tinytensor_features.h"
#include "tinytensor_net.h"
#include "tinytensor_tensor.h"

#define HAVE_NET (0)

#if HAVE_NET
#include "net.c" 
#endif 

#define SAMPLE_RATE  (16000)
#define FRAMES_PER_BUFFER (256)
#define NUM_SECONDS     (5)
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
    
    data->frameIndex += framesToCalc;
    return finished;
}

#define NUM_TIME_ELEMENTS (199)
typedef struct {
    ConstSequentialNetwork_t net;
    int8_t buf[NUM_TIME_ELEMENTS][NUM_MEL_BINS];
    uint32_t bufidx;
} FeatsCallbackContext;

static void feats_callback(void * context, int8_t * feats) {
    FeatsCallbackContext * pcontext = (FeatsCallbackContext *) context;
    const static uint32_t dims[4] = {1,1,NUM_MEL_BINS,NUM_TIME_ELEMENTS};

    //copy out new feats
   // printf("%d\n",pcontext->bufidx);
    memcpy(&(pcontext->buf[pcontext->bufidx][0]),feats,NUM_MEL_BINS*sizeof(int8_t));
    
    if (++pcontext->bufidx >= NUM_TIME_ELEMENTS) {
        pcontext->bufidx = 0;
    }
    
    if (pcontext->bufidx % 20 != 0) {
        return;
    }
    
    
#if HAVE_NET

    Tensor_t * tensor_in = tinytensor_create_new_tensor(dims);
    
    int8_t (*inmat)[NUM_TIME_ELEMENTS]  = (int8_t (*)[NUM_TIME_ELEMENTS])tensor_in->x;
    
  
    
    for (uint32_t j = 0; j < NUM_MEL_BINS; j++) {
        uint32_t bufidx = pcontext->bufidx;

        for (uint32_t t = 0; t < NUM_TIME_ELEMENTS; t++) {
            inmat[j][t] = pcontext->buf[bufidx][j];
        }
        
        if (++bufidx >= NUM_TIME_ELEMENTS) {
            bufidx = 0;
        }
    }
    

    Tensor_t * tensor_out = eval_net(&(pcontext->net),tensor_in);
    printf("%3.1f,%3.1f\n",tensor_out->x[0] / 128.0,tensor_out->x[1] / 128.0);
    tensor_out->delete_me(tensor_out);

#endif
}




/*******************************************************************/
int main(void) {
    FeatsCallbackContext featsContext;
    memset(&featsContext,0,sizeof(featsContext));
    
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

