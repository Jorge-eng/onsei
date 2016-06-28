#include "tinytensor_lstm_layer.h"
#include "tinytensor_memory.h"
#include "tinytensor_math.h"
#include <assert.h>
static void get_output_size(const void * context,uint32_t * dims) {
    const LstmLayer_t * lstm_layer = (const LstmLayer_t *) context;

    MEMCPY(dims,lstm_layer->output_dims,TENSOR_DIM*sizeof(uint32_t));
    
}


/*
 i = self.inner_activation(z0)
 f = self.inner_activation(z1)
 c = f * c_tm1 + i * self.activation(z2)
 o = self.inner_activation(z3)
 
 h = o * self.activation(c)
 return h, [h, c]
 
 */
//

//GATES SHOULD BE IN THIS ORDER
typedef enum {
    inputgate,
    forgetgate,
    celloutput,
    outputgate,
    NUM_GATES
} Gates_t;

static void lstm_time_step_forwards(int32_t * cell_state,
                                    Weight_t * output,
                                    const Weight_t * input_vec,
                                    const Weight_t * weights[NUM_GATES],
                                    const Weight_t * biases[NUM_GATES],
                                    const int8_t weights_scale[NUM_GATES],
                                    const int8_t biases_scale[NUM_GATES],
                                    const uint32_t num_cells,
                                    const uint32_t num_inputs,
                                    const int8_t input_scale,
                                    SquashFunc_t output_activation) {
    
    
  
    uint32_t igate;
    uint32_t icell;
    uint32_t ivec;
    uint32_t i;
    int32_t accumulator32;
    int32_t temp32;
    int8_t temp8;
    Weight_t h;
    int8_t tempscale;
    int32_t bias32;

    const Weight_t * weight_row_starts[NUM_GATES];
    const Weight_t * bias_row_starts[NUM_GATES];
    const uint32_t total_len = num_cells + num_inputs;
    int32_t pre_activations[NUM_GATES];
    Weight_t activations[NUM_GATES];
    
    const static SquashFunc_t internal_activation = tinytensor_hard_sigmoid;

    for (i = 0; i < NUM_GATES; i++) {
        //set up row starts for all weights
        weight_row_starts[i] = weights[i];
        bias_row_starts[i] = biases[i];

    }
    
    
    for (icell = 0; icell < num_cells; icell++) {
//        printf("cell=%d\n",icell);

        for (igate = 0; igate < NUM_GATES; igate++) {

            accumulator32 = 0;
            const Weight_t * w = weight_row_starts[igate];
            const Weight_t * b = bias_row_starts[igate];
            const int8_t w_scale = weights_scale[igate];
            const int8_t b_scale = biases_scale[igate];
            
//            printf("gate=%d\n",igate);
            for (ivec = 0; ivec < total_len; ivec++) {
//                if (ivec != 0) {
//                    printf("    ");
//                }
                //TODO optimize here
//                printf("%f,%f",w[ivec]/128.0,input_vec[ivec]/128.0);
                accumulator32 += w[ivec] * input_vec[ivec];
                
            }
//            printf("\n");
            
            bias32 = b[igate]  << QFIXEDPOINT; //to Q14 + QB FROM Q7 + QB

            temp8 = b_scale - input_scale - w_scale;
            
            if (temp8 > 0) {
                bias32 >>= temp8;
            }
            else if (temp8 < 0){
                bias32 <<= -temp8;
            }
            
            accumulator32 += bias32;
            
            if (w_scale > 0) {
                accumulator32 >>= w_scale;
            }
            else if (w_scale < 0) {
                accumulator32 <<= -w_scale;
            }
            
            //Q14 + QI
            pre_activations[igate] = accumulator32 >> QFIXEDPOINT;
            
            //update indices
            weight_row_starts[igate] += total_len;
            bias_row_starts[igate]++;

        }
        

        //now that we have our activations, process the gates
        internal_activation(&activations[forgetgate],&tempscale,pre_activations[forgetgate],input_scale);
        internal_activation(&activations[inputgate],&tempscale,pre_activations[inputgate],input_scale);
        internal_activation(&activations[outputgate],&tempscale,pre_activations[outputgate],input_scale);
        output_activation(&activations[celloutput],&tempscale,pre_activations[celloutput],input_scale);

        assert(tempscale == 0);

        
        
        /*
         i = self.inner_activation(z0)
         f = self.inner_activation(z1)
         c = f * c_tm1 + i * self.activation(z2)
         o = self.inner_activation(z3)
         
         h = o * self.activation(c)
         return h, [h, c]
         
         */
        
        
        //apply forget gate to prev cell state
        temp32 = activations[forgetgate] * cell_state[icell]; //Q7 x Q14 ---> Q21
        temp32 >>= QFIXEDPOINT; //Q14
        
        //and add gated cell input
        temp32 += (int32_t)activations[inputgate] * (int32_t)activations[celloutput]; //Q7 * Q7 --->Q14
        
        //"c"
        cell_state[icell] = temp32; //save result
        
        ///////////////////
        //output hidden state
        //take cell output "c", apply activation function to it
        
        //To Q7
        temp32 >>= QFIXEDPOINT;
        
        output_activation(&h,&tempscale,temp32,0);
        assert(tempscale == 0);
        //Q7 x Q7  = Q14
        temp32 = (int16_t)h * (int16_t)activations[outputgate];
        temp32 >>= QFIXEDPOINT;
        
        if (temp32 > MAX_WEIGHT) {
            temp32 = MAX_WEIGHT;
        }
        
        if (temp32 < -MAX_WEIGHT) {
            temp32 = -MAX_WEIGHT;
        }
        
        
        output[icell] = (Weight_t)temp32;
    }


}



/*
 
 The general idea here is that there is a bunch of vector inputs
 The dims are 1 x 1 x T x N
 // T is length of vector sequence
 // N is vector length
 //
 // the output is the number of hidden units in this layer
 // it's dims are [1 x 1 x T x M]
 //
 // so total input is [1 x 1 x T x (M + N)]
 // and output is [1 x 1 x T x M]
 //
 // cell state is intialized to zero
 
 */
static void eval(const void * context,Tensor_t * out,const Tensor_t * in) {
    const LstmLayer_t * lstm_layer = (const LstmLayer_t *) context;
    
    int32_t cell_state[LSTM_MAX_HIDDEN_UNITS];
    Weight_t input[LSTM_MAX_HIDDEN_UNITS];
    Weight_t output[LSTM_MAX_HIDDEN_UNITS];

    const uint32_t time_length = in->dims[2];
    const uint32_t num_inputs = in->dims[3];
    const uint32_t num_hidden_units = lstm_layer->output_dims[3];
    uint32_t t;
    uint32_t i;
    uint8_t current_gate = 0;
    Weight_t * out_row = out->x;
    const Weight_t * in_row = in->x;

    
    //arrange weights `n stuff
    const Weight_t * weights[NUM_GATES] = {
        lstm_layer->weights_input_gate->x,
        lstm_layer->weights_forget_gate->x,
        lstm_layer->weights_cell->x,
        lstm_layer->weights_output_gate->x};
    
    const int8_t weight_scales[NUM_GATES] = {
        lstm_layer->weights_input_gate->scale,
        lstm_layer->weights_forget_gate->scale,
        lstm_layer->weights_cell->scale,
        lstm_layer->weights_output_gate->scale};

    
    const Weight_t * biases[NUM_GATES] = {
        lstm_layer->biases_input_gate->x,
        lstm_layer->biases_forget_gate->x,
        lstm_layer->biases_cell->x,
        lstm_layer->biases_output_gate->x};
    
    const int8_t bias_scales[NUM_GATES] = {
        lstm_layer->biases_input_gate->scale,
        lstm_layer->biases_forget_gate->scale,
        lstm_layer->biases_cell->scale,
        lstm_layer->biases_output_gate->scale};
    
    
    MEMSET(cell_state,0,sizeof(cell_state));
    MEMSET(input,0,sizeof(input));
    MEMSET(output,0,sizeof(output));


    for (t = 0; t < time_length; t++) {

        MEMCPY(input,in_row,num_inputs*sizeof(Weight_t));
        MEMCPY(input + num_inputs,output,num_hidden_units*sizeof(Weight_t));
        
        lstm_time_step_forwards(cell_state,
                                output,
                                input,
                                weights,
                                biases,
                                weight_scales,
                                bias_scales,
                                num_hidden_units,
                                num_inputs,
                                in->scale,
                                lstm_layer->output_activation);
     
        for (i = 0; i < num_hidden_units; i++) {
            if (i != 0) printf(",");
            printf("%d",cell_state[i] >> QFIXEDPOINT);
        }
        printf("\n");
        
        MEMCPY(out_row,output,num_hidden_units * sizeof(Weight_t));

        current_gate++;
        out_row += num_hidden_units;
        in_row += num_inputs;
    }
    
    printf("\n");
    printf("\n");
    printf("\n");


}

ConstLayer_t tinytensor_create_lstm_layer(const LstmLayer_t * static_def) {
    ConstLayer_t layer = {eval,get_output_size,static_def};
    return layer;
}
