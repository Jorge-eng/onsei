#include "tinytensor_lstm_layer.h"
#include "tinytensor_memory.h"
#include "tinytensor_math.h"
#include <assert.h>
static void get_output_size(const void * context,uint32_t * dims) {
    const LstmLayer_t * lstm_layer = (const LstmLayer_t *) context;

    MEMCPY(dims,lstm_layer->output_dims,TENSOR_DIM*sizeof(uint32_t));
    
}

//GATES SHOULD BE IN THIS ORDER
typedef enum {
    forgetgate = 0,
    forgetgate_i,
    cellgate,
    cellgate_i,
    cellinput,
    cellinput_i,
    outputgate,
    outputgate_i,
    NUM_GATES
} Gates_t;

static void lstm_time_step_forwards(int32_t * cell_state,
                                    Weight_t * output,
                                    const Weight_t * inputs[NUM_GATES],
                                    const Weight_t * weights[NUM_GATES],
                                    const Weight_t * biases[NUM_GATES],
                                    const int8_t weights_scale[NUM_GATES],
                                    const int8_t biases_scale[NUM_GATES],
                                    const uint32_t vec_size,
                                    const uint32_t num_cells,
                                    const int8_t input_scale,
                                    SquashFunc_t output_activation) {
    
    
  
    uint32_t igate;
    uint32_t icell;
    uint32_t ivec;
    uint32_t i;
    int32_t accumulator32;
    int32_t temp32;
    int8_t temp8;
    int8_t tempscale;
    int32_t bias32;
    int8_t scale_diff;

    int8_t temp_scales[NUM_GATES];
    int8_t bias_scale_diffs[NUM_GATES];
    int32_t pre_activations[NUM_GATES];
    const Weight_t * weight_row_starts[NUM_GATES];
    const Weight_t * bias_row_starts[NUM_GATES];
    
    Weight_t activations[NUM_GATES/2];
    const static SquashFunc_t activation_funcs[NUM_GATES/2] = {tinytensor_sigmoid,tinytensor_sigmoid,tinytensor_tanh,tinytensor_sigmoid};

    for (i = 0; i < NUM_GATES; i++) {
        //for each matmul + bias, set up the scale differences between bias and weights
        bias_scale_diffs[i] = weights_scale[i] + input_scale - biases_scale[i];

        //set up row starts for all weights
        weight_row_starts[i] = weights[i];
        bias_row_starts[i] = bias_row_starts[i];
    }
    

    
    for (icell = 0; icell < num_cells; icell++) {

        for (igate = 0; igate < NUM_GATES; igate++) {
        
            accumulator32 = 0;
            const Weight_t * w = weight_row_starts[igate];
            const int8_t w_scale = weights_scale[igate];
            const Weight_t * input_vec = inputs[igate];

            for (ivec = 0; ivec < vec_size; ivec++) {
                
                //TODO optimize here
                accumulator32 += w[ivec] * input_vec[ivec];
                
            }
            
            scale_diff = bias_scale_diffs[igate];
            bias32 = (*(biases[igate]))  << QFIXEDPOINT; //to Q14 + QB FROM Q7 + QB
           
            //to Q14 + QW + QI
            if (scale_diff > 0) {
                bias32 <<= scale_diff;
            }
            else if (scale_diff < 0) {
                bias32 >>= -scale_diff;
            }
            
            accumulator32 += bias32;
            
            
            //FROM  Q14 + QW + QI
            //  TO  Q7 + QI
            accumulator32 >>= QFIXEDPOINT;
            accumulator32 >>= w_scale;

            pre_activations[igate] = accumulator32;
            
            //update indices
            weight_row_starts[igate] += vec_size;
            bias_row_starts[igate]++;
            
        }

        //perform activations, knowing that items 2N and 2N+1 are together
        //again this comes from the ordering of the enums
        for (igate = 0; igate < NUM_GATES/2; igate++) {
            //activate!
            temp32 = pre_activations[2*igate] + pre_activations[2*igate + 1];
            
            activation_funcs[igate](&activations[igate],&temp_scales[igate],temp32,input_scale);
        }

        
        
        //now that we have our activations, process the gates
        
        
        //forget the cell -- so multiply it by the gate
        //this is Q7 * Q14
        temp32 = activations[forgetgate/2] * cell_state[icell];
        temp32 >>= QFIXEDPOINT; //Q14 now
        
        //add cell input * gate (Q7 * Q7) = Q14
        temp32 += activations[cellgate/2] * activations[cellinput/2];
        cell_state[icell] = temp32; //save result
        
        //output hidden state
        //take cell output, apply activation function to it
        //To Q7
        temp32 >>= QFIXEDPOINT;
        
        //squash
        output_activation(&temp8,&tempscale,temp32,0);
        assert(tempscale == 0);
        //Q7 x Q7  = Q14
        temp32 = temp8 * activations[outputgate/2];
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
    Weight_t output1[LSTM_MAX_HIDDEN_UNITS];
    Weight_t output2[LSTM_MAX_HIDDEN_UNITS];

    Weight_t * outputs[2] = {&output1[0],&output2[1]};

    const uint32_t time_length = in->dims[2];
    const uint32_t num_inputs = in->dims[3];
    const uint32_t num_hidden_units = lstm_layer->output_dims[3];
    const uint32_t total_input_vec_len = num_hidden_units + num_inputs;
    uint32_t t;
    uint8_t i;
    uint32_t icell;
    uint8_t current_gate = 0;
    Weight_t * out_row = out->x;
    const Weight_t * in_row = in->x;
    const Weight_t * inputs[NUM_GATES];
    
    //arrange weights `n stuff
    const Weight_t * weights[NUM_GATES] = {
        lstm_layer->weights_forget_gate->x,
        lstm_layer->weights_forget_gate_i->x,
        lstm_layer->weights_cell_gate->x,
        lstm_layer->weights_cell_gate_i->x,
        lstm_layer->weights_cell_input->x,
        lstm_layer->weights_cell_input_i->x,
        lstm_layer->weights_output_gate->x,
        lstm_layer->weights_output_gate_i->x};
    
    const int8_t weight_scales[NUM_GATES] = {
        lstm_layer->weights_forget_gate->scale,
        lstm_layer->weights_forget_gate_i->scale,
        lstm_layer->weights_cell_gate->scale,
        lstm_layer->weights_cell_gate_i->scale,
        lstm_layer->weights_cell_input->scale,
        lstm_layer->weights_cell_input_i->scale,
        lstm_layer->weights_output_gate->scale,
        lstm_layer->weights_output_gate_i->scale};

    
    const Weight_t * biases[NUM_GATES] = {
        lstm_layer->biases_forget_gate->x,
        lstm_layer->biases_forget_gate->x,
        lstm_layer->biases_cell_gate->x,
        lstm_layer->biases_cell_gate->x,
        lstm_layer->biases_cell_input->x,
        lstm_layer->biases_cell_input->x,
        lstm_layer->biases_output_gate->x,
        lstm_layer->biases_output_gate->x};
    
    const int8_t bias_scales[NUM_GATES] = {
        lstm_layer->biases_forget_gate->scale,
        lstm_layer->biases_forget_gate->scale,
        lstm_layer->biases_cell_gate->scale,
        lstm_layer->biases_cell_gate->scale,
        lstm_layer->biases_cell_input->scale,
        lstm_layer->biases_cell_input->scale,
        lstm_layer->biases_output_gate->scale,
        lstm_layer->biases_output_gate->scale};
    
    
    MEMSET(cell_state,0,sizeof(cell_state));
    MEMSET(output1,0,sizeof(output1));
    
    for (t = 0; t < time_length; t++) {
        Weight_t * prev_output = outputs[current_gate & 0x01];
        Weight_t * output = outputs[ (current_gate + 1) & 0x01];
        
        //setup inputs, knowing that the gates
        //enums are set up as "from prev hidden, from input, from prev hidden ..."
        for (i = 0; i < NUM_GATES; i++) {
            if (i & 0x01) {
                inputs[i] = in_row;
            }
            else {
                inputs[i] = prev_output;
            }
        }
        
        lstm_time_step_forwards(cell_state,output,inputs,weights,biases,weight_scales,bias_scales,total_input_vec_len,num_hidden_units,in->scale,lstm_layer->output_activation);
     
        
        for (icell = 0; icell < num_hidden_units; icell++) {
            out_row[icell] = output[icell];
        }
        
        current_gate++;
        out_row += num_hidden_units;
        in_row += num_inputs;
    }
    
    

}

ConstLayer_t tinytensor_create_lstm_layer(const LstmLayer_t * static_def) {
    ConstLayer_t layer = {eval,get_output_size,static_def};
    return layer;
}
