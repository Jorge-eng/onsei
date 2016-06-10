#include "tinytensor_conv_layer.h"
#include "tinytensor_maxpool_layer.h"
#include "tinytensor_fullyconnected_layer.h"
#include "tinytensor_math.h"
#include "tinytensor_net.h"
const static Weight_t Convolution2D_01_conv_x[48] = {-41,-46,-34,57,-71,-21,71,25,-6,70,2,-68,72,17,-62,-47,-44,45,-47,5,-16,68,55,-1,26,-46,63,4,74,5,18,37,57,-71,-75,-42,-57,33,38,-18,70,27,-3,-26,43,69,-45,61};
const static uint32_t Convolution2D_01_conv_dims[4] = {4,2,2,3};
const static ConstTensor_t Convolution2D_01_conv = {&Convolution2D_01_conv_x[0],&Convolution2D_01_conv_dims[0]};

const static Weight_t Convolution2D_01_bias_x[4] = {0,0,0,0};
const static uint32_t Convolution2D_01_bias_dims[4] = {4,1,1,1};
const static ConstTensor_t Convolution2D_01_bias = {&Convolution2D_01_bias_x[0],&Convolution2D_01_bias_dims[0]};

const static uint32_t Convolution2D_01_input_dims[4] = {1,2,4,5};
const static uint32_t Convolution2D_01_output_dims[4] = {1,4,3,3};
const static uint32_t Convolution2D_01_max_pool_dims[2] = {1,1};
const static ConvLayer2D_t convolution2d_01 = {&Convolution2D_01_conv,&Convolution2D_01_bias,Convolution2D_01_output_dims,Convolution2D_01_input_dims,Convolution2D_01_max_pool_dims,TOFIX(0.000000),tinytensor_linear};



const static Weight_t Convolution2D_02_conv_x[16] = {-6,-72,-71,-19,77,86,8,52,91,26,-87,-46,56,-39,-72,-67};
const static uint32_t Convolution2D_02_conv_dims[4] = {2,4,1,2};
const static ConstTensor_t Convolution2D_02_conv = {&Convolution2D_02_conv_x[0],&Convolution2D_02_conv_dims[0]};

const static Weight_t Convolution2D_02_bias_x[2] = {0,0};
const static uint32_t Convolution2D_02_bias_dims[4] = {2,1,1,1};
const static ConstTensor_t Convolution2D_02_bias = {&Convolution2D_02_bias_x[0],&Convolution2D_02_bias_dims[0]};

const static uint32_t Convolution2D_02_input_dims[4] = {1,4,3,3};
const static uint32_t Convolution2D_02_output_dims[4] = {1,2,3,2};
const static uint32_t Convolution2D_02_max_pool_dims[2] = {1,1};


const static ConvLayer2D_t convolution2d_02 = {&Convolution2D_02_conv,&Convolution2D_02_bias,Convolution2D_02_output_dims,Convolution2D_02_input_dims,Convolution2D_02_max_pool_dims,TOFIX(0.000000),tinytensor_linear};






static ConstLayer_t _layers[2];
static ConstSequentialNetwork_t net = {&_layers[0],2};
static ConstSequentialNetwork_t initialize_network(void) {

  _layers[0] = tinytensor_create_conv_layer(&convolution2d_01);
  _layers[1] = tinytensor_create_conv_layer(&convolution2d_02);
  return net;

}