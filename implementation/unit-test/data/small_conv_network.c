#include "tinytensor_conv_layer.h"
#include "tinytensor_maxpool_layer.h"
#include "tinytensor_fullyconnected_layer.h"
#include "tinytensor_math.h"
#include "tinytensor_net.h"
const static Weight_t Convolution2D_01_conv_x[16] = {-41,-78,50,7,31,-89,84,15,87,-7,84,39,87,49,-59,-2};
const static uint32_t Convolution2D_01_conv_dims[4] = {2,2,2,2};
const static ConstTensor_t Convolution2D_01_conv = {&Convolution2D_01_conv_x[0],&Convolution2D_01_conv_dims[0]};

const static Weight_t Convolution2D_01_bias_x[2] = {0,0};
const static uint32_t Convolution2D_01_bias_dims[4] = {2,1,1,1};
const static ConstTensor_t Convolution2D_01_bias = {&Convolution2D_01_bias_x[0],&Convolution2D_01_bias_dims[0]};

const static uint32_t Convolution2D_01_input_dims[4] = {1,2,2,3};
const static uint32_t Convolution2D_01_output_dims[4] = {1,2,1,2};
const static ConvLayer2D_t convolution2d_01 = {&Convolution2D_01_conv,&Convolution2D_01_bias,Convolution2D_01_output_dims,Convolution2D_01_input_dims,TOFIX(0.000000),tinytensor_linear};






static ConstLayer_t _layers[1];
static ConstSequentialNetwork_t net = {&_layers[0],1};
static ConstSequentialNetwork_t initialize_network(void) {

  _layers[0] = tinytensor_create_conv_layer(&convolution2d_01);
  return net;

}