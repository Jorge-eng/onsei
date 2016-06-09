const static Weight_t weights2_weights[12] =
  {-87,-94,94,
    -55,99,17,
    
      82,-21,-73,
      -9,-33,51};
const static uint32_t weights2_dims[4] = {2,1,2,3};
const static ConstTensor_t weights2 = {&weights2_weights[0],&weights2_dims[0],0};


const static Weight_t weights2m_weights[24] =
{-87,-94,94,
    -55,99,17,

    -87,-94,94,
    -55,99,17,
    
    82,-21,-73,
    -9,-33,51,
    
    82,-21,-73,
    -9,-33,51,};
const static uint32_t weights2m_dims[4] = {2,2,2,3};
const static ConstTensor_t weights2m = {&weights2m_weights[0],&weights2m_dims[0],0};