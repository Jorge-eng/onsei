const static Weight_t image2_weights[9] =
{121,40,9,
 65,51,27,
 46,50,38};
const static uint32_t image2_dims[4] = {1,1,3,3};
const static ConstTensor_t image2 = {&image2_weights[0],&image2_dims[0],0};

const static Weight_t image2m_weights[18] =
{0,0,0,
    0,0,0,
    0,0,0,
    
    121,40,9,
    65,51,27,
    46,50,38};


const static Weight_t image2m2_weights[18] =
{121,40,9,
    65,51,27,
    46,50,38,
    
    0,0,0,
    0,0,0,
    0,0,0};

const static uint32_t image2m_dims[4] = {1,2,3,3};
const static ConstTensor_t image2m = {&image2m_weights[0],&image2m_dims[0],0};
const static ConstTensor_t image2m2 = {&image2m2_weights[0],&image2m_dims[0],0};


