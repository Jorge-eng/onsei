#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <functional>
#include "tinytensor_features.h"
#include "tinytensor_net.h"
#include "tinytensor_tensor.h"
#include "tinytensor_math.h"

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

#include "model_def.c"

std::string base64_decode(std::string const& encoded_string);


using namespace std;
using namespace rapidjson;


static void do_something_with_the_text(const std::string & str,const std::string desired_keyword) {
    Document d;
    d.Parse(str.c_str());
    
    std::string decoded = base64_decode(d["payload"].GetString());
    std::string id = d["id"].GetString();
    
    const size_t pos = id.find("+");
    if (pos == std::string::npos) {
        return;
    }

    const std::string keyword = id.substr(pos + 1,id.size() - pos + 1);
    
    if (keyword != desired_keyword) {
        return;
    }
    
    ConstSequentialNetwork_t net = initialize_network();
    SequentialNetworkStates_t state;
    static int count = 0;
    tinytensor_allocate_states(&state, &net);

    const static uint32_t dims[4] = {1,1,1,NUM_MEL_BINS};
    
    Tensor_t * tensor_in = tinytensor_create_new_tensor(dims);
    
    
    const int8_t * p = reinterpret_cast<const int8_t *>(decoded.data());
    
    std::vector<int16_t> feats;
    feats.reserve(decoded.size());
    
    int istart = decoded.size() % NUM_MEL_BINS;

    for (int i = istart; i < decoded.size(); i++) {
        int16_t value = p[i];
        value <<= 5; //from Q7 to Q12
        feats.push_back(value);
    }
    
    const int N = feats.size() / NUM_MEL_BINS;
    
    for (int i = 0; i < N; i++) {
        
        const int16_t * featvec = &feats[i*NUM_MEL_BINS];
        memcpy(tensor_in->x,featvec,NUM_MEL_BINS * sizeof(Weight_t));
        /*
        for (int j = 0; j < 40; j++) {
            if (j != 0) std::cout << ",";
            std::cout << featvec[j];
            
        }
        std::cout << std::endl;
        */
        
        Tensor_t * tensor_out = tinytensor_eval_stateful_net(&net,&state,tensor_in,NET_FLAG_LSTM_DAMPING);
        
        std::cout << count;
        
        for (int j = 0; j < tensor_out->dims[3]; j++) {
            std::cout << "," << tensor_out->x[j];
        }
        std::cout << std::endl;
        
    }
    
    count++;
        
    tinytensor_free_states(&state, &net);
    


}


int main(int argc, char * argv[]) {
    
    if (argc < 3) {
        std::cout << "need to have input file specified and target keyord (okay_sense)" << std::endl;
        return 0;
    }
    

    std::ifstream file(argv[1]);
    
    if (!file.is_open()) {
        std::cout << "unable to open file" << std::endl;
        return 0;
    }
    
    
    
    
    
    
    //Value::ConstValueIterator
    
    file.seekg(0,std::ios::end);
    size_t pos = file.tellg();
    file.seekg(0,std::ios::beg);
    
    
    char buf[pos];
    std::string text;

    text.resize(pos );
    
    file.read(buf,sizeof(buf));
    text.assign(buf,sizeof(buf));
    
    size_t prevpos = 0;
    
    while(1) {
        pos = text.find("}",prevpos);

        if (pos == std::string::npos) {
            break;
        }
        
        std::string objecttext = text.substr(prevpos,pos-prevpos + 1);
        
        do_something_with_the_text(objecttext,argv[2]);
        
        prevpos = pos + 1;
    }
    
    
    
    
    
    
    return 0;
}
