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

#include "hellomath/sha1.h"

#include "model_def.c"

#define FEATS_SAVE_THRESOLD_OKAY_SENSE (TOFIX(-0.1f))
#define NUM_TIMES_TO_HAVE_EXCEEDED_THRESHOLD (1)

std::string base64_decode(std::string const& encoded_string);
std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len);
bool Encode32(unsigned char* in, int inLen, unsigned char* out);

using namespace std;
using namespace rapidjson;



static void do_something_with_the_text(const std::string & str,const std::string desired_keyword,const size_t counter) {
    
    //string to json
    Document d;
    d.Parse(str.c_str());
    
    //get data from json
    std::string decoded = base64_decode(d["payload"].GetString());
    std::string id = d["id"].GetString();
    
    const size_t pos = id.find("+");
    if (pos == std::string::npos) {
        std::cerr << "failed to find + in id in iter" << counter << std::endl;
        return;
    }

    const std::string keyword = id.substr(pos + 1,id.size() - pos + 1);
    
    if (keyword != desired_keyword) {
        std::cerr << "undesired keyword" << keyword << " at iter " << counter << std::endl;
        return;
    }
    
    //setup neural net stuff
    ConstSequentialNetwork_t net = initialize_network();
    SequentialNetworkStates_t state;
    static int count = 0;
    tinytensor_allocate_states(&state, &net);

    const static uint32_t dims[4] = {1,1,1,NUM_MEL_BINS};
    
    Tensor_t * tensor_in = tinytensor_create_new_tensor(dims);
    
    //prepare data in 16 bit format
    const int8_t * p = reinterpret_cast<const int8_t *>(decoded.data());
    
    std::vector<int16_t> feats;
    feats.reserve(decoded.size());
    
    //since the data was uploaded from a circular buffer, the end of the buffer is aligned with the vectors
    //so the remainder is in the front
    int istart = decoded.size() % NUM_MEL_BINS;

    for (int i = istart; i < decoded.size(); i++) {
        int16_t value = p[i];
        value <<= 5; //from Q7 to Q12
        feats.push_back(value);
    }
    
    const int N = feats.size() / NUM_MEL_BINS;
    bool is_saved = false;
    int threshold_count = 0;
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
        
        if (tensor_out->x[1] > FEATS_SAVE_THRESOLD_OKAY_SENSE) {
            threshold_count++;
            
            if (threshold_count == NUM_TIMES_TO_HAVE_EXCEEDED_THRESHOLD) {
                is_saved = true;
            }
        }
        else {
            threshold_count = 0;
        }
        
    }
    
    if (!is_saved) {
        static int count = 1;
        std::cerr << "FAILED AT ITERATION " << count << std::endl;
        count++;
        int foo = 3;
        foo++;
    }
    
    if (is_saved) {
        //SHA the data so we can get a unique identifier for this set of features
        SHA1Context c;
        memset(&c,0,sizeof(c));
        
        const uint32_t data_size_bytes = feats.size()*sizeof(feats[0]);
        
        uint8_t message_digest[SHA1HashSize];
        memset(message_digest,0,sizeof(message_digest));
        
        SHA1Reset(&c);
        SHA1Input(&c,reinterpret_cast<uint8_t *>(feats.data()),data_size_bytes);
        SHA1Result(&c, message_digest);

        //hex conversion of binary data
        char idbuf[1 << 10] = {0};

        for (int ibyte = 0; ibyte < sizeof(message_digest); ibyte++) {
            char temp[10];
            snprintf(temp,sizeof(temp),"%02x",message_digest[ibyte]);
            idbuf[2*ibyte] = temp[0];
            idbuf[2*ibyte + 1] = temp[1];

        }

        int istart = 2*sizeof(message_digest);
        idbuf[istart] = '.';
        idbuf[istart + 1] = 'b';
        idbuf[istart + 2] = 'i';
        idbuf[istart + 3] = 'n';

        std::string id;// = base64_encode(message_digest,sizeof(message_digest)) + ".bin";
        id.assign(idbuf,2*sizeof(message_digest) + 5);
        
        
        std::ofstream fout(id.c_str());
        
        if (!fout.is_open()) {
            std::cerr << "FILE OUT " << id << " failed to open file" << std::endl;
        }
        
        fout.write(reinterpret_cast<char *>(feats.data()), data_size_bytes);
        fout.close();
        
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
    /*
    file.seekg(0,std::ios::end);
    size_t pos = file.tellg();
    file.seekg(0,std::ios::beg);
    
    
    char buf[pos];
    std::string text;

    text.resize(pos );
    
    file.read(buf,sizeof(buf));
    text.assign(buf,sizeof(buf));
    
    size_t prevpos = 0;
    */
    size_t counter = 0;
    std::string str;
    while (std::getline(file, str)) {
        do_something_with_the_text(str,argv[2],counter);
        counter++;
        
        if (counter % 100 == 0) {
            std::cerr << counter << std::endl;
        }
    }
    
    std::cerr << "FINISHED " << counter << " blobs" << std::endl;
    
    /*
    while(1) {
        pos = text.find("}",prevpos);

        if (pos == std::string::npos) {
            break;
        }
        
        std::string objecttext = text.substr(prevpos,pos-prevpos + 1);
        
        do_something_with_the_text(objecttext,argv[2]);
        
        prevpos = pos + 1;
    }
     */
    
    
    
    
    
    
    return 0;
}
