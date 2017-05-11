#include "radarmessageprotobuf.h"
#include "radar_messages.pb.h"
#include <cstdlib>
#include "pb_encode.h"
#include "proto_utils.h"
#include "log.h"

uint8_t * RadarMessageProtobuf::serialize_protobuf(const RadarMessage_t & message, size_t & size) {
    size_t max_bytes = message.vec.size() * sizeof(float) + message.id.size() + 4096;
    uint8_t * bytes = (uint8_t *) malloc(max_bytes);
    memset(bytes,0,max_bytes);

    pb_ostream_t ostream = pb_ostream_from_buffer(bytes, max_bytes);

    hello_FeatureVector vecstruct;
    memset(&vecstruct,0,sizeof(vecstruct));
    
    vecstruct.id.funcs.encode = encode_string_fields;
    vecstruct.id.arg = (void *)message.id.c_str();
    
    BufferInfo_t floatfeatsinfo = {(uint8_t *)message.vec.data(),message.vec.size()*sizeof(float),0};
    vecstruct.floatfeats.funcs.encode = encode_repeated_floats;
    vecstruct.floatfeats.arg = &floatfeatsinfo;
    
    vecstruct.sequence_number = message.sequence_number;
    vecstruct.device_id.funcs.encode = encode_string_fields;
    vecstruct.device_id.arg = (void *)message.device_id.c_str();
    
    vecstruct.has_sequence_number = true;
    vecstruct.sequence_number = message.sequence_number;

    if (!pb_encode(&ostream,hello_FeatureVector_fields,&vecstruct)) {
        LOG("failed to encode radar message %s %s",message.device_id.c_str(),message.id.c_str());
        free(bytes);
        return NULL;
    }
    
    size = ostream.bytes_written;
    
    return bytes;
}


bool RadarMessageProtobuf::deserialize_protobuf(const uint8_t * protobuf_bytes, const size_t protobuf_size, RadarMessage_t & message) {
    hello_FeatureVector featvec;
    
    //approximate
    float buf[protobuf_size/4]; //1024 range bins, should be enough
    
    
    pb_istream_t istream = pb_istream_from_buffer(protobuf_bytes, protobuf_size);
    
    memset(&featvec,0,sizeof(featvec));
    
    BufferInfo_t floatfeats_info = {reinterpret_cast<uint8_t * >(buf),sizeof(buf),0};
    featvec.floatfeats.funcs.decode = decode_repeated_floats;
    featvec.floatfeats.arg = (void *)&floatfeats_info;
    
    uint8_t idbuf[1024] = {0};
    BufferInfo_t id_info = {idbuf,sizeof(idbuf),0};
    featvec.id.funcs.decode = decode_string;
    featvec.id.arg = (void *)&id_info;
    
    uint8_t deviceidbuf[1024] = {0};
    BufferInfo_t deviceid_info = {deviceidbuf,sizeof(deviceidbuf),0};
    featvec.device_id.funcs.decode = decode_string;
    featvec.device_id.arg = (void *)&deviceid_info;
    
    if (!pb_decode(&istream,hello_FeatureVector_fields,&featvec)) {
        LOG("failed to decode");
        return false;
    }
    
    const size_t num_items_received = floatfeats_info.pos / sizeof(float);
    
    if (num_items_received == 0) {
        LOG("items received is zero");
        return false;
    }
    
    message.vec.clear();
    size_t num_elements = floatfeats_info.pos / sizeof(float);
    message.vec.reserve(num_elements);
    for (int i = 0; i < num_elements; i++) {
        message.vec.push_back(buf[i]);
    }
    
    message.id.assign((const char *)idbuf,strnlen((const char *)idbuf,sizeof(idbuf)));
    message.device_id.assign((const char *)deviceidbuf,strnlen((const char *)deviceidbuf,sizeof(deviceidbuf)));

    message.sequence_number = featvec.sequence_number;
    

    return true;

}
