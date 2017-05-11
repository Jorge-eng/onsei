#ifndef _RADARMESSAGEPROTOBUF_H_
#define _RADARMESSAGEPROTOBUF_H_

#include "haltija_types.h"

class RadarMessageProtobuf {
public:
    static uint8_t * serialize_protobuf(const RadarMessage_t & message,size_t & size);
    static bool deserialize_protobuf(const uint8_t * protobuf_bytes, const size_t protobuf_size, RadarMessage_t & message);

};

#endif //_RADARMESSAGEPROTOBUF_H_
