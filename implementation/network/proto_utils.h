#ifndef _PROTOUTILS_H_
#define _PROTOUTILS_H_

#include <pb.h>
#include <pb_encode.h>
#include <pb_decode.h>
#include <stdbool.h>


#ifdef __cplusplus
extern "C" {
#endif
    
    typedef struct {
        uint8_t * buffer;
        size_t buf_size_bytes;
        size_t pos;
    } BufferInfo_t;

    bool decode_repeated_floats(pb_istream_t *stream, const pb_field_t *field, void **arg);
    
    bool decode_repeated_doubles(pb_istream_t *stream, const pb_field_t *field, void **arg);
    
    bool decode_string(pb_istream_t *stream, const pb_field_t *field, void **arg);
    
    bool decode_repeated_bytes_fields(pb_istream_t *stream, const pb_field_t *field, void **arg);
    
    bool encode_buffer_fields(pb_ostream_t * stream, const pb_field_t * field, void * const * arg);
        
    bool encode_string_fields(pb_ostream_t * stream, const pb_field_t *field, void * const *arg);
    
    bool encode_repeated_floats(pb_ostream_t * stream, const pb_field_t *field, void * const *arg);

    
#ifdef __cplusplus
}
#endif
        

#endif
