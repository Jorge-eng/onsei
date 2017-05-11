#ifndef _PROTOBUFMATRIXUTILS_H_
#define _PROTOBUFMATRIXUTILS_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "simple_matrix.pb.h"

#ifdef __cplusplus
extern "C" {
#endif
    
    typedef enum {
        encode_status_success,
        encode_status_buffer_too_small,
        encode_status_invalid_input,
        encode_status_unexpected
    } EEncodeStatus_t;
    
    typedef long long Timestamp_t; //TODO move this to shared include

    typedef struct {
        SimpleMatrixDataType scalar_type;
        size_t num_rows;
        size_t num_cols;
        char id[1024];
        char device_id[1024];
    } DecodedSimpleMatrix_t;
    
    
    /* NOTE THAT THIS ASSUMES THE MATRIX IS ROW MAJOR, WHICH IS NOT THE CASE FOR THE DEFAULT FOR EIGEN  */
    
    //buffer is output
    //bytes_written is output
    //buffer_size is the max size of the buffer
    //the rest is are the stuff that gets encodeed into the protobuf
    EEncodeStatus_t protobuf_matrix_utils_create_and_write_protobuf(void * buffer,size_t * bytes_written,size_t buffer_size,const char * name, const char * device_id,const size_t num_rows, const size_t num_cols, const void * data, SimpleMatrixDataType mat_scalar_type, Timestamp_t timestamp_utc_millis, int tzoffset_millis);
    
    bool protobuf_matrix_utils_decode_protobuf(void * bufout, const size_t bufout_size, DecodedSimpleMatrix_t * decoded_mat,const void * protobuf_bytes, const size_t num_bytes);
    

#ifdef __cplusplus
}
#endif

#endif
