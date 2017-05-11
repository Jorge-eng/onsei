#include "protobuf_matrix_utils.h"
#include "proto_utils.h"
#include "simple_matrix.pb.h"
#include <stdbool.h>
#include "pb_encode.h"
#include "pb_decode.h"

#include <string.h>
#include <stdlib.h>


const static size_t _mat_scalar_sizes[] = {
    1,
    2,
    4,
    8,
    4,
    8,
    8,
    16
};




EEncodeStatus_t protobuf_matrix_utils_create_and_write_protobuf(void * buffer,size_t * bytes_written,size_t buffer_size,const char * name, const char * device_id,const size_t num_rows, const size_t num_cols, const void * data, SimpleMatrixDataType mat_scalar_type, Timestamp_t timestamp_utc_millis, int tzoffset_millis) {
        
    SimpleMatrix mat;
    const size_t scalar_size = _mat_scalar_sizes[mat_scalar_type];
    const size_t num_buf_bytes = scalar_size * num_rows * num_cols;

    memset(&mat,0,sizeof(mat));


    
    mat.has_data_type = true;
    mat.data_type = mat_scalar_type;
    
    mat.has_num_cols = true;
    mat.num_cols = num_cols;
    
    mat.has_timestamp_utc_millis = true;
    mat.timestamp_utc_millis = timestamp_utc_millis;
    
    mat.has_tz_offset_millis = true;
    mat.tz_offset_millis = tzoffset_millis;
    
    if (name) {
        mat.id.funcs.encode = encode_string_fields;
        mat.id.arg = (void *)name;
    }
    
    if (device_id) {
        mat.device_id.funcs.encode = encode_string_fields;
        mat.device_id.arg = (void *)device_id;
    }
    
    const BufferInfo_t info = {(void *)data,num_buf_bytes,0};
    
    mat.payload.funcs.encode = encode_buffer_fields;
    mat.payload.arg = (void *)&info;
    
    pb_ostream_t bufstream = pb_ostream_from_buffer(buffer,buffer_size);
    
    if (!pb_encode(&bufstream, SimpleMatrix_fields, &mat)) {
        return encode_status_buffer_too_small;
    }
    
    *bytes_written = bufstream.bytes_written;
    
    return encode_status_success;
}

bool protobuf_matrix_utils_decode_protobuf(void * bufout, const size_t bufout_size, DecodedSimpleMatrix_t * decoded_mat,const void * protobuf_bytes, const size_t num_bytes) {

    pb_istream_t istream = pb_istream_from_buffer(protobuf_bytes, num_bytes);
    SimpleMatrix mat;
    size_t scalar_size;
    size_t num_elements;
    
    memset(&mat,0,sizeof(mat));
    BufferInfo_t payload_info = {bufout,bufout_size};
    BufferInfo_t id_info = {(uint8_t *)&decoded_mat->id[0],sizeof(decoded_mat->id),0};
    BufferInfo_t device_id_info = {(uint8_t *)&decoded_mat->device_id[0],sizeof(decoded_mat->device_id),0};

    
    mat.payload.funcs.decode = decode_repeated_bytes_fields;
    mat.payload.arg = (void *)&payload_info;
    
    mat.id.funcs.decode = decode_string;
    mat.id.arg = (void *)&id_info;
    
    mat.device_id.funcs.decode = decode_string;
    mat.device_id.arg = (void *)&device_id_info;
    
    if (!pb_decode(&istream,SimpleMatrix_fields,&mat)) {
        return false;
    }
    
    if (!mat.has_data_type) {
        return false;
    }
    
    if (!mat.has_num_cols) {
        return false;
    }
    
    scalar_size = _mat_scalar_sizes[mat.data_type];
    
    if (scalar_size == 0) {
        return false;
    }
    
    num_elements = payload_info.pos / scalar_size;

    
    decoded_mat->num_rows = num_elements / mat.num_cols;
    decoded_mat->num_cols = mat.num_cols;
    decoded_mat->scalar_type = mat.data_type;
    
    return true;
}


