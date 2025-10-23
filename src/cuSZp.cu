#include "cuSZp.h"

// Wrap-up API for cuSZp compression.
void cuSZp_compress(void* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cuszp_type_t type, cuszp_mode_t mode, cudaStream_t stream)
{
    if (type == CUSZP_TYPE_FLOAT) {
        if (mode == CUSZP_MODE_PLAIN) {
            cuSZp_compress_1D_plain_f32((float*)d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
        } 
        else if (mode == CUSZP_MODE_OUTLIER) {
            cuSZp_compress_1D_outlier_f32((float*)d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
        }
        else {
            printf("Unsupported mode in cuSZp.\n");
        }
    } 
    else if (type == CUSZP_TYPE_DOUBLE) {
        double errorBound_f64 = (double)errorBound;
        if (mode == CUSZP_MODE_PLAIN) {
            cuSZp_compress_1D_plain_f64((double*)d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound_f64, stream);
        } 
        else if (mode == CUSZP_MODE_OUTLIER) {
            cuSZp_compress_1D_outlier_f64((double*)d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound_f64, stream);
        }
        else{
            printf("Unsupported mode in cuSZp.\n");
        }
    }
    else {
        printf("Unsupported type in cuSZp.\n");
    }
}

// Wrap-up API for cuSZp decompression.
void cuSZp_decompress(void* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cuszp_type_t type, cuszp_mode_t mode, cudaStream_t stream)
{
    if (type == CUSZP_TYPE_FLOAT) {
        if (mode == CUSZP_MODE_PLAIN) {
            cuSZp_decompress_1D_plain_f32((float*)d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
        } 
        else if (mode == CUSZP_MODE_OUTLIER) {
            cuSZp_decompress_1D_outlier_f32((float*)d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
        }
        else {
            printf("Unsupported mode in cuSZp.\n");
        }
    } 
    else if (type == CUSZP_TYPE_DOUBLE) {
        double errorBound_f64 = (double)errorBound;
        if (mode == CUSZP_MODE_PLAIN) {
            cuSZp_decompress_1D_plain_f64((double*)d_decData, d_cmpBytes, nbEle, cmpSize, errorBound_f64, stream);
        } 
        else if (mode == CUSZP_MODE_OUTLIER) {
            cuSZp_decompress_1D_outlier_f64((double*)d_decData, d_cmpBytes, nbEle, cmpSize, errorBound_f64, stream);
        }
        else {
            printf("Unsupported mode in cuSZp.\n");
        }
    }
    else {
        printf("Unsupported type in cuSZp.\n");
    }
}