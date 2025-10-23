#ifndef CUSZP_INCLUDE_CUSZP_CUSZP_ENTRY_1D_F64_H
#define CUSZP_INCLUDE_CUSZP_CUSZP_ENTRY_1D_F64_H

#include <cuda_runtime.h>

void cuSZp_compress_1D_plain_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, double errorBound, cudaStream_t stream = 0);
void cuSZp_decompress_1D_plain_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, double errorBound, cudaStream_t stream = 0);
void cuSZp_compress_1D_outlier_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, double errorBound, cudaStream_t stream = 0);
void cuSZp_decompress_1D_outlier_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, double errorBound, cudaStream_t stream = 0);

#endif // CUSZP_INCLUDE_CUSZP_CUSZP_ENTRY_1D_F64_H