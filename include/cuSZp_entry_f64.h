#ifndef CUSZP_INCLUDE_CUSZP_ENTRY_F64_H
#define CUSZP_INCLUDE_CUSZP_ENTRY_F64_H

#include <cuda_runtime.h>

void SZp_compress_hostptr_f64(double* oriData, unsigned char* cmpBytes, size_t nbEle, size_t* cmpSize, double errorBound);
void SZp_decompress_hostptr_f64(double* decData, unsigned char* cmpBytes, size_t nbEle, size_t cmpSize, double errorBound);
void SZp_compress_deviceptr_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, double errorBound, cudaStream_t stream = 0);
void SZp_decompress_deviceptr_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, double errorBound, cudaStream_t stream = 0);

#endif // CUSZP_INCLUDE_CUSZP_ENTRY_F64_H