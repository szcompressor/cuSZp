#ifndef CUSZP_INCLUDE_CUSZP_ENTRY_F32_H
#define CUSZP_INCLUDE_CUSZP_ENTRY_F32_H

#include <cuda_runtime.h>

void SZp_compress_hostptr_f32(float* oriData, unsigned char* cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound);
void SZp_decompress_hostptr_f32(float* decData, unsigned char* cmpBytes, size_t nbEle, size_t cmpSize, float errorBound);
void SZp_compress_deviceptr_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cudaStream_t stream = 0);
void SZp_decompress_deviceptr_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cudaStream_t stream = 0);

#endif // CUSZP_INCLUDE_CUSZP_ENTRY_F32_H