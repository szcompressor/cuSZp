#ifndef CUSZP_INCLUDE_CUSZP_CUSZP_ENTRY_3D_F32_H
#define CUSZP_INCLUDE_CUSZP_CUSZP_ENTRY_3D_F32_H

#include <cuda_runtime.h>

void cuSZp_compress_3D_fixed_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, float errorBound, cudaStream_t stream = 0);
void cuSZp_decompress_3D_fixed_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, float errorBound, cudaStream_t stream = 0);
void cuSZp_compress_3D_plain_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, float errorBound, cudaStream_t stream = 0);
void cuSZp_decompress_3D_plain_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, float errorBound, cudaStream_t stream = 0);
void cuSZp_compress_3D_outlier_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, float errorBound, cudaStream_t stream = 0);
void cuSZp_decompress_3D_outlier_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, float errorBound, cudaStream_t stream = 0);

#endif // CUSZP_INCLUDE_CUSZP_CUSZP_ENTRY_3D_F32_H