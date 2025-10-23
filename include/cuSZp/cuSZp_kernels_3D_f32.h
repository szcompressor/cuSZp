#ifndef CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_3D_F32_H
#define CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_3D_F32_H

static const int tblock_size = 32; // Fixed to 32, cannot be modified.
static const int block_per_thread = 16;

// // To do: Implement the following kernels. (set as a placeholder, not sure if needed)
// __global__ void cuSZp_compress_kernel_3D_plain_scalar_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const float eb);
// __global__ void cuSZp_decompress_kernel_3D_plain_scalar_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const float eb);
// __global__ void cuSZp_compress_kernel_3D_plain_vec2_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const float eb);
// __global__ void cuSZp_decompress_kernel_3D_plain_vec2_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const float eb);

// Yafan is working on this one.
__global__ void cuSZp_compress_kernel_3D_plain_vec4_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const float eb);
__global__ void cuSZp_decompress_kernel_3D_plain_vec4_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const float eb);
__global__ void cuSZp_compress_kernel_3D_outlier_vec4_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const float eb);
__global__ void cuSZp_decompress_kernel_3D_outlier_vec4_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, uint blockNum, const uint3 dims, const float eb);

#endif // CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_3D_F32_H