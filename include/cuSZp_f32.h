#ifndef CUSZP_INCLUDE_CUSZP_F32_H
#define CUSZP_INCLUDE_CUSZP_F32_H

static const int cmp_tblock_size_f32 = 32;
static const int dec_tblock_size_f32 = 32;
static const int cmp_chunk_f32 = 8192;
static const int dec_chunk_f32 = 8192;

__global__ void SZp_compress_kernel_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void SZp_decompress_kernel_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);

#endif // CUSZP_INCLUDE_CUSZP_F32_H