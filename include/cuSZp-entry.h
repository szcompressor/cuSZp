#ifndef CUSZP_INCLUDE_CUSZP_ENTRY_H
#define CUSZP_INCLUDE_CUSZP_ENTRY_H

#include <cstddef>
#include <cuda_runtime.h>

void cuSZp_compress(float* oridata, unsigned char* cmpBytes, size_t* outSize, float realPrecision, size_t nbEle, int blockSize, int predLevel, int shufLevel, cudaStream_t stream);
void cuSZp_decompress(float* decdata, unsigned char* cmpBytes, float realPrecision, size_t nbEle, int blockSize, int predLevel, int shufLevel, cudaStream_t stream);

#endif // CUSZP_INCLUDE_CUSZP_ENTRY_H