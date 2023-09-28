#include "cuSZp_entry.h"
#include "cuSZp.h"

void SZp_compress_hostptr(float* oriData, unsigned char* cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * cmp_chunk - 1) / (bsize * cmp_chunk);
    int msize = (cmp_tblock_size/32+1) * sizeof(unsigned int);
    int cmpOffSize = gsize + 1;
    int pad_nbEle = gsize * bsize * cmp_chunk;

    // Initializing global memory for GPU compression.
    float* d_oriData;
    unsigned char* d_cmpData;
    unsigned int* d_cmpOffset;
    int* d_flag;
    cudaMalloc((void**)&d_oriData, sizeof(float)*pad_nbEle);
    cudaMemcpy(d_oriData, oriData, sizeof(float)*pad_nbEle, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_cmpData, sizeof(float)*pad_nbEle);
    cudaMallocManaged((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);
    cudaMemset(d_oriData + nbEle, 0, (pad_nbEle - nbEle) * sizeof(float));

    // Initializing CUDA Stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    SZp_compress_kernel<<<gridSize, blockSize, msize, stream>>>(d_oriData, d_cmpData, d_cmpOffset, d_flag, errorBound, nbEle);
    cudaDeviceSynchronize();

    // Obtain compression ratio and move data back to CPU.  
    *cmpSize = (size_t)d_cmpOffset[cmpOffSize-1] + (nbEle+31)/32;
    cudaMemcpy(cmpBytes, d_cmpData, *cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free memory that is used.
    cudaFree(d_oriData);
    cudaFree(d_cmpData);
    cudaFree(d_cmpOffset);
    cudaFree(d_flag);
    cudaStreamDestroy(stream);
}


void SZp_decompress_hostptr(float* decData, unsigned char* cmpBytes, size_t nbEle, size_t cmpSize, float errorBound)
{
    // Data blocking.
    int bsize = dec_tblock_size;
    int gsize = (nbEle + bsize * dec_chunk - 1) / (bsize * dec_chunk);
    int cmpOffSize = gsize + 1;
    int pad_nbEle = gsize * bsize * dec_chunk;

    // Initializing global memory for GPU compression.
    float* d_decData;
    unsigned char* d_cmpData;
    unsigned int* d_cmpOffset;
    int* d_flag;
    cudaMalloc((void**)&d_decData, sizeof(float)*pad_nbEle);
    cudaMemset(d_decData, 0, sizeof(float)*pad_nbEle);
    cudaMalloc((void**)&d_cmpData, sizeof(float)*pad_nbEle);
    cudaMemcpy(d_cmpData, cmpBytes, sizeof(unsigned char)*cmpSize, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Initializing CUDA Stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    SZp_decompress_kernel<<<gridSize, blockSize, 0, stream>>>(d_decData, d_cmpData, d_cmpOffset, d_flag, errorBound, nbEle);
    cudaDeviceSynchronize();

    // Move data back to CPU.
    cudaMemcpy(decData, d_decData, sizeof(float)*nbEle, cudaMemcpyDeviceToHost);

    // Free memoy that is used.
    cudaFree(d_decData);
    cudaFree(d_cmpData);
    cudaFree(d_cmpOffset);
    cudaFree(d_flag);
    cudaStreamDestroy(stream);
}


void SZp_compress_deviceptr(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * cmp_chunk - 1) / (bsize * cmp_chunk);
    int msize = (cmp_tblock_size/32+1) * sizeof(unsigned int);
    int cmpOffSize = gsize + 1;
    int pad_nbEle = gsize * bsize * cmp_chunk;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    int* d_flag;
    cudaMallocManaged((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);
    cudaMemset(d_oriData + nbEle, 0, (pad_nbEle - nbEle) * sizeof(float));

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    SZp_compress_kernel<<<gridSize, blockSize, msize, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_flag, errorBound, nbEle);
    cudaDeviceSynchronize();

    // Obtain compression ratio and move data back to CPU.  
    *cmpSize = (size_t)d_cmpOffset[cmpOffSize-1] + (nbEle+31)/32;

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_flag);
}


void SZp_decompress_deviceptr(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = dec_tblock_size;
    int gsize = (nbEle + bsize * dec_chunk - 1) / (bsize * dec_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    int* d_flag;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    SZp_decompress_kernel<<<gridSize, blockSize, 0, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_flag, errorBound, nbEle);
    cudaDeviceSynchronize();

    // Free memoy that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_flag);
}