#include "cuSZp_entry_3D_f32.h"
#include "cuSZp_kernels_3D_f32.h"

/** ************************************************************************
* @brief cuSZp end-to-end compression API for device pointers
*        Compression is executed in GPU.
*        Original data is stored as device pointers (in GPU).
*        Compressed data is stored back as device pointers (in GPU).
* 
* @param   d_oriData       original data (device pointer)
* @param   d_cmpBytes      compressed data (device pointer)
* @param   nbEle           original data size (number of floating point)
* @param   cmpSize         compressed data size (number of unsigned char)
* @param   dims            dimensions of the original data
* @param   errorBound      user-defined error bound
* @param   stream          CUDA stream for executing compression kernel
* *********************************************************************** */
void cuSZp_compress_3D_plain_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    uint dimzBlock = (dims.z + 3) / 4;
    uint dimyBlock = (dims.y + 3) / 4;
    uint dimxBlock = (dims.x + 3) / 4;
    uint blockNum = dimzBlock * dimyBlock * dimxBlock;
    int bsize = tblock_size;
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    unsigned int glob_sync;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Compression (implement for now, will update later)
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_compress_kernel_3D_plain_vec4_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, errorBound);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + blockNum;

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
} 
 
/** ************************************************************************
* @brief cuSZp end-to-end decompression API for device pointers
*        Decompression is executed in GPU.
*        Compressed data is stored as device pointers (in GPU).
*        Reconstructed data is stored as device pointers (in GPU).
*        P.S. Reconstructed data and original data have the same shape.
* 
* @param   d_decData       reconstructed data (device pointer)
* @param   d_cmpBytes      compressed data (device pointer)
* @param   nbEle           reconstructed data size (number of floating point)
* @param   cmpSize         compressed data size (number of unsigned char)
* @param   dims            dimensions of the original data
* @param   errorBound      user-defined error bound
* @param   stream          CUDA stream for executing compression kernel
* *********************************************************************** */
void cuSZp_decompress_3D_plain_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    uint dimzBlock = (dims.z + 3) / 4;
    uint dimyBlock = (dims.y + 3) / 4;
    uint dimxBlock = (dims.x + 3) / 4;
    uint blockNum = dimzBlock * dimyBlock * dimxBlock;
    int bsize = tblock_size;
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Decompression (implement for now, will update later with adaptive vectorization)
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_decompress_kernel_3D_plain_vec4_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, errorBound);

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

/** ************************************************************************
* @brief cuSZp end-to-end compression API for device pointers
*        Compression is executed in GPU.
*        Original data is stored as device pointers (in GPU).
*        Compressed data is stored back as device pointers (in GPU).
* 
* @param   d_oriData       original data (device pointer)
* @param   d_cmpBytes      compressed data (device pointer)
* @param   nbEle           original data size (number of floating point)
* @param   cmpSize         compressed data size (number of unsigned char)
* @param   dims            dimensions of the original data
* @param   errorBound      user-defined error bound
* @param   stream          CUDA stream for executing compression kernel
* *********************************************************************** */
void cuSZp_compress_3D_outlier_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, uint3 dims, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    uint dimzBlock = (dims.z + 3) / 4;
    uint dimyBlock = (dims.y + 3) / 4;
    uint dimxBlock = (dims.x + 3) / 4;
    uint blockNum = dimzBlock * dimyBlock * dimxBlock;
    int bsize = tblock_size;
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    unsigned int glob_sync;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Compression (implement for now, will update later)
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_compress_kernel_3D_outlier_vec4_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, errorBound);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + blockNum;

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
} 
 
/** ************************************************************************
* @brief cuSZp end-to-end decompression API for device pointers
*        Decompression is executed in GPU.
*        Compressed data is stored as device pointers (in GPU).
*        Reconstructed data is stored as device pointers (in GPU).
*        P.S. Reconstructed data and original data have the same shape.
* 
* @param   d_decData       reconstructed data (device pointer)
* @param   d_cmpBytes      compressed data (device pointer)
* @param   nbEle           reconstructed data size (number of floating point)
* @param   cmpSize         compressed data size (number of unsigned char)
* @param   dims            dimensions of the original data
* @param   errorBound      user-defined error bound
* @param   stream          CUDA stream for executing compression kernel
* *********************************************************************** */
void cuSZp_decompress_3D_outlier_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, uint3 dims, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    uint dimzBlock = (dims.z + 3) / 4;
    uint dimyBlock = (dims.y + 3) / 4;
    uint dimxBlock = (dims.x + 3) / 4;
    uint blockNum = dimzBlock * dimyBlock * dimxBlock;
    int bsize = tblock_size;
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Decompression (implement for now, will update later with adaptive vectorization)
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_decompress_kernel_3D_outlier_vec4_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, errorBound);

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}