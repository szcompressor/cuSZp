#ifndef CUSZP_SRC_CUSZP_COMPRESS_H
#define CUSZP_SRC_CUSZP_COMPRESS_H

__device__ inline int quantization(float data, float recipPrecision);
__global__ void quant_1DLorenzo1Layer(float* oriData, int* quantArray, float recipPrecision, int chunk_size, int bunch, cudaStream_t stream);
__global__ void quant_1DLorenzo2Layer(float* oriData, int* quantArray, float recipPrecision, int chunk_size, int bunch, cudaStream_t stream);
__global__ void quant_1DLorenzo3Layer(float* oriData, int* quantArray, float recipPrecision, int chunk_size, int bunch, cudaStream_t stream);
__global__ void zeroBitmap_search(int* quantArray, unsigned char* int_typeArray, int blockSize, cudaStream_t stream);
__global__ void zeroOneByte_folding(unsigned char* d_int_typeArray, unsigned char* d_byte_typeData, cudaStream_t stream);
__global__ void unpredData_compress_rad(int* d_unpredData, unsigned int* d_sign, unsigned int* d_encoding, unsigned short* d_c, unsigned int* d_pos, unsigned int* d_bytes, unsigned int* d_values, int bunch, cudaStream_t stream);
__global__ void unpredData_compress_con(int* d_unpredData, unsigned int* d_sign, unsigned int* d_encoding, unsigned short* d_c, unsigned int* d_pos, unsigned int* d_bytes, unsigned int* d_values, int bunch, cudaStream_t stream);
#endif // CUSZP_SRC_CUSZP_COMPRESS_H