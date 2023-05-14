#ifndef CUSZP_SRC_CUSZP_DECOMPRESS_H
#define CUSZP_SRC_CUSZP_DECOMPRESS_H

__global__ void recover_quant_1DLorenzo1Layer(float* decData, int* decQuantArray, float e2, int chunk_size, int bunch, cudaStream_t stream);
__global__ void recover_quant_1DLorenzo2Layer(float* decData, int* decQuantArray, float e2, int chunk_size, int bunch, cudaStream_t stream);
__global__ void recover_quant_1DLorenzo3Layer(float* decData, int* decQuantArray, float e2, int chunk_size, int bunch, cudaStream_t stream);
__global__ void zeroOneByte_unfolding(unsigned char* d_byte_typeData, unsigned char* d_int_typeArray, cudaStream_t stream);
__global__ void unpredData_decompress_count(unsigned int* d_dencoding, unsigned int* d_dc, int bunch, cudaStream_t stream);
__global__ void unpredData_decompress(unsigned int* d_dencoding, unsigned int* d_dsign, int* d_dunpredData, unsigned int* d_dgs, unsigned int* d_dbs, unsigned short* d_fvalues, int bunch, cudaStream_t stream);

#endif // CUSZP_SRC_CUSZP_DECOMPRESS_H