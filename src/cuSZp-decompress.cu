#include "cuSZp-decompress.h"

__global__ void recover_quant_1DLorenzo1Layer(float* decData, int* decQuantArray, float e2, int chunk_size, int bunch, cudaStream_t stream)
{
    int index, currQuant, tempQuant;
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int baseIndex = threadIndex / chunk_size * (chunk_size * bunch) + threadIndex % chunk_size;
    int chunk_pos = threadIndex % chunk_size;
        
    for(int j=0; j<bunch; j++)
    {
        index =  baseIndex + j*chunk_size;
        currQuant = decQuantArray[index];

        // Recover 1-layer Lorenzo by partial-sum.
        for(int i=1; i<chunk_size; i=i*2)
        {
            tempQuant = __shfl_up_sync(0xffffffff, currQuant, i);
            if(chunk_pos >= i) currQuant += tempQuant;
        }

        // Recover quantization code.
        decData[index] = e2*currQuant;
    }        
}

__global__ void recover_quant_1DLorenzo2Layer(float* decData, int* decQuantArray, float e2, int chunk_size, int bunch, cudaStream_t stream)
{
    int index, currQuant, tempQuant;
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int baseIndex = threadIndex / chunk_size * (chunk_size * bunch) + threadIndex % chunk_size;
    int chunk_pos = threadIndex % chunk_size;
        
    for(int j=0; j<bunch; j++)
    {
        index =  baseIndex + j*chunk_size;
        currQuant = decQuantArray[index];

        // Recover 2-layer Lorenzo by partial-sum.
        for(int i=1; i<chunk_size; i=i*2)
        {
            tempQuant = __shfl_up_sync(0xffffffff, currQuant, i);
            if(chunk_pos >= i) currQuant += tempQuant;
        }
        for(int i=1; i<chunk_size; i=i*2)
        {
            tempQuant = __shfl_up_sync(0xffffffff, currQuant, i);
            if(chunk_pos >= i) currQuant += tempQuant;
        }

        // Recover quantization code.
        decData[index] = e2*currQuant;
    }    
}

__global__ void recover_quant_1DLorenzo3Layer(float* decData, int* decQuantArray, float e2, int chunk_size, int bunch, cudaStream_t stream)
{
    int index, currQuant, tempQuant;
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int baseIndex = threadIndex / chunk_size * (chunk_size * bunch) + threadIndex % chunk_size;
    int chunk_pos = threadIndex % chunk_size;
        
    for(int j=0; j<bunch; j++)
    {
        index =  baseIndex + j*chunk_size;
        currQuant = decQuantArray[index];

        // Recover 3-layer Lorenzo by partial-sum.
        for(int i=1; i<chunk_size; i=i*2)
        {
            tempQuant = __shfl_up_sync(0xffffffff, currQuant, i);
            if(chunk_pos >= i) currQuant += tempQuant;
        }
        for(int i=1; i<chunk_size; i=i*2)
        {
            tempQuant = __shfl_up_sync(0xffffffff, currQuant, i);
            if(chunk_pos >= i) currQuant += tempQuant;
        }
        for(int i=1; i<chunk_size; i=i*2)
        {
            tempQuant = __shfl_up_sync(0xffffffff, currQuant, i);
            if(chunk_pos >= i) currQuant += tempQuant;
        }

        // Recover quantization code.
        decData[index] = e2*currQuant;
    }   
}   

__global__ void zeroOneByte_unfolding(unsigned char* d_byte_typeData, unsigned char* d_int_typeArray, cudaStream_t stream)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int int_type_index = index * 8;

    int tmp = d_byte_typeData[index];
    d_int_typeArray[int_type_index+0] = (tmp & 0x80) >> 7;
    d_int_typeArray[int_type_index+1] = (tmp & 0x40) >> 6;
    d_int_typeArray[int_type_index+2] = (tmp & 0x20) >> 5;
    d_int_typeArray[int_type_index+3] = (tmp & 0x10) >> 4;
    d_int_typeArray[int_type_index+4] = (tmp & 0x08) >> 3;
    d_int_typeArray[int_type_index+5] = (tmp & 0x04) >> 2;
    d_int_typeArray[int_type_index+6] = (tmp & 0x02) >> 1;
    d_int_typeArray[int_type_index+7] = (tmp & 0x01) >> 0;
}

__global__ void unpredData_decompress_count(unsigned int* d_dencoding, unsigned int* d_dc, int bunch, cudaStream_t stream)
{
    unsigned int encode1 = 0, encode2 = 0, encode3 = 0;
    unsigned int c = 0;
    int code, i;
    unsigned int mask = 0x000007;
    unsigned int sindex = blockIdx.x * blockDim.x * 3;

    encode1 = d_dencoding[sindex+threadIdx.x];
    encode2 = d_dencoding[sindex+blockDim.x+threadIdx.x];
    encode3 = d_dencoding[sindex+2*blockDim.x+threadIdx.x];

    sindex = blockIdx.x * blockDim.x * bunch;
    for (i=0; i<10; i++){
        code = (encode1 >> (29-3*i)) & mask;
        if (code == 7){
            c++;
        }
    }

    code = (encode2 >> 31) & mask;
    code |= (encode1 << 1) & mask;
    if (code == 7){
        c++;
    }

    for (i=11; i<21; i++){
        code = (encode2 >> (28-3*(i-11))) & mask;
        if (code == 7){
            c++;
        }
    }

    code = (encode3 >> 30) & mask;
    code |= (encode2 << 2) & mask;
    if (code == 7){
        c++;
    }

    for (i=22; i<32; i++){
        code = (encode3 >> (27-3*(i-22))) & mask;
        if (code == 7){
            c++;
        }
    }
    d_dc[blockIdx.x*blockDim.x+threadIdx.x] = c;
}

__global__ void unpredData_decompress(unsigned int* d_dencoding, unsigned int* d_dsign, int* d_dunpredData, unsigned int* d_dgs, unsigned int* d_dbs, unsigned short* d_fvalues, int bunch, cudaStream_t stream)
{
    unsigned int encode1 = 0, encode2 = 0, encode3 = 0;
    unsigned int c = 0;
    int code, data, i;
    unsigned int mask = 0x000007;
    unsigned int sindex = blockIdx.x * blockDim.x * 3;
    extern __shared__ unsigned char shared[];
    unsigned short* sshared = (unsigned short*)&shared;

    unsigned int sign = d_dsign[blockIdx.x * blockDim.x + threadIdx.x];
    unsigned int start = d_dgs[blockIdx.x];
    unsigned int length = d_dgs[blockIdx.x+1] - start;
    for (i = 0; i < length; i+=blockDim.x) {
        if (threadIdx.x<(length-i)) sshared[i+threadIdx.x] = d_fvalues[start+i+threadIdx.x];     
    }
    __syncthreads();
    start = d_dbs[blockIdx.x * blockDim.x + threadIdx.x];


    encode1 = d_dencoding[sindex+threadIdx.x];
    encode2 = d_dencoding[sindex+blockDim.x+threadIdx.x];
    encode3 = d_dencoding[sindex+2*blockDim.x+threadIdx.x];

    sindex = blockIdx.x * blockDim.x * bunch;
    for (i=0; i<10; i++){
        data = 0;
        code = (encode1 >> (29-3*i)) & mask;
        if (code < 7){
            data = code;
        }else{
            data = (int)sshared[start+c];
            //data = (int)d_fvalues[d_dgs[blockIdx.x]+d_dbs[blockIdx.x*blockDim.x+threadIdx.x]+c];
            //data = 7;
            c++;
        }
        data |= (sign >> (31 - i) & 0x000001) == 0 ? 0 : 0x01000000;
        d_dunpredData[sindex+i*blockDim.x+threadIdx.x] = data;
    }

    data = 0;
    code = (encode2 >> 31) & mask;
    code |= (encode1 << 1) & mask;
    if (code < 7){
        data = code;
    }else{
        data = (int)sshared[start+c];
        //data = (int)d_fvalues[d_dgs[blockIdx.x]+d_dbs[blockIdx.x*blockDim.x+threadIdx.x]+c];
        //data = 7;
        c++;
    }
    data |= (sign >> (31 - 10) & 0x000001) == 0 ? 0 : 0x01000000;
    d_dunpredData[sindex+10*blockDim.x+threadIdx.x] = data;

    for (i=11; i<21; i++){
        data = 0;
        code = (encode2 >> (28-3*(i-11))) & mask;
        if (code < 7){
            data = code;
        }else{
            data = (int)sshared[start+c];
            //data = (int)d_fvalues[d_dgs[blockIdx.x]+d_dbs[blockIdx.x*blockDim.x+threadIdx.x]+c];
            //data = 7;
            c++;
        }
        data |= (sign >> (31 - i) & 0x000001) == 0 ? 0 : 0x01000000;
        d_dunpredData[sindex+i*blockDim.x+threadIdx.x] = data;
    }

    data = 0;
    code = (encode3 >> 30) & mask;
    code |= (encode2 << 2) & mask;
    if (code < 7){
        data = code;
    }else{
        data = (int)sshared[start+c];
        //data = (int)d_fvalues[d_dgs[blockIdx.x]+d_dbs[blockIdx.x*blockDim.x+threadIdx.x]+c];
        //data = 7;
        c++;
    }
    data |= (sign >> (31 - 21) & 0x000001) == 0 ? 0 : 0x01000000;
    d_dunpredData[sindex+21*blockDim.x+threadIdx.x] = data;

    for (i=22; i<32; i++){
        data = 0;
        code = (encode3 >> (27-3*(i-22))) & mask;
        if (code < 7){
            data = code;
        }else{
            data = (int)sshared[start+c];
            //data = (int)d_fvalues[d_dgs[blockIdx.x]+d_dbs[blockIdx.x*blockDim.x+threadIdx.x]+c];
            //data = 7;
            c++;
        }
        data |= (sign >> (31 - i) & 0x000001) == 0 ? 0 : 0x01000000;
        d_dunpredData[sindex+i*blockDim.x+threadIdx.x] = data;
    }
}