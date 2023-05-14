#include "cuSZp-compress.h"
#include <stdio.h>

__device__ inline int quantization(float data, float recipPrecision)
{
    float dataRecip = data*recipPrecision;
    int s = dataRecip>=-0.5f?0:1;
    return (int)(dataRecip+0.5f) - s;
}

__global__ void quant_1DLorenzo1Layer(float* oriData, int* quantArray, float recipPrecision, int chunk_size, int bunch, cudaStream_t stream)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * bunch;
    int currQuant, pre1Quant;
    int tempIdx;

    // Quantization and 1-layer Lorenzo.
    pre1Quant = index%(chunk_size)==0 ? 0 : quantization(oriData[index-1], recipPrecision);
    for(int i=0; i<bunch; i++)
    {
        tempIdx = index + i;
        currQuant = quantization(oriData[tempIdx], recipPrecision);
        quantArray[tempIdx] = currQuant - pre1Quant;
        pre1Quant = currQuant;
    }
}


__global__ void quant_1DLorenzo2Layer(float* oriData, int* quantArray, float recipPrecision, int chunk_size, int bunch, cudaStream_t stream)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * bunch;
    int currQuant, pre1Quant, pre2Quant;
    int tempIdx;

    // Quantization and 2-layer Lorenzo.
    pre1Quant = index%(chunk_size)==0 ? 0 : quantization(oriData[index-1], recipPrecision);
    pre2Quant = index%(chunk_size)<=1 ? 0 : quantization(oriData[index-2], recipPrecision);
    for(int i=0; i<bunch; i++)
    {
        tempIdx = index + i;
        currQuant = quantization(oriData[tempIdx], recipPrecision);
        quantArray[tempIdx] = currQuant - (2*pre1Quant - pre2Quant);
        pre2Quant = pre1Quant;
        pre1Quant = currQuant;
    }
}

__global__ void quant_1DLorenzo3Layer(float* oriData, int* quantArray, float recipPrecision, int chunk_size, int bunch, cudaStream_t stream)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * bunch;
    int currQuant, pre1Quant, pre2Quant, pre3Quant;
    int tempIdx;

    // Quantization and 3-layer Lorenzo.
    pre1Quant = index%(chunk_size)==0 ? 0 : quantization(oriData[index-1], recipPrecision);
    pre2Quant = index%(chunk_size)<=1 ? 0 : quantization(oriData[index-2], recipPrecision);
    pre3Quant = index%(chunk_size)<=2 ? 0 : quantization(oriData[index-3], recipPrecision);
    for(int i=0; i<bunch; i++)
    {
        tempIdx = index + i;
        currQuant = quantization(oriData[tempIdx], recipPrecision);
        quantArray[tempIdx] = currQuant - (3*pre1Quant - 3*pre2Quant + pre3Quant);
        pre3Quant = pre2Quant;
        pre2Quant = pre1Quant;
        pre1Quant = currQuant;
    }
}

// __global__ void quant_2DLorenzo1Layer(float* oriData, int* quantArray, float recipPrecision, int chunk_size, int bunch, cudaStream_t stream)
// {
//     int index = (threadIdx.x + blockIdx.x * blockDim.x) * bunch;
//     int currQuant, pre1Quant;
//     int tempIdx;

//     // Quantization and 1-layer Lorenzo.
//     pre1Quant = index%(chunk_size)==0 ? 0 : quantization(oriData[index-1], recipPrecision);
//     for(int i=0; i<bunch; i++)
//     {
//         tempIdx = index + i;
//         currQuant = quantization(oriData[tempIdx], recipPrecision);
//         quantArray[tempIdx] = currQuant - pre1Quant;
//         pre1Quant = currQuant;
//     }
// }


__global__ void zeroBitmap_search(int* quantArray, unsigned char* int_typeArray, int blockSize, cudaStream_t stream)
{
    // Variables for bitmap search.
    int int_type_index = threadIdx.x + blockIdx.x * blockDim.x;
    int quant_index = int_type_index * blockSize;

    int buffer = quantArray[quant_index];
    for(int i=1; i<blockSize; i++)
        buffer |= quantArray[quant_index+i];

    int_typeArray[int_type_index] = buffer==0 ? 0 : 1; 
}


__global__ void zeroOneByte_folding(unsigned char* d_int_typeArray, unsigned char* d_byte_typeData, cudaStream_t stream)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int int_type_index = index * 8;

    unsigned char tmp = 0;
    tmp = tmp | d_int_typeArray[int_type_index]   << 7
              | d_int_typeArray[int_type_index+1] << 6
              | d_int_typeArray[int_type_index+2] << 5
              | d_int_typeArray[int_type_index+3] << 4
              | d_int_typeArray[int_type_index+4] << 3
              | d_int_typeArray[int_type_index+5] << 2
              | d_int_typeArray[int_type_index+6] << 1
              | d_int_typeArray[int_type_index+7] << 0;
    d_byte_typeData[index] = tmp;
}


__global__ void unpredData_compress_rad(int* d_unpredData, unsigned int* d_sign, unsigned int* d_encoding, unsigned short* d_c, unsigned int* d_pos, unsigned int* d_bytes, unsigned int* d_values, int bunch, cudaStream_t stream)
{
    unsigned int sindex = blockIdx.x * blockDim.x * bunch;
    unsigned int tindex = threadIdx.x * bunch;
    int i, data, code;
    unsigned int uc = 0, bc = 0, sign = 0;
    extern __shared__ unsigned char shared[];
    unsigned char* cshared = shared;
    unsigned short* sshared = (unsigned short*)&cshared[blockDim.x];
    unsigned int* ishared = (unsigned int*)sshared;
    unsigned int encode1 = 0, encode2 = 0, encode3 = 0;
    unsigned int pos1 = 0, pos2 = 0, bytes1 = 0, bytes2 = 0;

    for (i=0; i<10; i++){
        data = d_unpredData[sindex+i*blockDim.x+threadIdx.x];
        sign |= (unsigned int)(data < 0) << (31 - i);
        data = abs(data);
        if (data < 7){
            code = data;
        }else{
            code = 7;
            sshared[tindex+uc] = (unsigned short)data;
            if (data > 0x0000ffff) {
                if (bc < 4) {
                    bytes1 |= (data >> 16 & 0x000000ff) << (3 - bc) * 8;  
                    pos1 |= i << (3 - bc) * 8;  
                }else if (bc < 8) {
                    bytes2 |= (data >> 16 & 0x000000ff) << (7 - bc) * 8;  
                    pos2 |= i << (7 - bc) * 8;  
                }
                bc++;
            }
            uc++;
        }
        encode1 |= (code << (29-3*i)); 
    }

    data = d_unpredData[sindex+10*blockDim.x+threadIdx.x];
    sign |= (unsigned int)(data < 0) << (31 - 10);
    data = abs(data);
    if (data < 7){
        code = data;
    }else{
        code = 7;
        sshared[tindex+uc] = (unsigned short)data;
        if (data > 0x0000ffff) {
            if (bc < 4) {
                bytes1 |= (data >> 16 & 0x000000ff) << (3 - bc) * 8;  
                pos1 |= 10 << (3 - bc) * 8;  
            }else if (bc < 8) {
                bytes2 |= (data >> 16 & 0x000000ff) << (7 - bc) * 8;  
                pos2 |= 10 << (7 - bc) * 8;  
            }
            bc++;
        }
        uc++;
    }
    encode1 |= (code >> 1); 
    encode2 |= (code << 31); 

    for (i=11; i<21; i++){
        data = d_unpredData[sindex+i*blockDim.x+threadIdx.x];
        sign |= (unsigned int)(data < 0) << (31 - i);
        data = abs(data);
        if (data < 7){
            code = data;
        }else{
            code = 7;
            sshared[tindex+uc] = (unsigned short)data;
            if (data > 0x0000ffff) {
                if (bc < 4) {
                    bytes1 |= (data >> 16 & 0x000000ff) << (3 - bc) * 8;  
                    pos1 |= i << (3 - bc) * 8;  
                }else if (bc < 8) {
                    bytes2 |= (data >> 16 & 0x000000ff) << (7 - bc) * 8;  
                    pos2 |= i << (7 - bc) * 8;  
                }
                bc++;
            }
            uc++;
        }
        encode2 |= (code << (28-3*(i-11))); 
    }

    data = d_unpredData[sindex+21*blockDim.x+threadIdx.x];
    sign |= (unsigned int)(data < 0) << (31 - 21);
    data = abs(data);
    if (data < 7){
        code = data;
    }else{
        code = 7;
        sshared[tindex+uc] = (unsigned short)data;
        if (data > 0x0000ffff) {
            if (bc < 4) {
                bytes1 |= (data >> 16 & 0x000000ff) << (3 - bc) * 8;  
                pos1 |= 21 << (3 - bc) * 8;  
            }else if (bc < 8) {
                bytes2 |= (data >> 16 & 0x000000ff) << (7 - bc) * 8;  
                pos2 |= 21 << (7 - bc) * 8;  
            }
            bc++;
        }
        uc++;
    }
    encode2 |= (code >> 2); 
    encode3 |= (code << 30); 

    for (i=22; i<32; i++){
        data = d_unpredData[sindex+i*blockDim.x+threadIdx.x];
        sign |= (unsigned int)(data < 0) << (31 - i);
        data = abs(data);
        if (data < 7){
            code = data;
        }else{
            code = 7;
            sshared[tindex+uc] = (unsigned short)data;
            if (data > 0x0000ffff) {
                if (bc < 4) {
                    bytes1 |= (data >> 16 & 0x000000ff) << (3 - bc) * 8;  
                    pos1 |= i << (3 - bc) * 8;  
                }else if (bc < 8) {
                    bytes2 |= (data >> 16 & 0x000000ff) << (7 - bc) * 8;  
                    pos2 |= i << (7 - bc) * 8;  
                }
                bc++;
            }
            uc++;
        }
        encode3 |= (code << (27-3*(i-22))); 
    }
    //cshared[threadIdx.x] = (unsigned char)uc;

    sindex = blockIdx.x * blockDim.x * 3;
    tindex = blockIdx.x * blockDim.x * bunch / 2;
    __syncthreads();

    d_sign[blockIdx.x*blockDim.x+threadIdx.x] = sign;
    d_encoding[sindex+threadIdx.x] = encode1;
    d_encoding[sindex+blockDim.x+threadIdx.x] = encode2;
    d_encoding[sindex+2*blockDim.x+threadIdx.x] = encode3;
    d_c[blockIdx.x*blockDim.x+threadIdx.x] = (unsigned short)(bc << 8 | uc);//cshared[threadIdx.x];
    d_pos[blockIdx.x*blockDim.x+threadIdx.x] = pos1;
    d_pos[(gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x] = pos2;
    d_bytes[blockIdx.x*blockDim.x+threadIdx.x] = bytes1;
    d_bytes[(gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x] = bytes2;
    for (i = 0; i < bunch / 2; i++) {
        d_values[tindex+i*blockDim.x+threadIdx.x] = ishared[i*blockDim.x+threadIdx.x];
    }
}

__global__ void unpredData_compress_con(int* d_unpredData, unsigned int* d_sign, unsigned int* d_encoding, unsigned short* d_c, unsigned int* d_pos, unsigned int* d_bytes, unsigned int* d_values, int bunch, cudaStream_t stream)
{
    unsigned int sindex = blockIdx.x * blockDim.x * bunch;
    unsigned int tindex = threadIdx.x * bunch;
    int i, data, code;
    unsigned int uc = 0, bc = 0, sign = 0;
    extern __shared__ unsigned char shared[];
    unsigned char* cshared = shared;
    unsigned short* sshared = (unsigned short*)&cshared[blockDim.x];
    unsigned int* ishared = (unsigned int*)sshared;
    unsigned int encode1 = 0, encode2 = 0, encode3 = 0;
    unsigned int pos1 = 0, pos2 = 0, bytes1 = 0, bytes2 = 0, bytes11 = 0, bytes22 = 0;


    // yafan add start
    int flag=0;
    // yafan add end
    for (i=0; i<10; i++){
        data = d_unpredData[sindex+i*blockDim.x+threadIdx.x];
        sign |= (unsigned int)(data < 0) << (31 - i);
        data = abs(data);
        if (data < 7){
            code = data;
        }else{
            code = 7;
            sshared[tindex+uc] = (unsigned short)data;
            // yafan add start
            if(threadIdx.x==72 && blockIdx.x==872 && i==8)
            {
                printf("I am here: %d\n", data);
                flag = 1;
            }
            // yafan add end
            if (data > 0x0000ffff) {
                if (bc < 4) {
                    bytes1 |= (data >> 16 & 0x000000ff) << (3 - bc) * 8;
                    bytes11|= (data >> 24 & 0x000000ff) << (3 - bc) * 8;
                    pos1 |= i << (3 - bc) * 8;
                    // yafan add start
                    if(flag == 1)
                    {
                        printf("value of data: %d, this point bc: %u, uc: %u!\n", data, bc, uc);
                        printf("%u %u %u\n", bytes1, bytes11, pos1);
                    }
                    // yafan add end
                }else if (bc < 8) {
                    bytes2 |= (data >> 16 & 0x000000ff) << (7 - bc) * 8;
                    bytes22|= (data >> 24 & 0x000000ff) << (7 - bc) * 8;
                    pos2 |= i << (7 - bc) * 8;  
                }
                bc++;
            }
            uc++;
        }
        encode1 |= (code << (29-3*i)); 
    }

    data = d_unpredData[sindex+10*blockDim.x+threadIdx.x];
    sign |= (unsigned int)(data < 0) << (31 - 10);
    data = abs(data);
    if (data < 7){
        code = data;
    }else{
        code = 7;
        sshared[tindex+uc] = (unsigned short)data;
        if (data > 0x0000ffff) {
            if (bc < 4) {
                bytes1 |= (data >> 16 & 0x000000ff) << (3 - bc) * 8;
                bytes11|= (data >> 24 & 0x000000ff) << (3 - bc) * 8; 
                pos1 |= 10 << (3 - bc) * 8;  
            }else if (bc < 8) {
                bytes2 |= (data >> 16 & 0x000000ff) << (7 - bc) * 8;
                bytes22|= (data >> 24 & 0x000000ff) << (7 - bc) * 8;  
                pos2 |= 10 << (7 - bc) * 8;  
            }
            bc++;
        }
        uc++;
    }
    encode1 |= (code >> 1); 
    encode2 |= (code << 31); 

    for (i=11; i<21; i++){
        data = d_unpredData[sindex+i*blockDim.x+threadIdx.x];
        sign |= (unsigned int)(data < 0) << (31 - i);
        data = abs(data);
        if (data < 7){
            code = data;
        }else{
            code = 7;
            sshared[tindex+uc] = (unsigned short)data;
            if (data > 0x0000ffff) {
                if (bc < 4) {
                    bytes1 |= (data >> 16 & 0x000000ff) << (3 - bc) * 8;
                    bytes11|= (data >> 24 & 0x000000ff) << (3 - bc) * 8;  
                    pos1 |= i << (3 - bc) * 8;  
                }else if (bc < 8) {
                    bytes2 |= (data >> 16 & 0x000000ff) << (7 - bc) * 8;
                    bytes22|= (data >> 24 & 0x000000ff) << (7 - bc) * 8;  
                    pos2 |= i << (7 - bc) * 8;  
                }
                bc++;
            }
            uc++;
        }
        encode2 |= (code << (28-3*(i-11))); 
    }

    data = d_unpredData[sindex+21*blockDim.x+threadIdx.x];
    sign |= (unsigned int)(data < 0) << (31 - 21);
    data = abs(data);
    if (data < 7){
        code = data;
    }else{
        code = 7;
        sshared[tindex+uc] = (unsigned short)data;
        if (data > 0x0000ffff) {
            if (bc < 4) {
                bytes1 |= (data >> 16 & 0x000000ff) << (3 - bc) * 8;
                bytes11|= (data >> 24 & 0x000000ff) << (3 - bc) * 8; 
                pos1 |= 21 << (3 - bc) * 8;  
            }else if (bc < 8) {
                bytes2 |= (data >> 16 & 0x000000ff) << (7 - bc) * 8;
                bytes22|= (data >> 24 & 0x000000ff) << (7 - bc) * 8;  
                pos2 |= 21 << (7 - bc) * 8;  
            }
            bc++;
        }
        uc++;
    }
    encode2 |= (code >> 2); 
    encode3 |= (code << 30); 

    for (i=22; i<32; i++){
        data = d_unpredData[sindex+i*blockDim.x+threadIdx.x];
        sign |= (unsigned int)(data < 0) << (31 - i);
        data = abs(data);
        if (data < 7){
            code = data;
        }else{
            code = 7;
            sshared[tindex+uc] = (unsigned short)data;
            if (data > 0x0000ffff) {
                if (bc < 4) {
                    bytes1 |= (data >> 16 & 0x000000ff) << (3 - bc) * 8;
                    bytes11|= (data >> 24 & 0x000000ff) << (3 - bc) * 8; 
                    pos1 |= i << (3 - bc) * 8;  
                }else if (bc < 8) {
                    bytes2 |= (data >> 16 & 0x000000ff) << (7 - bc) * 8;
                    bytes22|= (data >> 24 & 0x000000ff) << (7 - bc) * 8;  
                    pos2 |= i << (7 - bc) * 8;  
                }
                bc++;
            }
            uc++;
        }
        encode3 |= (code << (27-3*(i-22))); 
    }
    //cshared[threadIdx.x] = (unsigned char)uc;
    // yafan add start
    if(flag==1) printf("this point bc: %u, uc: %u!\n", bc, uc);
    // yafan add start
    sindex = blockIdx.x * blockDim.x * 3;
    tindex = blockIdx.x * blockDim.x * bunch / 2;
    __syncthreads();

    d_sign[blockIdx.x*blockDim.x+threadIdx.x] = sign;
    d_encoding[sindex+threadIdx.x] = encode1;
    d_encoding[sindex+blockDim.x+threadIdx.x] = encode2;
    d_encoding[sindex+2*blockDim.x+threadIdx.x] = encode3;
    d_c[blockIdx.x*blockDim.x+threadIdx.x] = (unsigned short)(bc << 8 | uc);//cshared[threadIdx.x];
    d_pos[blockIdx.x*blockDim.x+threadIdx.x] = pos1;
    d_pos[(gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x] = pos2;
    d_bytes[blockIdx.x*blockDim.x+threadIdx.x] = bytes1;
    d_bytes[(gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x] = bytes2;
    d_bytes[(2*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x] = bytes11;
    d_bytes[(3*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x] = bytes22;
    for (i = 0; i < bunch / 2; i++) {
        d_values[tindex+i*blockDim.x+threadIdx.x] = ishared[i*blockDim.x+threadIdx.x];
    }
}