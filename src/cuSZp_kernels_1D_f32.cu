#include "cuSZp_kernels_1D_f32.h"

__device__ inline int quantization(float data, float recipPrecision)
{
    int result;
    asm("{\n\t"
        ".reg .f32 dataRecip;\n\t"
        ".reg .f32 temp1;\n\t"
        ".reg .s32 s;\n\t"
        ".reg .pred p;\n\t"
        "mul.f32 dataRecip, %1, %2;\n\t"
        "setp.ge.f32 p, dataRecip, -0.5;\n\t"
        "selp.s32 s, 0, 1, p;\n\t"
        "add.f32 temp1, dataRecip, 0.5;\n\t"
        "cvt.rzi.s32.f32 %0, temp1;\n\t"
        "sub.s32 %0, %0, s;\n\t"
        "}": "=r"(result) : "f"(data), "f"(recipPrecision)
    );
    return result;
}


__device__ inline int get_bit_num(unsigned int x)
{
    int leading_zeros;
    asm("clz.b32 %0, %1;" : "=r"(leading_zeros) : "r"(x));
    return 32 - leading_zeros;
}


__global__ void cuSZp_compress_kernel_1D_fixed_f32(const float* const __restrict__ oriData, 
                                                unsigned char* const __restrict__ cmpData, 
                                                volatile unsigned int* const __restrict__ cmpOffset, 
                                                volatile unsigned int* const __restrict__ locOffset, 
                                                volatile int* const __restrict__ flag, 
                                                const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = thread_chunk >> 5;
    const int rate_ofs = (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;
    const float recipPrecision = 0.5f/eb;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    int block_idx;
    int currQuant, maxQuant;
    int absQuant[thread_chunk];
    unsigned int sign_flag[block_num];
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0; // Thread-level prefix-sum, double check for overflow in large data (can be resolved by using size_t type).
    float4 tmp_buffer;
    uchar4 tmp_char;

    base_start_idx = warp * thread_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        sign_flag[j] = 0;
        block_idx = base_block_start_idx/32;
        maxQuant = 0;

        if(base_block_end_idx < nbEle)
        {
            #pragma unroll 8
            for(int i=base_block_start_idx; i<base_block_end_idx; i+=4)
            {
                tmp_buffer = reinterpret_cast<const float4*>(oriData)[i/4];
                quant_chunk_idx = j * 32 + i % 32;

                currQuant = quantization(tmp_buffer.x, recipPrecision);
                sign_ofs = i % 32;
                sign_flag[j] |= (currQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx] = abs(currQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];

                currQuant = quantization(tmp_buffer.y, recipPrecision);
                sign_ofs = (i+1) % 32;
                sign_flag[j] |= (currQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+1] = abs(currQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+1] ? maxQuant : absQuant[quant_chunk_idx+1];

                currQuant = quantization(tmp_buffer.z, recipPrecision);
                sign_ofs = (i+2) % 32;
                sign_flag[j] |= (currQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+2] = abs(currQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+2] ? maxQuant : absQuant[quant_chunk_idx+2];

                currQuant = quantization(tmp_buffer.w, recipPrecision);
                sign_ofs = (i+3) % 32;
                sign_flag[j] |= (currQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+3] = abs(currQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+3] ? maxQuant : absQuant[quant_chunk_idx+3];
            }
        }
        else
        {
            if(base_block_start_idx >= nbEle)
            {
                quant_chunk_idx = j * 32 + base_block_start_idx % 32;
                for(int i=quant_chunk_idx; i<quant_chunk_idx+32; i++) absQuant[i] = 0;
            }
            else
            {
                int remainbEle = nbEle - base_block_start_idx;
                int zeronbEle = base_block_end_idx - nbEle;

                for(int i=base_block_start_idx; i<base_block_start_idx+remainbEle; i++)
                {
                    quant_chunk_idx = j * 32 + i % 32;
                    currQuant = quantization(oriData[i], recipPrecision);
                    sign_ofs = i % 32;
                    sign_flag[j] |= (currQuant < 0) << (31 - sign_ofs);
                    absQuant[quant_chunk_idx] = abs(currQuant);
                    maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
                }

                quant_chunk_idx = j * 32 + nbEle % 32;
                for(int i=quant_chunk_idx; i<quant_chunk_idx+zeronbEle; i++) absQuant[i] = 0;
            }  
        }

        fixed_rate[j] = get_bit_num(maxQuant);
        thread_ofs += (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        cmpData[block_idx] = (unsigned char)fixed_rate[j];
        __syncthreads();
    }

    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    for(int j=0; j<block_num; j++)
    {
        int chunk_idx_start = j*32;

        tmp_byte_ofs = (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(fixed_rate[j])
        {
            tmp_char.x = 0xff & (sign_flag[j] >> 24);
            tmp_char.y = 0xff & (sign_flag[j] >> 16);
            tmp_char.z = 0xff & (sign_flag[j] >> 8);
            tmp_char.w = 0xff & sign_flag[j];
            reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
            cmp_byte_ofs+=4;

            int mask = 1;
            for(int i=0; i<fixed_rate[j]; i++)
            {
                tmp_char.x = 0;
                tmp_char.y = 0;
                tmp_char.z = 0;
                tmp_char.w = 0;

                tmp_char.x = (((absQuant[chunk_idx_start+0] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+1] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+2] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+3] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+4] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+5] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+6] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+7] & mask) >> i) << 0);

                tmp_char.y = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                tmp_char.z = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                
                tmp_char.w = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                cmp_byte_ofs+=4;
                mask <<= 1;
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}


__global__ void cuSZp_decompress_kernel_1D_fixed_f32(float* const __restrict__ decData, 
                                                  const unsigned char* const __restrict__ cmpData, 
                                                  volatile unsigned int* const __restrict__ cmpOffset, 
                                                  volatile unsigned int* const __restrict__ locOffset, 
                                                  volatile int* const __restrict__ flag, 
                                                  const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = thread_chunk >> 5;
    const int rate_ofs = (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int block_idx;    
    int absQuant[32];
    int currQuant;
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char;
    float4 dec_buffer;

    for(int j=0; j<block_num; j++)
    {
        block_idx = warp * thread_chunk + j * 32 + lane;
        fixed_rate[j] = (int)cmpData[block_idx];
        thread_ofs += (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        __syncthreads();
    }

    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        if(!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if(!lane) flag[warp] = 2;
        __threadfence();  
    }
    __syncthreads();

    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    base_start_idx = warp * thread_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        unsigned int sign_flag = 0;

        tmp_byte_ofs = (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(fixed_rate[j])
        {
            tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
            sign_flag = (0xff000000 & (tmp_char.x << 24)) |
                        (0x00ff0000 & (tmp_char.y << 16)) |
                        (0x0000ff00 & (tmp_char.z << 8))  |
                        (0x000000ff & tmp_char.w);
            cmp_byte_ofs+=4;
            
            for(int i=0; i<32; i++) absQuant[i] = 0;
            for(int i=0; i<fixed_rate[j]; i++)
            {
                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                cmp_byte_ofs+=4;

                absQuant[0] |= ((tmp_char.x >> 7) & 0x00000001) << i;
                absQuant[1] |= ((tmp_char.x >> 6) & 0x00000001) << i;
                absQuant[2] |= ((tmp_char.x >> 5) & 0x00000001) << i;
                absQuant[3] |= ((tmp_char.x >> 4) & 0x00000001) << i;
                absQuant[4] |= ((tmp_char.x >> 3) & 0x00000001) << i;
                absQuant[5] |= ((tmp_char.x >> 2) & 0x00000001) << i;
                absQuant[6] |= ((tmp_char.x >> 1) & 0x00000001) << i;
                absQuant[7] |= ((tmp_char.x >> 0) & 0x00000001) << i;

                absQuant[8] |= ((tmp_char.y >> 7) & 0x00000001) << i;
                absQuant[9] |= ((tmp_char.y >> 6) & 0x00000001) << i;
                absQuant[10] |= ((tmp_char.y >> 5) & 0x00000001) << i;
                absQuant[11] |= ((tmp_char.y >> 4) & 0x00000001) << i;
                absQuant[12] |= ((tmp_char.y >> 3) & 0x00000001) << i;
                absQuant[13] |= ((tmp_char.y >> 2) & 0x00000001) << i;
                absQuant[14] |= ((tmp_char.y >> 1) & 0x00000001) << i;
                absQuant[15] |= ((tmp_char.y >> 0) & 0x00000001) << i;

                absQuant[16] |= ((tmp_char.z >> 7) & 0x00000001) << i;
                absQuant[17] |= ((tmp_char.z >> 6) & 0x00000001) << i;
                absQuant[18] |= ((tmp_char.z >> 5) & 0x00000001) << i;
                absQuant[19] |= ((tmp_char.z >> 4) & 0x00000001) << i;
                absQuant[20] |= ((tmp_char.z >> 3) & 0x00000001) << i;
                absQuant[21] |= ((tmp_char.z >> 2) & 0x00000001) << i;
                absQuant[22] |= ((tmp_char.z >> 1) & 0x00000001) << i;
                absQuant[23] |= ((tmp_char.z >> 0) & 0x00000001) << i;

                absQuant[24] |= ((tmp_char.w >> 7) & 0x00000001) << i;
                absQuant[25] |= ((tmp_char.w >> 6) & 0x00000001) << i;
                absQuant[26] |= ((tmp_char.w >> 5) & 0x00000001) << i;
                absQuant[27] |= ((tmp_char.w >> 4) & 0x00000001) << i;
                absQuant[28] |= ((tmp_char.w >> 3) & 0x00000001) << i;
                absQuant[29] |= ((tmp_char.w >> 2) & 0x00000001) << i;
                absQuant[30] |= ((tmp_char.w >> 1) & 0x00000001) << i;
                absQuant[31] |= ((tmp_char.w >> 0) & 0x00000001) << i;
            }
            
            if(base_block_end_idx < nbEle)
            {
                #pragma unroll 8
                for(int i=0; i<32; i+=4)
                {
                    sign_ofs = i % 32;
                    currQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                    dec_buffer.x = currQuant * eb * 2;

                    sign_ofs = (i+1) % 32;
                    currQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+1] * -1 : absQuant[i+1];
                    dec_buffer.y = currQuant * eb * 2;

                    sign_ofs = (i+2) % 32;
                    currQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+2] * -1 : absQuant[i+2];
                    dec_buffer.z = currQuant * eb * 2;

                    sign_ofs = (i+3) % 32;
                    currQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+3] * -1 : absQuant[i+3];
                    dec_buffer.w = currQuant * eb * 2;
                    
                    reinterpret_cast<float4*>(decData)[(base_block_start_idx+i)/4] = dec_buffer;
                }
            }
            else
            {
                for(int i=0; i<32; i++)
                {
                    sign_ofs = i % 32;
                    currQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                    if(base_block_start_idx+i < nbEle) decData[base_block_start_idx+i] = currQuant * eb * 2;
                }
            }      
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}


__global__ void cuSZp_compress_kernel_1D_plain_f32(const float* const __restrict__ oriData, 
                                                unsigned char* const __restrict__ cmpData, 
                                                volatile unsigned int* const __restrict__ cmpOffset, 
                                                volatile unsigned int* const __restrict__ locOffset, 
                                                volatile int* const __restrict__ flag, 
                                                const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = thread_chunk >> 5;
    const int rate_ofs = (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;
    const float recipPrecision = 0.5f/eb;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    int block_idx;
    int currQuant, lorenQuant, prevQuant, maxQuant;
    int absQuant[thread_chunk];
    unsigned int sign_flag[block_num];
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0; // Thread-level prefix-sum, double check for overflow in large data (can be resolved by using size_t type).
    float4 tmp_buffer;
    uchar4 tmp_char;

    base_start_idx = warp * thread_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        sign_flag[j] = 0;
        block_idx = base_block_start_idx/32;
        prevQuant = 0;
        maxQuant = 0;

        if(base_block_end_idx < nbEle)
        {
            #pragma unroll 8
            for(int i=base_block_start_idx; i<base_block_end_idx; i+=4)
            {
                tmp_buffer = reinterpret_cast<const float4*>(oriData)[i/4];
                quant_chunk_idx = j * 32 + i % 32;

                currQuant = quantization(tmp_buffer.x, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = i % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];

                currQuant = quantization(tmp_buffer.y, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+1) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+1] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+1] ? maxQuant : absQuant[quant_chunk_idx+1];

                currQuant = quantization(tmp_buffer.z, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+2) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+2] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+2] ? maxQuant : absQuant[quant_chunk_idx+2];

                currQuant = quantization(tmp_buffer.w, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+3) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+3] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+3] ? maxQuant : absQuant[quant_chunk_idx+3];
            }
        }
        else
        {
            if(base_block_start_idx >= nbEle)
            {
                quant_chunk_idx = j * 32 + base_block_start_idx % 32;
                for(int i=quant_chunk_idx; i<quant_chunk_idx+32; i++) absQuant[i] = 0;
            }
            else
            {
                int remainbEle = nbEle - base_block_start_idx;
                int zeronbEle = base_block_end_idx - nbEle;

                for(int i=base_block_start_idx; i<base_block_start_idx+remainbEle; i++)
                {
                    quant_chunk_idx = j * 32 + i % 32;
                    currQuant = quantization(oriData[i], recipPrecision);
                    lorenQuant = currQuant - prevQuant;
                    prevQuant = currQuant;
                    sign_ofs = i % 32;
                    sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                    absQuant[quant_chunk_idx] = abs(lorenQuant);
                    maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
                }

                quant_chunk_idx = j * 32 + nbEle % 32;
                for(int i=quant_chunk_idx; i<quant_chunk_idx+zeronbEle; i++) absQuant[i] = 0;
            }  
        }

        fixed_rate[j] = get_bit_num(maxQuant);
        thread_ofs += (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        cmpData[block_idx] = (unsigned char)fixed_rate[j];
        __syncthreads();
    }

    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    for(int j=0; j<block_num; j++)
    {
        int chunk_idx_start = j*32;

        tmp_byte_ofs = (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(fixed_rate[j])
        {
            tmp_char.x = 0xff & (sign_flag[j] >> 24);
            tmp_char.y = 0xff & (sign_flag[j] >> 16);
            tmp_char.z = 0xff & (sign_flag[j] >> 8);
            tmp_char.w = 0xff & sign_flag[j];
            reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
            cmp_byte_ofs+=4;

            int mask = 1;
            for(int i=0; i<fixed_rate[j]; i++)
            {
                tmp_char.x = 0;
                tmp_char.y = 0;
                tmp_char.z = 0;
                tmp_char.w = 0;

                tmp_char.x = (((absQuant[chunk_idx_start+0] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+1] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+2] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+3] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+4] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+5] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+6] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+7] & mask) >> i) << 0);

                tmp_char.y = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                tmp_char.z = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                
                tmp_char.w = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                cmp_byte_ofs+=4;
                mask <<= 1;
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}


__global__ void cuSZp_decompress_kernel_1D_plain_f32(float* const __restrict__ decData, 
                                                  const unsigned char* const __restrict__ cmpData, 
                                                  volatile unsigned int* const __restrict__ cmpOffset, 
                                                  volatile unsigned int* const __restrict__ locOffset, 
                                                  volatile int* const __restrict__ flag, 
                                                  const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = thread_chunk >> 5;
    const int rate_ofs = (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int block_idx;    
    int absQuant[32];
    int currQuant, lorenQuant, prevQuant;
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char;
    float4 dec_buffer;

    for(int j=0; j<block_num; j++)
    {
        block_idx = warp * thread_chunk + j * 32 + lane;
        fixed_rate[j] = (int)cmpData[block_idx];
        thread_ofs += (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        __syncthreads();
    }

    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        if(!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if(!lane) flag[warp] = 2;
        __threadfence();  
    }
    __syncthreads();

    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    base_start_idx = warp * thread_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        unsigned int sign_flag = 0;

        tmp_byte_ofs = (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(fixed_rate[j])
        {
            tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
            sign_flag = (0xff000000 & (tmp_char.x << 24)) |
                        (0x00ff0000 & (tmp_char.y << 16)) |
                        (0x0000ff00 & (tmp_char.z << 8))  |
                        (0x000000ff & tmp_char.w);
            cmp_byte_ofs+=4;
            
            for(int i=0; i<32; i++) absQuant[i] = 0;
            for(int i=0; i<fixed_rate[j]; i++)
            {
                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                cmp_byte_ofs+=4;

                absQuant[0] |= ((tmp_char.x >> 7) & 0x00000001) << i;
                absQuant[1] |= ((tmp_char.x >> 6) & 0x00000001) << i;
                absQuant[2] |= ((tmp_char.x >> 5) & 0x00000001) << i;
                absQuant[3] |= ((tmp_char.x >> 4) & 0x00000001) << i;
                absQuant[4] |= ((tmp_char.x >> 3) & 0x00000001) << i;
                absQuant[5] |= ((tmp_char.x >> 2) & 0x00000001) << i;
                absQuant[6] |= ((tmp_char.x >> 1) & 0x00000001) << i;
                absQuant[7] |= ((tmp_char.x >> 0) & 0x00000001) << i;

                absQuant[8] |= ((tmp_char.y >> 7) & 0x00000001) << i;
                absQuant[9] |= ((tmp_char.y >> 6) & 0x00000001) << i;
                absQuant[10] |= ((tmp_char.y >> 5) & 0x00000001) << i;
                absQuant[11] |= ((tmp_char.y >> 4) & 0x00000001) << i;
                absQuant[12] |= ((tmp_char.y >> 3) & 0x00000001) << i;
                absQuant[13] |= ((tmp_char.y >> 2) & 0x00000001) << i;
                absQuant[14] |= ((tmp_char.y >> 1) & 0x00000001) << i;
                absQuant[15] |= ((tmp_char.y >> 0) & 0x00000001) << i;

                absQuant[16] |= ((tmp_char.z >> 7) & 0x00000001) << i;
                absQuant[17] |= ((tmp_char.z >> 6) & 0x00000001) << i;
                absQuant[18] |= ((tmp_char.z >> 5) & 0x00000001) << i;
                absQuant[19] |= ((tmp_char.z >> 4) & 0x00000001) << i;
                absQuant[20] |= ((tmp_char.z >> 3) & 0x00000001) << i;
                absQuant[21] |= ((tmp_char.z >> 2) & 0x00000001) << i;
                absQuant[22] |= ((tmp_char.z >> 1) & 0x00000001) << i;
                absQuant[23] |= ((tmp_char.z >> 0) & 0x00000001) << i;

                absQuant[24] |= ((tmp_char.w >> 7) & 0x00000001) << i;
                absQuant[25] |= ((tmp_char.w >> 6) & 0x00000001) << i;
                absQuant[26] |= ((tmp_char.w >> 5) & 0x00000001) << i;
                absQuant[27] |= ((tmp_char.w >> 4) & 0x00000001) << i;
                absQuant[28] |= ((tmp_char.w >> 3) & 0x00000001) << i;
                absQuant[29] |= ((tmp_char.w >> 2) & 0x00000001) << i;
                absQuant[30] |= ((tmp_char.w >> 1) & 0x00000001) << i;
                absQuant[31] |= ((tmp_char.w >> 0) & 0x00000001) << i;
            }
            
            prevQuant = 0;
            if(base_block_end_idx < nbEle)
            {
                #pragma unroll 8
                for(int i=0; i<32; i+=4)
                {
                    sign_ofs = i % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.x = currQuant * eb * 2;

                    sign_ofs = (i+1) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+1] * -1 : absQuant[i+1];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.y = currQuant * eb * 2;

                    sign_ofs = (i+2) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+2] * -1 : absQuant[i+2];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.z = currQuant * eb * 2;

                    sign_ofs = (i+3) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+3] * -1 : absQuant[i+3];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.w = currQuant * eb * 2;
                    
                    reinterpret_cast<float4*>(decData)[(base_block_start_idx+i)/4] = dec_buffer;
                }
            }
            else
            {
                for(int i=0; i<32; i++)
                {
                    sign_ofs = i % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    if(base_block_start_idx+i < nbEle) decData[base_block_start_idx+i] = currQuant * eb * 2;
                }
            }      
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}


__global__ void cuSZp_compress_kernel_1D_outlier_f32(const float* const __restrict__ oriData, 
                                                    unsigned char* const __restrict__ cmpData, 
                                                    volatile unsigned int* const __restrict__ cmpOffset, 
                                                    volatile unsigned int* const __restrict__ locOffset, 
                                                    volatile int* const __restrict__ flag, 
                                                    const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = thread_chunk >> 5;
    const int rate_ofs = (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;
    const float recipPrecision = 0.5f/eb;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    int block_idx;
    int currQuant, lorenQuant, prevQuant;
    int absQuant[thread_chunk];
    unsigned int sign_flag[block_num];
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0; // Thread-level prefix-sum, double check for overflow in large data (can be resolved by using size_t type).
    float4 tmp_buffer;
    uchar4 tmp_char;

    base_start_idx = warp * thread_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        sign_flag[j] = 0;
        fixed_rate[j] = 0;
        block_idx = base_block_start_idx/32;
        prevQuant = 0;
        int maxQuant = 0;
        int maxQuan2 = 0;
        int outlier = 0;

        if(base_block_end_idx < nbEle)
        {
            #pragma unroll 8
            for(int i=base_block_start_idx; i<base_block_end_idx; i+=4)
            {
                tmp_buffer = reinterpret_cast<const float4*>(oriData)[i/4];
                quant_chunk_idx = j * 32 + i % 32;

                currQuant = quantization(tmp_buffer.x, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = i % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
                if(sign_ofs) maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx] ? maxQuan2 : absQuant[quant_chunk_idx];
                else outlier = absQuant[quant_chunk_idx];

                currQuant = quantization(tmp_buffer.y, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+1) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+1] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+1] ? maxQuant : absQuant[quant_chunk_idx+1];
                maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx+1] ? maxQuan2 : absQuant[quant_chunk_idx+1];

                currQuant = quantization(tmp_buffer.z, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+2) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+2] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+2] ? maxQuant : absQuant[quant_chunk_idx+2];
                maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx+2] ? maxQuan2 : absQuant[quant_chunk_idx+2];

                currQuant = quantization(tmp_buffer.w, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+3) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+3] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+3] ? maxQuant : absQuant[quant_chunk_idx+3];
                maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx+3] ? maxQuan2 : absQuant[quant_chunk_idx+3];
            }
        }
        else
        {
            if(base_block_start_idx >= nbEle)
            {
                quant_chunk_idx = j * 32 + base_block_start_idx % 32;
                for(int i=quant_chunk_idx; i<quant_chunk_idx+32; i++) absQuant[i] = 0;
            }
            else
            {
                int remainbEle = nbEle - base_block_start_idx;
                int zeronbEle = base_block_end_idx - nbEle;

                for(int i=base_block_start_idx; i<base_block_start_idx+remainbEle; i++)
                {
                    quant_chunk_idx = j * 32 + i % 32;
                    currQuant = quantization(oriData[i], recipPrecision);
                    lorenQuant = currQuant - prevQuant;
                    prevQuant = currQuant;
                    sign_ofs = i % 32;
                    sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                    absQuant[quant_chunk_idx] = abs(lorenQuant);
                    maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
                    if(sign_ofs) maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx] ? maxQuan2 : absQuant[quant_chunk_idx];
                    else outlier = absQuant[quant_chunk_idx];
                }

                quant_chunk_idx = j * 32 + nbEle % 32;
                for(int i=quant_chunk_idx; i<quant_chunk_idx+zeronbEle; i++) absQuant[i] = 0;
            }  
        }

        int fr1 = get_bit_num(maxQuant);
        int fr2 = get_bit_num(maxQuan2);
        outlier = (get_bit_num(outlier)+7)/8;
        int temp_rate = 0;
        int temp_ofs1 = fr1 ? 4 + fr1 * 4 : 0;
        int temp_ofs2 = fr2 ? 4 + fr2 * 4 + outlier: 4 + outlier;
        if(temp_ofs1<=temp_ofs2) 
        {
            thread_ofs += temp_ofs1;
            temp_rate = fr1;
        }
        else 
        {
            thread_ofs += temp_ofs2;
            temp_rate = fr2 | 0x80 | ((outlier-1) << 5);
        }

        fixed_rate[j] = temp_rate;
        cmpData[block_idx] = (unsigned char)fixed_rate[j];
        __syncthreads();
    }

    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        } 
    }
    __syncthreads();
    
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    for(int j=0; j<block_num; j++)
    {
        int encoding_selection = fixed_rate[j] >> 7;
        int outlier_byte_num = ((fixed_rate[j] & 0x60) >> 5) + 1;
        fixed_rate[j] &= 0x1f;
        int chunk_idx_start = j*32;

        if(!encoding_selection) tmp_byte_ofs = (fixed_rate[j]) ? (4 + fixed_rate[j] * 4) : 0;
        else tmp_byte_ofs = 4 + outlier_byte_num + fixed_rate[j] * 4;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(encoding_selection)
        {
            for(int i=0; i<outlier_byte_num; i++)
            {
                cmpData[cmp_byte_ofs++] = (unsigned char)(absQuant[chunk_idx_start] & 0xff);
                absQuant[chunk_idx_start] >>= 8;
            }

            if(!fixed_rate[j])
            {
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 16);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 8);
                cmpData[cmp_byte_ofs++] = 0xff & sign_flag[j];
            }
        }

        if(fixed_rate[j])
        {
            int vec_ofs = cmp_byte_ofs % 4;
            if(vec_ofs==0)
            {
                tmp_char.x = 0xff & (sign_flag[j] >> 24);
                tmp_char.y = 0xff & (sign_flag[j] >> 16);
                tmp_char.z = 0xff & (sign_flag[j] >> 8);
                tmp_char.w = 0xff & sign_flag[j];
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                cmp_byte_ofs+=4;

                int mask = 1;
                for(int i=0; i<fixed_rate[j]; i++)
                {
                    tmp_char.x = 0;
                    tmp_char.y = 0;
                    tmp_char.z = 0;
                    tmp_char.w = 0;

                    if(!encoding_selection) tmp_char.x = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
                    tmp_char.x = tmp_char.x | 
                                (((absQuant[chunk_idx_start+0] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+1] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+2] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+3] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+4] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+5] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+6] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+7] & mask) >> i) << 0);

                    tmp_char.y = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                    tmp_char.z = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                    
                    tmp_char.w = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                    cmp_byte_ofs+=4;
                    mask <<= 1;
                }
            }
            else if(vec_ofs==1)
            {
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 16);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 8);

                tmp_char.x = 0;
                tmp_char.y = 0;
                tmp_char.z = 0;
                tmp_char.w = 0;
                tmp_char.x = 0xff & sign_flag[j];
                if(!encoding_selection) tmp_char.y = ((absQuant[chunk_idx_start] & 1) << 7);
                tmp_char.y = tmp_char.y | 
                            ((absQuant[chunk_idx_start+1] & 1) << 6) |
                            ((absQuant[chunk_idx_start+2] & 1) << 5) |
                            ((absQuant[chunk_idx_start+3] & 1) << 4) |
                            ((absQuant[chunk_idx_start+4] & 1) << 3) |
                            ((absQuant[chunk_idx_start+5] & 1) << 2) |
                            ((absQuant[chunk_idx_start+6] & 1) << 1) |
                            ((absQuant[chunk_idx_start+7] & 1) << 0);
                tmp_char.z = ((absQuant[chunk_idx_start+8] & 1) << 7) |
                            ((absQuant[chunk_idx_start+9] & 1) << 6) |
                            ((absQuant[chunk_idx_start+10] & 1) << 5) |
                            ((absQuant[chunk_idx_start+11] & 1) << 4) |
                            ((absQuant[chunk_idx_start+12] & 1) << 3) |
                            ((absQuant[chunk_idx_start+13] & 1) << 2) |
                            ((absQuant[chunk_idx_start+14] & 1) << 1) |
                            ((absQuant[chunk_idx_start+15] & 1) << 0);
                tmp_char.w = ((absQuant[chunk_idx_start+16] & 1) << 7) |
                            ((absQuant[chunk_idx_start+17] & 1) << 6) |
                            ((absQuant[chunk_idx_start+18] & 1) << 5) |
                            ((absQuant[chunk_idx_start+19] & 1) << 4) |
                            ((absQuant[chunk_idx_start+20] & 1) << 3) |
                            ((absQuant[chunk_idx_start+21] & 1) << 2) |
                            ((absQuant[chunk_idx_start+22] & 1) << 1) |
                            ((absQuant[chunk_idx_start+23] & 1) << 0);
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                cmp_byte_ofs+=4;

                int mask = 1;
                for(int i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char.x = 0;
                    tmp_char.y = 0;
                    tmp_char.z = 0;
                    tmp_char.w = 0;
                    
                    tmp_char.x = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);
                    mask <<= 1;

                    if(!encoding_selection) tmp_char.y = (((absQuant[chunk_idx_start] & mask) >> (i+1)) << 7);
                    tmp_char.y = tmp_char.y | 
                                (((absQuant[chunk_idx_start+0] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+1] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+2] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+3] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+4] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+5] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+6] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+7] & mask) >> (i+1)) << 0);

                    tmp_char.z = (((absQuant[chunk_idx_start+8] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+9] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+10] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+11] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+12] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+13] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+14] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+15] & mask) >> (i+1)) << 0);

                    tmp_char.w = (((absQuant[chunk_idx_start+16] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+17] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+18] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+19] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+20] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+21] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+22] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+23] & mask) >> (i+1)) << 0);

                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                    cmp_byte_ofs+=4;
                }

                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+24] & mask) >> (fixed_rate[j]-1)) << 7) |
                                        (((absQuant[chunk_idx_start+25] & mask) >> (fixed_rate[j]-1)) << 6) |
                                        (((absQuant[chunk_idx_start+26] & mask) >> (fixed_rate[j]-1)) << 5) |
                                        (((absQuant[chunk_idx_start+27] & mask) >> (fixed_rate[j]-1)) << 4) |
                                        (((absQuant[chunk_idx_start+28] & mask) >> (fixed_rate[j]-1)) << 3) |
                                        (((absQuant[chunk_idx_start+29] & mask) >> (fixed_rate[j]-1)) << 2) |
                                        (((absQuant[chunk_idx_start+30] & mask) >> (fixed_rate[j]-1)) << 1) |
                                        (((absQuant[chunk_idx_start+31] & mask) >> (fixed_rate[j]-1)) << 0);
            }
            else if(vec_ofs==2)
            {
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 16);

                tmp_char.x = 0;
                tmp_char.y = 0;
                tmp_char.z = 0;
                tmp_char.w = 0;
                tmp_char.x = 0xff & (sign_flag[j] >> 8);
                tmp_char.y = 0xff & sign_flag[j];
                if(!encoding_selection) tmp_char.z = ((absQuant[chunk_idx_start] & 1) << 7);
                tmp_char.z = tmp_char.z | 
                            ((absQuant[chunk_idx_start+1] & 1) << 6) |
                            ((absQuant[chunk_idx_start+2] & 1) << 5) |
                            ((absQuant[chunk_idx_start+3] & 1) << 4) |
                            ((absQuant[chunk_idx_start+4] & 1) << 3) |
                            ((absQuant[chunk_idx_start+5] & 1) << 2) |
                            ((absQuant[chunk_idx_start+6] & 1) << 1) |
                            ((absQuant[chunk_idx_start+7] & 1) << 0);
                tmp_char.w = ((absQuant[chunk_idx_start+8] & 1) << 7) |
                            ((absQuant[chunk_idx_start+9] & 1) << 6) |
                            ((absQuant[chunk_idx_start+10] & 1) << 5) |
                            ((absQuant[chunk_idx_start+11] & 1) << 4) |
                            ((absQuant[chunk_idx_start+12] & 1) << 3) |
                            ((absQuant[chunk_idx_start+13] & 1) << 2) |
                            ((absQuant[chunk_idx_start+14] & 1) << 1) |
                            ((absQuant[chunk_idx_start+15] & 1) << 0);
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                cmp_byte_ofs+=4;

                int mask = 1;
                for(int i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char.x = 0;
                    tmp_char.y = 0;
                    tmp_char.z = 0;
                    tmp_char.w = 0;

                    tmp_char.x = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                    
                    tmp_char.y = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);
                    mask <<= 1;

                    if(!encoding_selection) tmp_char.z = (((absQuant[chunk_idx_start] & mask) >> (i+1)) << 7);
                    tmp_char.z = tmp_char.z | 
                                (((absQuant[chunk_idx_start+0] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+1] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+2] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+3] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+4] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+5] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+6] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+7] & mask) >> (i+1)) << 0);

                    tmp_char.w = (((absQuant[chunk_idx_start+8] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+9] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+10] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+11] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+12] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+13] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+14] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+15] & mask) >> (i+1)) << 0);

                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                    cmp_byte_ofs+=4;
                }

                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+16] & mask) >> (fixed_rate[j]-1)) << 7) |
                                        (((absQuant[chunk_idx_start+17] & mask) >> (fixed_rate[j]-1)) << 6) |
                                        (((absQuant[chunk_idx_start+18] & mask) >> (fixed_rate[j]-1)) << 5) |
                                        (((absQuant[chunk_idx_start+19] & mask) >> (fixed_rate[j]-1)) << 4) |
                                        (((absQuant[chunk_idx_start+20] & mask) >> (fixed_rate[j]-1)) << 3) |
                                        (((absQuant[chunk_idx_start+21] & mask) >> (fixed_rate[j]-1)) << 2) |
                                        (((absQuant[chunk_idx_start+22] & mask) >> (fixed_rate[j]-1)) << 1) |
                                        (((absQuant[chunk_idx_start+23] & mask) >> (fixed_rate[j]-1)) << 0);
                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+24] & mask) >> (fixed_rate[j]-1)) << 7) |
                                        (((absQuant[chunk_idx_start+25] & mask) >> (fixed_rate[j]-1)) << 6) |
                                        (((absQuant[chunk_idx_start+26] & mask) >> (fixed_rate[j]-1)) << 5) |
                                        (((absQuant[chunk_idx_start+27] & mask) >> (fixed_rate[j]-1)) << 4) |
                                        (((absQuant[chunk_idx_start+28] & mask) >> (fixed_rate[j]-1)) << 3) |
                                        (((absQuant[chunk_idx_start+29] & mask) >> (fixed_rate[j]-1)) << 2) |
                                        (((absQuant[chunk_idx_start+30] & mask) >> (fixed_rate[j]-1)) << 1) |
                                        (((absQuant[chunk_idx_start+31] & mask) >> (fixed_rate[j]-1)) << 0);
            }
            else
            {
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);

                tmp_char.x = 0;
                tmp_char.y = 0;
                tmp_char.z = 0;
                tmp_char.w = 0;
                tmp_char.x = 0xff & (sign_flag[j] >> 16);
                tmp_char.y = 0xff & (sign_flag[j] >> 8);
                tmp_char.z = 0xff & sign_flag[j];
                if(!encoding_selection) tmp_char.w = ((absQuant[chunk_idx_start] & 1) << 7);
                tmp_char.w = tmp_char.w | 
                            ((absQuant[chunk_idx_start+1] & 1) << 6) |
                            ((absQuant[chunk_idx_start+2] & 1) << 5) |
                            ((absQuant[chunk_idx_start+3] & 1) << 4) |
                            ((absQuant[chunk_idx_start+4] & 1) << 3) |
                            ((absQuant[chunk_idx_start+5] & 1) << 2) |
                            ((absQuant[chunk_idx_start+6] & 1) << 1) |
                            ((absQuant[chunk_idx_start+7] & 1) << 0);
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                cmp_byte_ofs+=4;

                int mask = 1;
                for(int i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char.x = 0;
                    tmp_char.y = 0;
                    tmp_char.z = 0;
                    tmp_char.w = 0;

                    tmp_char.x = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                    tmp_char.y = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                    
                    tmp_char.z = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);
                    mask <<= 1;

                    if(!encoding_selection) tmp_char.w = (((absQuant[chunk_idx_start] & mask) >> (i+1)) << 7);
                    tmp_char.w = (((absQuant[chunk_idx_start+0] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+1] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+2] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+3] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+4] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+5] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+6] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+7] & mask) >> (i+1)) << 0);

                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                    cmp_byte_ofs+=4;
                }

                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+8] & mask) >> (fixed_rate[j]-1)) << 7) |
                                        (((absQuant[chunk_idx_start+9] & mask) >> (fixed_rate[j]-1)) << 6) |
                                        (((absQuant[chunk_idx_start+10] & mask) >> (fixed_rate[j]-1)) << 5) |
                                        (((absQuant[chunk_idx_start+11] & mask) >> (fixed_rate[j]-1)) << 4) |
                                        (((absQuant[chunk_idx_start+12] & mask) >> (fixed_rate[j]-1)) << 3) |
                                        (((absQuant[chunk_idx_start+13] & mask) >> (fixed_rate[j]-1)) << 2) |
                                        (((absQuant[chunk_idx_start+14] & mask) >> (fixed_rate[j]-1)) << 1) |
                                        (((absQuant[chunk_idx_start+15] & mask) >> (fixed_rate[j]-1)) << 0);
                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+16] & mask) >> (fixed_rate[j]-1)) << 7) |
                                        (((absQuant[chunk_idx_start+17] & mask) >> (fixed_rate[j]-1)) << 6) |
                                        (((absQuant[chunk_idx_start+18] & mask) >> (fixed_rate[j]-1)) << 5) |
                                        (((absQuant[chunk_idx_start+19] & mask) >> (fixed_rate[j]-1)) << 4) |
                                        (((absQuant[chunk_idx_start+20] & mask) >> (fixed_rate[j]-1)) << 3) |
                                        (((absQuant[chunk_idx_start+21] & mask) >> (fixed_rate[j]-1)) << 2) |
                                        (((absQuant[chunk_idx_start+22] & mask) >> (fixed_rate[j]-1)) << 1) |
                                        (((absQuant[chunk_idx_start+23] & mask) >> (fixed_rate[j]-1)) << 0);
                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+24] & mask) >> (fixed_rate[j]-1)) << 7) |
                                        (((absQuant[chunk_idx_start+25] & mask) >> (fixed_rate[j]-1)) << 6) |
                                        (((absQuant[chunk_idx_start+26] & mask) >> (fixed_rate[j]-1)) << 5) |
                                        (((absQuant[chunk_idx_start+27] & mask) >> (fixed_rate[j]-1)) << 4) |
                                        (((absQuant[chunk_idx_start+28] & mask) >> (fixed_rate[j]-1)) << 3) |
                                        (((absQuant[chunk_idx_start+29] & mask) >> (fixed_rate[j]-1)) << 2) |
                                        (((absQuant[chunk_idx_start+30] & mask) >> (fixed_rate[j]-1)) << 1) |
                                        (((absQuant[chunk_idx_start+31] & mask) >> (fixed_rate[j]-1)) << 0);
            }
        }
        
        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}


__global__ void cuSZp_decompress_kernel_1D_outlier_f32(float* const __restrict__ decData, 
                                                    const unsigned char* const __restrict__ cmpData, 
                                                    volatile unsigned int* const __restrict__ cmpOffset, 
                                                    volatile unsigned int* const __restrict__ locOffset, 
                                                    volatile int* const __restrict__ flag, 
                                                    const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = thread_chunk >> 5;
    const int rate_ofs = (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int block_idx;    
    int absQuant[32];
    int currQuant, lorenQuant, prevQuant;
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char;
    float4 dec_buffer;

    for(int j=0; j<block_num; j++)
    {
        block_idx = warp * thread_chunk + j * 32 + lane;
        fixed_rate[j] = (int)cmpData[block_idx];

        int encoding_selection = fixed_rate[j] >> 7;
        int outlier = ((fixed_rate[j] & 0x60) >> 5) + 1;
        int temp_rate = fixed_rate[j] & 0x1f;
        if(!encoding_selection) thread_ofs += temp_rate ? (4 + temp_rate * 4) : 0;
        else thread_ofs += 4 + temp_rate * 4 + outlier;
        __syncthreads();
    }

    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        if(!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if(!lane) flag[warp] = 2;
        __threadfence();  
    }
    __syncthreads();

    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    base_start_idx = warp * thread_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        int encoding_selection = fixed_rate[j] >> 7;
        int outlier_byte_num = ((fixed_rate[j] & 0x60) >> 5) + 1;
        fixed_rate[j] &= 0x1f;
        int outlier_buffer = 0;
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        unsigned int sign_flag = 0;

        if(!encoding_selection) tmp_byte_ofs = (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        else tmp_byte_ofs = 4 + outlier_byte_num + fixed_rate[j] * 4;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(encoding_selection)
        {
            for(int i=0; i<outlier_byte_num; i++)
            {
                int buffer = cmpData[cmp_byte_ofs++] << (8*i);
                outlier_buffer |= buffer;
            }

            if(!fixed_rate[j])
            {
                sign_flag = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                            (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16)) |
                            (0x0000ff00 & (cmpData[cmp_byte_ofs++] << 8))  |
                            (0x000000ff & cmpData[cmp_byte_ofs++]);
                absQuant[0] = outlier_buffer;
                for(int i=1; i<32; i++) absQuant[i] = 0;

                prevQuant = 0;
                if(base_block_end_idx < nbEle)
                {
                    #pragma unroll 8
                    for(int i=0; i<32; i+=4)
                    {
                        sign_ofs = i % 32;
                        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                        currQuant = lorenQuant + prevQuant;
                        prevQuant = currQuant;
                        dec_buffer.x = currQuant * eb * 2;

                        sign_ofs = (i+1) % 32;
                        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+1] * -1 : absQuant[i+1];
                        currQuant = lorenQuant + prevQuant;
                        prevQuant = currQuant;
                        dec_buffer.y = currQuant * eb * 2;

                        sign_ofs = (i+2) % 32;
                        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+2] * -1 : absQuant[i+2];
                        currQuant = lorenQuant + prevQuant;
                        prevQuant = currQuant;
                        dec_buffer.z = currQuant * eb * 2;

                        sign_ofs = (i+3) % 32;
                        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+3] * -1 : absQuant[i+3];
                        currQuant = lorenQuant + prevQuant;
                        prevQuant = currQuant;
                        dec_buffer.w = currQuant * eb * 2;
                        
                        reinterpret_cast<float4*>(decData)[(base_block_start_idx+i)/4] = dec_buffer;
                    }
                }
                else
                {
                    for(int i=0; i<32; i++)
                    {
                        sign_ofs = i % 32;
                        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                        currQuant = lorenQuant + prevQuant;
                        prevQuant = currQuant;
                        if(base_block_start_idx+i < nbEle) decData[base_block_start_idx+i] = currQuant * eb * 2;
                    }
                }
            }
        }

        if(fixed_rate[j])
        {
            int vec_ofs = cmp_byte_ofs % 4;
            if(vec_ofs==0)
            {
                for(int i=0; i<32; i++) absQuant[i] = 0;

                if(encoding_selection) absQuant[0] = outlier_buffer;

                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                sign_flag = (0xff000000 & (tmp_char.x << 24)) |
                            (0x00ff0000 & (tmp_char.y << 16)) |
                            (0x0000ff00 & (tmp_char.z << 8))  |
                            (0x000000ff & tmp_char.w);
                cmp_byte_ofs+=4;

                for(int i=0; i<fixed_rate[j]; i++)
                {
                    tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                    cmp_byte_ofs+=4;

                    if(!encoding_selection) absQuant[0] |= ((tmp_char.x >> 7) & 0x00000001) << i;
                    absQuant[1] |= ((tmp_char.x >> 6) & 0x00000001) << i;
                    absQuant[2] |= ((tmp_char.x >> 5) & 0x00000001) << i;
                    absQuant[3] |= ((tmp_char.x >> 4) & 0x00000001) << i;
                    absQuant[4] |= ((tmp_char.x >> 3) & 0x00000001) << i;
                    absQuant[5] |= ((tmp_char.x >> 2) & 0x00000001) << i;
                    absQuant[6] |= ((tmp_char.x >> 1) & 0x00000001) << i;
                    absQuant[7] |= ((tmp_char.x >> 0) & 0x00000001) << i;

                    absQuant[8] |= ((tmp_char.y >> 7) & 0x00000001) << i;
                    absQuant[9] |= ((tmp_char.y >> 6) & 0x00000001) << i;
                    absQuant[10] |= ((tmp_char.y >> 5) & 0x00000001) << i;
                    absQuant[11] |= ((tmp_char.y >> 4) & 0x00000001) << i;
                    absQuant[12] |= ((tmp_char.y >> 3) & 0x00000001) << i;
                    absQuant[13] |= ((tmp_char.y >> 2) & 0x00000001) << i;
                    absQuant[14] |= ((tmp_char.y >> 1) & 0x00000001) << i;
                    absQuant[15] |= ((tmp_char.y >> 0) & 0x00000001) << i;

                    absQuant[16] |= ((tmp_char.z >> 7) & 0x00000001) << i;
                    absQuant[17] |= ((tmp_char.z >> 6) & 0x00000001) << i;
                    absQuant[18] |= ((tmp_char.z >> 5) & 0x00000001) << i;
                    absQuant[19] |= ((tmp_char.z >> 4) & 0x00000001) << i;
                    absQuant[20] |= ((tmp_char.z >> 3) & 0x00000001) << i;
                    absQuant[21] |= ((tmp_char.z >> 2) & 0x00000001) << i;
                    absQuant[22] |= ((tmp_char.z >> 1) & 0x00000001) << i;
                    absQuant[23] |= ((tmp_char.z >> 0) & 0x00000001) << i;

                    absQuant[24] |= ((tmp_char.w >> 7) & 0x00000001) << i;
                    absQuant[25] |= ((tmp_char.w >> 6) & 0x00000001) << i;
                    absQuant[26] |= ((tmp_char.w >> 5) & 0x00000001) << i;
                    absQuant[27] |= ((tmp_char.w >> 4) & 0x00000001) << i;
                    absQuant[28] |= ((tmp_char.w >> 3) & 0x00000001) << i;
                    absQuant[29] |= ((tmp_char.w >> 2) & 0x00000001) << i;
                    absQuant[30] |= ((tmp_char.w >> 1) & 0x00000001) << i;
                    absQuant[31] |= ((tmp_char.w >> 0) & 0x00000001) << i;
                }
            }
            else if(vec_ofs==1)
            {
                for(int i=0; i<32; i++) absQuant[i] = 0;

                if(encoding_selection) absQuant[0] = outlier_buffer;

                sign_flag = (0xff000000 & cmpData[cmp_byte_ofs++] << 24) |
                            (0x00ff0000 & cmpData[cmp_byte_ofs++] << 16) |
                            (0x0000ff00 & cmpData[cmp_byte_ofs++] << 8);

                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                sign_flag |= (0x000000ff & tmp_char.x);

                if(!encoding_selection) absQuant[0] |= ((tmp_char.y >> 7) & 0x00000001);
                absQuant[1] |= ((tmp_char.y >> 6) & 0x00000001);
                absQuant[2] |= ((tmp_char.y >> 5) & 0x00000001);
                absQuant[3] |= ((tmp_char.y >> 4) & 0x00000001);
                absQuant[4] |= ((tmp_char.y >> 3) & 0x00000001);
                absQuant[5] |= ((tmp_char.y >> 2) & 0x00000001);
                absQuant[6] |= ((tmp_char.y >> 1) & 0x00000001);
                absQuant[7] |= ((tmp_char.y >> 0) & 0x00000001);

                absQuant[8] |= ((tmp_char.z >> 7) & 0x00000001);
                absQuant[9] |= ((tmp_char.z >> 6) & 0x00000001);
                absQuant[10] |= ((tmp_char.z >> 5) & 0x00000001);
                absQuant[11] |= ((tmp_char.z >> 4) & 0x00000001);
                absQuant[12] |= ((tmp_char.z >> 3) & 0x00000001);
                absQuant[13] |= ((tmp_char.z >> 2) & 0x00000001);
                absQuant[14] |= ((tmp_char.z >> 1) & 0x00000001);
                absQuant[15] |= ((tmp_char.z >> 0) & 0x00000001);

                absQuant[16] |= ((tmp_char.w >> 7) & 0x00000001);
                absQuant[17] |= ((tmp_char.w >> 6) & 0x00000001);
                absQuant[18] |= ((tmp_char.w >> 5) & 0x00000001);
                absQuant[19] |= ((tmp_char.w >> 4) & 0x00000001);
                absQuant[20] |= ((tmp_char.w >> 3) & 0x00000001);
                absQuant[21] |= ((tmp_char.w >> 2) & 0x00000001);
                absQuant[22] |= ((tmp_char.w >> 1) & 0x00000001);
                absQuant[23] |= ((tmp_char.w >> 0) & 0x00000001);
                cmp_byte_ofs+=4;

                for(int i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                    cmp_byte_ofs+=4;

                    absQuant[24] |= ((tmp_char.x >> 7) & 0x00000001) << i;
                    absQuant[25] |= ((tmp_char.x >> 6) & 0x00000001) << i;
                    absQuant[26] |= ((tmp_char.x >> 5) & 0x00000001) << i;
                    absQuant[27] |= ((tmp_char.x >> 4) & 0x00000001) << i;
                    absQuant[28] |= ((tmp_char.x >> 3) & 0x00000001) << i;
                    absQuant[29] |= ((tmp_char.x >> 2) & 0x00000001) << i;
                    absQuant[30] |= ((tmp_char.x >> 1) & 0x00000001) << i;
                    absQuant[31] |= ((tmp_char.x >> 0) & 0x00000001) << i;

                    if(!encoding_selection) absQuant[0] |= ((tmp_char.y >> 7) & 0x00000001) << (i+1);
                    absQuant[1] |= ((tmp_char.y >> 6) & 0x00000001) << (i+1);
                    absQuant[2] |= ((tmp_char.y >> 5) & 0x00000001) << (i+1);
                    absQuant[3] |= ((tmp_char.y >> 4) & 0x00000001) << (i+1);
                    absQuant[4] |= ((tmp_char.y >> 3) & 0x00000001) << (i+1);
                    absQuant[5] |= ((tmp_char.y >> 2) & 0x00000001) << (i+1);
                    absQuant[6] |= ((tmp_char.y >> 1) & 0x00000001) << (i+1);
                    absQuant[7] |= ((tmp_char.y >> 0) & 0x00000001) << (i+1);

                    absQuant[8] |= ((tmp_char.z >> 7) & 0x00000001) << (i+1);
                    absQuant[9] |= ((tmp_char.z >> 6) & 0x00000001) << (i+1);
                    absQuant[10] |= ((tmp_char.z >> 5) & 0x00000001) << (i+1);
                    absQuant[11] |= ((tmp_char.z >> 4) & 0x00000001) << (i+1);
                    absQuant[12] |= ((tmp_char.z >> 3) & 0x00000001) << (i+1);
                    absQuant[13] |= ((tmp_char.z >> 2) & 0x00000001) << (i+1);
                    absQuant[14] |= ((tmp_char.z >> 1) & 0x00000001) << (i+1);
                    absQuant[15] |= ((tmp_char.z >> 0) & 0x00000001) << (i+1);

                    absQuant[16] |= ((tmp_char.w >> 7) & 0x00000001) << (i+1);
                    absQuant[17] |= ((tmp_char.w >> 6) & 0x00000001) << (i+1);
                    absQuant[18] |= ((tmp_char.w >> 5) & 0x00000001) << (i+1);
                    absQuant[19] |= ((tmp_char.w >> 4) & 0x00000001) << (i+1);
                    absQuant[20] |= ((tmp_char.w >> 3) & 0x00000001) << (i+1);
                    absQuant[21] |= ((tmp_char.w >> 2) & 0x00000001) << (i+1);
                    absQuant[22] |= ((tmp_char.w >> 1) & 0x00000001) << (i+1);
                    absQuant[23] |= ((tmp_char.w >> 0) & 0x00000001) << (i+1);
                }

                int uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[24] |= ((uchar_buffer >> 7) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[25] |= ((uchar_buffer >> 6) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[26] |= ((uchar_buffer >> 5) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[27] |= ((uchar_buffer >> 4) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[28] |= ((uchar_buffer >> 3) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[29] |= ((uchar_buffer >> 2) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[30] |= ((uchar_buffer >> 1) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[31] |= ((uchar_buffer >> 0) & 0x00000001) << (fixed_rate[j]-1);
            }
            else if(vec_ofs==2)
            {
                for(int i=0; i<32; i++) absQuant[i] = 0;

                if(encoding_selection) absQuant[0] = outlier_buffer;

                sign_flag = (0xff000000 & cmpData[cmp_byte_ofs++] << 24) |
                            (0x00ff0000 & cmpData[cmp_byte_ofs++] << 16);

                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                sign_flag |= (0x0000ff00 & tmp_char.x << 8) |
                             (0x000000ff & tmp_char.y);

                if(!encoding_selection) absQuant[0] |= ((tmp_char.z >> 7) & 0x00000001);
                absQuant[1] |= ((tmp_char.z >> 6) & 0x00000001);
                absQuant[2] |= ((tmp_char.z >> 5) & 0x00000001);
                absQuant[3] |= ((tmp_char.z >> 4) & 0x00000001);
                absQuant[4] |= ((tmp_char.z >> 3) & 0x00000001);
                absQuant[5] |= ((tmp_char.z >> 2) & 0x00000001);
                absQuant[6] |= ((tmp_char.z >> 1) & 0x00000001);
                absQuant[7] |= ((tmp_char.z >> 0) & 0x00000001);

                absQuant[8] |= ((tmp_char.w >> 7) & 0x00000001);
                absQuant[9] |= ((tmp_char.w >> 6) & 0x00000001);
                absQuant[10] |= ((tmp_char.w >> 5) & 0x00000001);
                absQuant[11] |= ((tmp_char.w >> 4) & 0x00000001);
                absQuant[12] |= ((tmp_char.w >> 3) & 0x00000001);
                absQuant[13] |= ((tmp_char.w >> 2) & 0x00000001);
                absQuant[14] |= ((tmp_char.w >> 1) & 0x00000001);
                absQuant[15] |= ((tmp_char.w >> 0) & 0x00000001);
                cmp_byte_ofs+=4;

                for(int i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                    cmp_byte_ofs+=4;

                    absQuant[16] |= ((tmp_char.x >> 7) & 0x00000001) << i;
                    absQuant[17] |= ((tmp_char.x >> 6) & 0x00000001) << i;
                    absQuant[18] |= ((tmp_char.x >> 5) & 0x00000001) << i;
                    absQuant[19] |= ((tmp_char.x >> 4) & 0x00000001) << i;
                    absQuant[20] |= ((tmp_char.x >> 3) & 0x00000001) << i;
                    absQuant[21] |= ((tmp_char.x >> 2) & 0x00000001) << i;
                    absQuant[22] |= ((tmp_char.x >> 1) & 0x00000001) << i;
                    absQuant[23] |= ((tmp_char.x >> 0) & 0x00000001) << i;

                    absQuant[24] |= ((tmp_char.y >> 7) & 0x00000001) << i;
                    absQuant[25] |= ((tmp_char.y >> 6) & 0x00000001) << i;
                    absQuant[26] |= ((tmp_char.y >> 5) & 0x00000001) << i;
                    absQuant[27] |= ((tmp_char.y >> 4) & 0x00000001) << i;
                    absQuant[28] |= ((tmp_char.y >> 3) & 0x00000001) << i;
                    absQuant[29] |= ((tmp_char.y >> 2) & 0x00000001) << i;
                    absQuant[30] |= ((tmp_char.y >> 1) & 0x00000001) << i;
                    absQuant[31] |= ((tmp_char.y >> 0) & 0x00000001) << i;

                    if(!encoding_selection) absQuant[0] |= ((tmp_char.z >> 7) & 0x00000001) << (i+1);
                    absQuant[1] |= ((tmp_char.z >> 6) & 0x00000001) << (i+1);
                    absQuant[2] |= ((tmp_char.z >> 5) & 0x00000001) << (i+1);
                    absQuant[3] |= ((tmp_char.z >> 4) & 0x00000001) << (i+1);
                    absQuant[4] |= ((tmp_char.z >> 3) & 0x00000001) << (i+1);
                    absQuant[5] |= ((tmp_char.z >> 2) & 0x00000001) << (i+1);
                    absQuant[6] |= ((tmp_char.z >> 1) & 0x00000001) << (i+1);
                    absQuant[7] |= ((tmp_char.z >> 0) & 0x00000001) << (i+1);

                    absQuant[8] |= ((tmp_char.w >> 7) & 0x00000001) << (i+1);
                    absQuant[9] |= ((tmp_char.w >> 6) & 0x00000001) << (i+1);
                    absQuant[10] |= ((tmp_char.w >> 5) & 0x00000001) << (i+1);
                    absQuant[11] |= ((tmp_char.w >> 4) & 0x00000001) << (i+1);
                    absQuant[12] |= ((tmp_char.w >> 3) & 0x00000001) << (i+1);
                    absQuant[13] |= ((tmp_char.w >> 2) & 0x00000001) << (i+1);
                    absQuant[14] |= ((tmp_char.w >> 1) & 0x00000001) << (i+1);
                    absQuant[15] |= ((tmp_char.w >> 0) & 0x00000001) << (i+1);                    
                }

                int uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[16] |= ((uchar_buffer >> 7) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[17] |= ((uchar_buffer >> 6) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[18] |= ((uchar_buffer >> 5) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[19] |= ((uchar_buffer >> 4) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[20] |= ((uchar_buffer >> 3) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[21] |= ((uchar_buffer >> 2) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[22] |= ((uchar_buffer >> 1) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[23] |= ((uchar_buffer >> 0) & 0x00000001) << (fixed_rate[j]-1);

                uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[24] |= ((uchar_buffer >> 7) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[25] |= ((uchar_buffer >> 6) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[26] |= ((uchar_buffer >> 5) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[27] |= ((uchar_buffer >> 4) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[28] |= ((uchar_buffer >> 3) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[29] |= ((uchar_buffer >> 2) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[30] |= ((uchar_buffer >> 1) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[31] |= ((uchar_buffer >> 0) & 0x00000001) << (fixed_rate[j]-1);
            }
            else
            {
                for(int i=0; i<32; i++) absQuant[i] = 0;

                if(encoding_selection) absQuant[0] = outlier_buffer;

                sign_flag = (0xff000000 & cmpData[cmp_byte_ofs++] << 24);                            

                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                sign_flag |= (0x00ff0000 & tmp_char.x << 16) |
                             (0x0000ff00 & tmp_char.y << 8)  |
                             (0x000000ff & tmp_char.z);

                if(!encoding_selection) absQuant[0] |= ((tmp_char.w >> 7) & 0x00000001);
                absQuant[1] |= ((tmp_char.w >> 6) & 0x00000001);
                absQuant[2] |= ((tmp_char.w >> 5) & 0x00000001);
                absQuant[3] |= ((tmp_char.w >> 4) & 0x00000001);
                absQuant[4] |= ((tmp_char.w >> 3) & 0x00000001);
                absQuant[5] |= ((tmp_char.w >> 2) & 0x00000001);
                absQuant[6] |= ((tmp_char.w >> 1) & 0x00000001);
                absQuant[7] |= ((tmp_char.w >> 0) & 0x00000001);
                cmp_byte_ofs+=4;

                for(int i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                    cmp_byte_ofs+=4;

                    absQuant[8] |= ((tmp_char.x >> 7) & 0x00000001) << i;
                    absQuant[9] |= ((tmp_char.x >> 6) & 0x00000001) << i;
                    absQuant[10] |= ((tmp_char.x >> 5) & 0x00000001) << i;
                    absQuant[11] |= ((tmp_char.x >> 4) & 0x00000001) << i;
                    absQuant[12] |= ((tmp_char.x >> 3) & 0x00000001) << i;
                    absQuant[13] |= ((tmp_char.x >> 2) & 0x00000001) << i;
                    absQuant[14] |= ((tmp_char.x >> 1) & 0x00000001) << i;
                    absQuant[15] |= ((tmp_char.x >> 0) & 0x00000001) << i;

                    absQuant[16] |= ((tmp_char.y >> 7) & 0x00000001) << i;
                    absQuant[17] |= ((tmp_char.y >> 6) & 0x00000001) << i;
                    absQuant[18] |= ((tmp_char.y >> 5) & 0x00000001) << i;
                    absQuant[19] |= ((tmp_char.y >> 4) & 0x00000001) << i;
                    absQuant[20] |= ((tmp_char.y >> 3) & 0x00000001) << i;
                    absQuant[21] |= ((tmp_char.y >> 2) & 0x00000001) << i;
                    absQuant[22] |= ((tmp_char.y >> 1) & 0x00000001) << i;
                    absQuant[23] |= ((tmp_char.y >> 0) & 0x00000001) << i;

                    absQuant[24] |= ((tmp_char.z >> 7) & 0x00000001) << i;
                    absQuant[25] |= ((tmp_char.z >> 6) & 0x00000001) << i;
                    absQuant[26] |= ((tmp_char.z >> 5) & 0x00000001) << i;
                    absQuant[27] |= ((tmp_char.z >> 4) & 0x00000001) << i;
                    absQuant[28] |= ((tmp_char.z >> 3) & 0x00000001) << i;
                    absQuant[29] |= ((tmp_char.z >> 2) & 0x00000001) << i;
                    absQuant[30] |= ((tmp_char.z >> 1) & 0x00000001) << i;
                    absQuant[31] |= ((tmp_char.z >> 0) & 0x00000001) << i;

                    if(!encoding_selection) absQuant[0] |= ((tmp_char.w >> 7) & 0x00000001) << (i+1);
                    absQuant[1] |= ((tmp_char.w >> 6) & 0x00000001) << (i+1);
                    absQuant[2] |= ((tmp_char.w >> 5) & 0x00000001) << (i+1);
                    absQuant[3] |= ((tmp_char.w >> 4) & 0x00000001) << (i+1);
                    absQuant[4] |= ((tmp_char.w >> 3) & 0x00000001) << (i+1);
                    absQuant[5] |= ((tmp_char.w >> 2) & 0x00000001) << (i+1);
                    absQuant[6] |= ((tmp_char.w >> 1) & 0x00000001) << (i+1);
                    absQuant[7] |= ((tmp_char.w >> 0) & 0x00000001) << (i+1);                    
                }

                int uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[8] |= ((uchar_buffer >> 7) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[9] |= ((uchar_buffer >> 6) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[10] |= ((uchar_buffer >> 5) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[11] |= ((uchar_buffer >> 4) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[12] |= ((uchar_buffer >> 3) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[13] |= ((uchar_buffer >> 2) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[14] |= ((uchar_buffer >> 1) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[15] |= ((uchar_buffer >> 0) & 0x00000001) << (fixed_rate[j]-1);

                uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[16] |= ((uchar_buffer >> 7) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[17] |= ((uchar_buffer >> 6) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[18] |= ((uchar_buffer >> 5) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[19] |= ((uchar_buffer >> 4) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[20] |= ((uchar_buffer >> 3) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[21] |= ((uchar_buffer >> 2) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[22] |= ((uchar_buffer >> 1) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[23] |= ((uchar_buffer >> 0) & 0x00000001) << (fixed_rate[j]-1);

                uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[24] |= ((uchar_buffer >> 7) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[25] |= ((uchar_buffer >> 6) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[26] |= ((uchar_buffer >> 5) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[27] |= ((uchar_buffer >> 4) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[28] |= ((uchar_buffer >> 3) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[29] |= ((uchar_buffer >> 2) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[30] |= ((uchar_buffer >> 1) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[31] |= ((uchar_buffer >> 0) & 0x00000001) << (fixed_rate[j]-1);
            }
            
            prevQuant = 0;
            if(base_block_end_idx < nbEle)
            {
                #pragma unroll 8
                for(int i=0; i<32; i+=4)
                {
                    sign_ofs = i % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.x = currQuant * eb * 2;

                    sign_ofs = (i+1) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+1] * -1 : absQuant[i+1];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.y = currQuant * eb * 2;

                    sign_ofs = (i+2) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+2] * -1 : absQuant[i+2];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.z = currQuant * eb * 2;

                    sign_ofs = (i+3) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+3] * -1 : absQuant[i+3];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.w = currQuant * eb * 2;
                    
                    reinterpret_cast<float4*>(decData)[(base_block_start_idx+i)/4] = dec_buffer;
                }
            }
            else
            {
                for(int i=0; i<32; i++)
                {
                    sign_ofs = i % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    if(base_block_start_idx+i < nbEle) decData[base_block_start_idx+i] = currQuant * eb * 2;
                }
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}


// __device__ inline int quantization(float data, float recipPrecision)
// {
//     float dataRecip = data*recipPrecision;
//     int s = dataRecip>=-0.5f?0:1;
//     return (int)(dataRecip+0.5f) - s;
// }


// __device__ inline int get_bit_num(unsigned int x)
// {
//     return (sizeof(unsigned int)*8) - __clz(x);
// }