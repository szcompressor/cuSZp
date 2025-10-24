#include "cuSZp_kernels_3D_f64.h"

__device__ inline int quantization(double data, double recipPrecision)
{
    double dataRecip = data*recipPrecision;
    int s = dataRecip>=-0.5?0:1;
    return (int)(dataRecip+0.5) - s;
}

__device__ inline int get_bit_num(unsigned int x)
{
    return (sizeof(unsigned int)*8) - __clz(x);
}


__global__ void cuSZp_compress_kernel_3D_fixed_vec4_f64(const double* const __restrict__ oriData, 
                                                        unsigned char* const __restrict__ cmpData, 
                                                        volatile unsigned int* const __restrict__ cmpOffset, 
                                                        volatile unsigned int* const __restrict__ locOffset, 
                                                        volatile int* const __restrict__ flag, 
                                                        uint blockNum, const uint3 dims, const double eb)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 3) / 4;
    const uint dimxBlock = (dims.x + 3) / 4;
    const double recipPrecision = 0.5f / eb;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    uint base_start_block_idx;
    uint quant_chunk_idx;
    unsigned int absQuant[block_per_thread * 64];
    unsigned int sign_flag[block_per_thread * 2];
    int sign_ofs;
    unsigned char fixed_rate[block_per_thread];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char1, tmp_char2;

    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        uint block_idx = base_start_block_idx + j * 32 + lane; // Data block that is currently being processed
        uint block_stride_per_slice = dimyBlock * dimxBlock;
        uint block_idx_z = block_idx / block_stride_per_slice;
        uint block_idx_y = (block_idx % block_stride_per_slice) / dimxBlock;
        uint block_idx_x = (block_idx % block_stride_per_slice) % dimxBlock;
        fixed_rate[j] = 0;

        if(block_idx < blockNum)
        {
            sign_flag[j * 2] = 0;
            sign_flag[j * 2 + 1] = 0;
            unsigned int maxQuant = 0;
            int currQuant;

            for(uint i=0; i<4; i++)
            {
                uint data_idx_z = block_idx_z * 4 + i;
                
                if(data_idx_z < dims.z)
                {
                    for(uint k=0; k<4; k++)
                    {
                        uint data_idx_y = block_idx_y * 4 + k;
                        uint block_ofs = i * 16 + k * 4;
                        quant_chunk_idx = j * 64 + block_ofs;

                        if(data_idx_y < dims.y)
                        {
                            uint data_idx_x = block_idx_x * 4;
                            uint data_idx = data_idx_z * dims.y * dims.x + data_idx_y * dims.x + data_idx_x;
                            double4 tmp_buffer = reinterpret_cast<const double4*>(oriData)[data_idx/4];

                            currQuant = quantization(tmp_buffer.x, recipPrecision);
                            sign_ofs = block_ofs % 32;
                            sign_flag[2*j+i/2] |= (currQuant < 0) << (31 - sign_ofs);
                            absQuant[quant_chunk_idx] = abs(currQuant);
                            maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
                            
                            currQuant = quantization(tmp_buffer.y, recipPrecision);
                            sign_ofs = (block_ofs + 1) % 32;
                            sign_flag[2*j+i/2] |= (currQuant < 0) << (31 - sign_ofs);
                            absQuant[quant_chunk_idx+1] = abs(currQuant);
                            maxQuant = maxQuant > absQuant[quant_chunk_idx+1] ? maxQuant : absQuant[quant_chunk_idx+1];
                            
                            currQuant = quantization(tmp_buffer.z, recipPrecision);
                            sign_ofs = (block_ofs + 2) % 32;
                            sign_flag[2*j+i/2] |= (currQuant < 0) << (31 - sign_ofs);
                            absQuant[quant_chunk_idx+2] = abs(currQuant);
                            maxQuant = maxQuant > absQuant[quant_chunk_idx+2] ? maxQuant : absQuant[quant_chunk_idx+2];
                            
                            currQuant = quantization(tmp_buffer.w, recipPrecision);
                            sign_ofs = (block_ofs + 3) % 32;
                            sign_flag[2*j+i/2] |= (currQuant < 0) << (31 - sign_ofs);
                            absQuant[quant_chunk_idx+3] = abs(currQuant);
                            maxQuant = maxQuant > absQuant[quant_chunk_idx+3] ? maxQuant : absQuant[quant_chunk_idx+3];
                        }
                        else
                        {
                            absQuant[quant_chunk_idx] = 0;
                            absQuant[quant_chunk_idx+1] = 0;
                            absQuant[quant_chunk_idx+2] = 0;
                            absQuant[quant_chunk_idx+3] = 0;
                        }
                    }
                }
                else
                {
                    quant_chunk_idx = j * 64 + i * 16;
                    for(uint k=0; k<16; k++) absQuant[quant_chunk_idx+k] = 0;
                }
            }

            fixed_rate[j] = (unsigned char)get_bit_num(maxQuant);
            thread_ofs += (fixed_rate[j]) ? (8+fixed_rate[j]*8) : 0;
            cmpData[block_idx] = fixed_rate[j];
        }
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
    for(uint j=0; j<block_per_thread; j++)
    {
        uint block_idx = base_start_block_idx + j * 32 + lane; // Data block that is currently being processed
        uint chunk_idx_start = j * 64;

        tmp_byte_ofs = (fixed_rate[j]) ? (8+fixed_rate[j]*8) : 0;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(block_idx < blockNum && fixed_rate[j])
        {
            tmp_char1.x = 0xff & (sign_flag[2*j] >> 24);
            tmp_char1.y = 0xff & (sign_flag[2*j] >> 16);
            tmp_char1.z = 0xff & (sign_flag[2*j] >> 8);
            tmp_char1.w = 0xff & sign_flag[2*j];
            tmp_char2.x = 0xff & (sign_flag[2*j+1] >> 24);
            tmp_char2.y = 0xff & (sign_flag[2*j+1] >> 16);
            tmp_char2.z = 0xff & (sign_flag[2*j+1] >> 8);
            tmp_char2.w = 0xff & sign_flag[2*j+1];
            reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
            reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4+1] = tmp_char2;
            cmp_byte_ofs+=8;

            int mask = 1;
            for(uint i=0; i<fixed_rate[j]; i++)
            {
                tmp_char1 = make_uchar4(0, 0, 0, 0);
                tmp_char2 = make_uchar4(0, 0, 0, 0);

                tmp_char1.x = (((absQuant[chunk_idx_start+0] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+1] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+2] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+3] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+4] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+5] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+6] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+7] & mask) >> i) << 0);

                tmp_char1.y = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                tmp_char1.z = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                
                tmp_char1.w = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                tmp_char2.x = (((absQuant[chunk_idx_start+32] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+33] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+34] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+35] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+36] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+37] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+38] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+39] & mask) >> i) << 0);

                tmp_char2.y = (((absQuant[chunk_idx_start+40] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+41] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+42] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+43] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+44] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+45] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+46] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+47] & mask) >> i) << 0);

                tmp_char2.z = (((absQuant[chunk_idx_start+48] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+49] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+50] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+51] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+52] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+53] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+54] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+55] & mask) >> i) << 0);
                
                tmp_char2.w = (((absQuant[chunk_idx_start+56] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+57] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+58] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+59] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+60] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+61] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+62] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+63] & mask) >> i) << 0);
                
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4+1] = tmp_char2;
                cmp_byte_ofs+=8;
                mask <<= 1;
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}

__global__ void cuSZp_decompress_kernel_3D_fixed_vec4_f64(double* const __restrict__ decData, 
                                                        const unsigned char* const __restrict__ cmpData, 
                                                        volatile unsigned int* const __restrict__ cmpOffset, 
                                                        volatile unsigned int* const __restrict__ locOffset, 
                                                        volatile int* const __restrict__ flag, 
                                                        uint blockNum, const uint3 dims, const double eb)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 3) / 4;
    const uint dimxBlock = (dims.x + 3) / 4;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    uint base_start_block_idx;
    int absQuant[64];
    int sign_ofs;
    unsigned char fixed_rate[block_per_thread];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char1, tmp_char2;

    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        uint block_idx = base_start_block_idx + j * 32 + lane; // Data block that is currently being processed

        if(block_idx < blockNum)
        {
            fixed_rate[j] = cmpData[block_idx];
            thread_ofs += (fixed_rate[j]) ? (8+fixed_rate[j]*8) : 0;
        }
        else fixed_rate[j] = 0;
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
    for(uint j=0; j<block_per_thread; j++)
    {
        uint block_idx = base_start_block_idx + j * 32 + lane; // Data block that is currently being processed
        uint block_stride_per_slice = dimyBlock * dimxBlock;
        uint block_idx_z = block_idx / block_stride_per_slice;
        uint block_idx_y = (block_idx % block_stride_per_slice) / dimxBlock;
        uint block_idx_x = (block_idx % block_stride_per_slice) % dimxBlock;
        unsigned int sign_flag[2] = {0, 0};

        tmp_byte_ofs = (fixed_rate[j]) ? (8+fixed_rate[j]*8) : 0;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(block_idx < blockNum && fixed_rate[j])
        {
            tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
            tmp_char2 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4+1];
            sign_flag[0] = (0xff000000 & (tmp_char1.x << 24)) |
                           (0x00ff0000 & (tmp_char1.y << 16)) |
                           (0x0000ff00 & (tmp_char1.z << 8))  |
                           (0x000000ff & tmp_char1.w);
            sign_flag[1] = (0xff000000 & (tmp_char2.x << 24)) |
                           (0x00ff0000 & (tmp_char2.y << 16)) |
                           (0x0000ff00 & (tmp_char2.z << 8))  |
                           (0x000000ff & tmp_char2.w);
            cmp_byte_ofs+=8;

            for(uint i=0; i<64; i++) absQuant[i] = 0;
            for(uint i=0; i<fixed_rate[j]; i++)
            {
                tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                tmp_char2 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4+1];
                cmp_byte_ofs+=8;

                absQuant[0] |= ((tmp_char1.x >> 7) & 0x00000001) << i;
                absQuant[1] |= ((tmp_char1.x >> 6) & 0x00000001) << i;
                absQuant[2] |= ((tmp_char1.x >> 5) & 0x00000001) << i;
                absQuant[3] |= ((tmp_char1.x >> 4) & 0x00000001) << i;
                absQuant[4] |= ((tmp_char1.x >> 3) & 0x00000001) << i;
                absQuant[5] |= ((tmp_char1.x >> 2) & 0x00000001) << i;
                absQuant[6] |= ((tmp_char1.x >> 1) & 0x00000001) << i;
                absQuant[7] |= ((tmp_char1.x >> 0) & 0x00000001) << i;

                absQuant[8] |= ((tmp_char1.y >> 7) & 0x00000001) << i;
                absQuant[9] |= ((tmp_char1.y >> 6) & 0x00000001) << i;
                absQuant[10] |= ((tmp_char1.y >> 5) & 0x00000001) << i;
                absQuant[11] |= ((tmp_char1.y >> 4) & 0x00000001) << i;
                absQuant[12] |= ((tmp_char1.y >> 3) & 0x00000001) << i;
                absQuant[13] |= ((tmp_char1.y >> 2) & 0x00000001) << i;
                absQuant[14] |= ((tmp_char1.y >> 1) & 0x00000001) << i;
                absQuant[15] |= ((tmp_char1.y >> 0) & 0x00000001) << i;

                absQuant[16] |= ((tmp_char1.z >> 7) & 0x00000001) << i;
                absQuant[17] |= ((tmp_char1.z >> 6) & 0x00000001) << i;
                absQuant[18] |= ((tmp_char1.z >> 5) & 0x00000001) << i;
                absQuant[19] |= ((tmp_char1.z >> 4) & 0x00000001) << i;
                absQuant[20] |= ((tmp_char1.z >> 3) & 0x00000001) << i;
                absQuant[21] |= ((tmp_char1.z >> 2) & 0x00000001) << i;
                absQuant[22] |= ((tmp_char1.z >> 1) & 0x00000001) << i;
                absQuant[23] |= ((tmp_char1.z >> 0) & 0x00000001) << i;

                absQuant[24] |= ((tmp_char1.w >> 7) & 0x00000001) << i;
                absQuant[25] |= ((tmp_char1.w >> 6) & 0x00000001) << i;
                absQuant[26] |= ((tmp_char1.w >> 5) & 0x00000001) << i;
                absQuant[27] |= ((tmp_char1.w >> 4) & 0x00000001) << i;
                absQuant[28] |= ((tmp_char1.w >> 3) & 0x00000001) << i;
                absQuant[29] |= ((tmp_char1.w >> 2) & 0x00000001) << i;
                absQuant[30] |= ((tmp_char1.w >> 1) & 0x00000001) << i;
                absQuant[31] |= ((tmp_char1.w >> 0) & 0x00000001) << i;

                absQuant[32] |= ((tmp_char2.x >> 7) & 0x00000001) << i;
                absQuant[33] |= ((tmp_char2.x >> 6) & 0x00000001) << i;
                absQuant[34] |= ((tmp_char2.x >> 5) & 0x00000001) << i;
                absQuant[35] |= ((tmp_char2.x >> 4) & 0x00000001) << i;
                absQuant[36] |= ((tmp_char2.x >> 3) & 0x00000001) << i;
                absQuant[37] |= ((tmp_char2.x >> 2) & 0x00000001) << i;
                absQuant[38] |= ((tmp_char2.x >> 1) & 0x00000001) << i;
                absQuant[39] |= ((tmp_char2.x >> 0) & 0x00000001) << i;

                absQuant[40] |= ((tmp_char2.y >> 7) & 0x00000001) << i;
                absQuant[41] |= ((tmp_char2.y >> 6) & 0x00000001) << i;
                absQuant[42] |= ((tmp_char2.y >> 5) & 0x00000001) << i;
                absQuant[43] |= ((tmp_char2.y >> 4) & 0x00000001) << i;
                absQuant[44] |= ((tmp_char2.y >> 3) & 0x00000001) << i;
                absQuant[45] |= ((tmp_char2.y >> 2) & 0x00000001) << i;
                absQuant[46] |= ((tmp_char2.y >> 1) & 0x00000001) << i;
                absQuant[47] |= ((tmp_char2.y >> 0) & 0x00000001) << i;

                absQuant[48] |= ((tmp_char2.z >> 7) & 0x00000001) << i;
                absQuant[49] |= ((tmp_char2.z >> 6) & 0x00000001) << i;
                absQuant[50] |= ((tmp_char2.z >> 5) & 0x00000001) << i;
                absQuant[51] |= ((tmp_char2.z >> 4) & 0x00000001) << i;
                absQuant[52] |= ((tmp_char2.z >> 3) & 0x00000001) << i;
                absQuant[53] |= ((tmp_char2.z >> 2) & 0x00000001) << i;
                absQuant[54] |= ((tmp_char2.z >> 1) & 0x00000001) << i;
                absQuant[55] |= ((tmp_char2.z >> 0) & 0x00000001) << i;

                absQuant[56] |= ((tmp_char2.w >> 7) & 0x00000001) << i;
                absQuant[57] |= ((tmp_char2.w >> 6) & 0x00000001) << i;
                absQuant[58] |= ((tmp_char2.w >> 5) & 0x00000001) << i;
                absQuant[59] |= ((tmp_char2.w >> 4) & 0x00000001) << i;
                absQuant[60] |= ((tmp_char2.w >> 3) & 0x00000001) << i;
                absQuant[61] |= ((tmp_char2.w >> 2) & 0x00000001) << i;
                absQuant[62] |= ((tmp_char2.w >> 1) & 0x00000001) << i;
                absQuant[63] |= ((tmp_char2.w >> 0) & 0x00000001) << i;
            }

            int currQuant;
            for(uint i=0; i<4; i++)
            {
                uint data_idx_z = block_idx_z * 4 + i;

                if(data_idx_z < dims.z)
                {
                    for(uint k=0; k<4; k++)
                    {
                        uint data_idx_y = block_idx_y * 4 + k;
                        uint block_ofs = i * 16 + k * 4;

                        if(data_idx_y < dims.y)
                        {
                            uint data_idx_x = block_idx_x * 4;
                            uint data_idx = data_idx_z * dims.y * dims.x + data_idx_y * dims.x + data_idx_x;
                            double4 dec_buffer;

                            sign_ofs = block_ofs % 32;
                            currQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs] * -1 : absQuant[block_ofs];
                            dec_buffer.x = currQuant * eb * 2;

                            sign_ofs = (block_ofs + 1) % 32;
                            currQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs+1] * -1 : absQuant[block_ofs+1];
                            dec_buffer.y = currQuant * eb * 2;

                            sign_ofs = (block_ofs + 2) % 32;
                            currQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs+2] * -1 : absQuant[block_ofs+2];
                            dec_buffer.z = currQuant * eb * 2;

                            sign_ofs = (block_ofs + 3) % 32;
                            currQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs+3] * -1 : absQuant[block_ofs+3];
                            dec_buffer.w = currQuant * eb * 2;
                            
                            reinterpret_cast<double4*>(decData)[data_idx/4] = dec_buffer;
                        }
                    }
                }
            } 
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}


__global__ void cuSZp_compress_kernel_3D_plain_vec4_f64(const double* const __restrict__ oriData, 
                                                        unsigned char* const __restrict__ cmpData, 
                                                        volatile unsigned int* const __restrict__ cmpOffset, 
                                                        volatile unsigned int* const __restrict__ locOffset, 
                                                        volatile int* const __restrict__ flag, 
                                                        uint blockNum, const uint3 dims, const double eb)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 3) / 4;
    const uint dimxBlock = (dims.x + 3) / 4;
    const double recipPrecision = 0.5f / eb;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    uint base_start_block_idx;
    uint quant_chunk_idx;
    unsigned int absQuant[block_per_thread * 64];
    unsigned int sign_flag[block_per_thread * 2];
    int sign_ofs;
    unsigned char fixed_rate[block_per_thread];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char1, tmp_char2;

    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        uint block_idx = base_start_block_idx + j * 32 + lane; // Data block that is currently being processed
        uint block_stride_per_slice = dimyBlock * dimxBlock;
        uint block_idx_z = block_idx / block_stride_per_slice;
        uint block_idx_y = (block_idx % block_stride_per_slice) / dimxBlock;
        uint block_idx_x = (block_idx % block_stride_per_slice) % dimxBlock;
        fixed_rate[j] = 0;

        if(block_idx < blockNum)
        {
            sign_flag[j * 2] = 0;
            sign_flag[j * 2 + 1] = 0;
            unsigned int maxQuant = 0;
            int currQuant, lorenQuant;
            int prevQuant_z = 0;

            for(uint i=0; i<4; i++)
            {
                uint data_idx_z = block_idx_z * 4 + i;
                int prevQuant_y = 0;
                
                if(data_idx_z < dims.z)
                {
                    for(uint k=0; k<4; k++)
                    {
                        uint data_idx_y = block_idx_y * 4 + k;
                        uint block_ofs = i * 16 + k * 4;
                        quant_chunk_idx = j * 64 + block_ofs;
                        int prevQuant_x = 0;

                        if(data_idx_y < dims.y)
                        {
                            uint data_idx_x = block_idx_x * 4;
                            uint data_idx = data_idx_z * dims.y * dims.x + data_idx_y * dims.x + data_idx_x;
                            double4 tmp_buffer = reinterpret_cast<const double4*>(oriData)[data_idx/4];

                            currQuant = quantization(tmp_buffer.x, recipPrecision);
                            if(k) lorenQuant = currQuant - prevQuant_y; // Y-delta
                            else {
                                lorenQuant = currQuant - prevQuant_z;   // Z-delta
                                prevQuant_z = currQuant;
                            }
                            prevQuant_y = currQuant;
                            prevQuant_x = currQuant;
                            sign_ofs = block_ofs % 32;
                            sign_flag[2*j+i/2] |= (lorenQuant < 0) << (31 - sign_ofs);
                            absQuant[quant_chunk_idx] = abs(lorenQuant);
                            maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
                            
                            currQuant = quantization(tmp_buffer.y, recipPrecision);
                            lorenQuant = currQuant - prevQuant_x;
                            prevQuant_x = currQuant;
                            sign_ofs = (block_ofs + 1) % 32;
                            sign_flag[2*j+i/2] |= (lorenQuant < 0) << (31 - sign_ofs);
                            absQuant[quant_chunk_idx+1] = abs(lorenQuant);
                            maxQuant = maxQuant > absQuant[quant_chunk_idx+1] ? maxQuant : absQuant[quant_chunk_idx+1];
                            
                            currQuant = quantization(tmp_buffer.z, recipPrecision);
                            lorenQuant = currQuant - prevQuant_x;
                            prevQuant_x = currQuant;
                            sign_ofs = (block_ofs + 2) % 32;
                            sign_flag[2*j+i/2] |= (lorenQuant < 0) << (31 - sign_ofs);
                            absQuant[quant_chunk_idx+2] = abs(lorenQuant);
                            maxQuant = maxQuant > absQuant[quant_chunk_idx+2] ? maxQuant : absQuant[quant_chunk_idx+2];
                            
                            currQuant = quantization(tmp_buffer.w, recipPrecision);
                            lorenQuant = currQuant - prevQuant_x;
                            prevQuant_x = currQuant;
                            sign_ofs = (block_ofs + 3) % 32;
                            sign_flag[2*j+i/2] |= (lorenQuant < 0) << (31 - sign_ofs);
                            absQuant[quant_chunk_idx+3] = abs(lorenQuant);
                            maxQuant = maxQuant > absQuant[quant_chunk_idx+3] ? maxQuant : absQuant[quant_chunk_idx+3];
                        }
                        else
                        {
                            absQuant[quant_chunk_idx] = 0;
                            absQuant[quant_chunk_idx+1] = 0;
                            absQuant[quant_chunk_idx+2] = 0;
                            absQuant[quant_chunk_idx+3] = 0;
                        }
                    }
                }
                else
                {
                    quant_chunk_idx = j * 64 + i * 16;
                    for(uint k=0; k<16; k++) absQuant[quant_chunk_idx+k] = 0;
                }
            }

            fixed_rate[j] = (unsigned char)get_bit_num(maxQuant);
            thread_ofs += (fixed_rate[j]) ? (8+fixed_rate[j]*8) : 0;
            cmpData[block_idx] = fixed_rate[j];
        }
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
    for(uint j=0; j<block_per_thread; j++)
    {
        uint block_idx = base_start_block_idx + j * 32 + lane; // Data block that is currently being processed
        uint chunk_idx_start = j * 64;

        tmp_byte_ofs = (fixed_rate[j]) ? (8+fixed_rate[j]*8) : 0;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(block_idx < blockNum && fixed_rate[j])
        {
            tmp_char1.x = 0xff & (sign_flag[2*j] >> 24);
            tmp_char1.y = 0xff & (sign_flag[2*j] >> 16);
            tmp_char1.z = 0xff & (sign_flag[2*j] >> 8);
            tmp_char1.w = 0xff & sign_flag[2*j];
            tmp_char2.x = 0xff & (sign_flag[2*j+1] >> 24);
            tmp_char2.y = 0xff & (sign_flag[2*j+1] >> 16);
            tmp_char2.z = 0xff & (sign_flag[2*j+1] >> 8);
            tmp_char2.w = 0xff & sign_flag[2*j+1];
            reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
            reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4+1] = tmp_char2;
            cmp_byte_ofs+=8;

            int mask = 1;
            for(uint i=0; i<fixed_rate[j]; i++)
            {
                tmp_char1 = make_uchar4(0, 0, 0, 0);
                tmp_char2 = make_uchar4(0, 0, 0, 0);

                tmp_char1.x = (((absQuant[chunk_idx_start+0] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+1] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+2] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+3] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+4] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+5] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+6] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+7] & mask) >> i) << 0);

                tmp_char1.y = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                tmp_char1.z = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                
                tmp_char1.w = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                tmp_char2.x = (((absQuant[chunk_idx_start+32] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+33] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+34] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+35] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+36] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+37] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+38] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+39] & mask) >> i) << 0);

                tmp_char2.y = (((absQuant[chunk_idx_start+40] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+41] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+42] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+43] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+44] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+45] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+46] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+47] & mask) >> i) << 0);

                tmp_char2.z = (((absQuant[chunk_idx_start+48] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+49] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+50] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+51] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+52] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+53] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+54] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+55] & mask) >> i) << 0);
                
                tmp_char2.w = (((absQuant[chunk_idx_start+56] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+57] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+58] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+59] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+60] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+61] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+62] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+63] & mask) >> i) << 0);
                
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4+1] = tmp_char2;
                cmp_byte_ofs+=8;
                mask <<= 1;
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}

__global__ void cuSZp_decompress_kernel_3D_plain_vec4_f64(double* const __restrict__ decData, 
                                                        const unsigned char* const __restrict__ cmpData, 
                                                        volatile unsigned int* const __restrict__ cmpOffset, 
                                                        volatile unsigned int* const __restrict__ locOffset, 
                                                        volatile int* const __restrict__ flag, 
                                                        uint blockNum, const uint3 dims, const double eb)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 3) / 4;
    const uint dimxBlock = (dims.x + 3) / 4;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    uint base_start_block_idx;
    int absQuant[64];
    int sign_ofs;
    unsigned char fixed_rate[block_per_thread];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char1, tmp_char2;

    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        uint block_idx = base_start_block_idx + j * 32 + lane; // Data block that is currently being processed

        if(block_idx < blockNum)
        {
            fixed_rate[j] = cmpData[block_idx];
            thread_ofs += (fixed_rate[j]) ? (8+fixed_rate[j]*8) : 0;
        }
        else fixed_rate[j] = 0;
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
    for(uint j=0; j<block_per_thread; j++)
    {
        uint block_idx = base_start_block_idx + j * 32 + lane; // Data block that is currently being processed
        uint block_stride_per_slice = dimyBlock * dimxBlock;
        uint block_idx_z = block_idx / block_stride_per_slice;
        uint block_idx_y = (block_idx % block_stride_per_slice) / dimxBlock;
        uint block_idx_x = (block_idx % block_stride_per_slice) % dimxBlock;
        unsigned int sign_flag[2] = {0, 0};

        tmp_byte_ofs = (fixed_rate[j]) ? (8+fixed_rate[j]*8) : 0;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(block_idx < blockNum && fixed_rate[j])
        {
            tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
            tmp_char2 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4+1];
            sign_flag[0] = (0xff000000 & (tmp_char1.x << 24)) |
                           (0x00ff0000 & (tmp_char1.y << 16)) |
                           (0x0000ff00 & (tmp_char1.z << 8))  |
                           (0x000000ff & tmp_char1.w);
            sign_flag[1] = (0xff000000 & (tmp_char2.x << 24)) |
                           (0x00ff0000 & (tmp_char2.y << 16)) |
                           (0x0000ff00 & (tmp_char2.z << 8))  |
                           (0x000000ff & tmp_char2.w);
            cmp_byte_ofs+=8;

            for(uint i=0; i<64; i++) absQuant[i] = 0;
            for(uint i=0; i<fixed_rate[j]; i++)
            {
                tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                tmp_char2 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4+1];
                cmp_byte_ofs+=8;

                absQuant[0] |= ((tmp_char1.x >> 7) & 0x00000001) << i;
                absQuant[1] |= ((tmp_char1.x >> 6) & 0x00000001) << i;
                absQuant[2] |= ((tmp_char1.x >> 5) & 0x00000001) << i;
                absQuant[3] |= ((tmp_char1.x >> 4) & 0x00000001) << i;
                absQuant[4] |= ((tmp_char1.x >> 3) & 0x00000001) << i;
                absQuant[5] |= ((tmp_char1.x >> 2) & 0x00000001) << i;
                absQuant[6] |= ((tmp_char1.x >> 1) & 0x00000001) << i;
                absQuant[7] |= ((tmp_char1.x >> 0) & 0x00000001) << i;

                absQuant[8] |= ((tmp_char1.y >> 7) & 0x00000001) << i;
                absQuant[9] |= ((tmp_char1.y >> 6) & 0x00000001) << i;
                absQuant[10] |= ((tmp_char1.y >> 5) & 0x00000001) << i;
                absQuant[11] |= ((tmp_char1.y >> 4) & 0x00000001) << i;
                absQuant[12] |= ((tmp_char1.y >> 3) & 0x00000001) << i;
                absQuant[13] |= ((tmp_char1.y >> 2) & 0x00000001) << i;
                absQuant[14] |= ((tmp_char1.y >> 1) & 0x00000001) << i;
                absQuant[15] |= ((tmp_char1.y >> 0) & 0x00000001) << i;

                absQuant[16] |= ((tmp_char1.z >> 7) & 0x00000001) << i;
                absQuant[17] |= ((tmp_char1.z >> 6) & 0x00000001) << i;
                absQuant[18] |= ((tmp_char1.z >> 5) & 0x00000001) << i;
                absQuant[19] |= ((tmp_char1.z >> 4) & 0x00000001) << i;
                absQuant[20] |= ((tmp_char1.z >> 3) & 0x00000001) << i;
                absQuant[21] |= ((tmp_char1.z >> 2) & 0x00000001) << i;
                absQuant[22] |= ((tmp_char1.z >> 1) & 0x00000001) << i;
                absQuant[23] |= ((tmp_char1.z >> 0) & 0x00000001) << i;

                absQuant[24] |= ((tmp_char1.w >> 7) & 0x00000001) << i;
                absQuant[25] |= ((tmp_char1.w >> 6) & 0x00000001) << i;
                absQuant[26] |= ((tmp_char1.w >> 5) & 0x00000001) << i;
                absQuant[27] |= ((tmp_char1.w >> 4) & 0x00000001) << i;
                absQuant[28] |= ((tmp_char1.w >> 3) & 0x00000001) << i;
                absQuant[29] |= ((tmp_char1.w >> 2) & 0x00000001) << i;
                absQuant[30] |= ((tmp_char1.w >> 1) & 0x00000001) << i;
                absQuant[31] |= ((tmp_char1.w >> 0) & 0x00000001) << i;

                absQuant[32] |= ((tmp_char2.x >> 7) & 0x00000001) << i;
                absQuant[33] |= ((tmp_char2.x >> 6) & 0x00000001) << i;
                absQuant[34] |= ((tmp_char2.x >> 5) & 0x00000001) << i;
                absQuant[35] |= ((tmp_char2.x >> 4) & 0x00000001) << i;
                absQuant[36] |= ((tmp_char2.x >> 3) & 0x00000001) << i;
                absQuant[37] |= ((tmp_char2.x >> 2) & 0x00000001) << i;
                absQuant[38] |= ((tmp_char2.x >> 1) & 0x00000001) << i;
                absQuant[39] |= ((tmp_char2.x >> 0) & 0x00000001) << i;

                absQuant[40] |= ((tmp_char2.y >> 7) & 0x00000001) << i;
                absQuant[41] |= ((tmp_char2.y >> 6) & 0x00000001) << i;
                absQuant[42] |= ((tmp_char2.y >> 5) & 0x00000001) << i;
                absQuant[43] |= ((tmp_char2.y >> 4) & 0x00000001) << i;
                absQuant[44] |= ((tmp_char2.y >> 3) & 0x00000001) << i;
                absQuant[45] |= ((tmp_char2.y >> 2) & 0x00000001) << i;
                absQuant[46] |= ((tmp_char2.y >> 1) & 0x00000001) << i;
                absQuant[47] |= ((tmp_char2.y >> 0) & 0x00000001) << i;

                absQuant[48] |= ((tmp_char2.z >> 7) & 0x00000001) << i;
                absQuant[49] |= ((tmp_char2.z >> 6) & 0x00000001) << i;
                absQuant[50] |= ((tmp_char2.z >> 5) & 0x00000001) << i;
                absQuant[51] |= ((tmp_char2.z >> 4) & 0x00000001) << i;
                absQuant[52] |= ((tmp_char2.z >> 3) & 0x00000001) << i;
                absQuant[53] |= ((tmp_char2.z >> 2) & 0x00000001) << i;
                absQuant[54] |= ((tmp_char2.z >> 1) & 0x00000001) << i;
                absQuant[55] |= ((tmp_char2.z >> 0) & 0x00000001) << i;

                absQuant[56] |= ((tmp_char2.w >> 7) & 0x00000001) << i;
                absQuant[57] |= ((tmp_char2.w >> 6) & 0x00000001) << i;
                absQuant[58] |= ((tmp_char2.w >> 5) & 0x00000001) << i;
                absQuant[59] |= ((tmp_char2.w >> 4) & 0x00000001) << i;
                absQuant[60] |= ((tmp_char2.w >> 3) & 0x00000001) << i;
                absQuant[61] |= ((tmp_char2.w >> 2) & 0x00000001) << i;
                absQuant[62] |= ((tmp_char2.w >> 1) & 0x00000001) << i;
                absQuant[63] |= ((tmp_char2.w >> 0) & 0x00000001) << i;
            }

            int currQuant, lorenQuant;
            int prevQuant_z = 0;
            for(uint i=0; i<4; i++)
            {
                uint data_idx_z = block_idx_z * 4 + i;
                int prevQuant_y = 0;

                if(data_idx_z < dims.z)
                {
                    for(uint k=0; k<4; k++)
                    {
                        uint data_idx_y = block_idx_y * 4 + k;
                        uint block_ofs = i * 16 + k * 4;
                        int prevQuant_x = 0;

                        if(data_idx_y < dims.y)
                        {
                            uint data_idx_x = block_idx_x * 4;
                            uint data_idx = data_idx_z * dims.y * dims.x + data_idx_y * dims.x + data_idx_x;
                            double4 dec_buffer;

                            sign_ofs = block_ofs % 32;
                            lorenQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs] * -1 : absQuant[block_ofs];
                            if(k) currQuant = lorenQuant + prevQuant_y; // Y-de-delta
                            else { 
                                currQuant = lorenQuant + prevQuant_z;   // Z-de-delta
                                prevQuant_z = currQuant;
                            }
                            prevQuant_y = currQuant;
                            prevQuant_x = currQuant;
                            dec_buffer.x = currQuant * eb * 2;

                            sign_ofs = (block_ofs + 1) % 32;
                            lorenQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs+1] * -1 : absQuant[block_ofs+1];
                            currQuant = lorenQuant + prevQuant_x;
                            prevQuant_x = currQuant;
                            dec_buffer.y = currQuant * eb * 2;

                            sign_ofs = (block_ofs + 2) % 32;
                            lorenQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs+2] * -1 : absQuant[block_ofs+2];
                            currQuant = lorenQuant + prevQuant_x;
                            prevQuant_x = currQuant;
                            dec_buffer.z = currQuant * eb * 2;

                            sign_ofs = (block_ofs + 3) % 32;
                            lorenQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs+3] * -1 : absQuant[block_ofs+3];
                            currQuant = lorenQuant + prevQuant_x;
                            prevQuant_x = currQuant;
                            dec_buffer.w = currQuant * eb * 2;
                            
                            reinterpret_cast<double4*>(decData)[data_idx/4] = dec_buffer;
                        }
                    }
                }
            } 
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}


__global__ void cuSZp_compress_kernel_3D_outlier_vec4_f64(const double* const __restrict__ oriData, 
                                                        unsigned char* const __restrict__ cmpData, 
                                                        volatile unsigned int* const __restrict__ cmpOffset, 
                                                        volatile unsigned int* const __restrict__ locOffset, 
                                                        volatile int* const __restrict__ flag, 
                                                        uint blockNum, const uint3 dims, const double eb)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 3) / 4;
    const uint dimxBlock = (dims.x + 3) / 4;
    const double recipPrecision = 0.5f / eb;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    uint base_start_block_idx;
    uint quant_chunk_idx;
    unsigned int absQuant[block_per_thread * 64];
    unsigned int sign_flag[block_per_thread * 2];
    int sign_ofs;
    unsigned char fixed_rate[block_per_thread];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char1, tmp_char2;

    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        uint block_idx = base_start_block_idx + j * 32 + lane; // Data block that is currently being processed
        uint block_stride_per_slice = dimyBlock * dimxBlock;
        uint block_idx_z = block_idx / block_stride_per_slice;
        uint block_idx_y = (block_idx % block_stride_per_slice) / dimxBlock;
        uint block_idx_x = (block_idx % block_stride_per_slice) % dimxBlock;
        fixed_rate[j] = 0;

        if(block_idx < blockNum)
        {
            sign_flag[j * 2] = 0;
            sign_flag[j * 2 + 1] = 0;
            unsigned int maxQuant = 0;
            unsigned int maxQuan2 = 0;
            int outlier = 0;
            int currQuant, lorenQuant;
            int prevQuant_z = 0;

            for(uint i=0; i<4; i++)
            {
                uint data_idx_z = block_idx_z * 4 + i;
                int prevQuant_y = 0;
                
                if(data_idx_z < dims.z)
                {
                    for(uint k=0; k<4; k++)
                    {
                        uint data_idx_y = block_idx_y * 4 + k;
                        uint block_ofs = i * 16 + k * 4;
                        quant_chunk_idx = j * 64 + block_ofs;
                        int prevQuant_x = 0;

                        if(data_idx_y < dims.y)
                        {
                            uint data_idx_x = block_idx_x * 4;
                            uint data_idx = data_idx_z * dims.y * dims.x + data_idx_y * dims.x + data_idx_x;
                            double4 tmp_buffer = reinterpret_cast<const double4*>(oriData)[data_idx/4];

                            currQuant = quantization(tmp_buffer.x, recipPrecision);
                            if(k) lorenQuant = currQuant - prevQuant_y; // Y-delta
                            else {
                                lorenQuant = currQuant - prevQuant_z;   // Z-delta
                                prevQuant_z = currQuant;
                            }
                            prevQuant_y = currQuant;
                            prevQuant_x = currQuant;
                            sign_ofs = block_ofs % 32;
                            sign_flag[2*j+i/2] |= (lorenQuant < 0) << (31 - sign_ofs);
                            absQuant[quant_chunk_idx] = abs(lorenQuant);
                            maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
                            if(i||k) maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx] ? maxQuan2 : absQuant[quant_chunk_idx];
                            else outlier = absQuant[quant_chunk_idx];
                            
                            currQuant = quantization(tmp_buffer.y, recipPrecision);
                            lorenQuant = currQuant - prevQuant_x;
                            prevQuant_x = currQuant;
                            sign_ofs = (block_ofs + 1) % 32;
                            sign_flag[2*j+i/2] |= (lorenQuant < 0) << (31 - sign_ofs);
                            absQuant[quant_chunk_idx+1] = abs(lorenQuant);
                            maxQuant = maxQuant > absQuant[quant_chunk_idx+1] ? maxQuant : absQuant[quant_chunk_idx+1];
                            maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx+1] ? maxQuan2 : absQuant[quant_chunk_idx+1];
                            
                            currQuant = quantization(tmp_buffer.z, recipPrecision);
                            lorenQuant = currQuant - prevQuant_x;
                            prevQuant_x = currQuant;
                            sign_ofs = (block_ofs + 2) % 32;
                            sign_flag[2*j+i/2] |= (lorenQuant < 0) << (31 - sign_ofs);
                            absQuant[quant_chunk_idx+2] = abs(lorenQuant);
                            maxQuant = maxQuant > absQuant[quant_chunk_idx+2] ? maxQuant : absQuant[quant_chunk_idx+2];
                            maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx+2] ? maxQuan2 : absQuant[quant_chunk_idx+2];
                            
                            currQuant = quantization(tmp_buffer.w, recipPrecision);
                            lorenQuant = currQuant - prevQuant_x;
                            prevQuant_x = currQuant;
                            sign_ofs = (block_ofs + 3) % 32;
                            sign_flag[2*j+i/2] |= (lorenQuant < 0) << (31 - sign_ofs);
                            absQuant[quant_chunk_idx+3] = abs(lorenQuant);
                            maxQuant = maxQuant > absQuant[quant_chunk_idx+3] ? maxQuant : absQuant[quant_chunk_idx+3];
                            maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx+3] ? maxQuan2 : absQuant[quant_chunk_idx+3];
                        }
                        else
                        {
                            absQuant[quant_chunk_idx] = 0;
                            absQuant[quant_chunk_idx+1] = 0;
                            absQuant[quant_chunk_idx+2] = 0;
                            absQuant[quant_chunk_idx+3] = 0;
                        }
                    }
                }
                else
                {
                    quant_chunk_idx = j * 64 + i * 16;
                    for(uint k=0; k<16; k++) absQuant[quant_chunk_idx+k] = 0;
                }
            }

            int fr1 = get_bit_num(maxQuant);
            int fr2 = get_bit_num(maxQuan2);
            outlier = (get_bit_num(outlier)+7)/8;
            int temp_rate = 0;
            int temp_ofs1 = fr1 ? 8 + fr1 * 8 : 0;
            int temp_ofs2 = fr2 ? 8 + fr2 * 8 + outlier: 8 + outlier;
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

            fixed_rate[j] = (unsigned char)temp_rate;
            cmpData[block_idx] = fixed_rate[j];
        }
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
    for(uint j=0; j<block_per_thread; j++)
    {
        int encoding_selection = fixed_rate[j] >> 7;
        int outlier_byte_num = ((fixed_rate[j] & 0x60) >> 5) + 1;
        fixed_rate[j] &= 0x1f;
        uint block_idx = base_start_block_idx + j * 32 + lane; // Data block that is currently being processed
        uint chunk_idx_start = j * 64;

        if(!encoding_selection) tmp_byte_ofs = (fixed_rate[j]) ? (8 + fixed_rate[j] * 8) : 0;
        else tmp_byte_ofs = 8 + outlier_byte_num + fixed_rate[j] * 8;
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
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[2*j] >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[2*j] >> 16);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[2*j] >> 8);
                cmpData[cmp_byte_ofs++] = 0xff & sign_flag[2*j];
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[2*j+1] >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[2*j+1] >> 16);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[2*j+1] >> 8);
                cmpData[cmp_byte_ofs++] = 0xff & sign_flag[2*j+1];
            }
        }

        if(block_idx < blockNum && fixed_rate[j])
        {
            int vec_ofs = cmp_byte_ofs % 4;
            if(vec_ofs==0)
            {
                tmp_char1.x = 0xff & (sign_flag[2*j] >> 24);
                tmp_char1.y = 0xff & (sign_flag[2*j] >> 16);
                tmp_char1.z = 0xff & (sign_flag[2*j] >> 8);
                tmp_char1.w = 0xff & sign_flag[2*j];
                tmp_char2.x = 0xff & (sign_flag[2*j+1] >> 24);
                tmp_char2.y = 0xff & (sign_flag[2*j+1] >> 16);
                tmp_char2.z = 0xff & (sign_flag[2*j+1] >> 8);
                tmp_char2.w = 0xff & sign_flag[2*j+1];
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4+1] = tmp_char2;
                cmp_byte_ofs+=8;

                int mask = 1;
                for(uint i=0; i<fixed_rate[j]; i++)
                {
                    tmp_char1 = make_uchar4(0, 0, 0, 0);
                    tmp_char2 = make_uchar4(0, 0, 0, 0);

                    if(!encoding_selection) tmp_char1.x = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
                    tmp_char1.x = tmp_char1.x |
                                (((absQuant[chunk_idx_start+1] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+2] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+3] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+4] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+5] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+6] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+7] & mask) >> i) << 0);

                    tmp_char1.y = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                    tmp_char1.z = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                    
                    tmp_char1.w = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                    tmp_char2.x = (((absQuant[chunk_idx_start+32] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+33] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+34] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+35] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+36] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+37] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+38] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+39] & mask) >> i) << 0);

                    tmp_char2.y = (((absQuant[chunk_idx_start+40] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+41] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+42] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+43] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+44] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+45] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+46] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+47] & mask) >> i) << 0);

                    tmp_char2.z = (((absQuant[chunk_idx_start+48] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+49] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+50] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+51] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+52] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+53] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+54] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+55] & mask) >> i) << 0);
                    
                    tmp_char2.w = (((absQuant[chunk_idx_start+56] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+57] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+58] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+59] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+60] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+61] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+62] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+63] & mask) >> i) << 0);
                    
                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4+1] = tmp_char2;
                    cmp_byte_ofs+=8;
                    mask <<= 1;
                }
            }
            else if(vec_ofs==1)
            {
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[2*j] >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[2*j] >> 16);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[2*j] >> 8);
                
                tmp_char2 = make_uchar4(0, 0, 0, 0);
                tmp_char1.x = 0xff & sign_flag[2*j];
                tmp_char1.y = 0xff & (sign_flag[2*j+1] >> 24);
                tmp_char1.z = 0xff & (sign_flag[2*j+1] >> 16);
                tmp_char1.w = 0xff & (sign_flag[2*j+1] >> 8);
                tmp_char2.x = 0xff & sign_flag[2*j+1];

                if(!encoding_selection) tmp_char2.y = ((absQuant[chunk_idx_start] & 1) << 7);
                tmp_char2.y = tmp_char2.y |
                            ((absQuant[chunk_idx_start+1] & 1) << 6) |
                            ((absQuant[chunk_idx_start+2] & 1) << 5) |
                            ((absQuant[chunk_idx_start+3] & 1) << 4) |
                            ((absQuant[chunk_idx_start+4] & 1) << 3) |
                            ((absQuant[chunk_idx_start+5] & 1) << 2) |
                            ((absQuant[chunk_idx_start+6] & 1) << 1) |
                            ((absQuant[chunk_idx_start+7] & 1) << 0);

                tmp_char2.z = ((absQuant[chunk_idx_start+8] & 1) << 7) |
                            ((absQuant[chunk_idx_start+9] & 1) << 6) |
                            ((absQuant[chunk_idx_start+10] & 1) << 5) |
                            ((absQuant[chunk_idx_start+11] & 1) << 4) |
                            ((absQuant[chunk_idx_start+12] & 1) << 3) |
                            ((absQuant[chunk_idx_start+13] & 1) << 2) |
                            ((absQuant[chunk_idx_start+14] & 1) << 1) |
                            ((absQuant[chunk_idx_start+15] & 1) << 0);

                tmp_char2.w = ((absQuant[chunk_idx_start+16] & 1) << 7) |
                            ((absQuant[chunk_idx_start+17] & 1) << 6) |
                            ((absQuant[chunk_idx_start+18] & 1) << 5) |
                            ((absQuant[chunk_idx_start+19] & 1) << 4) |
                            ((absQuant[chunk_idx_start+20] & 1) << 3) |
                            ((absQuant[chunk_idx_start+21] & 1) << 2) |
                            ((absQuant[chunk_idx_start+22] & 1) << 1) |
                            ((absQuant[chunk_idx_start+23] & 1) << 0);
                
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4+1] = tmp_char2;
                cmp_byte_ofs+=8;

                int mask = 1;
                for(uint i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char1 = make_uchar4(0, 0, 0, 0);
                    tmp_char2 = make_uchar4(0, 0, 0, 0);

                    tmp_char1.x = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                    tmp_char1.y = (((absQuant[chunk_idx_start+32] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+33] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+34] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+35] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+36] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+37] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+38] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+39] & mask) >> i) << 0);

                    tmp_char1.z = (((absQuant[chunk_idx_start+40] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+41] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+42] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+43] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+44] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+45] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+46] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+47] & mask) >> i) << 0);

                    tmp_char1.w = (((absQuant[chunk_idx_start+48] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+49] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+50] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+51] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+52] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+53] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+54] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+55] & mask) >> i) << 0);
                    
                    tmp_char2.x = (((absQuant[chunk_idx_start+56] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+57] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+58] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+59] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+60] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+61] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+62] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+63] & mask) >> i) << 0);
                    mask <<= 1;
                    
                    if(!encoding_selection) tmp_char2.y = (((absQuant[chunk_idx_start] & mask) >> (i+1)) << 7);
                    tmp_char2.y = tmp_char2.y |
                                (((absQuant[chunk_idx_start+1] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+2] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+3] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+4] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+5] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+6] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+7] & mask) >> (i+1)) << 0);

                    tmp_char2.z = (((absQuant[chunk_idx_start+8] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+9] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+10] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+11] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+12] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+13] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+14] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+15] & mask) >> (i+1)) << 0);

                    tmp_char2.w = (((absQuant[chunk_idx_start+16] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+17] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+18] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+19] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+20] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+21] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+22] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+23] & mask) >> (i+1)) << 0);
                    
                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4+1] = tmp_char2;
                    cmp_byte_ofs+=8;
                }

                tmp_char1 = make_uchar4(0, 0, 0, 0);
                int i = fixed_rate[j] - 1;
                tmp_char1.x = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                tmp_char1.y = (((absQuant[chunk_idx_start+32] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+33] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+34] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+35] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+36] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+37] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+38] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+39] & mask) >> i) << 0);

                tmp_char1.z = (((absQuant[chunk_idx_start+40] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+41] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+42] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+43] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+44] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+45] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+46] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+47] & mask) >> i) << 0);

                tmp_char1.w = (((absQuant[chunk_idx_start+48] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+49] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+50] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+51] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+52] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+53] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+54] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+55] & mask) >> i) << 0);
                
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                cmp_byte_ofs+=4;
    
                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+56] & mask) >> i) << 7) |
                                        (((absQuant[chunk_idx_start+57] & mask) >> i) << 6) |
                                        (((absQuant[chunk_idx_start+58] & mask) >> i) << 5) |
                                        (((absQuant[chunk_idx_start+59] & mask) >> i) << 4) |
                                        (((absQuant[chunk_idx_start+60] & mask) >> i) << 3) |
                                        (((absQuant[chunk_idx_start+61] & mask) >> i) << 2) |
                                        (((absQuant[chunk_idx_start+62] & mask) >> i) << 1) |
                                        (((absQuant[chunk_idx_start+63] & mask) >> i) << 0);
            }
            else if(vec_ofs==2)
            {
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[2*j] >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[2*j] >> 16);
                
                tmp_char2 = make_uchar4(0, 0, 0, 0);
                tmp_char1.x = 0xff & (sign_flag[2*j] >> 8);
                tmp_char1.y = 0xff & sign_flag[2*j];
                tmp_char1.z = 0xff & (sign_flag[2*j+1] >> 24);
                tmp_char1.w = 0xff & (sign_flag[2*j+1] >> 16);
                tmp_char2.x = 0xff & (sign_flag[2*j+1] >> 8);
                tmp_char2.y = 0xff & sign_flag[2*j+1];

                if(!encoding_selection) tmp_char2.z = ((absQuant[chunk_idx_start] & 1) << 7);
                tmp_char2.z = tmp_char2.z |
                            ((absQuant[chunk_idx_start+1] & 1) << 6) |
                            ((absQuant[chunk_idx_start+2] & 1) << 5) |
                            ((absQuant[chunk_idx_start+3] & 1) << 4) |
                            ((absQuant[chunk_idx_start+4] & 1) << 3) |
                            ((absQuant[chunk_idx_start+5] & 1) << 2) |
                            ((absQuant[chunk_idx_start+6] & 1) << 1) |
                            ((absQuant[chunk_idx_start+7] & 1) << 0);

                tmp_char2.w = ((absQuant[chunk_idx_start+8] & 1) << 7) |
                            ((absQuant[chunk_idx_start+9] & 1) << 6) |
                            ((absQuant[chunk_idx_start+10] & 1) << 5) |
                            ((absQuant[chunk_idx_start+11] & 1) << 4) |
                            ((absQuant[chunk_idx_start+12] & 1) << 3) |
                            ((absQuant[chunk_idx_start+13] & 1) << 2) |
                            ((absQuant[chunk_idx_start+14] & 1) << 1) |
                            ((absQuant[chunk_idx_start+15] & 1) << 0);

                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4+1] = tmp_char2;
                cmp_byte_ofs+=8;

                int mask = 1;
                for(uint i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char1 = make_uchar4(0, 0, 0, 0);
                    tmp_char2 = make_uchar4(0, 0, 0, 0);

                    tmp_char1.x = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);

                    tmp_char1.y = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                    tmp_char1.z = (((absQuant[chunk_idx_start+32] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+33] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+34] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+35] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+36] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+37] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+38] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+39] & mask) >> i) << 0);

                    tmp_char1.w = (((absQuant[chunk_idx_start+40] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+41] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+42] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+43] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+44] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+45] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+46] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+47] & mask) >> i) << 0);

                    tmp_char2.x = (((absQuant[chunk_idx_start+48] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+49] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+50] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+51] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+52] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+53] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+54] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+55] & mask) >> i) << 0);
                    
                    tmp_char2.y = (((absQuant[chunk_idx_start+56] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+57] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+58] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+59] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+60] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+61] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+62] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+63] & mask) >> i) << 0);
                    mask <<= 1;
                    
                    if(!encoding_selection) tmp_char2.z = (((absQuant[chunk_idx_start] & mask) >> (i+1)) << 7);
                    tmp_char2.z = tmp_char2.z |
                                (((absQuant[chunk_idx_start+1] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+2] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+3] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+4] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+5] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+6] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+7] & mask) >> (i+1)) << 0);

                    tmp_char2.w = (((absQuant[chunk_idx_start+8] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+9] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+10] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+11] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+12] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+13] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+14] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+15] & mask) >> (i+1)) << 0);
                    
                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4+1] = tmp_char2;
                    cmp_byte_ofs+=8;
                }

                tmp_char1 = make_uchar4(0, 0, 0, 0);
                int i = fixed_rate[j] - 1;
                tmp_char1.x = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);

                tmp_char1.y = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                tmp_char1.z = (((absQuant[chunk_idx_start+32] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+33] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+34] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+35] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+36] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+37] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+38] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+39] & mask) >> i) << 0);

                tmp_char1.w = (((absQuant[chunk_idx_start+40] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+41] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+42] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+43] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+44] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+45] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+46] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+47] & mask) >> i) << 0);
                
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                cmp_byte_ofs+=4;

                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+48] & mask) >> i) << 7) |
                                        (((absQuant[chunk_idx_start+49] & mask) >> i) << 6) |
                                        (((absQuant[chunk_idx_start+50] & mask) >> i) << 5) |
                                        (((absQuant[chunk_idx_start+51] & mask) >> i) << 4) |
                                        (((absQuant[chunk_idx_start+52] & mask) >> i) << 3) |
                                        (((absQuant[chunk_idx_start+53] & mask) >> i) << 2) |
                                        (((absQuant[chunk_idx_start+54] & mask) >> i) << 1) |
                                        (((absQuant[chunk_idx_start+55] & mask) >> i) << 0);
    
                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+56] & mask) >> i) << 7) |
                                        (((absQuant[chunk_idx_start+57] & mask) >> i) << 6) |
                                        (((absQuant[chunk_idx_start+58] & mask) >> i) << 5) |
                                        (((absQuant[chunk_idx_start+59] & mask) >> i) << 4) |
                                        (((absQuant[chunk_idx_start+60] & mask) >> i) << 3) |
                                        (((absQuant[chunk_idx_start+61] & mask) >> i) << 2) |
                                        (((absQuant[chunk_idx_start+62] & mask) >> i) << 1) |
                                        (((absQuant[chunk_idx_start+63] & mask) >> i) << 0);
            }
            else
            {
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[2*j] >> 24);
                
                tmp_char2 = make_uchar4(0, 0, 0, 0);
                tmp_char1.x = 0xff & (sign_flag[2*j] >> 16);
                tmp_char1.y = 0xff & (sign_flag[2*j] >> 8);
                tmp_char1.z = 0xff & sign_flag[2*j];
                tmp_char1.w = 0xff & (sign_flag[2*j+1] >> 24);
                tmp_char2.x = 0xff & (sign_flag[2*j+1] >> 16);
                tmp_char2.y = 0xff & (sign_flag[2*j+1] >> 8);
                tmp_char2.z = 0xff & sign_flag[2*j+1];

                if(!encoding_selection) tmp_char2.w = ((absQuant[chunk_idx_start] & 1) << 7);
                tmp_char2.w = tmp_char2.w |
                            ((absQuant[chunk_idx_start+1] & 1) << 6) |
                            ((absQuant[chunk_idx_start+2] & 1) << 5) |
                            ((absQuant[chunk_idx_start+3] & 1) << 4) |
                            ((absQuant[chunk_idx_start+4] & 1) << 3) |
                            ((absQuant[chunk_idx_start+5] & 1) << 2) |
                            ((absQuant[chunk_idx_start+6] & 1) << 1) |
                            ((absQuant[chunk_idx_start+7] & 1) << 0);

                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4+1] = tmp_char2;
                cmp_byte_ofs+=8;

                int mask = 1;
                for(uint i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char1 = make_uchar4(0, 0, 0, 0);
                    tmp_char2 = make_uchar4(0, 0, 0, 0);

                    tmp_char1.x = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                    tmp_char1.y = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);

                    tmp_char1.z = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                    tmp_char1.w = (((absQuant[chunk_idx_start+32] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+33] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+34] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+35] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+36] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+37] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+38] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+39] & mask) >> i) << 0);

                    tmp_char2.x = (((absQuant[chunk_idx_start+40] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+41] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+42] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+43] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+44] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+45] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+46] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+47] & mask) >> i) << 0);

                    tmp_char2.y = (((absQuant[chunk_idx_start+48] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+49] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+50] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+51] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+52] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+53] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+54] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+55] & mask) >> i) << 0);
                    
                    tmp_char2.z = (((absQuant[chunk_idx_start+56] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+57] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+58] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+59] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+60] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+61] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+62] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+63] & mask) >> i) << 0);
                    mask <<= 1;
                    
                    if(!encoding_selection) tmp_char2.w = (((absQuant[chunk_idx_start] & mask) >> (i+1)) << 7);
                    tmp_char2.w = tmp_char2.w |
                                (((absQuant[chunk_idx_start+1] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+2] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+3] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+4] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+5] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+6] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+7] & mask) >> (i+1)) << 0);
                    
                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4+1] = tmp_char2;
                    cmp_byte_ofs+=8;
                }

                tmp_char1 = make_uchar4(0, 0, 0, 0);
                int i = fixed_rate[j] - 1;
                tmp_char1.x = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                tmp_char1.y = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);

                tmp_char1.z = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                tmp_char1.w = (((absQuant[chunk_idx_start+32] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+33] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+34] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+35] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+36] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+37] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+38] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+39] & mask) >> i) << 0);

                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char1;
                cmp_byte_ofs+=4;

                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+40] & mask) >> i) << 7) |
                                        (((absQuant[chunk_idx_start+41] & mask) >> i) << 6) |
                                        (((absQuant[chunk_idx_start+42] & mask) >> i) << 5) |
                                        (((absQuant[chunk_idx_start+43] & mask) >> i) << 4) |
                                        (((absQuant[chunk_idx_start+44] & mask) >> i) << 3) |
                                        (((absQuant[chunk_idx_start+45] & mask) >> i) << 2) |
                                        (((absQuant[chunk_idx_start+46] & mask) >> i) << 1) |
                                        (((absQuant[chunk_idx_start+47] & mask) >> i) << 0);
                
                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+48] & mask) >> i) << 7) |
                                        (((absQuant[chunk_idx_start+49] & mask) >> i) << 6) |
                                        (((absQuant[chunk_idx_start+50] & mask) >> i) << 5) |
                                        (((absQuant[chunk_idx_start+51] & mask) >> i) << 4) |
                                        (((absQuant[chunk_idx_start+52] & mask) >> i) << 3) |
                                        (((absQuant[chunk_idx_start+53] & mask) >> i) << 2) |
                                        (((absQuant[chunk_idx_start+54] & mask) >> i) << 1) |
                                        (((absQuant[chunk_idx_start+55] & mask) >> i) << 0);
    
                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+56] & mask) >> i) << 7) |
                                        (((absQuant[chunk_idx_start+57] & mask) >> i) << 6) |
                                        (((absQuant[chunk_idx_start+58] & mask) >> i) << 5) |
                                        (((absQuant[chunk_idx_start+59] & mask) >> i) << 4) |
                                        (((absQuant[chunk_idx_start+60] & mask) >> i) << 3) |
                                        (((absQuant[chunk_idx_start+61] & mask) >> i) << 2) |
                                        (((absQuant[chunk_idx_start+62] & mask) >> i) << 1) |
                                        (((absQuant[chunk_idx_start+63] & mask) >> i) << 0);
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}


__global__ void cuSZp_decompress_kernel_3D_outlier_vec4_f64(double* const __restrict__ decData, 
                                                            const unsigned char* const __restrict__ cmpData, 
                                                            volatile unsigned int* const __restrict__ cmpOffset, 
                                                            volatile unsigned int* const __restrict__ locOffset, 
                                                            volatile int* const __restrict__ flag, 
                                                            uint blockNum, const uint3 dims, const double eb)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 3) / 4;
    const uint dimxBlock = (dims.x + 3) / 4;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    uint base_start_block_idx;
    int absQuant[64];
    int sign_ofs;
    unsigned char fixed_rate[block_per_thread];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char1, tmp_char2;

    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        uint block_idx = base_start_block_idx + j * 32 + lane; // Data block that is currently being processed

        if(block_idx < blockNum)
        {
            fixed_rate[j] = cmpData[block_idx];

            int encoding_selection = fixed_rate[j] >> 7;
            int outlier = ((fixed_rate[j] & 0x60) >> 5) + 1;
            int temp_rate = fixed_rate[j] & 0x1f;
            if(!encoding_selection) thread_ofs += temp_rate ? (8 + temp_rate * 8) : 0;
            else thread_ofs += 8 + temp_rate * 8 + outlier;
        }
        else fixed_rate[j] = 0;
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
    for(uint j=0; j<block_per_thread; j++)
    {
        uint block_idx = base_start_block_idx + j * 32 + lane; // Data block that is currently being processed
        uint block_stride_per_slice = dimyBlock * dimxBlock;
        uint block_idx_z = block_idx / block_stride_per_slice;
        uint block_idx_y = (block_idx % block_stride_per_slice) / dimxBlock;
        uint block_idx_x = (block_idx % block_stride_per_slice) % dimxBlock;
        unsigned int sign_flag[2] = {0, 0};

        int encoding_selection = fixed_rate[j] >> 7;
        int outlier_byte_num = ((fixed_rate[j] & 0x60) >> 5) + 1;
        fixed_rate[j] &= 0x1f;
        int outlier_buffer = 0;

        if(!encoding_selection) tmp_byte_ofs = (fixed_rate[j]) ? (8 + fixed_rate[j] * 8) : 0;
        else tmp_byte_ofs = 8 + outlier_byte_num + fixed_rate[j] * 8;
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
                sign_flag[0] = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                               (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16)) |
                               (0x0000ff00 & (cmpData[cmp_byte_ofs++] << 8))  |
                               (0x000000ff & cmpData[cmp_byte_ofs++]);
                sign_flag[1] = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                               (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16)) |
                               (0x0000ff00 & (cmpData[cmp_byte_ofs++] << 8))  |
                               (0x000000ff & cmpData[cmp_byte_ofs++]);
                absQuant[0] = outlier_buffer;
                for(uint i=1; i<64; i++) absQuant[i] = 0;
                
                int currQuant, lorenQuant;
                int prevQuant_z = 0;
                for(uint i=0; i<4; i++)
                {
                    uint data_idx_z = block_idx_z * 4 + i;
                    int prevQuant_y = 0;

                    if(data_idx_z < dims.z)
                    {
                        for(uint k=0; k<4; k++)
                        {
                            uint data_idx_y = block_idx_y * 4 + k;
                            uint block_ofs = i * 16 + k * 4;
                            int prevQuant_x = 0;

                            if(data_idx_y < dims.y)
                            {
                                uint data_idx_x = block_idx_x * 4;
                                uint data_idx = data_idx_z * dims.y * dims.x + data_idx_y * dims.x + data_idx_x;
                                double4 dec_buffer;

                                sign_ofs = block_ofs % 32;
                                lorenQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs] * -1 : absQuant[block_ofs];
                                if(k) currQuant = lorenQuant + prevQuant_y; // Y-de-delta
                                else { 
                                    currQuant = lorenQuant + prevQuant_z;   // Z-de-delta
                                    prevQuant_z = currQuant;
                                }
                                prevQuant_y = currQuant;
                                prevQuant_x = currQuant;
                                dec_buffer.x = currQuant * eb * 2;

                                sign_ofs = (block_ofs + 1) % 32;
                                lorenQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs+1] * -1 : absQuant[block_ofs+1];
                                currQuant = lorenQuant + prevQuant_x;
                                prevQuant_x = currQuant;
                                dec_buffer.y = currQuant * eb * 2;

                                sign_ofs = (block_ofs + 2) % 32;
                                lorenQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs+2] * -1 : absQuant[block_ofs+2];
                                currQuant = lorenQuant + prevQuant_x;
                                prevQuant_x = currQuant;
                                dec_buffer.z = currQuant * eb * 2;

                                sign_ofs = (block_ofs + 3) % 32;
                                lorenQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs+3] * -1 : absQuant[block_ofs+3];
                                currQuant = lorenQuant + prevQuant_x;
                                prevQuant_x = currQuant;
                                dec_buffer.w = currQuant * eb * 2;
                                
                                reinterpret_cast<double4*>(decData)[data_idx/4] = dec_buffer;
                            }
                        }
                    }
                }
            }
        }

        if(block_idx < blockNum && fixed_rate[j])
        {
            int vec_ofs = cmp_byte_ofs % 4;
            if(vec_ofs==0)
            {
                for(uint i=0; i<64; i++) absQuant[i] = 0;

                if(encoding_selection) absQuant[0] = outlier_buffer;

                tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                tmp_char2 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4+1];
                sign_flag[0] = (0xff000000 & (tmp_char1.x << 24)) |
                               (0x00ff0000 & (tmp_char1.y << 16)) |
                               (0x0000ff00 & (tmp_char1.z << 8))  |
                               (0x000000ff & tmp_char1.w);
                sign_flag[1] = (0xff000000 & (tmp_char2.x << 24)) |
                               (0x00ff0000 & (tmp_char2.y << 16)) |
                               (0x0000ff00 & (tmp_char2.z << 8))  |
                               (0x000000ff & tmp_char2.w);
                cmp_byte_ofs+=8;

                for(uint i=0; i<fixed_rate[j]; i++)
                {
                    tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                    tmp_char2 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4+1];
                    cmp_byte_ofs+=8;

                    if(!encoding_selection) absQuant[0] |= ((tmp_char1.x >> 7) & 0x00000001) << i;
                    absQuant[1] |= ((tmp_char1.x >> 6) & 0x00000001) << i;
                    absQuant[2] |= ((tmp_char1.x >> 5) & 0x00000001) << i;
                    absQuant[3] |= ((tmp_char1.x >> 4) & 0x00000001) << i;
                    absQuant[4] |= ((tmp_char1.x >> 3) & 0x00000001) << i;
                    absQuant[5] |= ((tmp_char1.x >> 2) & 0x00000001) << i;
                    absQuant[6] |= ((tmp_char1.x >> 1) & 0x00000001) << i;
                    absQuant[7] |= ((tmp_char1.x >> 0) & 0x00000001) << i;

                    absQuant[8] |= ((tmp_char1.y >> 7) & 0x00000001) << i;
                    absQuant[9] |= ((tmp_char1.y >> 6) & 0x00000001) << i;
                    absQuant[10] |= ((tmp_char1.y >> 5) & 0x00000001) << i;
                    absQuant[11] |= ((tmp_char1.y >> 4) & 0x00000001) << i;
                    absQuant[12] |= ((tmp_char1.y >> 3) & 0x00000001) << i;
                    absQuant[13] |= ((tmp_char1.y >> 2) & 0x00000001) << i;
                    absQuant[14] |= ((tmp_char1.y >> 1) & 0x00000001) << i;
                    absQuant[15] |= ((tmp_char1.y >> 0) & 0x00000001) << i;

                    absQuant[16] |= ((tmp_char1.z >> 7) & 0x00000001) << i;
                    absQuant[17] |= ((tmp_char1.z >> 6) & 0x00000001) << i;
                    absQuant[18] |= ((tmp_char1.z >> 5) & 0x00000001) << i;
                    absQuant[19] |= ((tmp_char1.z >> 4) & 0x00000001) << i;
                    absQuant[20] |= ((tmp_char1.z >> 3) & 0x00000001) << i;
                    absQuant[21] |= ((tmp_char1.z >> 2) & 0x00000001) << i;
                    absQuant[22] |= ((tmp_char1.z >> 1) & 0x00000001) << i;
                    absQuant[23] |= ((tmp_char1.z >> 0) & 0x00000001) << i;

                    absQuant[24] |= ((tmp_char1.w >> 7) & 0x00000001) << i;
                    absQuant[25] |= ((tmp_char1.w >> 6) & 0x00000001) << i;
                    absQuant[26] |= ((tmp_char1.w >> 5) & 0x00000001) << i;
                    absQuant[27] |= ((tmp_char1.w >> 4) & 0x00000001) << i;
                    absQuant[28] |= ((tmp_char1.w >> 3) & 0x00000001) << i;
                    absQuant[29] |= ((tmp_char1.w >> 2) & 0x00000001) << i;
                    absQuant[30] |= ((tmp_char1.w >> 1) & 0x00000001) << i;
                    absQuant[31] |= ((tmp_char1.w >> 0) & 0x00000001) << i;

                    absQuant[32] |= ((tmp_char2.x >> 7) & 0x00000001) << i;
                    absQuant[33] |= ((tmp_char2.x >> 6) & 0x00000001) << i;
                    absQuant[34] |= ((tmp_char2.x >> 5) & 0x00000001) << i;
                    absQuant[35] |= ((tmp_char2.x >> 4) & 0x00000001) << i;
                    absQuant[36] |= ((tmp_char2.x >> 3) & 0x00000001) << i;
                    absQuant[37] |= ((tmp_char2.x >> 2) & 0x00000001) << i;
                    absQuant[38] |= ((tmp_char2.x >> 1) & 0x00000001) << i;
                    absQuant[39] |= ((tmp_char2.x >> 0) & 0x00000001) << i;

                    absQuant[40] |= ((tmp_char2.y >> 7) & 0x00000001) << i;
                    absQuant[41] |= ((tmp_char2.y >> 6) & 0x00000001) << i;
                    absQuant[42] |= ((tmp_char2.y >> 5) & 0x00000001) << i;
                    absQuant[43] |= ((tmp_char2.y >> 4) & 0x00000001) << i;
                    absQuant[44] |= ((tmp_char2.y >> 3) & 0x00000001) << i;
                    absQuant[45] |= ((tmp_char2.y >> 2) & 0x00000001) << i;
                    absQuant[46] |= ((tmp_char2.y >> 1) & 0x00000001) << i;
                    absQuant[47] |= ((tmp_char2.y >> 0) & 0x00000001) << i;

                    absQuant[48] |= ((tmp_char2.z >> 7) & 0x00000001) << i;
                    absQuant[49] |= ((tmp_char2.z >> 6) & 0x00000001) << i;
                    absQuant[50] |= ((tmp_char2.z >> 5) & 0x00000001) << i;
                    absQuant[51] |= ((tmp_char2.z >> 4) & 0x00000001) << i;
                    absQuant[52] |= ((tmp_char2.z >> 3) & 0x00000001) << i;
                    absQuant[53] |= ((tmp_char2.z >> 2) & 0x00000001) << i;
                    absQuant[54] |= ((tmp_char2.z >> 1) & 0x00000001) << i;
                    absQuant[55] |= ((tmp_char2.z >> 0) & 0x00000001) << i;

                    absQuant[56] |= ((tmp_char2.w >> 7) & 0x00000001) << i;
                    absQuant[57] |= ((tmp_char2.w >> 6) & 0x00000001) << i;
                    absQuant[58] |= ((tmp_char2.w >> 5) & 0x00000001) << i;
                    absQuant[59] |= ((tmp_char2.w >> 4) & 0x00000001) << i;
                    absQuant[60] |= ((tmp_char2.w >> 3) & 0x00000001) << i;
                    absQuant[61] |= ((tmp_char2.w >> 2) & 0x00000001) << i;
                    absQuant[62] |= ((tmp_char2.w >> 1) & 0x00000001) << i;
                    absQuant[63] |= ((tmp_char2.w >> 0) & 0x00000001) << i;
                }
            }
            else if(vec_ofs==1)
            {
                for(uint i=0; i<64; i++) absQuant[i] = 0;

                if(encoding_selection) absQuant[0] = outlier_buffer;

                sign_flag[0] = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                               (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16)) |
                               (0x0000ff00 & (cmpData[cmp_byte_ofs++] << 8));

                tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                tmp_char2 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4+1];
                sign_flag[0] |= (0x000000ff & tmp_char1.x);
                sign_flag[1] = (0xff000000 & (tmp_char1.y << 24)) |
                               (0x00ff0000 & (tmp_char1.z << 16)) |
                               (0x0000ff00 & (tmp_char1.w << 8))  |
                               (0x000000ff & tmp_char2.x);
                cmp_byte_ofs+=8;

                if(!encoding_selection) absQuant[0] |= ((tmp_char2.y >> 7) & 0x00000001);
                absQuant[1] |= ((tmp_char2.y >> 6) & 0x00000001);
                absQuant[2] |= ((tmp_char2.y >> 5) & 0x00000001);
                absQuant[3] |= ((tmp_char2.y >> 4) & 0x00000001);
                absQuant[4] |= ((tmp_char2.y >> 3) & 0x00000001);
                absQuant[5] |= ((tmp_char2.y >> 2) & 0x00000001);
                absQuant[6] |= ((tmp_char2.y >> 1) & 0x00000001);
                absQuant[7] |= ((tmp_char2.y >> 0) & 0x00000001);

                absQuant[8] |= ((tmp_char2.z >> 7) & 0x00000001);
                absQuant[9] |= ((tmp_char2.z >> 6) & 0x00000001);
                absQuant[10] |= ((tmp_char2.z >> 5) & 0x00000001);
                absQuant[11] |= ((tmp_char2.z >> 4) & 0x00000001);
                absQuant[12] |= ((tmp_char2.z >> 3) & 0x00000001);
                absQuant[13] |= ((tmp_char2.z >> 2) & 0x00000001);
                absQuant[14] |= ((tmp_char2.z >> 1) & 0x00000001);
                absQuant[15] |= ((tmp_char2.z >> 0) & 0x00000001);

                absQuant[16] |= ((tmp_char2.w >> 7) & 0x00000001);
                absQuant[17] |= ((tmp_char2.w >> 6) & 0x00000001);
                absQuant[18] |= ((tmp_char2.w >> 5) & 0x00000001);
                absQuant[19] |= ((tmp_char2.w >> 4) & 0x00000001);
                absQuant[20] |= ((tmp_char2.w >> 3) & 0x00000001);
                absQuant[21] |= ((tmp_char2.w >> 2) & 0x00000001);
                absQuant[22] |= ((tmp_char2.w >> 1) & 0x00000001);
                absQuant[23] |= ((tmp_char2.w >> 0) & 0x00000001);

                for(uint i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                    tmp_char2 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4+1];
                    cmp_byte_ofs+=8;

                    absQuant[24] |= ((tmp_char1.x >> 7) & 0x00000001) << i;
                    absQuant[25] |= ((tmp_char1.x >> 6) & 0x00000001) << i;
                    absQuant[26] |= ((tmp_char1.x >> 5) & 0x00000001) << i;
                    absQuant[27] |= ((tmp_char1.x >> 4) & 0x00000001) << i;
                    absQuant[28] |= ((tmp_char1.x >> 3) & 0x00000001) << i;
                    absQuant[29] |= ((tmp_char1.x >> 2) & 0x00000001) << i;
                    absQuant[30] |= ((tmp_char1.x >> 1) & 0x00000001) << i;
                    absQuant[31] |= ((tmp_char1.x >> 0) & 0x00000001) << i;

                    absQuant[32] |= ((tmp_char1.y >> 7) & 0x00000001) << i;
                    absQuant[33] |= ((tmp_char1.y >> 6) & 0x00000001) << i;
                    absQuant[34] |= ((tmp_char1.y >> 5) & 0x00000001) << i;
                    absQuant[35] |= ((tmp_char1.y >> 4) & 0x00000001) << i;
                    absQuant[36] |= ((tmp_char1.y >> 3) & 0x00000001) << i;
                    absQuant[37] |= ((tmp_char1.y >> 2) & 0x00000001) << i;
                    absQuant[38] |= ((tmp_char1.y >> 1) & 0x00000001) << i;
                    absQuant[39] |= ((tmp_char1.y >> 0) & 0x00000001) << i;

                    absQuant[40] |= ((tmp_char1.z >> 7) & 0x00000001) << i;
                    absQuant[41] |= ((tmp_char1.z >> 6) & 0x00000001) << i;
                    absQuant[42] |= ((tmp_char1.z >> 5) & 0x00000001) << i;
                    absQuant[43] |= ((tmp_char1.z >> 4) & 0x00000001) << i;
                    absQuant[44] |= ((tmp_char1.z >> 3) & 0x00000001) << i;
                    absQuant[45] |= ((tmp_char1.z >> 2) & 0x00000001) << i;
                    absQuant[46] |= ((tmp_char1.z >> 1) & 0x00000001) << i;
                    absQuant[47] |= ((tmp_char1.z >> 0) & 0x00000001) << i;

                    absQuant[48] |= ((tmp_char1.w >> 7) & 0x00000001) << i;
                    absQuant[49] |= ((tmp_char1.w >> 6) & 0x00000001) << i;
                    absQuant[50] |= ((tmp_char1.w >> 5) & 0x00000001) << i;
                    absQuant[51] |= ((tmp_char1.w >> 4) & 0x00000001) << i;
                    absQuant[52] |= ((tmp_char1.w >> 3) & 0x00000001) << i;
                    absQuant[53] |= ((tmp_char1.w >> 2) & 0x00000001) << i;
                    absQuant[54] |= ((tmp_char1.w >> 1) & 0x00000001) << i;
                    absQuant[55] |= ((tmp_char1.w >> 0) & 0x00000001) << i;

                    absQuant[56] |= ((tmp_char2.x >> 7) & 0x00000001) << i;
                    absQuant[57] |= ((tmp_char2.x >> 6) & 0x00000001) << i;
                    absQuant[58] |= ((tmp_char2.x >> 5) & 0x00000001) << i;
                    absQuant[59] |= ((tmp_char2.x >> 4) & 0x00000001) << i;
                    absQuant[60] |= ((tmp_char2.x >> 3) & 0x00000001) << i;
                    absQuant[61] |= ((tmp_char2.x >> 2) & 0x00000001) << i;
                    absQuant[62] |= ((tmp_char2.x >> 1) & 0x00000001) << i;
                    absQuant[63] |= ((tmp_char2.x >> 0) & 0x00000001) << i;

                    if(!encoding_selection) absQuant[0] |= ((tmp_char2.y >> 7) & 0x00000001) << (i+1);
                    absQuant[1] |= ((tmp_char2.y >> 6) & 0x00000001) << (i+1);
                    absQuant[2] |= ((tmp_char2.y >> 5) & 0x00000001) << (i+1);
                    absQuant[3] |= ((tmp_char2.y >> 4) & 0x00000001) << (i+1);
                    absQuant[4] |= ((tmp_char2.y >> 3) & 0x00000001) << (i+1);
                    absQuant[5] |= ((tmp_char2.y >> 2) & 0x00000001) << (i+1);
                    absQuant[6] |= ((tmp_char2.y >> 1) & 0x00000001) << (i+1);
                    absQuant[7] |= ((tmp_char2.y >> 0) & 0x00000001) << (i+1);

                    absQuant[8] |= ((tmp_char2.z >> 7) & 0x00000001) << (i+1);
                    absQuant[9] |= ((tmp_char2.z >> 6) & 0x00000001) << (i+1);
                    absQuant[10] |= ((tmp_char2.z >> 5) & 0x00000001) << (i+1);
                    absQuant[11] |= ((tmp_char2.z >> 4) & 0x00000001) << (i+1);
                    absQuant[12] |= ((tmp_char2.z >> 3) & 0x00000001) << (i+1);
                    absQuant[13] |= ((tmp_char2.z >> 2) & 0x00000001) << (i+1);
                    absQuant[14] |= ((tmp_char2.z >> 1) & 0x00000001) << (i+1);
                    absQuant[15] |= ((tmp_char2.z >> 0) & 0x00000001) << (i+1);

                    absQuant[16] |= ((tmp_char2.w >> 7) & 0x00000001) << (i+1);
                    absQuant[17] |= ((tmp_char2.w >> 6) & 0x00000001) << (i+1);
                    absQuant[18] |= ((tmp_char2.w >> 5) & 0x00000001) << (i+1);
                    absQuant[19] |= ((tmp_char2.w >> 4) & 0x00000001) << (i+1);
                    absQuant[20] |= ((tmp_char2.w >> 3) & 0x00000001) << (i+1);
                    absQuant[21] |= ((tmp_char2.w >> 2) & 0x00000001) << (i+1);
                    absQuant[22] |= ((tmp_char2.w >> 1) & 0x00000001) << (i+1);
                    absQuant[23] |= ((tmp_char2.w >> 0) & 0x00000001) << (i+1);
                }

                tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                cmp_byte_ofs+=4;
                int i = fixed_rate[j]-1;

                absQuant[24] |= ((tmp_char1.x >> 7) & 0x00000001) << i;
                absQuant[25] |= ((tmp_char1.x >> 6) & 0x00000001) << i;
                absQuant[26] |= ((tmp_char1.x >> 5) & 0x00000001) << i;
                absQuant[27] |= ((tmp_char1.x >> 4) & 0x00000001) << i;
                absQuant[28] |= ((tmp_char1.x >> 3) & 0x00000001) << i;
                absQuant[29] |= ((tmp_char1.x >> 2) & 0x00000001) << i;
                absQuant[30] |= ((tmp_char1.x >> 1) & 0x00000001) << i;
                absQuant[31] |= ((tmp_char1.x >> 0) & 0x00000001) << i;

                absQuant[32] |= ((tmp_char1.y >> 7) & 0x00000001) << i;
                absQuant[33] |= ((tmp_char1.y >> 6) & 0x00000001) << i;
                absQuant[34] |= ((tmp_char1.y >> 5) & 0x00000001) << i;
                absQuant[35] |= ((tmp_char1.y >> 4) & 0x00000001) << i;
                absQuant[36] |= ((tmp_char1.y >> 3) & 0x00000001) << i;
                absQuant[37] |= ((tmp_char1.y >> 2) & 0x00000001) << i;
                absQuant[38] |= ((tmp_char1.y >> 1) & 0x00000001) << i;
                absQuant[39] |= ((tmp_char1.y >> 0) & 0x00000001) << i;

                absQuant[40] |= ((tmp_char1.z >> 7) & 0x00000001) << i;
                absQuant[41] |= ((tmp_char1.z >> 6) & 0x00000001) << i;
                absQuant[42] |= ((tmp_char1.z >> 5) & 0x00000001) << i;
                absQuant[43] |= ((tmp_char1.z >> 4) & 0x00000001) << i;
                absQuant[44] |= ((tmp_char1.z >> 3) & 0x00000001) << i;
                absQuant[45] |= ((tmp_char1.z >> 2) & 0x00000001) << i;
                absQuant[46] |= ((tmp_char1.z >> 1) & 0x00000001) << i;
                absQuant[47] |= ((tmp_char1.z >> 0) & 0x00000001) << i;

                absQuant[48] |= ((tmp_char1.w >> 7) & 0x00000001) << i;
                absQuant[49] |= ((tmp_char1.w >> 6) & 0x00000001) << i;
                absQuant[50] |= ((tmp_char1.w >> 5) & 0x00000001) << i;
                absQuant[51] |= ((tmp_char1.w >> 4) & 0x00000001) << i;
                absQuant[52] |= ((tmp_char1.w >> 3) & 0x00000001) << i;
                absQuant[53] |= ((tmp_char1.w >> 2) & 0x00000001) << i;
                absQuant[54] |= ((tmp_char1.w >> 1) & 0x00000001) << i;
                absQuant[55] |= ((tmp_char1.w >> 0) & 0x00000001) << i;

                int uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[56] |= ((uchar_buffer >> 7) & 0x00000001) << i;
                absQuant[57] |= ((uchar_buffer >> 6) & 0x00000001) << i;
                absQuant[58] |= ((uchar_buffer >> 5) & 0x00000001) << i;
                absQuant[59] |= ((uchar_buffer >> 4) & 0x00000001) << i;
                absQuant[60] |= ((uchar_buffer >> 3) & 0x00000001) << i;
                absQuant[61] |= ((uchar_buffer >> 2) & 0x00000001) << i;
                absQuant[62] |= ((uchar_buffer >> 1) & 0x00000001) << i;
                absQuant[63] |= ((uchar_buffer >> 0) & 0x00000001) << i;
            }
            else if(vec_ofs==2)
            {
                for(uint i=0; i<64; i++) absQuant[i] = 0;
                
                if(encoding_selection) absQuant[0] = outlier_buffer;

                sign_flag[0] = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                               (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16));

                tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                tmp_char2 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4+1];
                sign_flag[0] |= (0x0000ff00 & (tmp_char1.x << 8)) |
                                (0x000000ff & tmp_char1.y);
                sign_flag[1] = (0xff000000 & (tmp_char1.z << 24)) |
                               (0x00ff0000 & (tmp_char1.w << 16)) |
                               (0x0000ff00 & (tmp_char2.x << 8))  |
                               (0x000000ff & tmp_char2.y);
                cmp_byte_ofs+=8;

                if(!encoding_selection) absQuant[0] |= ((tmp_char2.z >> 7) & 0x00000001);
                absQuant[1] |= ((tmp_char2.z >> 6) & 0x00000001);
                absQuant[2] |= ((tmp_char2.z >> 5) & 0x00000001);
                absQuant[3] |= ((tmp_char2.z >> 4) & 0x00000001);
                absQuant[4] |= ((tmp_char2.z >> 3) & 0x00000001);
                absQuant[5] |= ((tmp_char2.z >> 2) & 0x00000001);
                absQuant[6] |= ((tmp_char2.z >> 1) & 0x00000001);
                absQuant[7] |= ((tmp_char2.z >> 0) & 0x00000001);

                absQuant[8] |= ((tmp_char2.w >> 7) & 0x00000001);
                absQuant[9] |= ((tmp_char2.w >> 6) & 0x00000001);
                absQuant[10] |= ((tmp_char2.w >> 5) & 0x00000001);
                absQuant[11] |= ((tmp_char2.w >> 4) & 0x00000001);
                absQuant[12] |= ((tmp_char2.w >> 3) & 0x00000001);
                absQuant[13] |= ((tmp_char2.w >> 2) & 0x00000001);
                absQuant[14] |= ((tmp_char2.w >> 1) & 0x00000001);
                absQuant[15] |= ((tmp_char2.w >> 0) & 0x00000001);

                for(uint i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                    tmp_char2 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4+1];
                    cmp_byte_ofs+=8;

                    absQuant[16] |= ((tmp_char1.x >> 7) & 0x00000001) << i;
                    absQuant[17] |= ((tmp_char1.x >> 6) & 0x00000001) << i;
                    absQuant[18] |= ((tmp_char1.x >> 5) & 0x00000001) << i;
                    absQuant[19] |= ((tmp_char1.x >> 4) & 0x00000001) << i;
                    absQuant[20] |= ((tmp_char1.x >> 3) & 0x00000001) << i;
                    absQuant[21] |= ((tmp_char1.x >> 2) & 0x00000001) << i;
                    absQuant[22] |= ((tmp_char1.x >> 1) & 0x00000001) << i;
                    absQuant[23] |= ((tmp_char1.x >> 0) & 0x00000001) << i;

                    absQuant[24] |= ((tmp_char1.y >> 7) & 0x00000001) << i;
                    absQuant[25] |= ((tmp_char1.y >> 6) & 0x00000001) << i;
                    absQuant[26] |= ((tmp_char1.y >> 5) & 0x00000001) << i;
                    absQuant[27] |= ((tmp_char1.y >> 4) & 0x00000001) << i;
                    absQuant[28] |= ((tmp_char1.y >> 3) & 0x00000001) << i;
                    absQuant[29] |= ((tmp_char1.y >> 2) & 0x00000001) << i;
                    absQuant[30] |= ((tmp_char1.y >> 1) & 0x00000001) << i;
                    absQuant[31] |= ((tmp_char1.y >> 0) & 0x00000001) << i;

                    absQuant[32] |= ((tmp_char1.z >> 7) & 0x00000001) << i;
                    absQuant[33] |= ((tmp_char1.z >> 6) & 0x00000001) << i;
                    absQuant[34] |= ((tmp_char1.z >> 5) & 0x00000001) << i;
                    absQuant[35] |= ((tmp_char1.z >> 4) & 0x00000001) << i;
                    absQuant[36] |= ((tmp_char1.z >> 3) & 0x00000001) << i;
                    absQuant[37] |= ((tmp_char1.z >> 2) & 0x00000001) << i;
                    absQuant[38] |= ((tmp_char1.z >> 1) & 0x00000001) << i;
                    absQuant[39] |= ((tmp_char1.z >> 0) & 0x00000001) << i;

                    absQuant[40] |= ((tmp_char1.w >> 7) & 0x00000001) << i;
                    absQuant[41] |= ((tmp_char1.w >> 6) & 0x00000001) << i;
                    absQuant[42] |= ((tmp_char1.w >> 5) & 0x00000001) << i;
                    absQuant[43] |= ((tmp_char1.w >> 4) & 0x00000001) << i;
                    absQuant[44] |= ((tmp_char1.w >> 3) & 0x00000001) << i;
                    absQuant[45] |= ((tmp_char1.w >> 2) & 0x00000001) << i;
                    absQuant[46] |= ((tmp_char1.w >> 1) & 0x00000001) << i;
                    absQuant[47] |= ((tmp_char1.w >> 0) & 0x00000001) << i;

                    absQuant[48] |= ((tmp_char2.x >> 7) & 0x00000001) << i;
                    absQuant[49] |= ((tmp_char2.x >> 6) & 0x00000001) << i;
                    absQuant[50] |= ((tmp_char2.x >> 5) & 0x00000001) << i;
                    absQuant[51] |= ((tmp_char2.x >> 4) & 0x00000001) << i;
                    absQuant[52] |= ((tmp_char2.x >> 3) & 0x00000001) << i;
                    absQuant[53] |= ((tmp_char2.x >> 2) & 0x00000001) << i;
                    absQuant[54] |= ((tmp_char2.x >> 1) & 0x00000001) << i;
                    absQuant[55] |= ((tmp_char2.x >> 0) & 0x00000001) << i;

                    absQuant[56] |= ((tmp_char2.y >> 7) & 0x00000001) << i;
                    absQuant[57] |= ((tmp_char2.y >> 6) & 0x00000001) << i;
                    absQuant[58] |= ((tmp_char2.y >> 5) & 0x00000001) << i;
                    absQuant[59] |= ((tmp_char2.y >> 4) & 0x00000001) << i;
                    absQuant[60] |= ((tmp_char2.y >> 3) & 0x00000001) << i;
                    absQuant[61] |= ((tmp_char2.y >> 2) & 0x00000001) << i;
                    absQuant[62] |= ((tmp_char2.y >> 1) & 0x00000001) << i;
                    absQuant[63] |= ((tmp_char2.y >> 0) & 0x00000001) << i;

                    if(!encoding_selection) absQuant[0] |= ((tmp_char2.z >> 7) & 0x00000001) << (i+1);
                    absQuant[1] |= ((tmp_char2.z >> 6) & 0x00000001) << (i+1);
                    absQuant[2] |= ((tmp_char2.z >> 5) & 0x00000001) << (i+1);
                    absQuant[3] |= ((tmp_char2.z >> 4) & 0x00000001) << (i+1);
                    absQuant[4] |= ((tmp_char2.z >> 3) & 0x00000001) << (i+1);
                    absQuant[5] |= ((tmp_char2.z >> 2) & 0x00000001) << (i+1);
                    absQuant[6] |= ((tmp_char2.z >> 1) & 0x00000001) << (i+1);
                    absQuant[7] |= ((tmp_char2.z >> 0) & 0x00000001) << (i+1);

                    absQuant[8] |= ((tmp_char2.w >> 7) & 0x00000001) << (i+1);
                    absQuant[9] |= ((tmp_char2.w >> 6) & 0x00000001) << (i+1);
                    absQuant[10] |= ((tmp_char2.w >> 5) & 0x00000001) << (i+1);
                    absQuant[11] |= ((tmp_char2.w >> 4) & 0x00000001) << (i+1);
                    absQuant[12] |= ((tmp_char2.w >> 3) & 0x00000001) << (i+1);
                    absQuant[13] |= ((tmp_char2.w >> 2) & 0x00000001) << (i+1);
                    absQuant[14] |= ((tmp_char2.w >> 1) & 0x00000001) << (i+1);
                    absQuant[15] |= ((tmp_char2.w >> 0) & 0x00000001) << (i+1);
                }

                tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                cmp_byte_ofs+=4;
                int i = fixed_rate[j]-1;

                absQuant[16] |= ((tmp_char1.x >> 7) & 0x00000001) << i;
                absQuant[17] |= ((tmp_char1.x >> 6) & 0x00000001) << i;
                absQuant[18] |= ((tmp_char1.x >> 5) & 0x00000001) << i;
                absQuant[19] |= ((tmp_char1.x >> 4) & 0x00000001) << i;
                absQuant[20] |= ((tmp_char1.x >> 3) & 0x00000001) << i;
                absQuant[21] |= ((tmp_char1.x >> 2) & 0x00000001) << i;
                absQuant[22] |= ((tmp_char1.x >> 1) & 0x00000001) << i;
                absQuant[23] |= ((tmp_char1.x >> 0) & 0x00000001) << i;

                absQuant[24] |= ((tmp_char1.y >> 7) & 0x00000001) << i;
                absQuant[25] |= ((tmp_char1.y >> 6) & 0x00000001) << i;
                absQuant[26] |= ((tmp_char1.y >> 5) & 0x00000001) << i;
                absQuant[27] |= ((tmp_char1.y >> 4) & 0x00000001) << i;
                absQuant[28] |= ((tmp_char1.y >> 3) & 0x00000001) << i;
                absQuant[29] |= ((tmp_char1.y >> 2) & 0x00000001) << i;
                absQuant[30] |= ((tmp_char1.y >> 1) & 0x00000001) << i;
                absQuant[31] |= ((tmp_char1.y >> 0) & 0x00000001) << i;

                absQuant[32] |= ((tmp_char1.z >> 7) & 0x00000001) << i;
                absQuant[33] |= ((tmp_char1.z >> 6) & 0x00000001) << i;
                absQuant[34] |= ((tmp_char1.z >> 5) & 0x00000001) << i;
                absQuant[35] |= ((tmp_char1.z >> 4) & 0x00000001) << i;
                absQuant[36] |= ((tmp_char1.z >> 3) & 0x00000001) << i;
                absQuant[37] |= ((tmp_char1.z >> 2) & 0x00000001) << i;
                absQuant[38] |= ((tmp_char1.z >> 1) & 0x00000001) << i;
                absQuant[39] |= ((tmp_char1.z >> 0) & 0x00000001) << i;

                absQuant[40] |= ((tmp_char1.w >> 7) & 0x00000001) << i;
                absQuant[41] |= ((tmp_char1.w >> 6) & 0x00000001) << i;
                absQuant[42] |= ((tmp_char1.w >> 5) & 0x00000001) << i;
                absQuant[43] |= ((tmp_char1.w >> 4) & 0x00000001) << i;
                absQuant[44] |= ((tmp_char1.w >> 3) & 0x00000001) << i;
                absQuant[45] |= ((tmp_char1.w >> 2) & 0x00000001) << i;
                absQuant[46] |= ((tmp_char1.w >> 1) & 0x00000001) << i;
                absQuant[47] |= ((tmp_char1.w >> 0) & 0x00000001) << i;

                int uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[48] |= ((uchar_buffer >> 7) & 0x00000001) << i;
                absQuant[49] |= ((uchar_buffer >> 6) & 0x00000001) << i;
                absQuant[50] |= ((uchar_buffer >> 5) & 0x00000001) << i;
                absQuant[51] |= ((uchar_buffer >> 4) & 0x00000001) << i;
                absQuant[52] |= ((uchar_buffer >> 3) & 0x00000001) << i;
                absQuant[53] |= ((uchar_buffer >> 2) & 0x00000001) << i;
                absQuant[54] |= ((uchar_buffer >> 1) & 0x00000001) << i;
                absQuant[55] |= ((uchar_buffer >> 0) & 0x00000001) << i;

                uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[56] |= ((uchar_buffer >> 7) & 0x00000001) << i;
                absQuant[57] |= ((uchar_buffer >> 6) & 0x00000001) << i;
                absQuant[58] |= ((uchar_buffer >> 5) & 0x00000001) << i;
                absQuant[59] |= ((uchar_buffer >> 4) & 0x00000001) << i;
                absQuant[60] |= ((uchar_buffer >> 3) & 0x00000001) << i;
                absQuant[61] |= ((uchar_buffer >> 2) & 0x00000001) << i;
                absQuant[62] |= ((uchar_buffer >> 1) & 0x00000001) << i;
                absQuant[63] |= ((uchar_buffer >> 0) & 0x00000001) << i;
            }
            else
            {
                for(uint i=0; i<64; i++) absQuant[i] = 0;
                
                if(encoding_selection) absQuant[0] = outlier_buffer;
                
                sign_flag[0] = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24));

                tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                tmp_char2 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4+1];
                sign_flag[0] |= (0x00ff0000 & (tmp_char1.x << 16)) |
                                (0x0000ff00 & (tmp_char1.y << 8)) |
                                (0x000000ff & tmp_char1.z);
                sign_flag[1] = (0xff000000 & (tmp_char1.w << 24)) |
                               (0x00ff0000 & (tmp_char2.x << 16)) |
                               (0x0000ff00 & (tmp_char2.y << 8))  |
                               (0x000000ff & tmp_char2.z);
                cmp_byte_ofs+=8;

                if(!encoding_selection) absQuant[0] |= ((tmp_char2.w >> 7) & 0x00000001);
                absQuant[1] |= ((tmp_char2.w >> 6) & 0x00000001);
                absQuant[2] |= ((tmp_char2.w >> 5) & 0x00000001);
                absQuant[3] |= ((tmp_char2.w >> 4) & 0x00000001);
                absQuant[4] |= ((tmp_char2.w >> 3) & 0x00000001);
                absQuant[5] |= ((tmp_char2.w >> 2) & 0x00000001);
                absQuant[6] |= ((tmp_char2.w >> 1) & 0x00000001);
                absQuant[7] |= ((tmp_char2.w >> 0) & 0x00000001);

                for(uint i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                    tmp_char2 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4+1];
                    cmp_byte_ofs+=8;

                    absQuant[8] |= ((tmp_char1.x >> 7) & 0x00000001) << i;
                    absQuant[9] |= ((tmp_char1.x >> 6) & 0x00000001) << i;
                    absQuant[10] |= ((tmp_char1.x >> 5) & 0x00000001) << i;
                    absQuant[11] |= ((tmp_char1.x >> 4) & 0x00000001) << i;
                    absQuant[12] |= ((tmp_char1.x >> 3) & 0x00000001) << i;
                    absQuant[13] |= ((tmp_char1.x >> 2) & 0x00000001) << i;
                    absQuant[14] |= ((tmp_char1.x >> 1) & 0x00000001) << i;
                    absQuant[15] |= ((tmp_char1.x >> 0) & 0x00000001) << i;

                    absQuant[16] |= ((tmp_char1.y >> 7) & 0x00000001) << i;
                    absQuant[17] |= ((tmp_char1.y >> 6) & 0x00000001) << i;
                    absQuant[18] |= ((tmp_char1.y >> 5) & 0x00000001) << i;
                    absQuant[19] |= ((tmp_char1.y >> 4) & 0x00000001) << i;
                    absQuant[20] |= ((tmp_char1.y >> 3) & 0x00000001) << i;
                    absQuant[21] |= ((tmp_char1.y >> 2) & 0x00000001) << i;
                    absQuant[22] |= ((tmp_char1.y >> 1) & 0x00000001) << i;
                    absQuant[23] |= ((tmp_char1.y >> 0) & 0x00000001) << i;

                    absQuant[24] |= ((tmp_char1.z >> 7) & 0x00000001) << i;
                    absQuant[25] |= ((tmp_char1.z >> 6) & 0x00000001) << i;
                    absQuant[26] |= ((tmp_char1.z >> 5) & 0x00000001) << i;
                    absQuant[27] |= ((tmp_char1.z >> 4) & 0x00000001) << i;
                    absQuant[28] |= ((tmp_char1.z >> 3) & 0x00000001) << i;
                    absQuant[29] |= ((tmp_char1.z >> 2) & 0x00000001) << i;
                    absQuant[30] |= ((tmp_char1.z >> 1) & 0x00000001) << i;
                    absQuant[31] |= ((tmp_char1.z >> 0) & 0x00000001) << i;

                    absQuant[32] |= ((tmp_char1.w >> 7) & 0x00000001) << i;
                    absQuant[33] |= ((tmp_char1.w >> 6) & 0x00000001) << i;
                    absQuant[34] |= ((tmp_char1.w >> 5) & 0x00000001) << i;
                    absQuant[35] |= ((tmp_char1.w >> 4) & 0x00000001) << i;
                    absQuant[36] |= ((tmp_char1.w >> 3) & 0x00000001) << i;
                    absQuant[37] |= ((tmp_char1.w >> 2) & 0x00000001) << i;
                    absQuant[38] |= ((tmp_char1.w >> 1) & 0x00000001) << i;
                    absQuant[39] |= ((tmp_char1.w >> 0) & 0x00000001) << i;

                    absQuant[40] |= ((tmp_char2.x >> 7) & 0x00000001) << i;
                    absQuant[41] |= ((tmp_char2.x >> 6) & 0x00000001) << i;
                    absQuant[42] |= ((tmp_char2.x >> 5) & 0x00000001) << i;
                    absQuant[43] |= ((tmp_char2.x >> 4) & 0x00000001) << i;
                    absQuant[44] |= ((tmp_char2.x >> 3) & 0x00000001) << i;
                    absQuant[45] |= ((tmp_char2.x >> 2) & 0x00000001) << i;
                    absQuant[46] |= ((tmp_char2.x >> 1) & 0x00000001) << i;
                    absQuant[47] |= ((tmp_char2.x >> 0) & 0x00000001) << i;

                    absQuant[48] |= ((tmp_char2.y >> 7) & 0x00000001) << i;
                    absQuant[49] |= ((tmp_char2.y >> 6) & 0x00000001) << i;
                    absQuant[50] |= ((tmp_char2.y >> 5) & 0x00000001) << i;
                    absQuant[51] |= ((tmp_char2.y >> 4) & 0x00000001) << i;
                    absQuant[52] |= ((tmp_char2.y >> 3) & 0x00000001) << i;
                    absQuant[53] |= ((tmp_char2.y >> 2) & 0x00000001) << i;
                    absQuant[54] |= ((tmp_char2.y >> 1) & 0x00000001) << i;
                    absQuant[55] |= ((tmp_char2.y >> 0) & 0x00000001) << i;

                    absQuant[56] |= ((tmp_char2.z >> 7) & 0x00000001) << i;
                    absQuant[57] |= ((tmp_char2.z >> 6) & 0x00000001) << i;
                    absQuant[58] |= ((tmp_char2.z >> 5) & 0x00000001) << i;
                    absQuant[59] |= ((tmp_char2.z >> 4) & 0x00000001) << i;
                    absQuant[60] |= ((tmp_char2.z >> 3) & 0x00000001) << i;
                    absQuant[61] |= ((tmp_char2.z >> 2) & 0x00000001) << i;
                    absQuant[62] |= ((tmp_char2.z >> 1) & 0x00000001) << i;
                    absQuant[63] |= ((tmp_char2.z >> 0) & 0x00000001) << i;

                    if(!encoding_selection) absQuant[0] |= ((tmp_char2.w >> 7) & 0x00000001) << (i+1);
                    absQuant[1] |= ((tmp_char2.w >> 6) & 0x00000001) << (i+1);
                    absQuant[2] |= ((tmp_char2.w >> 5) & 0x00000001) << (i+1);
                    absQuant[3] |= ((tmp_char2.w >> 4) & 0x00000001) << (i+1);
                    absQuant[4] |= ((tmp_char2.w >> 3) & 0x00000001) << (i+1);
                    absQuant[5] |= ((tmp_char2.w >> 2) & 0x00000001) << (i+1);
                    absQuant[6] |= ((tmp_char2.w >> 1) & 0x00000001) << (i+1);
                    absQuant[7] |= ((tmp_char2.w >> 0) & 0x00000001) << (i+1);
                }

                tmp_char1 = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                cmp_byte_ofs+=4;
                int i = fixed_rate[j]-1;

                absQuant[8] |= ((tmp_char1.x >> 7) & 0x00000001) << i;
                absQuant[9] |= ((tmp_char1.x >> 6) & 0x00000001) << i;
                absQuant[10] |= ((tmp_char1.x >> 5) & 0x00000001) << i;
                absQuant[11] |= ((tmp_char1.x >> 4) & 0x00000001) << i;
                absQuant[12] |= ((tmp_char1.x >> 3) & 0x00000001) << i;
                absQuant[13] |= ((tmp_char1.x >> 2) & 0x00000001) << i;
                absQuant[14] |= ((tmp_char1.x >> 1) & 0x00000001) << i;
                absQuant[15] |= ((tmp_char1.x >> 0) & 0x00000001) << i;

                absQuant[16] |= ((tmp_char1.y >> 7) & 0x00000001) << i;
                absQuant[17] |= ((tmp_char1.y >> 6) & 0x00000001) << i;
                absQuant[18] |= ((tmp_char1.y >> 5) & 0x00000001) << i;
                absQuant[19] |= ((tmp_char1.y >> 4) & 0x00000001) << i;
                absQuant[20] |= ((tmp_char1.y >> 3) & 0x00000001) << i;
                absQuant[21] |= ((tmp_char1.y >> 2) & 0x00000001) << i;
                absQuant[22] |= ((tmp_char1.y >> 1) & 0x00000001) << i;
                absQuant[23] |= ((tmp_char1.y >> 0) & 0x00000001) << i;

                absQuant[24] |= ((tmp_char1.z >> 7) & 0x00000001) << i;
                absQuant[25] |= ((tmp_char1.z >> 6) & 0x00000001) << i;
                absQuant[26] |= ((tmp_char1.z >> 5) & 0x00000001) << i;
                absQuant[27] |= ((tmp_char1.z >> 4) & 0x00000001) << i;
                absQuant[28] |= ((tmp_char1.z >> 3) & 0x00000001) << i;
                absQuant[29] |= ((tmp_char1.z >> 2) & 0x00000001) << i;
                absQuant[30] |= ((tmp_char1.z >> 1) & 0x00000001) << i;
                absQuant[31] |= ((tmp_char1.z >> 0) & 0x00000001) << i;

                absQuant[32] |= ((tmp_char1.w >> 7) & 0x00000001) << i;
                absQuant[33] |= ((tmp_char1.w >> 6) & 0x00000001) << i;
                absQuant[34] |= ((tmp_char1.w >> 5) & 0x00000001) << i;
                absQuant[35] |= ((tmp_char1.w >> 4) & 0x00000001) << i;
                absQuant[36] |= ((tmp_char1.w >> 3) & 0x00000001) << i;
                absQuant[37] |= ((tmp_char1.w >> 2) & 0x00000001) << i;
                absQuant[38] |= ((tmp_char1.w >> 1) & 0x00000001) << i;
                absQuant[39] |= ((tmp_char1.w >> 0) & 0x00000001) << i;

                int uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[40] |= ((uchar_buffer >> 7) & 0x00000001) << i;
                absQuant[41] |= ((uchar_buffer >> 6) & 0x00000001) << i;
                absQuant[42] |= ((uchar_buffer >> 5) & 0x00000001) << i;
                absQuant[43] |= ((uchar_buffer >> 4) & 0x00000001) << i;
                absQuant[44] |= ((uchar_buffer >> 3) & 0x00000001) << i;
                absQuant[45] |= ((uchar_buffer >> 2) & 0x00000001) << i;
                absQuant[46] |= ((uchar_buffer >> 1) & 0x00000001) << i;
                absQuant[47] |= ((uchar_buffer >> 0) & 0x00000001) << i;

                uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[48] |= ((uchar_buffer >> 7) & 0x00000001) << i;
                absQuant[49] |= ((uchar_buffer >> 6) & 0x00000001) << i;
                absQuant[50] |= ((uchar_buffer >> 5) & 0x00000001) << i;
                absQuant[51] |= ((uchar_buffer >> 4) & 0x00000001) << i;
                absQuant[52] |= ((uchar_buffer >> 3) & 0x00000001) << i;
                absQuant[53] |= ((uchar_buffer >> 2) & 0x00000001) << i;
                absQuant[54] |= ((uchar_buffer >> 1) & 0x00000001) << i;
                absQuant[55] |= ((uchar_buffer >> 0) & 0x00000001) << i;

                uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[56] |= ((uchar_buffer >> 7) & 0x00000001) << i;
                absQuant[57] |= ((uchar_buffer >> 6) & 0x00000001) << i;
                absQuant[58] |= ((uchar_buffer >> 5) & 0x00000001) << i;
                absQuant[59] |= ((uchar_buffer >> 4) & 0x00000001) << i;
                absQuant[60] |= ((uchar_buffer >> 3) & 0x00000001) << i;
                absQuant[61] |= ((uchar_buffer >> 2) & 0x00000001) << i;
                absQuant[62] |= ((uchar_buffer >> 1) & 0x00000001) << i;
                absQuant[63] |= ((uchar_buffer >> 0) & 0x00000001) << i;
            }

            int currQuant, lorenQuant;
            int prevQuant_z = 0;
            for(uint i=0; i<4; i++)
            {
                uint data_idx_z = block_idx_z * 4 + i;
                int prevQuant_y = 0;

                if(data_idx_z < dims.z)
                {
                    for(uint k=0; k<4; k++)
                    {
                        uint data_idx_y = block_idx_y * 4 + k;
                        uint block_ofs = i * 16 + k * 4;
                        int prevQuant_x = 0;

                        if(data_idx_y < dims.y)
                        {
                            uint data_idx_x = block_idx_x * 4;
                            uint data_idx = data_idx_z * dims.y * dims.x + data_idx_y * dims.x + data_idx_x;
                            double4 dec_buffer;

                            sign_ofs = block_ofs % 32;
                            lorenQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs] * -1 : absQuant[block_ofs];
                            if(k) currQuant = lorenQuant + prevQuant_y; // Y-de-delta
                            else { 
                                currQuant = lorenQuant + prevQuant_z;   // Z-de-delta
                                prevQuant_z = currQuant;
                            }
                            prevQuant_y = currQuant;
                            prevQuant_x = currQuant;
                            dec_buffer.x = currQuant * eb * 2;

                            sign_ofs = (block_ofs + 1) % 32;
                            lorenQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs+1] * -1 : absQuant[block_ofs+1];
                            currQuant = lorenQuant + prevQuant_x;
                            prevQuant_x = currQuant;
                            dec_buffer.y = currQuant * eb * 2;

                            sign_ofs = (block_ofs + 2) % 32;
                            lorenQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs+2] * -1 : absQuant[block_ofs+2];
                            currQuant = lorenQuant + prevQuant_x;
                            prevQuant_x = currQuant;
                            dec_buffer.z = currQuant * eb * 2;

                            sign_ofs = (block_ofs + 3) % 32;
                            lorenQuant = sign_flag[i/2] & (1 << (31 - sign_ofs)) ? absQuant[block_ofs+3] * -1 : absQuant[block_ofs+3];
                            currQuant = lorenQuant + prevQuant_x;
                            prevQuant_x = currQuant;
                            dec_buffer.w = currQuant * eb * 2;
                            
                            reinterpret_cast<double4*>(decData)[data_idx/4] = dec_buffer;
                        }
                    }
                }
            } 
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}