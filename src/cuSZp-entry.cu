#include <stdio.h>
#include "cuSZp-entry.h"
#include "cuSZp-compress.h"
#include "cuSZp-decompress.h"
#include "errorCheck.h"
#include "timingGPU.h"

TimingGPU timer_GPU;
int chunk_size_1D = 32;
int chunk_size_2D = 256;
int chunk_size_3D = 512;

void cuSZp_compress(float* oridata, unsigned char* cmpBytes, size_t* outSize, float realPrecision, size_t nbEle, int blockSize, int predLevel, int shufLevel, cudaStream_t stream)
{
    float timerCMP = 0.0;

    // GPU kernel func0: Prequantization and 1D Lorenzo
    float recipPrecision = 0.5f/realPrecision;
    int bsize0 = 256, bunch0 = 2;
    int gsize0 = nbEle / (bsize0 * bunch0) + (nbEle % (bsize0 * bunch0) ==0 ? 0 : 1);
    int pad_nbEle = gsize0 * bsize0 * bunch0;
    float* oriData = (float*)malloc(sizeof(float)*pad_nbEle);
    memcpy(oriData, oridata, sizeof(float)*pad_nbEle);

    float* d_oriData;
    int* d_quantArray;
    
    CUCHK(cudaMalloc((void**)&d_oriData, sizeof(int)*pad_nbEle));
    CUCHK(cudaMalloc((void**)&d_quantArray, sizeof(int)*pad_nbEle)); 
    CUCHK(cudaMemcpy(d_oriData, oriData, sizeof(int)*pad_nbEle, cudaMemcpyHostToDevice)); 

    dim3 blockSize0(bsize0);
    dim3 gridSize0(gsize0);
    timer_GPU.StartCounter(); // set timer
    if(predLevel == 1)
    {
        quant_1DLorenzo1Layer<<<gridSize0, blockSize0>>>(d_oriData, d_quantArray, recipPrecision, chunk_size_1D, bunch0, stream);
        
    }
    else if(predLevel == 2)
    {
        quant_1DLorenzo2Layer<<<gridSize0, blockSize0>>>(d_oriData, d_quantArray, recipPrecision, chunk_size_1D, bunch0, stream);
        
    }
    else if(predLevel == 3)
    {
        quant_1DLorenzo3Layer<<<gridSize0, blockSize0>>>(d_oriData, d_quantArray, recipPrecision, chunk_size_1D, bunch0, stream);
        
    }
    else
        printf("Other layers of 1D-Lorenzo is not supported!\n");
    timerCMP += timer_GPU.GetCounter();
    printf("compression kernel0 speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/timer_GPU.GetCounter()); // print speed



    // GPU kernel func1: Zero bitmap search.
    size_t int_typeNbEle = nbEle%blockSize==0?nbEle/blockSize:nbEle/blockSize+1;
    int bsize1 = 256;
    int gsize1 = int_typeNbEle / bsize1 + (int_typeNbEle % bsize1 ==0 ? 0 : 1);
    unsigned char* d_int_typeArray;

    CUCHK(cudaMalloc((void**)&d_int_typeArray, sizeof(unsigned char)*int_typeNbEle)); 
    
    dim3 blockSize1(bsize1);
    dim3 gridSize1(gsize1);
    timer_GPU.StartCounter(); // set timer
    zeroBitmap_search<<<gridSize1, blockSize1>>>(d_quantArray, d_int_typeArray, blockSize, stream); 
    timerCMP += timer_GPU.GetCounter();
    printf("compression kernel1 speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/timer_GPU.GetCounter()); // print speed



    // GPU kernel func2: Convert d_int_typeArray to BitArray
    float folding_timer = 0.0;
    size_t byte_typeNbEle = int_typeNbEle%8==0?int_typeNbEle/8:int_typeNbEle/8+1;
    int bsize21 = 256;
    int gsize21 = byte_typeNbEle / bsize21 + (byte_typeNbEle % bsize21 ==0 ? 0 : 1);
    unsigned char* d_byte_typeData;

    CUCHK(cudaMalloc((void**)&d_byte_typeData, sizeof(unsigned char)*byte_typeNbEle)); 

    dim3 blockSize21(bsize21);
    dim3 gridSize21(gsize21);
    timer_GPU.StartCounter(); // set timer
    zeroOneByte_folding<<<gridSize21, blockSize21>>>(d_int_typeArray, d_byte_typeData, stream); 
    timerCMP += timer_GPU.GetCounter();
    folding_timer += timer_GPU.GetCounter();

    size_t dual_byte_typeNbEle = byte_typeNbEle%8==0?byte_typeNbEle/8:byte_typeNbEle/8+1;
    unsigned char* fir_level_byte_typeData = (unsigned char*)malloc(sizeof(unsigned char)*byte_typeNbEle);
    unsigned char* sec_level_byte_typeData = (unsigned char*)malloc(sizeof(unsigned char)*(dual_byte_typeNbEle+byte_typeNbEle));
    size_t non_zero_counter = 0;
    unsigned char* d_01_byte_typeData;
    unsigned char* d_dual_byte_typeData;
    int bsize22 = 256;
    int gsize22 = dual_byte_typeNbEle / bsize22 + (dual_byte_typeNbEle * bsize22 == 0 ? 0 : 1);

    CUCHK(cudaMemcpy(fir_level_byte_typeData, d_byte_typeData, sizeof(unsigned char)*byte_typeNbEle, cudaMemcpyDeviceToHost)); 
    for(size_t i=0; i<byte_typeNbEle; i++)
    {
        if(fir_level_byte_typeData[i]!=0)
        {
            sec_level_byte_typeData[dual_byte_typeNbEle+non_zero_counter] = fir_level_byte_typeData[i];
            fir_level_byte_typeData[i] = 1;
            non_zero_counter++;
        }
    }
    CUCHK(cudaMalloc((void**)&d_01_byte_typeData, sizeof(unsigned char)*byte_typeNbEle)); 
    CUCHK(cudaMalloc((void**)&d_dual_byte_typeData, sizeof(unsigned char)*dual_byte_typeNbEle)); 
    CUCHK(cudaMemcpy(d_01_byte_typeData, fir_level_byte_typeData, sizeof(unsigned char)*byte_typeNbEle, cudaMemcpyHostToDevice)); 
    
    dim3 blockSize22(bsize22);
    dim3 gridSize22(gsize22);
    timer_GPU.StartCounter(); // set timer
    zeroOneByte_folding<<<gridSize22, blockSize22>>>(d_01_byte_typeData, d_dual_byte_typeData, stream); 
    timerCMP += timer_GPU.GetCounter();
    folding_timer += timer_GPU.GetCounter();
    printf("compression kernel2 speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/folding_timer); // print speed

    CUCHK(cudaMemcpy(sec_level_byte_typeData, d_dual_byte_typeData, sizeof(unsigned char)*dual_byte_typeNbEle, cudaMemcpyDeviceToHost)); 



    // GPU kernel func3: Compress unpredicted data.
    int* tempQuantArray = (int*)malloc(sizeof(int)*nbEle);
    unsigned char* tempIntTypeArray = (unsigned char*)malloc(sizeof(unsigned char)*int_typeNbEle);
    CUCHK(cudaMemcpy(tempQuantArray, d_quantArray, sizeof(int)*nbEle, cudaMemcpyDeviceToHost)); 
    CUCHK(cudaMemcpy(tempIntTypeArray, d_int_typeArray,  sizeof(unsigned char)*int_typeNbEle, cudaMemcpyDeviceToHost)); 
    int* unpreddata = (int*)malloc(sizeof(int)*nbEle);
    size_t unpredNbEle = 0;
    for(size_t i=0; i<int_typeNbEle; i++)
    {
        if(tempIntTypeArray[i]==1)
        {
            int tempIDX = i*blockSize;
            for(int k=0;k<blockSize;k++)
            {
                unpreddata[unpredNbEle] = tempQuantArray[tempIDX+k];
                unpredNbEle++;
            }
        }
    }
    
    int bsize3 = 256, bunch3 = 32;
    int bnum3 = unpredNbEle / (bsize3 * bunch3) + (unpredNbEle % (bsize3 * bunch3) ==0 ? 0 : 1); 
    int pad_unpredNbEle = bnum3 * bsize3 * bunch3;
    int gsize3 = bnum3 > 65536 ? (bnum3 % 2 == 0 ? bnum3 / 2 : bnum3 /2 + 1) : bnum3;

    int* unpredData = (int*)malloc(sizeof(int)*pad_unpredNbEle);
    memcpy(unpredData, unpreddata, sizeof(int)*pad_unpredNbEle);

    int* d_unpredData;
    unsigned int* d_sign;
    unsigned int* d_encoding;
    unsigned short* d_c;
    unsigned int* d_pos;
    unsigned int* d_bytes;
    unsigned int* d_values; 
    CUCHK(cudaMalloc((void**)&d_unpredData, sizeof(int)*pad_unpredNbEle));
    CUCHK(cudaMemcpy(d_unpredData, unpredData, sizeof(int)*pad_unpredNbEle, cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_sign, sizeof(unsigned int)*pad_unpredNbEle/bunch3));
    CUCHK(cudaMemset(d_sign, 0, sizeof(unsigned int)*pad_unpredNbEle/bunch3));
    CUCHK(cudaMalloc((void**)&d_encoding, sizeof(unsigned int)*pad_unpredNbEle*3/32));
    CUCHK(cudaMemset(d_encoding, 0, sizeof(unsigned int)*pad_unpredNbEle*3/32));
    CUCHK(cudaMalloc((void**)&d_c, sizeof(unsigned short)*pad_unpredNbEle/bunch3));
    CUCHK(cudaMemset(d_c, 0, sizeof(unsigned short)*pad_unpredNbEle/bunch3));
    CUCHK(cudaMalloc((void**)&d_pos, sizeof(unsigned int)*pad_unpredNbEle/bunch3*2));
    CUCHK(cudaMemset(d_pos, 0, sizeof(unsigned short)*pad_unpredNbEle/bunch3*2));
    if(shufLevel==1)
    {
        CUCHK(cudaMalloc((void**)&d_bytes, sizeof(unsigned int)*pad_unpredNbEle/bunch3*2));
        CUCHK(cudaMemset(d_bytes, 0, sizeof(unsigned short)*pad_unpredNbEle/bunch3*2));
    }
    else if(shufLevel==0)
    {
        CUCHK(cudaMalloc((void**)&d_bytes, sizeof(unsigned int)*pad_unpredNbEle/bunch3*4));
        CUCHK(cudaMemset(d_bytes, 0, sizeof(unsigned short)*pad_unpredNbEle/bunch3*4));
    }
    
    CUCHK(cudaMalloc((void**)&d_values, sizeof(unsigned int)*pad_unpredNbEle/2));
    CUCHK(cudaMemset(d_values, 0, sizeof(unsigned int)*pad_unpredNbEle/2));

    dim3 dimBlock3(bsize3);
    dim3 dimGrid3(gsize3);
    timer_GPU.StartCounter(); // set timer
    const int sMemsize3 = bsize3 * (bunch3 * sizeof(unsigned short) + 1);
    if(shufLevel==1)
        unpredData_compress_rad<<<dimGrid3, dimBlock3, sMemsize3>>>(d_unpredData, d_sign, d_encoding, d_c, d_pos, d_bytes, d_values, bunch3, stream);
    else if(shufLevel==0)
        unpredData_compress_con<<<dimGrid3, dimBlock3, sMemsize3>>>(d_unpredData, d_sign, d_encoding, d_c, d_pos, d_bytes, d_values, bunch3, stream);
    timerCMP += timer_GPU.GetCounter();
    printf("compression kernel3 speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/timer_GPU.GetCounter()); // print speed

    // Post processing of compressed unpredicted data.
    unsigned int* sign = (unsigned int*)malloc(sizeof(unsigned int)*pad_unpredNbEle/bunch3);
    unsigned int* encoding = (unsigned int*)malloc(sizeof(unsigned int)*pad_unpredNbEle*3/32);
    unsigned short* c = (unsigned short*)malloc(sizeof(unsigned short)*pad_unpredNbEle/bunch3);
    unsigned int* pos = (unsigned int*)malloc(sizeof(unsigned int)*pad_unpredNbEle/bunch3*2);
    unsigned short* values = (unsigned short*)malloc(sizeof(unsigned short)*pad_unpredNbEle);
    unsigned int* bytes;
    if(shufLevel==1)
    {
        bytes = (unsigned int*)malloc(sizeof(unsigned int)*pad_unpredNbEle/bunch3*2);
        CUCHK(cudaMemcpy(bytes, d_bytes, sizeof(unsigned int)*pad_unpredNbEle/bunch3*2, cudaMemcpyDeviceToHost));
    }
    else if((shufLevel==0))
    {
        bytes = (unsigned int*)malloc(sizeof(unsigned int)*pad_unpredNbEle/bunch3*4);
        CUCHK(cudaMemcpy(bytes, d_bytes, sizeof(unsigned int)*pad_unpredNbEle/bunch3*4, cudaMemcpyDeviceToHost));
    }
    CUCHK(cudaMemcpy(sign, d_sign, sizeof(unsigned int)*pad_unpredNbEle/bunch3, cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(encoding, d_encoding, sizeof(unsigned int)*pad_unpredNbEle*3/32, cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(c, d_c, sizeof(unsigned short)*pad_unpredNbEle/bunch3, cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(pos, d_pos, sizeof(unsigned int)*pad_unpredNbEle/bunch3*2, cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(values, d_values, sizeof(unsigned short)*pad_unpredNbEle, cudaMemcpyDeviceToHost));

    unsigned int total = 0, total2 = 0;
    unsigned short* fvalues = (unsigned short*)malloc(sizeof(unsigned short)*pad_unpredNbEle); 
    unsigned char* bvalues = (unsigned char*)malloc(sizeof(unsigned char)*pad_unpredNbEle); 
    unsigned int* pvalues = (unsigned int*)malloc(sizeof(unsigned int)*pad_unpredNbEle); 
    unsigned short* tmp_values = fvalues; 
    unsigned short uc = 0, bc = 0;
    unsigned cc = 0;
    for (int i = 0; i < pad_unpredNbEle/bunch3; i++) {
        if (sign[i]>0) {
            cc++; 
        }
        uc = c[i] & 0x00ff;
        bc = c[i] >> 8;
        if (uc) {
            memcpy (tmp_values, values, sizeof(unsigned short)*uc);
            tmp_values += uc;
            total += uc; 
        }
        if (bc>0) {
            unsigned gpos = i / bsize3;
            unsigned bpos = i % bsize3;
            int bound = bc < 4 ? bc : 4;
            for (int j = 0; j < bound; j++) {
                pvalues[total2] = gpos * bsize3 * bunch3 + (pos[i] >> (3 - j) * 8 & 0x000000ff) * bsize3 + bpos; 
                bvalues[total2++] = bytes[i] >> (3 - j) * 8 & 0x000000ff;
                if(shufLevel==0)
                    bvalues[total2++] = bytes[pad_unpredNbEle/bunch3*2 + i] >> (3 - j) * 8 & 0x0000ff00;
            }
            bound = bc > 4 ? bc : 4;
            for (int j = 0; j < (bound - 4); j++) {
                pvalues[total2] = gpos * bsize3 * bunch3 + (pos[pad_unpredNbEle/bunch3+i] >> (3 - j) * 8 & 0x000000ff) * bsize3 + bpos; 
                bvalues[total2++] = bytes[pad_unpredNbEle/bunch3+i] >> (3 - j) * 8 & 0x000000ff;
                if(shufLevel==0)
                    bvalues[total2++] = bytes[pad_unpredNbEle/bunch3*3 + i] >> (3 - j) * 8 & 0x0000ff00;
            }
            //printf("test:%i, %i\n", i, bc); 
        }
        values += bunch3;
    }
    fvalues = (unsigned short*)realloc(fvalues, sizeof(unsigned short)*total); 
    bvalues = (unsigned char*)realloc(bvalues, sizeof(unsigned char)*total2); 
    pvalues = (unsigned int*)realloc(pvalues, sizeof(unsigned int)*total2);

    *outSize = sizeof(unsigned char)  * (dual_byte_typeNbEle+non_zero_counter)    // size of dual folded zero bit map
             + sizeof(unsigned int)   * pad_unpredNbEle*3/32                      // size of encoding
             + sizeof(unsigned int)   * pad_unpredNbEle/bunch3                    // size of sign
             + sizeof(unsigned short) * total                                     // size of fvalues
             + sizeof(unsigned char)  * total2                                    // size of bvalues
             + sizeof(unsigned int)   * total2                                    // size of pvalues
             + sizeof(size_t)         * 2                                         // non_zero_counter and unpredNbEle
             + sizeof(unsigned int)   * 2;                                        // total and total2

    printf("compression total kernel speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/timerCMP); // print final compression speed
    printf("compression ratio: %f\n\n", (float)nbEle*sizeof(float)/(*outSize)); // print compression ratio

    
    size_t cmpBytes_counter = 0;
    memcpy(cmpBytes+cmpBytes_counter, &non_zero_counter, sizeof(size_t)); cmpBytes_counter += sizeof(size_t);
    memcpy(cmpBytes+cmpBytes_counter, &unpredNbEle, sizeof(size_t)); cmpBytes_counter += sizeof(size_t);
    memcpy(cmpBytes+cmpBytes_counter, &total, sizeof(unsigned int)); cmpBytes_counter += sizeof(unsigned int);
    memcpy(cmpBytes+cmpBytes_counter, &total2, sizeof(unsigned int)); cmpBytes_counter += sizeof(unsigned int);
    memcpy(cmpBytes+cmpBytes_counter, sec_level_byte_typeData, sizeof(unsigned char)*dual_byte_typeNbEle); cmpBytes_counter += sizeof(unsigned char)*dual_byte_typeNbEle;
    memcpy(cmpBytes+cmpBytes_counter, sec_level_byte_typeData+dual_byte_typeNbEle, sizeof(unsigned char)*non_zero_counter); cmpBytes_counter += sizeof(unsigned char)*non_zero_counter;
    memcpy(cmpBytes+cmpBytes_counter, encoding, sizeof(unsigned int)*pad_unpredNbEle*3/32); cmpBytes_counter += sizeof(unsigned int)*pad_unpredNbEle*3/32;
    memcpy(cmpBytes+cmpBytes_counter, sign, sizeof(unsigned int)*pad_unpredNbEle/bunch3); cmpBytes_counter += sizeof(unsigned int)*pad_unpredNbEle/bunch3;
    memcpy(cmpBytes+cmpBytes_counter, fvalues, sizeof(unsigned short)*total); cmpBytes_counter += sizeof(unsigned short)*total;
    memcpy(cmpBytes+cmpBytes_counter, bvalues, sizeof(unsigned char)*total2); cmpBytes_counter += sizeof(unsigned char)*total2;
    memcpy(cmpBytes+cmpBytes_counter, pvalues, sizeof(unsigned int)*total2); 


    free(oriData);
    free(fir_level_byte_typeData);
    free(sec_level_byte_typeData);
    free(tempQuantArray);
    free(tempIntTypeArray);
    free(unpreddata);
    free(unpredData);
    free(sign);
    free(encoding);
    free(c);
    free(pos);
    free(bytes);
    free(fvalues);
    free(bvalues);
    free(pvalues);
    CUCHK(cudaFree(d_oriData));
    CUCHK(cudaFree(d_quantArray));
    CUCHK(cudaFree(d_int_typeArray));
    CUCHK(cudaFree(d_byte_typeData));
    CUCHK(cudaFree(d_01_byte_typeData));
    CUCHK(cudaFree(d_dual_byte_typeData));
    CUCHK(cudaFree(d_unpredData));
    CUCHK(cudaFree(d_sign));
    CUCHK(cudaFree(d_encoding));
    CUCHK(cudaFree(d_c));
    CUCHK(cudaFree(d_pos));
    CUCHK(cudaFree(d_bytes));
    CUCHK(cudaFree(d_values));

    // printf("Compressed Data Component:\n");
    // printf("bitmap:     %f\n", (float)sizeof(unsigned char) * (dual_byte_typeNbEle+non_zero_counter)/(float)*outSize);
    // printf("encoding:   %f\n", (float)sizeof(unsigned int) * pad_unpredNbEle*3/32/(float)*outSize);
    // printf("sign:       %f\n", (float)sizeof(unsigned int) * pad_unpredNbEle/bunch3/(float)*outSize);
    // printf("fvalues:    %f\n", (float)sizeof(unsigned short) * total/(float)*outSize);
    // printf("bvalues:    %f\n", (float)sizeof(unsigned char) * total2/(float)*outSize);
    // printf("pvalues:    %f\n", (float)sizeof(unsigned int) * total2/(float)*outSize);
}




void cuSZp_decompress(float* decdata, unsigned char* cmpBytes, float realPrecision, size_t nbEle, int blockSize, int predLevel, int shufLevel, cudaStream_t stream)
{
    float timerDEC = 0.0;

    // Preparation: Varaibles for decompression.
    size_t cmpBytes_counter = 0;
    size_t non_zero_counter;
    size_t unpredNbEle;
    unsigned int total, total2;
    size_t int_typeNbEle = nbEle%blockSize==0?nbEle/blockSize:nbEle/blockSize+1;
    size_t byte_typeNbEle = int_typeNbEle%8==0?int_typeNbEle/8:int_typeNbEle/8+1;
    size_t dual_byte_typeNbEle = byte_typeNbEle%8==0?byte_typeNbEle/8:byte_typeNbEle/8+1;
    memcpy(&non_zero_counter, cmpBytes+cmpBytes_counter, sizeof(size_t)); cmpBytes_counter += sizeof(size_t);
    memcpy(&unpredNbEle, cmpBytes+cmpBytes_counter, sizeof(size_t)); cmpBytes_counter += sizeof(size_t);
    memcpy(&total, cmpBytes+cmpBytes_counter, sizeof(unsigned int)); cmpBytes_counter += sizeof(unsigned int);
    memcpy(&total2, cmpBytes+cmpBytes_counter, sizeof(unsigned int)); cmpBytes_counter += sizeof(unsigned int);
    
    // Variables. Compressed bitmap.
    unsigned char* sec_level_byte_typeData = (unsigned char*)malloc(sizeof(unsigned char)*(dual_byte_typeNbEle+non_zero_counter));
    memcpy(sec_level_byte_typeData, cmpBytes+cmpBytes_counter, sizeof(unsigned char)*(dual_byte_typeNbEle+non_zero_counter)); cmpBytes_counter += sizeof(unsigned char)*(dual_byte_typeNbEle+non_zero_counter);
  
    // Variables. Compressed unpredicted data.
    int bsize0 = 256, bunch0 = 32;
    int bnum0 = unpredNbEle / (bsize0 * bunch0) + (unpredNbEle % (bsize0 * bunch0) ==0 ? 0 : 1); 
    int pad_unpredNbEle = bnum0 * bsize0 * bunch0;
    int gsize0 = bnum0 > 65536 ? (bnum0 % 2 == 0 ? bnum0 / 2 : bnum0 /2 + 1) : bnum0;
    const int sMemsize0 = bsize0 * (bunch0 * sizeof(unsigned short) + 1);
    unsigned int* encoding = (unsigned int*)malloc(sizeof(unsigned int)*pad_unpredNbEle*3/32);
    unsigned int* sign = (unsigned int*)malloc(sizeof(unsigned int)*pad_unpredNbEle/bunch0);
    unsigned short* fvalues = (unsigned short*)malloc(sizeof(unsigned short)*total); 
    unsigned char* bvalues = (unsigned char*)malloc(sizeof(unsigned char)*total2); 
    unsigned int* pvalues = (unsigned int*)malloc(sizeof(unsigned int)*total2);
    memcpy(encoding, cmpBytes+cmpBytes_counter, sizeof(unsigned int)*pad_unpredNbEle*3/32); cmpBytes_counter += sizeof(unsigned int)*pad_unpredNbEle*3/32;
    memcpy(sign, cmpBytes+cmpBytes_counter, sizeof(unsigned int)*pad_unpredNbEle/bunch0); cmpBytes_counter += sizeof(unsigned int)*pad_unpredNbEle/bunch0;
    memcpy(fvalues, cmpBytes+cmpBytes_counter, sizeof(unsigned short)*total); cmpBytes_counter += sizeof(unsigned short)*total;
    memcpy(bvalues, cmpBytes+cmpBytes_counter, sizeof(unsigned char)*total2); cmpBytes_counter += sizeof(unsigned char)*total2;
    memcpy(pvalues, cmpBytes+cmpBytes_counter, sizeof(unsigned int)*total2);
    

    
    // GPU kernel func0: Unpredicted data counter.
    unsigned int* d_dencoding;
    int* d_dunpredData;
    unsigned int* d_dc;

    CUCHK(cudaMalloc((void**)&d_dencoding, sizeof(unsigned int)*pad_unpredNbEle*3/32));
    CUCHK(cudaMemcpy(d_dencoding, encoding, sizeof(unsigned int)*pad_unpredNbEle*3/32, cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_dunpredData, sizeof(int)*pad_unpredNbEle));
    CUCHK(cudaMemset(d_dunpredData, -1, sizeof(int)*pad_unpredNbEle));
    CUCHK(cudaMalloc((void**)&d_dc, sizeof(unsigned int)*pad_unpredNbEle/bunch0));
    CUCHK(cudaMemset(d_dc, 0, sizeof(unsigned int)*pad_unpredNbEle/bunch0));

    dim3 dimBlock0(bsize0);
    dim3 dimGrid0(gsize0);
    timer_GPU.StartCounter(); // set timer
    unpredData_decompress_count<<<dimGrid0, dimBlock0, sMemsize0>>>(d_dencoding, d_dc, bunch0, stream);
    timerDEC += timer_GPU.GetCounter();
    printf("decompression kernel0 speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/timer_GPU.GetCounter()); // print speed



    // GPU kernel func1: Decompress unpredicted data.
    unsigned int* dc = (unsigned int*)malloc(sizeof(unsigned int)*pad_unpredNbEle/bunch0);
    CUCHK(cudaMemcpy(dc, d_dc, sizeof(unsigned int)*pad_unpredNbEle/bunch0, cudaMemcpyDeviceToHost));
    unsigned int* dgs = (unsigned int*)malloc(sizeof(unsigned int)*(gsize0+1));
    unsigned int* dbs = (unsigned int*)malloc(sizeof(unsigned int)*pad_unpredNbEle/bunch0);
    dgs[0] = 0;
    for (int i = 0; i < gsize0; i++) {
        dbs[i*bsize0] = 0;
        for (int j = 1; j < bsize0; j++) {
            dbs[i*bsize0+j] = dbs[i*bsize0+j-1] + dc[i*bsize0+j-1];
        }
        dgs[i+1] = dgs[i] + dbs[i*bsize0+bsize0-1] + dc[i*bsize0+bsize0-1]; 
    }

    unsigned int* d_dsign;
    unsigned int* d_dgs;
    unsigned int* d_dbs;
    unsigned short* d_fvalues;

    CUCHK(cudaMalloc((void**)&d_dsign, sizeof(unsigned int)*pad_unpredNbEle/bunch0));
    CUCHK(cudaMemcpy(d_dsign, sign, sizeof(unsigned int)*pad_unpredNbEle/bunch0, cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_dgs, sizeof(unsigned int)*(gsize0+1)));
    CUCHK(cudaMemcpy(d_dgs, dgs, sizeof(unsigned int)*(gsize0+1), cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_dbs, sizeof(unsigned int)*pad_unpredNbEle/bunch0));
    CUCHK(cudaMemcpy(d_dbs, dbs, sizeof(unsigned int)*pad_unpredNbEle/bunch0, cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_fvalues, sizeof(unsigned short)*total));
    CUCHK(cudaMemcpy(d_fvalues, fvalues, sizeof(unsigned short)*total, cudaMemcpyHostToDevice));

    timer_GPU.StartCounter(); // set timer
    unpredData_decompress<<<dimGrid0, dimBlock0, sMemsize0>>>(d_dencoding, d_dsign, d_dunpredData, d_dgs, d_dbs, d_fvalues, bunch0, stream);
    timerDEC += timer_GPU.GetCounter();
    printf("decompression kernel1 speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/timer_GPU.GetCounter()); // print speed

    int* dunpredData = (int*)malloc(sizeof(int)*pad_unpredNbEle);
    CUCHK(cudaMemcpy(dunpredData, d_dunpredData, sizeof(int)*pad_unpredNbEle, cudaMemcpyDeviceToHost));
    if(shufLevel==1)
    {
        for (int i = 0; i < total2; i++) {
            dunpredData[pvalues[i]] |= bvalues[i] << 16;
        }
    }
    else if(shufLevel==0)
    {
        for (int i = 0; i < total2; i+=2) {
            dunpredData[pvalues[i]] = dunpredData[pvalues[i]] | (bvalues[i] << 16) | (bvalues[i+1] << 24);
        }
    }
    for (int i = 0; i < pad_unpredNbEle; i++) {
        dunpredData[i] = ((dunpredData[i] < 0x01000000) ? 1 : -1) * (dunpredData[i] & 0x00ffffff);
    }
    dunpredData = (int*)realloc(dunpredData, sizeof(int)*unpredNbEle);



    // GPU Kernel func2: Unfolding compressed BitArray.
    float unfolding_timer = 0.0;
    int bsize21 = 256;
    int gsize21 = dual_byte_typeNbEle / bsize21 + (dual_byte_typeNbEle * bsize21 == 0 ? 0 : 1);
    unsigned char* d_int_typeArray;
    unsigned char* d_01_byte_typeData;
    unsigned char* d_dual_byte_typeData;
    
    CUCHK(cudaMalloc((void**)&d_dual_byte_typeData, sizeof(unsigned char)*dual_byte_typeNbEle));
    CUCHK(cudaMalloc((void**)&d_01_byte_typeData, sizeof(unsigned char)*byte_typeNbEle));
    CUCHK(cudaMalloc((void**)&d_int_typeArray, sizeof(unsigned char)*int_typeNbEle));
    CUCHK(cudaMemcpy(d_dual_byte_typeData, sec_level_byte_typeData, sizeof(unsigned char)*dual_byte_typeNbEle, cudaMemcpyHostToDevice));

    dim3 blockSize21(bsize21);
    dim3 gridSize21(gsize21);
    timer_GPU.StartCounter(); // set timer
    zeroOneByte_unfolding<<<gridSize21, blockSize21>>>(d_dual_byte_typeData, d_01_byte_typeData, stream);
    timerDEC += timer_GPU.GetCounter();
    unfolding_timer += timer_GPU.GetCounter();

    size_t non_zero_counter1 = 0;
    int bsize22 = 256;
    int gsize22 = byte_typeNbEle / bsize22 + (byte_typeNbEle % bsize22 ==0 ? 0 : 1);
    unsigned char* fir_level_byte_typeData = (unsigned char*)malloc(sizeof(unsigned char)*byte_typeNbEle);
    CUCHK(cudaMemcpy(fir_level_byte_typeData, d_01_byte_typeData, sizeof(unsigned char)*byte_typeNbEle, cudaMemcpyDeviceToHost));
    for(size_t i=0; i<byte_typeNbEle; i++)
    {
        if(fir_level_byte_typeData[i]!=0)
        {
            fir_level_byte_typeData[i] = sec_level_byte_typeData[dual_byte_typeNbEle+non_zero_counter1];
            non_zero_counter1++;
            
        }
    }
    CUCHK(cudaMemcpy(d_01_byte_typeData, fir_level_byte_typeData, sizeof(unsigned char)*byte_typeNbEle, cudaMemcpyHostToDevice));

    dim3 blockSize22(bsize22);
    dim3 gridSize22(gsize22);
    timer_GPU.StartCounter(); // set timer
    zeroOneByte_unfolding<<<gridSize22, blockSize22>>>(d_01_byte_typeData, d_int_typeArray, stream);
    timerDEC += timer_GPU.GetCounter();
    unfolding_timer += timer_GPU.GetCounter();

    printf("decompression kernel2 speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/unfolding_timer); // print speed


    // GPU Kernel func3: Delorenzo and dequantization
    int bsize3 = 256, bunch3 = 2;
    int gsize3 = nbEle / (bsize3 * bunch3) + (nbEle % (bsize3 * bunch3) ==0 ? 0 : 1);
    size_t pad_nbEle = gsize3 * bsize3 * bunch3;
    unsigned char* tempIntTypeArray = (unsigned char*)malloc(sizeof(unsigned char)*int_typeNbEle);
    CUCHK(cudaMemcpy(tempIntTypeArray, d_int_typeArray,  sizeof(unsigned char)*int_typeNbEle, cudaMemcpyDeviceToHost));
    int* tempDeQuantArray = (int*)malloc(sizeof(int)*pad_nbEle);
    memset(tempDeQuantArray, 0, pad_nbEle);
    size_t tempDeQuantArray_counter = 0;
    for(size_t i=0; i<int_typeNbEle; i++)
    {
        if(tempIntTypeArray[i]==1)
        {
            int tempIDX = i*blockSize;
            for(int k=0; k<blockSize; k++)
            {
                tempDeQuantArray[tempIDX+k] = dunpredData[tempDeQuantArray_counter];
                tempDeQuantArray_counter++;
            }
        }
    }
    
    float e2 = realPrecision*2;
    float* d_decData;
    int* d_decQuantArray;

    CUCHK(cudaMalloc((void**)&d_decData, sizeof(float)*pad_nbEle)); 
    CUCHK(cudaMalloc((void**)&d_decQuantArray, sizeof(int)*pad_nbEle)); 
    CUCHK(cudaMemcpy(d_decQuantArray, tempDeQuantArray, sizeof(int)*pad_nbEle, cudaMemcpyHostToDevice)); 

    dim3 blockSize3(bsize3);
    dim3 gridSize3(gsize3);
    timer_GPU.StartCounter(); // set timer
    if(predLevel == 1)
    {
        recover_quant_1DLorenzo1Layer<<<gridSize3, blockSize3>>>(d_decData, d_decQuantArray, e2, chunk_size_1D, bunch3, stream);
        
    }
    else if(predLevel == 2)
    {
        recover_quant_1DLorenzo2Layer<<<gridSize3, blockSize3>>>(d_decData, d_decQuantArray, e2, chunk_size_1D, bunch3, stream);
        
    }
    else if(predLevel == 3)
    {
        recover_quant_1DLorenzo3Layer<<<gridSize3, blockSize3>>>(d_decData, d_decQuantArray, e2, chunk_size_1D, bunch3, stream);
        
    }
    else
        printf("Other layers of 1D-Lorenzo is not supported!\n");

    
    timerDEC += timer_GPU.GetCounter();
    printf("decompression kernel3 speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/timer_GPU.GetCounter()); // print speed


    CUCHK(cudaMemcpy(decdata, d_decData, sizeof(float)*nbEle, cudaMemcpyDeviceToHost));
    printf("decompression total kernel speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/timerDEC); // print final compression speed


    free(sec_level_byte_typeData);
    free(encoding);
    free(sign);
    free(fvalues);
    free(bvalues);
    free(pvalues);
    free(dc);
    free(dgs);
    free(dbs);
    free(dunpredData);
    free(fir_level_byte_typeData);
    free(tempIntTypeArray);
    free(tempDeQuantArray);
    CUCHK(cudaFree(d_dencoding)); 
    CUCHK(cudaFree(d_dunpredData)); 
    CUCHK(cudaFree(d_dc)); 
    CUCHK(cudaFree(d_dsign));
    CUCHK(cudaFree(d_dgs));
    CUCHK(cudaFree(d_dbs));
    CUCHK(cudaFree(d_fvalues));
    CUCHK(cudaFree(d_int_typeArray));
    CUCHK(cudaFree(d_01_byte_typeData));
    CUCHK(cudaFree(d_dual_byte_typeData));
    CUCHK(cudaFree(d_decData));
    CUCHK(cudaFree(d_decQuantArray));
}