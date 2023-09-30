#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuSZp_utility.h>
#include <cuSZp_entry_f64.h>
#include <cuSZp_timer.h>

int main(int argc, char* argv[])
{
    // Read input information.
    char oriFilePath[640];
    char errorMode[20];
    int status=0;
    if(argc != 4)
    {
        printf("Usage: cuSZp_gpu_f64_api [srcFilePath] [errorMode] [errBound] # errorMode can only be ABS or REL\n");
        printf("Example: cuSZp_gpu_f64_api testdouble_8_8_128.dat ABS 1E-2     # compress dataset with absolute 1E-2 error bound\n");
        printf("         cuSZp_gpu_f64_api testdouble_8_8_128.dat REL 1e-3     # compress dataset with relative 1E-3 error bound\n");
        exit(0);
    }
    sprintf(oriFilePath, "%s", argv[1]);
    sprintf(errorMode, "%s", argv[2]);
    double errorBound = atof(argv[3]);

    // For measuring the end-to-end throughput.
    TimingGPU timer_GPU;

    // Input data preparation on CPU.
    double* oriData = NULL;
    double* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 0;
    size_t cmpSize = 0;
    oriData = readDoubleData_Yafan(oriFilePath, &nbEle, &status);
    decData = (double*)malloc(nbEle*sizeof(double));
    cmpBytes = (unsigned char*)malloc(nbEle*sizeof(double));

    // Generating error bounds.
    if(strcmp(errorMode, "REL")==0)
    {
        double max_val = oriData[0];
        double min_val = oriData[0];
        for(size_t i=0; i<nbEle; i++)
        {
            if(oriData[i]>max_val)
                max_val = oriData[i];
            else if(oriData[i]<min_val)
                min_val = oriData[i];
        }
        errorBound = errorBound * (max_val - min_val);
    }
    else if(strcmp(errorMode, "ABS")!=0)
    {
        printf("invalid errorMode! errorMode can only be ABS or REL.\n");
        exit(0);
    }

    // Input data preparation on GPU.
    double* d_oriData;
    double* d_decData;
    unsigned char* d_cmpBytes;
    size_t pad_nbEle = (nbEle + 262144 - 1) / 262144 * 262144; // A temp demo, will add more block sizes in future implementation.
    cudaMalloc((void**)&d_oriData, sizeof(double)*pad_nbEle);
    cudaMemcpy(d_oriData, oriData, sizeof(double)*pad_nbEle, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_decData, sizeof(double)*pad_nbEle);
    cudaMemset(d_decData, 0, sizeof(double)*pad_nbEle);
    cudaMalloc((void**)&d_cmpBytes, sizeof(double)*pad_nbEle);

    // Initializing CUDA Stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Just a warmup.
    for(int i=0; i<3; i++)
        SZp_compress_deviceptr_f64(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, stream);

    // cuSZp compression.
    timer_GPU.StartCounter(); // set timer
    SZp_compress_deviceptr_f64(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, stream);
    float cmpTime = timer_GPU.GetCounter();
    
    // cuSZp decompression.
    timer_GPU.StartCounter(); // set timer
    SZp_decompress_deviceptr_f64(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
    float decTime = timer_GPU.GetCounter();

    // Print result.
    printf("cuSZp finished!\n");
    printf("cuSZp compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/cmpTime);
    printf("cuSZp decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/decTime);
    printf("cuSZp compression ratio: %f\n\n", (nbEle*sizeof(double)/1024.0/1024.0)/(cmpSize*sizeof(unsigned char)/1024.0/1024.0));

    // Error check
    cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(decData, d_decData, sizeof(double)*nbEle, cudaMemcpyDeviceToHost);
    int not_bound = 0;
    for(size_t i=0; i<nbEle; i+=1)
    {
        if(abs(oriData[i]-decData[i]) > errorBound*1.1)
        {
            not_bound++;
            // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound: %f\n", i, oriData[i], decData[i], abs(oriData[i]-decData[i]), errBound);
        }
    }
    if(!not_bound) printf("\033[0;32mPass error check!\033[0m\n");
    else printf("\033[0;31mFail error check!\033[0m\n");
    
    // Free allocated data.
    free(oriData);
    free(decData);
    free(cmpBytes);
    cudaFree(d_oriData);
    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);

    return 0;
}