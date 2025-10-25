#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuSZp.h>

int main()
{
    // For measuring the end-to-end throughput.
    TimingGPU timer_GPU;

    // Input data preparation on CPU.
    double* oriData = NULL;
    double* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 1024*1024*512; // 4 GB fp64 data.
    uint3 dims = {512, 1024, 1024}; // assuming 512 is the fastest changing dimension.
    size_t cmpSize1 = 0;
    size_t cmpSize2 = 0;
    size_t cmpSize3 = 0;
    oriData = (double*)malloc(nbEle*sizeof(double));
    decData = (double*)malloc(nbEle*sizeof(double));
    cmpBytes = (unsigned char*)malloc(nbEle*sizeof(double));

    // Initialize oriData.
    printf("Generating test data...\n\n");
    double startValue = -20.0f;
    double step = 0.1f;
    double endValue = 20.0f;
    size_t idx = 0;
    double value = startValue;
    while (idx < nbEle) 
    {
        oriData[idx++] = value;
        value += step;
        if (value > endValue)
        {
            value = startValue;
        }
    }

    // Get value range, making it a REL errMode test -- remove this will be ABS errMode.
    double max_val = oriData[0];
    double min_val = oriData[0];
    for(size_t i=0; i<nbEle; i++)
    {
        if(oriData[i]>max_val)
            max_val = oriData[i];
        else if(oriData[i]<min_val)
            min_val = oriData[i];
    }
    double errorBound = (max_val - min_val) * 1E-2f;

    // Input data preparation on GPU.
    double* d_oriData;
    double* d_decData;
    unsigned char* d_cmpBytes;
    cudaMalloc((void**)&d_oriData, sizeof(double)*nbEle);
    cudaMemcpy(d_oriData, oriData, sizeof(double)*nbEle, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_decData, sizeof(double)*nbEle);
    cudaMalloc((void**)&d_cmpBytes, sizeof(double)*nbEle);

    // Initializing CUDA Stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup for NVIDIA GPU.
    for(int i=0; i<3; i++)
    {
        cuSZp_compress(d_oriData, d_cmpBytes, nbEle, &cmpSize1, errorBound, CUSZP_DIM_3D, dims, CUSZP_TYPE_DOUBLE, CUSZP_MODE_FIXED, stream);
    }

    // cuSZp-f testing.
    printf("=================================================\n");
    printf("========Testing cuSZp-f-3D-f64 on REL 1E-2=======\n");
    printf("=================================================\n");
    // cuSZp compression.
    timer_GPU.StartCounter(); // set timer
    cuSZp_compress_3D_fixed_f64(d_oriData, d_cmpBytes, nbEle, &cmpSize1, dims, errorBound, stream);
    float cmpTime = timer_GPU.GetCounter();

    // Transfer compressed data to CPU then back to GPU, making sure compression ratio is correct.
    // No need to add this part for real-world usages, this is only for testing compresion ratio correcness.
    unsigned char* cmpBytes_dup1 = (unsigned char*)malloc(cmpSize1*sizeof(unsigned char));
    cudaMemcpy(cmpBytes_dup1, d_cmpBytes, cmpSize1*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemset(d_cmpBytes, 0, sizeof(double)*nbEle); // set to zero for double check.
    cudaMemcpy(d_cmpBytes, cmpBytes_dup1, cmpSize1*sizeof(unsigned char), cudaMemcpyHostToDevice);
        
    // cuSZp decompression.
    timer_GPU.StartCounter(); // set timer
    cuSZp_decompress_3D_fixed_f64(d_decData, d_cmpBytes, nbEle, cmpSize1, dims, errorBound, stream);
    float decTime = timer_GPU.GetCounter();

    // Print result.
    printf("cuSZp-f finished!\n");
    printf("cuSZp-f compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/cmpTime);
    printf("cuSZp-f decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/decTime);
    printf("cuSZp-f compression ratio: %f\n", (nbEle*sizeof(double)/1024.0/1024.0)/(cmpSize1*sizeof(unsigned char)/1024.0/1024.0));
    
    // Error check
    cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize1*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(decData, d_decData, sizeof(double)*nbEle, cudaMemcpyDeviceToHost);
    int not_bound = 0;
    for(size_t i=0; i<nbEle; i++)
    {
        if(fabs(oriData[i]-decData[i]) > errorBound*1.1)
        {
            not_bound++;
            // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound: %f\n", i, oriData[i], decData[i], fabs(oriData[i]-decData[i]), errorBound);
        }
    }
    if(!not_bound) printf("\033[0;32mPass error check!\033[0m\n");
    else printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n", not_bound);
    printf("\033[1mDone with testing cuSZp-f on REL 1E-2!\033[0m\n\n");

    // cuSZp-p testing.
    printf("=================================================\n");
    printf("========Testing cuSZp-p-3D-f64 on REL 1E-2=======\n");
    printf("=================================================\n");
    // cuSZp compression.
    timer_GPU.StartCounter(); // set timer
    cuSZp_compress_3D_plain_f64(d_oriData, d_cmpBytes, nbEle, &cmpSize2, dims, errorBound, stream);
    cmpTime = timer_GPU.GetCounter();

    // Transfer compressed data to CPU then back to GPU, making sure compression ratio is correct.
    // No need to add this part for real-world usages, this is only for testing compresion ratio correcness.
    unsigned char* cmpBytes_dup2 = (unsigned char*)malloc(cmpSize2*sizeof(unsigned char));
    cudaMemcpy(cmpBytes_dup2, d_cmpBytes, cmpSize2*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemset(d_cmpBytes, 0, sizeof(double)*nbEle); // set to zero for double check.
    cudaMemcpy(d_cmpBytes, cmpBytes_dup2, cmpSize2*sizeof(unsigned char), cudaMemcpyHostToDevice);
        
    // cuSZp decompression.
    timer_GPU.StartCounter(); // set timer
    cuSZp_decompress_3D_plain_f64(d_decData, d_cmpBytes, nbEle, cmpSize2, dims, errorBound, stream);
    decTime = timer_GPU.GetCounter();

    // Print result.
    printf("cuSZp-p finished!\n");
    printf("cuSZp-p compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/cmpTime);
    printf("cuSZp-p decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/decTime);
    printf("cuSZp-p compression ratio: %f\n", (nbEle*sizeof(double)/1024.0/1024.0)/(cmpSize2*sizeof(unsigned char)/1024.0/1024.0));
    
    // Error check
    cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize2*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(decData, d_decData, sizeof(double)*nbEle, cudaMemcpyDeviceToHost);
    not_bound = 0;
    for(size_t i=0; i<nbEle; i++)
    {
        if(fabs(oriData[i]-decData[i]) > errorBound*1.1)
        {
            not_bound++;
            // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound: %f\n", i, oriData[i], decData[i], fabs(oriData[i]-decData[i]), errorBound);
        }
    }
    if(!not_bound) printf("\033[0;32mPass error check!\033[0m\n");
    else printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n", not_bound);
    printf("\033[1mDone with testing cuSZp-p on REL 1E-2!\033[0m\n\n");

    // cuSZp-o testing.
    printf("=================================================\n");
    printf("========Testing cuSZp-o-3D-f64 on REL 1E-2=======\n");
    printf("=================================================\n");
    // cuSZp compression.
    timer_GPU.StartCounter(); // set timer
    cuSZp_compress_3D_outlier_f64(d_oriData, d_cmpBytes, nbEle, &cmpSize3, dims, errorBound, stream);
    cmpTime = timer_GPU.GetCounter();

    // Transfer compressed data to CPU then back to GPU, making sure compression ratio is correct.
    // No need to add this part for real-world usages, this is only for testing compresion ratio correcness.
    unsigned char* cmpBytes_dup3 = (unsigned char*)malloc(cmpSize3*sizeof(unsigned char));
    cudaMemcpy(cmpBytes_dup3, d_cmpBytes, cmpSize3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemset(d_cmpBytes, 0, sizeof(double)*nbEle); // set to zero for double check.
    cudaMemcpy(d_cmpBytes, cmpBytes_dup3, cmpSize3*sizeof(unsigned char), cudaMemcpyHostToDevice);
        
    // cuSZp decompression.
    timer_GPU.StartCounter(); // set timer
    cuSZp_decompress_3D_outlier_f64(d_decData, d_cmpBytes, nbEle, cmpSize3, dims, errorBound, stream);
    decTime = timer_GPU.GetCounter();

    // Print result.
    printf("cuSZp-o finished!\n");
    printf("cuSZp-o compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/cmpTime);
    printf("cuSZp-o decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/decTime);
    printf("cuSZp-o compression ratio: %f\n", (nbEle*sizeof(double)/1024.0/1024.0)/(cmpSize3*sizeof(unsigned char)/1024.0/1024.0));

    // Error check
    cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(decData, d_decData, sizeof(double)*nbEle, cudaMemcpyDeviceToHost);
    not_bound = 0;
    for(size_t i=0; i<nbEle; i++)
    {
        if(fabs(oriData[i]-decData[i]) > errorBound*1.1)
        {
            not_bound++;
            // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound: %f\n", i, oriData[i], decData[i], fabs(oriData[i]-decData[i]), errorBound);
        }
    }
    if(!not_bound) printf("\033[0;32mPass error check!\033[0m\n");
    else printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n", not_bound);
    printf("\033[1mDone with testing cuSZp-o on REL 1E-2!\033[0m\n");

    free(oriData);
    free(decData);
    free(cmpBytes);
    free(cmpBytes_dup1);
    free(cmpBytes_dup2);
    free(cmpBytes_dup3);
    cudaFree(d_oriData);
    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);

    return 0;
}