#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuSZp.h>

void printUsage() {
    printf("Usage: ./cuSZp_2D -i [input_file_path] -d [dim_z] [dim_y] [dim_x] -eb [error_mode] [error_bound] [-x cmpFilePath] [-o decFilePath]\n");
    printf("    -i  : Input file path (required)\n");
    printf("    -d  : Dimensions (required, order: dim_z dim_y dim_x, dim_x is the fastest dim)\n");
    printf("    -eb : Error bound mode (\"rel\" or \"abs\") followed by error bound value (required)\n");
    printf("    -x  : Compressed file path (optional)\n");
    printf("    -o  : Decompressed file path (optional)\n");
}

int main(int argc, char *argv[])
{
    // Variables for user-proposed input.
    char oriFilePath[640] = {0};
    char cmpFilePath[640] = {0};
    char decFilePath[640] = {0};
    uint3 dims = {0, 0, 0};
    char errorBoundMode[4] = {0};
    double errorBound = 0.0f;

    // Flags to check if required arguments are provided
    int hasInput = 0, hasDims = 0, hasErrorBound = 0;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            strncpy(oriFilePath, argv[i + 1], 639);
            hasInput = 1;
            i++;
        } else if (strcmp(argv[i], "-d") == 0 && i + 3 < argc) {
            dims.z = (unsigned int)atoi(argv[i + 1]);
            dims.y = (unsigned int)atoi(argv[i + 2]);
            dims.x = (unsigned int)atoi(argv[i + 3]);
            hasDims = 1;
            i += 3;
        } else if (strcmp(argv[i], "-eb") == 0 && i + 2 < argc) {
            if (strcmp(argv[i + 1], "rel") == 0 || strcmp(argv[i + 1], "abs") == 0) {
                strncpy(errorBoundMode, argv[i + 1], 3);
                errorBound = atof(argv[i + 2]);
                hasErrorBound = 1;
                i += 2;
            } else {
                printf("Error: Invalid error bound mode. Use \"rel\" or \"abs\".\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            strncpy(cmpFilePath, argv[i + 1], 639);
            i++;
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            strncpy(decFilePath, argv[i + 1], 639);
            i++;
        } else {
            printf("Error: Unrecognized or incomplete argument: %s\n", argv[i]);
            printUsage();
            return 1;
        }
    }

    // Check if required arguments are provided
    if (!hasInput || !hasDims || !hasErrorBound) {
        printf("Error: Missing required arguments.\n");
        printUsage();
        return 1;
    }

    // // Yafan is checking input by printing parsed values
    // printf("Input File Path: %s\n", oriFilePath);
    // printf("Dimensions: z=%u, y=%u, x=%u\n", dims.z, dims.y, dims.x);
    // printf("Error Bound Mode: %s\n", errorBoundMode);
    // printf("Error Bound Value: %f\n", errorBound);
    // if (cmpFilePath[0] != '\0') {
    //     printf("Compressed File Path: %s\n", cmpFilePath);
    // }
    // if (decFilePath[0] != '\0') {
    //     printf("Decompressed File Path: %s\n", decFilePath);
    // }

    // Data preparation on CPU.
    double* oriData = NULL;
    double* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 0;
    size_t cmpSize = 0;
    int status = 0;
    oriData = readDoubleData_Yafan(oriFilePath, &nbEle, &status);
    decData = (double*)malloc(nbEle*sizeof(double));
    cmpBytes = (unsigned char*)malloc(nbEle*sizeof(double));
    if(nbEle != (size_t)dims.x * (size_t)dims.y * (size_t)dims.z) {
        fprintf(stderr, "Error: The number of elements in the original data does not match the dimensions\n");
        return 1;
    }
    
    // Updating error bound if rel mode.
    if(strcmp(errorBoundMode, "rel") == 0) {
        double max_val = oriData[0];
        double min_val = oriData[0];
        for(size_t i=0; i<nbEle; i++) {
            if(oriData[i]>max_val)
                max_val = oriData[i];
            else if(oriData[i]<min_val)
                min_val = oriData[i];
        }
        errorBound = (max_val - min_val) * errorBound;
    }

    // For measuring the end-to-end throughput.
    TimingGPU timer_GPU;

    // Data preparation on GPU.
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
    for(int i=0; i<3; i++) {
        cuSZp_compress_2D_plain_f64(d_oriData, d_cmpBytes, nbEle, &cmpSize, dims, errorBound, stream);
        // cuSZp_compress_2D_outlier_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, dims, errorBound, stream);
    }

    // cuSZp compression
    timer_GPU.StartCounter(); // set timer
    cuSZp_compress_2D_plain_f64(d_oriData, d_cmpBytes, nbEle, &cmpSize, dims, errorBound, stream);
    // cuSZp_compress_2D_outlier_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, dims, errorBound, stream);
    float cmpTime = timer_GPU.GetCounter();

    // Transfer compressed data to CPU then back to GPU, making sure compression ratio is correct.
    // No need to add this part for real-world usages, this is only for testing compresion ratio correcness.
    unsigned char* cmpBytes_dup = (unsigned char*)malloc(cmpSize*sizeof(unsigned char));
    cudaMemcpy(cmpBytes_dup, d_cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemset(d_cmpBytes, 0, sizeof(double)*nbEle); // set to zero for double check.
    cudaMemcpy(d_cmpBytes, cmpBytes_dup, cmpSize*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // cuSZp decompression.
    timer_GPU.StartCounter(); // set timer
    cuSZp_decompress_2D_plain_f64(d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
    // cuSZp_decompress_2D_outlier_f32(d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
    float decTime = timer_GPU.GetCounter();

    // Print result.
    printf("cuSZp finished! (2D implementation)\n");
    printf("cuSZp compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/cmpTime);
    printf("cuSZp decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/decTime);
    printf("cuSZp compression ratio: %f\n\n", (nbEle*sizeof(double)/1024.0/1024.0)/(cmpSize*sizeof(unsigned char)/1024.0/1024.0));

    // Error check.
    int not_bound = 0;
    cudaMemcpy(decData, d_decData, sizeof(double)*nbEle, cudaMemcpyDeviceToHost);
    for(size_t i=0; i<nbEle; i++) {
        if(fabs(oriData[i]-decData[i]) > errorBound*1.1) {
            not_bound++;
            // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound: %f\n", i, oriData[i], decData[i], fabs(oriData[i]-decData[i]), errorBound);
        }
    }
    if(!not_bound) printf("\033[0;32mPass error check!\033[0m\n");
    else printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n", not_bound);

    free(oriData);
    free(decData);
    free(cmpBytes);
    free(cmpBytes_dup);
    cudaFree(d_oriData);
    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);
    return 0;
}