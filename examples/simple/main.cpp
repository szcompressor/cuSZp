#include <cstdlib>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuSZp-entry.h>
#include "utility.h"

int main(int argc, char*argv[]) {
    int status = 0;
    char oriFilePath[1024];

    // Parse command line arguments
    if (argc != 6) {
        printf("Usage: <program name> <input data file path> <absolute error bound> <block size> <lorenzo prediction level> <bit-shuffle level(0/1)>\n");
        exit(EXIT_FAILURE);
    }
    sprintf(oriFilePath, "%s", argv[1]);
    float errBound = atof(argv[2]);
    int blockSize = atoi(argv[3]);
    int predLevel = atoi(argv[4]);
    int shufLevel = atoi(argv[5]);
    printf("Input data file path: %s\n", oriFilePath);
    printf("Absolute error bound: %f\n", errBound);
    printf("Block size: %d\n", blockSize);
    printf("Lorenzo prediction level: %d\n", predLevel);
    printf("Bit shuffle level: %d\n", shufLevel);

    float *oriData = NULL;
    size_t nbEle = 0;
    printf("Reading data from %s \n", oriFilePath);
    oriData = readFloatData_Yafan(oriFilePath, &nbEle, &status);
    printf("Data size: %f GB\n", (nbEle * sizeof(float)) / 1024. / 1024. / 1024.);
    printf("Number of elements: %zu\n", nbEle);

    // Compress data
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t outSize = 0;
    unsigned char *cmpBytes = (unsigned char *) malloc(sizeof(float) * nbEle);
    cuSZp_compress(oriData, cmpBytes, &outSize, errBound, nbEle, blockSize, predLevel, shufLevel, stream);

    // Decompress data
    float *decData = (float *) malloc(sizeof(float) * nbEle);
    cuSZp_decompress(decData, cmpBytes, errBound, nbEle, blockSize, predLevel, shufLevel, stream);

    // Check absolute error
    size_t counter = 0;
    for (size_t i = 0; i < nbEle; i++) {
        float absError = abs(oriData[i] - decData[i]);
        if (absError > errBound*1.1) {
            counter++;
            printf("%zu, %.20f, %.20f, %.20f, %.20f\n", i, oriData[i], decData[i], absError, errBound);
        }
    }
    if (counter) {
        printf("Test failed. Number of elements exceeding absolute error bound is %zu\n", counter);
    }

    // Free memory
    free(oriData);
    free(cmpBytes);
    free(decData);

    return 0;
}

