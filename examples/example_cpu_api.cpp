#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuSZp_utility.h>
#include <cuSZp_entry.h>

int main(int argc, char* argv[])
{
    // Read input information.
    char oriFilePath[640];
    int status=0;
    if(argc != 3)
    {
        printf("Usage: cuSZp [srcFilePath] [abs err bound]\n");
        printf("Example: cuSZp testfloat_8_8_128.dat 1e-3\n");
        exit(0);
    }
    sprintf(oriFilePath, "%s", argv[1]);
    float errorBound = atof(argv[2]);

    // Input data preparation.
    float* oriData = NULL;
    float* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 0;
    size_t cmpSize = 0;
    oriData = readFloatData_Yafan(oriFilePath, &nbEle, &status);
    decData = (float*)malloc(nbEle*sizeof(float));
    cmpBytes = (unsigned char*)malloc(nbEle*sizeof(float));

    // cuSZp compression.
    SZp_compress_hostptr(oriData, cmpBytes, nbEle, &cmpSize, errorBound);
    
    // cuSZp decompression.
    SZp_decompress_hostptr(decData, cmpBytes, nbEle, cmpSize, errorBound);

    // Print result.
    printf("cuSZp finished!\n");
    printf("compression ratios: %f\n\n", (nbEle*sizeof(float)/1024.0/1024.0)/(cmpSize*sizeof(unsigned char)/1024.0/1024.0));

    // Error check
    int not_bound = 0;
    for(size_t i=0; i<nbEle; i+=1)
    {
        if(abs(oriData[i]-decData[i]) > errorBound*1.1)
        {
            not_bound++;
            // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound: %f\n", i, oriData[i], decData[i], abs(oriData[i]-decData[i]), errBound);
        }
    }
    if(!not_bound) printf("Pass error check!\n");
    else printf("Fail error check!\n");
    
    free(oriData);
    free(decData);
    free(cmpBytes);
    return 0;
}