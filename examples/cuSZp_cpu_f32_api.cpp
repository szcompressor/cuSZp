#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <cuSZp_utility.h>
#include <cuSZp_entry_f32.h>

int main(int argc, char* argv[])
{
    // Read input information.
    char oriFilePath[640];
    char errorMode[20];
    int status=0;
    if(argc != 4)
    {
        printf("Usage: cuSZp_cpu_f32_api [srcFilePath] [errorMode] [errBound] # errorMode can only be ABS or REL\n");
        printf("Example: cuSZp_cpu_f32_api testfloat_8_8_128.dat ABS 1E-2     # compress dataset with absolute 1E-2 error bound\n");
        printf("         cuSZp_cpu_f32_api testfloat_8_8_128.dat REL 1e-3     # compress dataset with relative 1E-3 error bound\n");
        exit(0);
    }
    sprintf(oriFilePath, "%s", argv[1]);
    sprintf(errorMode, "%s", argv[2]);
    float errorBound = atof(argv[3]);

    // Input data preparation.
    float* oriData = NULL;
    float* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 0;
    size_t cmpSize = 0;
    oriData = readFloatData_Yafan(oriFilePath, &nbEle, &status);
    decData = (float*)malloc(nbEle*sizeof(float));
    cmpBytes = (unsigned char*)malloc(nbEle*sizeof(float));

    // Generating error bounds.
    if(strcmp(errorMode, "REL")==0)
    {
        float max_val = oriData[0];
        float min_val = oriData[0];
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

    // cuSZp compression.
    SZp_compress_hostptr_f32(oriData, cmpBytes, nbEle, &cmpSize, errorBound);
    
    // cuSZp decompression.
    SZp_decompress_hostptr_f32(decData, cmpBytes, nbEle, cmpSize, errorBound);

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
    if(!not_bound) printf("\033[0;32mPass error check!\033[0m\n");
    else printf("\033[0;31mFail error check!\033[0m\n");
    
    // Free allocated data.
    free(oriData);
    free(decData);
    free(cmpBytes);
    return 0;
}