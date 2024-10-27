#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuSZp.h>

int main(int argc, char* argv[])
{
    // Variables for user-proposed input.
    char oriFilePath[640] = {0};
    char cmpFilePath[640] = {0};
    char decFilePath[640] = {0};
    cuszp_type_t dataType;
    cuszp_mode_t encodingMode;
    char errorBoundMode[4];
    float errorBound = 0.0;

    // Check if there are enough arguments
    if (argc < 9) {
        printf("cuSZp: A GPU error-bounded lossy compressor for floating-point data\n");
        printf("\n");
        printf("Usage:\n");
        printf("  ./cuSZp -i [oriFilePath] -t [dataType] -m [encodingMode] -eb [errorBoundMode] [errorBound] -x [cmpFilePath] -o [decFilePath]\n");
        printf("\n");
        printf("Options:\n");
        printf("  -i  [oriFilePath]    Path to the input file containing the original data\n");
        printf("  -t  [dataType]       Data type of the input data. Options:\n");
        printf("                       f32      : Single precision (float)\n");
        printf("                       f64      : Double precision (double)\n");
        printf("  -m  [encodingMode]   Encoding mode to use. Options:\n");
        printf("                       plain    : Plain fixed-length encoding mode\n");
        printf("                       outlier  : Outlier fixed-length encoding mode\n");
        printf("  -eb [errorBoundMode] [errorBound]\n");
        printf("                       errorBoundMode can only be:\n");
        printf("                       abs      : Absolute error bound\n");
        printf("                       rel      : Relative error bound\n");
        printf("                       errorBound is a floating-point number representing the error bound, e.g. 1E-4, 0.03.\n");
        printf("  -x  [cmpFilePath]    Path to the compressed output file (optional)\n");
        printf("  -o  [decFilePath]    Path to the decompressed output file (optional)\n");
        printf("\n");
        printf("Examples:\n");
        printf("  ./cuSZp -i pressure_3000 -t f32 -m plain -eb abs 1E-4 -x pressure_3000.cuszp.cmp -o pressure_3000.cuszp.dec\n");
        printf("  ./cuSZp -i ccd-tst.bin.d64 -t f64 -m outlier -eb abs 0.01\n");
        printf("  ./cuSZp -i velocity_x.f32 -t f32 -m outlier -eb rel 0.01 -x velocity_x.f32.cuszp.cmp\n");
        printf("  ./cuSZp -i xx.f32 -m outlier -eb rel 1e-4 -t f32\n");
        exit(EXIT_FAILURE);
    }

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            // Input file path
            if (i + 1 < argc) {
                strncpy(oriFilePath, argv[i + 1], sizeof(oriFilePath) - 1);
                i++;
            }
        } else if (strcmp(argv[i], "-t") == 0) {
            // Data type
            if (i + 1 < argc) {
                if (strcmp(argv[i + 1], "f32") == 0) {
                    dataType = CUSZP_TYPE_FLOAT;
                } else if (strcmp(argv[i + 1], "f64") == 0) {
                    dataType = CUSZP_TYPE_DOUBLE;
                } else {
                    printf("Error: Unsupported data type. Use 'f32' or 'f64'.\n");
                    exit(EXIT_FAILURE);
                }
                i++;
            }
        } else if (strcmp(argv[i], "-m") == 0) {
            // Encoding mode
            if (i + 1 < argc) {
                if (strcmp(argv[i + 1], "plain") == 0) {
                    encodingMode = CUSZP_MODE_PLAIN;
                } else if (strcmp(argv[i + 1], "outlier") == 0) {
                    encodingMode = CUSZP_MODE_OUTLIER;
                } else {
                    printf("Error: Unsupported encoding mode. Use 'plain' or 'outlier'.\n");
                    exit(EXIT_FAILURE);
                }
                i++;
            }
        } else if (strcmp(argv[i], "-eb") == 0) {
            // Error bound mode and value
            if (i + 2 < argc) {
                strncpy(errorBoundMode, argv[i + 1], sizeof(errorBoundMode) - 1);
                if (strcmp(errorBoundMode, "rel") != 0 && strcmp(errorBoundMode, "abs") != 0) {
                    printf("Error: Unsupported error bound mode. Use 'rel' or 'abs'.\n");
                    exit(EXIT_FAILURE);
                }
                errorBound = atof(argv[i + 2]);
                i += 2;
            }
        } else if (strcmp(argv[i], "-x") == 0) {
            // Compressed file path (optional)
            if (i + 1 < argc) {
                strncpy(cmpFilePath, argv[i + 1], sizeof(cmpFilePath) - 1);
                i++;
            }
        } else if (strcmp(argv[i], "-o") == 0) {
            // Decompressed file path (optional)
            if (i + 1 < argc) {
                strncpy(decFilePath, argv[i + 1], sizeof(decFilePath) - 1);
                i++;
            }
        } else {
            printf("Error: Unrecognized option '%s'.\n", argv[i]);
            exit(EXIT_FAILURE);
        }
    }

    // Ensure required arguments are provided
    if (strlen(oriFilePath) == 0 || errorBound == 0.0) {
        printf("Error: Missing required arguments.\n");
        exit(EXIT_FAILURE);
    }

    // For measuring the end-to-end throughput.
    TimingGPU timer_GPU;

    // Input data preparation on CPU.
    void* oriData = NULL;
    void* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 0;
    size_t cmpSize = 0;
    int status=0;
    if(dataType == CUSZP_TYPE_FLOAT) {
        oriData = (void*)readFloatData_Yafan(oriFilePath, &nbEle, &status);
        decData = (void*)malloc(nbEle*sizeof(float));
        cmpBytes = (unsigned char*)malloc(nbEle*sizeof(float));
        
        // Value range calculation f32
        if(strcmp(errorBoundMode, "rel") == 0) {
            float* oriData_f32 = (float*)oriData;
            float max_val = oriData_f32[0];
            float min_val = oriData_f32[0];
            for(size_t i=0; i<nbEle; i++) {
                if(oriData_f32[i]>max_val)
                    max_val = oriData_f32[i];
                else if(oriData_f32[i]<min_val)
                    min_val = oriData_f32[i];
            }
            errorBound = errorBound * (max_val - min_val);
        }
    }
    else if(dataType == CUSZP_TYPE_DOUBLE) {
        oriData = (void*)readDoubleData_Yafan(oriFilePath, &nbEle, &status);
        decData = (void*)malloc(nbEle*sizeof(double));
        cmpBytes = (unsigned char*)malloc(nbEle*sizeof(double));

        // Value range calculation f64
        if(strcmp(errorBoundMode, "rel") == 0) {
            double* oriData_f64 = (double*)oriData;
            double max_val = oriData_f64[0];
            double min_val = oriData_f64[0];
            for(size_t i=0; i<nbEle; i++) {
                if(oriData_f64[i]>max_val)
                    max_val = oriData_f64[i];
                else if(oriData_f64[i]<min_val)
                    min_val = oriData_f64[i];
            }
            errorBound = errorBound * (float)(max_val - min_val);
        }
    }

    // Input data preparation on GPU.
    void* d_oriData;
    void* d_decData;
    unsigned char* d_cmpBytes;
    if(dataType == CUSZP_TYPE_FLOAT) {
        cudaMalloc((void**)&d_oriData, sizeof(float)*nbEle);
        cudaMemcpy(d_oriData, oriData, sizeof(float)*nbEle, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_decData, sizeof(float)*nbEle);
        cudaMalloc((void**)&d_cmpBytes, sizeof(float)*nbEle);
        cudaMemset(d_decData, 0, sizeof(float)*nbEle); // Initialize to 0 array in decompressed data.
    }
    else if(dataType == CUSZP_TYPE_DOUBLE) {
        cudaMalloc((void**)&d_oriData, sizeof(double)*nbEle);
        cudaMemcpy(d_oriData, oriData, sizeof(double)*nbEle, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_decData, sizeof(double)*nbEle);
        cudaMalloc((void**)&d_cmpBytes, sizeof(double)*nbEle);
        cudaMemset(d_decData, 0, sizeof(double)*nbEle); // Initialize to 0 array in decompressed data.
    }

    // Initializing CUDA Stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Warmup for NVIDIA GPU.
    for(int i=0; i<10; i++) {
        cuSZp_compress(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, dataType, encodingMode, stream);
    }

    // cuSZp compression.
    timer_GPU.StartCounter(); // set timer
    cuSZp_compress(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, dataType, encodingMode, stream);
    float cmpTime = timer_GPU.GetCounter();

    // Transfer compressed data to CPU then back to GPU, making sure compression ratio is correct.
    // No need to add this part for real-world usages, this is only for testing compresion ratio correcness.
    unsigned char* cmpBytes_dup = (unsigned char*)malloc(cmpSize*sizeof(unsigned char));
    cudaMemcpy(cmpBytes_dup, d_cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if(dataType == CUSZP_TYPE_FLOAT) {
        cudaMemset(d_cmpBytes, 0, sizeof(float)*nbEle); // set to zero for double check.
    }
    else if(dataType == CUSZP_TYPE_DOUBLE) {
        cudaMemset(d_cmpBytes, 0, sizeof(double)*nbEle); // set to zero for double check.
    }
    cudaMemcpy(d_cmpBytes, cmpBytes_dup, cmpSize*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // cuSZp decompression.
    timer_GPU.StartCounter(); // set timer
    cuSZp_decompress(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, dataType, encodingMode, stream);
    float decTime = timer_GPU.GetCounter();

    // Print result.
    printf("cuSZp finished!\n");
    if(dataType == CUSZP_TYPE_FLOAT) {
        printf("cuSZp compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/cmpTime);
        printf("cuSZp decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/decTime);
        printf("cuSZp compression ratio: %f\n\n", (nbEle*sizeof(float)/1024.0/1024.0)/(cmpSize*sizeof(unsigned char)/1024.0/1024.0));
    }
    else if(dataType == CUSZP_TYPE_DOUBLE) {
        printf("cuSZp compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/cmpTime);
        printf("cuSZp decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(double)/1024.0/1024.0)/decTime);
        printf("cuSZp compression ratio: %f\n\n", (nbEle*sizeof(double)/1024.0/1024.0)/(cmpSize*sizeof(unsigned char)/1024.0/1024.0));
    }

    // Error check
    if(dataType == CUSZP_TYPE_FLOAT) {
        cudaMemcpy(decData, d_decData, sizeof(float)*nbEle, cudaMemcpyDeviceToHost);
    }
    else if(dataType == CUSZP_TYPE_DOUBLE) {
        cudaMemcpy(decData, d_decData, sizeof(double)*nbEle, cudaMemcpyDeviceToHost);
    }
    int not_bound = 0;
    for(size_t i=0; i<nbEle; i++) {
        if(dataType == CUSZP_TYPE_FLOAT) {
            float* oriData_f32 = (float*)oriData;
            float* decData_f32 = (float*)decData;
            if(fabs(oriData_f32[i]-decData_f32[i]) > errorBound*1.1) {
                not_bound++;
                // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound: %f\n", i, ((float*)oriData)[i], ((float*)decData)[i], fabs(((float*)oriData)[i]-((float*)decData)[i]), errorBound);
            }
        }
        else if(dataType == CUSZP_TYPE_DOUBLE) {
            double* oriData_f64 = (double*)oriData;
            double* decData_f64 = (double*)decData;
            if(fabs(oriData_f64[i]-decData_f64[i]) > errorBound*1.1) {
                not_bound++;
                // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound: %f\n", i, ((double*)oriData)[i], ((double*)decData)[i], fabs(((double*)oriData)[i]-((double*)decData)[i]), errorBound);
            }
        }
    }
    if(!not_bound) printf("\033[0;32mPass error check!\033[0m\n");
    else printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n", not_bound);

    // Write compressed or reconstructed files if needed.
    if (strlen(cmpFilePath) > 0) {
        cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
        writeByteData_Yafan(cmpBytes, cmpSize, cmpFilePath, &status);
    }
    if (strlen(decFilePath) > 0) {
        if(dataType == CUSZP_TYPE_FLOAT) {
            writeFloatData_inBytes_Yafan((float*)decData, nbEle, decFilePath, &status);
        }
        else if(dataType == CUSZP_TYPE_DOUBLE) {
            writeDoubleData_inBytes_Yafan((double*)decData, nbEle, decFilePath, &status);
        }
    }

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