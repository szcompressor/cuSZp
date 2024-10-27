#ifndef CUSZP_H
#define CUSZP_H

#include "cuSZp/cuSZp_utility.h"
#include "cuSZp/cuSZp_timer.h"
#include "cuSZp/cuSZp_entry_f32.h"
#include "cuSZp/cuSZp_entry_f64.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CUSZP_MODE_PLAIN   = 0, // Plain   fixed-length encoding mode
    CUSZP_MODE_OUTLIER = 1  // Outlier fixed-length encoding mode
} cuszp_mode_t;

typedef enum {
    CUSZP_TYPE_FLOAT  = 0,  // Single precision floating point (f32)
    CUSZP_TYPE_DOUBLE = 1   // Double precision floating point (f64)
} cuszp_type_t;

void cuSZp_compress(void* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cuszp_type_t type, cuszp_mode_t mode, cudaStream_t stream = 0);
void cuSZp_decompress(void* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cuszp_type_t type, cuszp_mode_t mode, cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif

#endif // CUSZP_H