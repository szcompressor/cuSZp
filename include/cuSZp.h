#ifndef CUSZP_H
#define CUSZP_H

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus

#include "cuSZp/cuSZp_entry_1D_f32.h"
#include "cuSZp/cuSZp_entry_1D_f64.h"
#include "cuSZp/cuSZp_entry_2D_f32.h"
#include "cuSZp/cuSZp_entry_2D_f64.h"
#include "cuSZp/cuSZp_entry_3D_f32.h"
#include "cuSZp/cuSZp_entry_3D_f64.h"
#include "cuSZp/cuSZp_timer.h"
#include "cuSZp/cuSZp_utility.h"

extern "C" {
#endif

typedef enum {
  CUSZP_DIM_1D = 1, // 1D Processing Manner (can be used for all datasets)
  CUSZP_DIM_2D = 2, // 2D Processing Manner (can be used for both 2D and 3D)
  CUSZP_DIM_3D = 3  // 3d Processing Manner (can be used for only 3D datasets)
} cuszp_dim_t;

typedef enum {
  CUSZP_MODE_FIXED = 0,   // No-delta           fixed-length encoding mode
  CUSZP_MODE_PLAIN = 1,   // Plain (with delta) fixed-length encoding mode
  CUSZP_MODE_OUTLIER = 2, // Outlier            fixed-length encoding mode
  // CUSZP_MODE_AATROX  = 3  // AaTrox (ICS'25)    fixed-length encoding mode
} cuszp_mode_t;

typedef enum {
  CUSZP_TYPE_FLOAT = 0, // Single precision floating point (f32)
  CUSZP_TYPE_DOUBLE = 1 // Double precision floating point (f64)
} cuszp_type_t;

void cuSZp_compress(void *d_oriData, unsigned char *d_cmpBytes, size_t nbEle,
                    size_t *cmpSize, float errorBound, cuszp_dim_t dim,
                    uint3 dims, cuszp_type_t type, cuszp_mode_t mode,
                    cudaStream_t stream);
void cuSZp_decompress(void *d_decData, unsigned char *d_cmpBytes, size_t nbEle,
                      size_t cmpSize, float errorBound, cuszp_dim_t dim,
                      uint3 dims, cuszp_type_t type, cuszp_mode_t mode,
                      cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // CUSZP_H
