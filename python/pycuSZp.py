import ctypes
import numpy as np

# Load the shared library (using default installation path, replace if neccessary)
lib = ctypes.CDLL('../install/lib/libcuSZp.so')

# Define uint3 in ctypes
class Uint3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]

# Update function signatures with dim and dims
lib.cuSZp_compress.argtypes = [
    ctypes.c_void_p,                   # d_oriData
    ctypes.c_void_p,                   # d_cmpBytes
    ctypes.c_size_t,                   # nbEle
    ctypes.POINTER(ctypes.c_size_t),  # cmpSize
    ctypes.c_float,                    # errorBound
    ctypes.c_int,                      # dim (cuszp_dim_t),   1 for 1D, 2 for 2D, 3 for 3D
    Uint3,                             # dims (uint3)
    ctypes.c_int,                      # type (cuszp_type_t), 0 for f32, 1 for f64
    ctypes.c_int,                      # mode (cuszp_mode_t), 0 for fixed, 1 for plain, 2 for outlier
    ctypes.c_void_p                    # stream
]

lib.cuSZp_decompress.argtypes = [
    ctypes.c_void_p,                   # d_decData
    ctypes.c_void_p,                   # d_cmpBytes
    ctypes.c_size_t,                   # nbEle
    ctypes.c_size_t,                   # cmpSize
    ctypes.c_float,                    # errorBound
    ctypes.c_int,                      # dim (cuszp_dim_t)
    Uint3,                             # dims
    ctypes.c_int,                      # type
    ctypes.c_int,                      # mode
    ctypes.c_void_p                    # stream
]

class cuSZp:
    def __init__(self):
        pass

    def compress(self, d_oriData, d_cmpBytes, num_elements, error_bound=0.01,
                 dim=1, dims=(0, 0, 0), data_type=0, mode=0, stream=None):
        compressed_size = ctypes.c_size_t(0)
        dims_struct = Uint3(dims[0], dims[1], dims[2])  # Convert tuple to uint3

        lib.cuSZp_compress(
            d_oriData,
            d_cmpBytes,
            num_elements,
            ctypes.byref(compressed_size),
            ctypes.c_float(error_bound),
            ctypes.c_int(dim),
            dims_struct,
            ctypes.c_int(data_type),
            ctypes.c_int(mode),
            ctypes.c_void_p(0) if stream is None else stream
        )

        return compressed_size.value

    def decompress(self, d_decData, d_cmpBytes, num_elements, compressed_size,
                   error_bound=0.01, dim=1, dims=(0, 0, 0), data_type=0, mode=0, stream=None):
        dims_struct = Uint3(dims[0], dims[1], dims[2])

        lib.cuSZp_decompress(
            d_decData,
            d_cmpBytes,
            num_elements,
            compressed_size,
            ctypes.c_float(error_bound),
            ctypes.c_int(dim),
            dims_struct,
            ctypes.c_int(data_type),
            ctypes.c_int(mode),
            ctypes.c_void_p(0) if stream is None else stream
        )
