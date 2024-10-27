import ctypes
import numpy as np

# Load the shared library (using default installation path, replace if neccessary)
lib = ctypes.CDLL('../install/lib/libcuSZp.so')

# Define ctypes for the C functions with void pointers for flexible input/output data
lib.cuSZp_compress.argtypes = [
    ctypes.c_void_p,  # GPU pointer for input data (void* for flexibility)
    ctypes.c_void_p,  # GPU pointer for compressed output (interpreted as unsigned char*)
    ctypes.c_size_t,  # Number of elements
    ctypes.POINTER(ctypes.c_size_t),  # Compressed size output
    ctypes.c_float,  # Error bound
    ctypes.c_int,    # Data type (0 for float, 1 for double)
    ctypes.c_int,    # Mode (0 for plain, 1 for outlier)
    ctypes.c_void_p  # Stream (optional, can be None)
]

lib.cuSZp_decompress.argtypes = [
    ctypes.c_void_p,  # GPU pointer for decompressed output (void*)
    ctypes.c_void_p,  # Compressed input buffer (unsigned char*)
    ctypes.c_size_t,  # Number of elements
    ctypes.c_size_t,  # Compressed size
    ctypes.c_float,   # Error bound
    ctypes.c_int,     # Data type (0 for float, 1 for double)
    ctypes.c_int,     # Mode (0 for plain, 1 for outlier)
    ctypes.c_void_p   # Stream (optional, can be None)
]

class cuSZp:
    def __init__(self):
        pass

    def compress(self, d_oriData, d_cmpBytes, num_elements, error_bound=0.01, data_type=0, mode=0):
        # Initialize size for compressed data
        compressed_size = ctypes.c_size_t(0)

        # Call the C function
        lib.cuSZp_compress(
            d_oriData,  # GPU pointer for input data
            d_cmpBytes,  # GPU pointer for compressed output (unsigned char*)
            num_elements,
            ctypes.byref(compressed_size),
            ctypes.c_float(error_bound),
            ctypes.c_int(data_type),
            ctypes.c_int(mode),
            None  # Stream (optional, set to None)
        )
        
        return compressed_size.value

    def decompress(self, d_decData, d_cmpBytes, num_elements, compressed_size, error_bound=0.01, data_type=0, mode=0):
        # Call the C function
        lib.cuSZp_decompress(
            d_decData,  # GPU pointer for decompressed output
            d_cmpBytes,  # GPU pointer for compressed input
            num_elements,
            compressed_size,
            ctypes.c_float(error_bound),
            ctypes.c_int(data_type),
            ctypes.c_int(mode),
            None  # Stream (optional, set to None)
        )
