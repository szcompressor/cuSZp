import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA driver
import ctypes
import sys
from pycuSZp import cuSZp

def read_dataset(filename, dtype):
    return np.fromfile(filename, dtype=dtype)

def main():
    if len(sys.argv) < 3:
        print("Usage: python example-hpc.py <dataset_file> <data_type (f32 or f64)>")
        sys.exit(1)

    filename = sys.argv[1]
    data_type = sys.argv[2]
    error_bound = 1E-2


    if data_type == 'f32':
        data = read_dataset(filename, np.float32)
        dtype = 0  # float
    elif data_type == 'f64':
        data = read_dataset(filename, np.float64)
        dtype = 1  # double
    else:
        print("Invalid data type. Use 'f32' or 'f64'.")
        sys.exit(1)


    compressor = cuSZp()

    # Calculate original data size
    original_size = data.nbytes  # in bytes

    # Allocate GPU memory and move input data
    d_oriData = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(d_oriData, data)

    # Allocate GPU memory for the compressed output
    d_cmpBytes = cuda.mem_alloc(data.nbytes)  # Output buffer for compressed data

    # cuSZp compression
    compressed_size = compressor.compress(
        ctypes.c_void_p(int(d_oriData)),
        ctypes.c_void_p(int(d_cmpBytes)),
        data.size,
        error_bound,
        data_type=dtype,
        mode=0                                # Plain mode, 1 for outlier mode     
    )

    # Print compression results
    print(f"Original data size:   {original_size} bytes")
    print(f"Compressed data size: {compressed_size} bytes")
    compression_ratio = original_size / compressed_size
    print(f"Compression Ratio: {compression_ratio:.2f}")

    # Allocate GPU memory for decompressed output
    d_decData = cuda.mem_alloc(data.nbytes)

    # cuSZp decompression
    compressor.decompress(
        ctypes.c_void_p(int(d_decData)),
        ctypes.c_void_p(int(d_cmpBytes)),
        data.size,
        compressed_size,
        error_bound,
        data_type=dtype,
        mode=0                              # Plain mode, 1 for outlier mode
    )

    # Copy decompressed data back to CPU
    decompressed = np.empty_like(data)
    cuda.memcpy_dtoh(decompressed, d_decData)

    # Error checking for decompression
    print(f"Decompressed data matches original within error bound: {np.allclose(data, decompressed, atol=error_bound)}")

    # Free GPU memory
    d_oriData.free()
    d_cmpBytes.free()
    d_decData.free()

if __name__ == "__main__":
    main()
