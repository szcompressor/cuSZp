import torch
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA driver
import ctypes
from pycuSZp import cuSZp

def main():
    # Set parameters
    error_bound = 1E-2
    tensor_size = int(4 * 1024**3 / 4)  # 4GB in float32, 4 bytes per float32

    # Generate a 4GB tensor of random float32 values on GPU
    data = torch.rand(tensor_size, dtype=torch.float32, device='cuda')

    # Define data type flag for compression API (0 for float, 1 for double)
    data_type_flag = 0

    compressor = cuSZp()

    # Calculate original data size in bytes
    original_size = data.numel() * data.element_size()

    # Allocate GPU memory for the compressed output
    d_cmpBytes = cuda.mem_alloc(original_size)

    # cuSZp compression
    compressed_size = compressor.compress(
        ctypes.c_void_p(data.data_ptr()),  # Input data pointer on GPU
        ctypes.c_void_p(int(d_cmpBytes)),  # Output buffer on GPU
        data.numel(),                      # Number of elements
        error_bound,
        data_type=data_type_flag,
        mode=0                              # Plain mode, 1 for outlier mode
    )

    # Print compression results
    print(f"Original data size:   {original_size} bytes")
    print(f"Compressed data size: {compressed_size} bytes")
    compression_ratio = original_size / compressed_size
    print(f"Compression Ratio: {compression_ratio:.2f}")

    # Allocate GPU memory for decompressed output
    d_decData = torch.empty_like(data, device='cuda')

    # cuSZp decompression
    compressor.decompress(
        ctypes.c_void_p(d_decData.data_ptr()),  # Output data pointer for decompressed data
        ctypes.c_void_p(int(d_cmpBytes)),       # Compressed data on GPU
        data.numel(),
        compressed_size,
        error_bound,
        data_type=data_type_flag,
        mode=0                                   # Plain mode, 1 for outlier mode
    )

    # Error checking for decompression
    print(f"Decompressed data matches original within error bound: {torch.allclose(data, d_decData, atol=error_bound)}")

    # Free GPU memory for compressed data
    d_cmpBytes.free()

if __name__ == "__main__":
    main()
