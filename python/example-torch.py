import torch
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA driver
import ctypes
import time
from pycuSZp import cuSZp, Uint3

def main():
    # ========== Configuration ==========
    error_bound = 1E-2
    data_type_flag = 0       # 0 = float32, 1 = float64
    mode_flag = 1            # 1 = plain, 2 = outlier
    dim = 1                  # 1D for this tensor case
    dims = (1, 1, 1)         # placeholder (uint3), unused in 1D
    tensor_size = int(1024**3)  # 4 GB (in float32)
    # ==================================

    # Generate a random tensor on GPU
    data = torch.rand(tensor_size, dtype=torch.float32, device='cuda')

    compressor = cuSZp()

    # Allocate GPU memory for compressed output
    original_size = data.numel() * data.element_size()
    d_cmpBytes = cuda.mem_alloc(original_size)
    d_decData = torch.empty_like(data, device='cuda')

    # ---------- Compression ----------
    print("\n[cuSZp Compression on Torch Tensor]")
    start_c = time.perf_counter()
    compressed_size = compressor.compress(
        ctypes.c_void_p(data.data_ptr()),   # GPU input pointer
        ctypes.c_void_p(int(d_cmpBytes)),   # GPU compressed buffer
        data.numel(),
        error_bound,
        dim=dim,
        dims=dims,                          # uint3(1,1,1) placeholder for 1D
        data_type=data_type_flag,
        mode=mode_flag
    )
    end_c = time.perf_counter()

    print(f"Original size:   {original_size / (1024**2):.2f} MB")
    print(f"Compressed size: {compressed_size / (1024**2):.2f} MB")
    print(f"Compression ratio: {original_size / compressed_size:.2f}x")

    # ---------- Decompression ----------
    start_d = time.perf_counter()
    compressor.decompress(
        ctypes.c_void_p(d_decData.data_ptr()),  # GPU output tensor
        ctypes.c_void_p(int(d_cmpBytes)),       # Compressed bytes on GPU
        data.numel(),
        compressed_size,
        error_bound,
        dim=dim,
        dims=dims,
        data_type=data_type_flag,
        mode=mode_flag
    )
    end_d = time.perf_counter()

    throughput_comp = (data.nbytes / (end_c - start_c)) / (1024 ** 3)
    throughput_decomp = (data.nbytes / (end_d - start_d)) / (1024 ** 3)
    print(f"Compression speed: {throughput_comp:.2f} GB/s")
    print(f"Decompression speed: {throughput_decomp:.2f} GB/s")

    # ---------- Verification ----------
    print(f"Data matches within error bound: {torch.allclose(data, d_decData, atol=error_bound)}")

    # Free compressed GPU memory
    d_cmpBytes.free()


if __name__ == "__main__":
    main()
