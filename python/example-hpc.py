import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
import sys
import time
from pycuSZp import cuSZp

def read_dataset(filename, dtype):
    return np.fromfile(filename, dtype=dtype)

def main():
    if len(sys.argv) < 7:
        print("Usage: python example-hpc.py <dataset_file> <data_type: f32|f64> <mode: fixed|plain|outlier> <dim: 1|2|3> <dim_z> <dim_y> <dim_x> <error_bound>")
        print("       if dim=1, dim_z, dim_y, and dim_x can be any value (e.g., 0).")
        print("       dim_x is the fastest changing dimension.")
        print("Example: python example-hpc.py data.f32 f32 plain 3 100 100 100 1e-3")
        print("         python example-hpc.py data.f64 f64 outlier 2 0 1000 1000 1e-4")
        print("         python example-hpc.py data.f32 f32 fixed 1 0 0 0 1e-2")
        sys.exit(1)

    filename = sys.argv[1]
    dtype_str = sys.argv[2]
    mode_str = sys.argv[3]
    dim = int(sys.argv[4])
    dim_z, dim_y, dim_x = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    error_bound = float(sys.argv[8])

    dims = (dim_x, dim_y, dim_z)

    # Data type
    if dtype_str == 'f32':
        data = read_dataset(filename, np.float32)
        data_type = 0
    elif dtype_str == 'f64':
        data = read_dataset(filename, np.float64)
        data_type = 1
    else:
        print("Invalid data type.")
        sys.exit(1)

    # Mode
    if mode_str == 'plain':
        mode = 1
    elif mode_str == 'outlier':
        mode = 2
    elif mode_str == 'fixed':
        mode = 0
    else:
        print("Invalid mode.")
        sys.exit(1)

    compressor = cuSZp()

    # Allocate GPU memory
    d_oriData = cuda.mem_alloc(data.nbytes)
    d_cmpBytes = cuda.mem_alloc(data.nbytes)
    d_decData = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(d_oriData, data)

    # Compression
    start_c = time.perf_counter()
    cmp_size = compressor.compress(
        ctypes.c_void_p(int(d_oriData)),
        ctypes.c_void_p(int(d_cmpBytes)),
        data.size,
        error_bound,
        dim=dim,
        dims=dims,
        data_type=data_type,
        mode=mode
    )
    end_c = time.perf_counter()

    # Decompression
    start_d = time.perf_counter()
    compressor.decompress(
        ctypes.c_void_p(int(d_decData)),
        ctypes.c_void_p(int(d_cmpBytes)),
        data.size,
        cmp_size,
        error_bound,
        dim=dim,
        dims=dims,
        data_type=data_type,
        mode=mode
    )
    end_d = time.perf_counter()

    # Evaluate performance
    comp_time = end_c - start_c
    decomp_time = end_d - start_d
    throughput_comp = (data.nbytes / comp_time) / (1024 ** 3)
    throughput_decomp = (data.nbytes / decomp_time) / (1024 ** 3)

    print(f"\n===== cuSZp GPU Compression Test =====")
    print(f"Data file:         {filename}")
    print(f"Type:              {dtype_str}")
    print(f"Mode:              {mode_str}")
    print(f"Dim:               {dim}D ({dim_z}, {dim_y}, {dim_x})")
    print(f"Error bound:       {error_bound}")
    print(f"Original size:     {data.nbytes / (1024**2):.2f} MB")
    print(f"Compressed size:   {cmp_size / (1024**2):.2f} MB")
    print(f"Compression ratio: {data.nbytes / cmp_size:.2f}x")
    print(f"Compression speed: {throughput_comp:.2f} GB/s")
    print(f"Decompression speed: {throughput_decomp:.2f} GB/s")

    # Optional correctness check
    decompressed = np.empty_like(data)
    cuda.memcpy_dtoh(decompressed, d_decData)
    print(f"Within bound: {np.allclose(data, decompressed, atol=error_bound)}")

    # Free
    d_oriData.free()
    d_cmpBytes.free()
    d_decData.free()

if __name__ == "__main__":
    main()
