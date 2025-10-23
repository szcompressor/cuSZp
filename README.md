# cuSZp
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a> 

cuSZp is an ultra-fast and user-friendly GPU error-bounded lossy compressor for floating-point data array (both single- and double-precision). In short, cuSZp has several key features:

1. Fusing entire compression/decompression phase into **one CUDA kernel function**.
2. Efficient latency control and memory access -- targeting **ultra-fast end-to-end throughput**.
3. Two encoding modes (plain or outlier modes) supported, **high compression ratio** for different data patterns. In general, if your dataset is sparse (consisting lots of 0s) -- plain mode will be a good choice; if your dataset exhibits non-sparse and high smoothness -- outlier mode will be a good choice.
4. Executable binary, C/C++ API, Python API are provided.


## Environment Requirements
- Linux OS with NVIDIA GPUs
- Git >= 2.15
- CMake >= 3.21
- Cuda Toolkit >= 11.0
- GCC >= 7.3.0

## Compile and Use cuSZp

You can compile and install cuSZp with following commands.
```shell
$ git clone https://github.com/szcompressor/cuSZp.git
$ cd cuSZp
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ ..
$ make -j
$ make install
```
After installation, you will see executable binary generated in ```cuSZp/install/bin/``` and shared/static library generated in ``cuSZp/install/lib/``.
You can also try ```ctest -V``` in ```cuSZp/build/``` folder (Optional).

### Using cuSZp Prepared Executable Binary
After installation, you will see ```cuSZp``` binary generated in ```cuSZp/install/bin/```. The detailed usage of this binary are explained in the code block below.

```text
Usage:
  ./cuSZp -i [oriFilePath] -t [dataType] -m [encodingMode] -d [dim] [dim_z] [dim_y] [dim_x] -eb [errorBoundMode] [errorBound] -x [cmpFilePath] -o [decFilePath]

Options:
  -i  [oriFilePath]    Path to the input file containing the original data
  -t  [dataType]       Data type of the input data. Options:
                       f32      : Single precision (float)
                       f64      : Double precision (double)
  -m  [encodingMode]   Encoding mode to use. Options:
                       fixed    : No-delta fixed-length encoding mode
                       plain    : Plain fixed-length encoding mode (with delta)
                       outlier  : Outlier fixed-length encoding mode (with delta and outlier preservation)
  -d  [dim]            Data dimensionality and processing manner. Options:
                       1                         : 1D Processing Manner (can be used for all datasets)
                       2 [dim_z] [dim_y] [dim_z] : 2D Processing Manner (can be used for both 2D and 3D datasets)
                       3 [dim_z] [dim_y] [dim_z] : 3D Processing Manner (can be used for only 3D datasets)
                       Note: dim_x is the fastest varying dimension.
                       Hint: 2D processing can be used for 3D datasets by compressing each [dim_y x dim_x] slice independently.
  -eb [errorBoundMode] [errorBound]
                       errorBoundMode can only be:
                       abs      : Absolute error bound
                       rel      : Relative error bound
                       errorBound is a floating-point number representing the error bound, e.g. 1E-4, 0.03.
  -x  [cmpFilePath]    Path to the compressed output file (optional)
  -o  [decFilePath]    Path to the decompressed output file (optional)
```

Some example commands can be found here:
```shell
./cuSZp -i pressure_3000 -t f32 -m plain -d 1 -eb abs 1E-4 -x pressure_3000.cuszp.cmp -o pressure_3000.cuszp.dec
./cuSZp -i pressure_3000 -t f32 -m plain -d 1 -eb rel 1e-4
./cuSZp -i pressure_3000 -t f32 -m plain -d 2 1008 1008 352 -eb rel 1e-4
./cuSZp -i pressure_3000 -t f32 -m plain -d 3 1008 1008 352 -eb rel 1e-4
./cuSZp -i ccd-tst.bin.d64 -t f64 -m outlier -d 1 -eb abs 0.01
./cuSZp -i velocity_x.f32 -t f32 -m outlier -d 1 -eb rel 0.01 -x velocity_x.f32.cuszp.cmp
./cuSZp -i xx.f32 -m outlier -eb rel 1e-4 -t f32 -d 1
```

Some results measured on one NVIDIA A100 GPU can be shown as below:
```shell
$ ./cuSZp -i pressure_2000 -t f32 -m plain -eb rel 1e-3 -d 1
cuSZp finished!
cuSZp compression   end-to-end speed: 410.416846 GB/s
cuSZp decompression end-to-end speed: 627.771706 GB/s
cuSZp compression ratio: 22.328032

Pass error check!

$ ./cuSZp -i pressure_2000 -t f32 -m plain -eb rel 1e-3 -d 3 1008 1008 352
cuSZp finished!
cuSZp compression   end-to-end speed: 418.239408 GB/s
cuSZp decompression end-to-end speed: 879.286882 GB/s
cuSZp compression ratio: 28.165690

Pass error check!

$ ./cuSZp -i xx.f32 -t f32 -m outlier -d 1 -eb rel 1e-4 
cuSZp finished!
cuSZp compression   end-to-end speed: 341.494199 GB/s
cuSZp decompression end-to-end speed: 412.994599 GB/s
cuSZp compression ratio: 6.277573

Pass error check!

$ ./cuSZp -i acd-tst.bin.d64 -t f64 -m outlier -d 1 -eb rel 0.0001 
cuSZp finished!
cuSZp compression   end-to-end speed: 552.982586 GB/s
cuSZp decompression end-to-end speed: 550.156870 GB/s
cuSZp compression ratio: 13.737053

Pass error check!
```

You will also see two binaries generated this folder, named ```cuSZp_test_f32``` and ```cuSZp_test_f64```. They are used for functionality test and can be executed by ```./cuSZp_test_f32``` and ```./cuSZp_test_f64```. (to yafan: will be updated with higher-dim tests)

### Using cuSZp as C/C++ Interal API
If you want to use cuSZp as a C/C++ interal API, there are two ways.

1. Use the cuSZp generic API.
    
    ```C
    // Other headers.
    #include <cuSZp.h>
    
    int main (int argc, char* argv[]) {
        
        // Other code.

        // For measuring the end-to-end throughput.
        TimingGPU timer_GPU;

        // cuSZp compression for device pointer.
        timer_GPU.StartCounter(); // set timer
        cuSZp_compress(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, processingDim, dims, dataType, encodingMode, stream);
        float cmpTime = timer_GPU.GetCounter();

        // cuSZp decompression for device pointer.
        timer_GPU.StartCounter(); // set timer
        cuSZp_decompress(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, processingDim, dims, dataType, encodingMode, stream);
        float decTime = timer_GPU.GetCounter();

        // Other code.

        return 0;
    }
    ```

    Here, ```d_oriData```, ```d_cmpBytes```, and ```d_decData``` are device pointers (array on GPU), representing original data, compressed byte, and reconstructed data, respectively.
    ```processingDim```, ```dataType```, and ```encodingMode``` can be defined as below:
    ```C
    cuszp_dim_t processingDim = CUSZP_DIM_1; // or CUSZP_DIM_2, CUSZP_DIM_3 (or just a number: 1, 2, 3)
    cuszp_type_t dataType = CUSZP_TYPE_FLOAT; // or CUSZP_TYPE_DOUBLE
    cuszp_mode_t encodingMode = CUSZP_MODE_PLAIN; // or CUSZP_MODE_OUTLIER, CUSZP_MODE_FIXED
    ```

    A detailed example can be seen in ```cuSZp/examples/cuSZp.cpp```.

2. Use a specific encoding mode and floating-point data type (f32 or f64).

    ```C
    #include <cuSZp.h> // Still the only header you need.

    // Compression and decompression for float type data array using plain mode and 1D processing manner.
    cuSZp_compress_1D_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, stream);
    cuSZp_decompress_1D_plain_f32(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
    // In this case, d_oriData and d_Decdata are float* array on GPU.
    ```
    
    <details>
    <summary>Other modes and data type usages</summary>
    1D Processing APIs, using outlier mode and single-precision data as an example.
    ```C
    #include <cuSZp.h> // Still the only header you need.

    cuSZp_compress_1D_outlier_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, stream);
    cuSZp_decompress_1D_outlier_f32(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
    ```

    2D Processing APIs, using plain mode and single-precision data as an example.
    ```C
    #include <cuSZp.h> // Still the only header you need.

    cuSZp_compress_2D_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, dims, errorBound, stream);
    cuSZp_decompress_2D_plain_f32(d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
    ```

    3D Processing APIs, using plain mode and double-precision data as an example.
    ```C
    #include <cuSZp.h> // Still the only header you need.

    cuSZp_compress_3D_plain_f64(d_oriData, d_cmpBytes, nbEle, &cmpSize, dims, errorBound, stream);
    cuSZp_decompress_3D_plain_f64(d_decData, d_cmpBytes, nbEle, cmpSize, dims, errorBound, stream);
    ```
    </details>

    
    In this case, you do not need to set ```cuszp_dim_t```, ```cuszp_type_t```, and ```cuszp_mode_t```.
    More detaild examples can be found in ```cuSZp/examples/cuSZp_test_f32.cpp``` and ```cuSZp/examples/cuSZp_test_f64.cpp```. (to yafan: will be updated with higher-dim tests)


### Using cuSZp as Python API
cuSZp also supports Python bindings for fast compression on GPU array.
Examples can be found in ```cuSZp/python/```. 
The required Python packages include ```ctypes, numpy, pycuda```. ```pytorch``` is optional unless you want to use cuSZp compress a ```torch``` tensor.

We provide two examples for showing how cuSZp can be used to compress/decompress a HPC field in numpy format (see ```python/example-hpc.py```) and a torch tensor (see ```python/example-torch.py```).
To execute them:
```shell
# This shows cuSZp compresses a HPC field with f32 format.
python example-hpc.py ./pressure_3000 f32

# This shows cuSZp compresses a 4 GB f32 torch tensor.
python example-torch.py
```

cuSZp Python API also preseves very high throughput. Taking compressing and decompressing 4 GB torch tensor on NVIDIA A100 GPU as an example.
Similarly, we measure throughput in an end-to-end manner (e.g. following code block shows compression measurement).
```python
compressor = cuSZp()
# cuSZp compression.
start_time = time.time()                    # set cuSZp timer start
compressed_size = compressor.compress(
    ctypes.c_void_p(data.data_ptr()),       # Input data pointer on GPU
    ctypes.c_void_p(int(d_cmpBytes)),       # Output buffer on GPU
    data.numel(),                           # Number of elements
    1E-2,                                   # Set 1E-2 as error bound.
    data_type=0,                            # float 32, 1 for float64 (i.e. double)
    mode=0                                  # Plain mode, 1 for outlier mode
)
compression_time = time.time() - start_time # set cuSZp timer end
```

This throughput measurement can be shown as below:
```shell
$ python example-torch.py 
Original data size:   4294967296 bytes
Compressed data size: 971519248 bytes
Compression Ratio: 4.42
Compression Throughput:   214.07 GB/s
Decompression Throughput: 345.08 GB/s
Decompressed data matches original within error bound: True
```



## Documentation
A more detailed documentation with some intrinsic usage descriptions will be updated soon.

## Authors and Citation

cuSZp was developed and contributed by following authors.

- Developers: Yafan Huang (kernels, entries, and examples), Sheng Di (utility), Boyuan Zhang (AaTrox mode).
- Contributors: Franck Cappello, Xiaodong Yu, Robert Underwood, Guanpeng Li.

If you find cuSZp is useful, following papers can be considered for citing.
- **[SC'23]** cuSZp: An Ultra-fast GPU Error-bounded Lossy Compression Framework with Optimized End-to-End Performance
    ```bibtex
    @inproceedings{huang2023cuszp,
        title={cuSZp: An Ultra-fast GPU Error-bounded Lossy Compression Framework with Optimized End-to-End Performance},
        author={Huang, Yafan and Di, Sheng and Yu, Xiaodong and Li, Guanpeng and Cappello, Franck},
        booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
        pages={1--13},
        year={2023}
    }
    ```
- **[SC'24]** cuSZp2: A GPU Lossy Compressor with Extreme Throughput and Optimized Compression Ratio
    ```bibtex
    @inproceedings{huang2024cuszp2,
        title={cuSZp2: A GPU Lossy Compressor with Extreme Throughput and Optimized Compression Ratio},
        author={Huang, Yafan and Di, Sheng and Li, Guanpeng and Cappello, Franck},
        booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
        pages={1--18},
        year={2024}
    }
    ```
- **[SC'25]** GPU Lossy Compression for HPC Can Be Versatile and Ultra-Fast
    ```bibtex
    @inproceedings{huang2025gpu,
        title={GPU Lossy Compression for HPC Can Be Versatile and Ultra-Fast},
        author={Huang, Yafan and Di, Sheng and Li, Guanpeng and Cappello, Franck},
        booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
        pages={1--20},
        year={2025}
    }
    ```
- **[ICS'25]** Pushing the Limits of GPU Lossy Compression: A Hierarchical Delta Approach
    ```bibtex
    @inproceedings{zhang2025pushing,
        title={Pushing the Limits of GPU Lossy Compression: A Hierarchical Delta Approach},
        author={Zhang, Boyuan and Huang, Yafan and Di, Sheng and Song, Fengguang and Li, Guanpeng and Cappello, Franck},
        booktitle={Proceedings of the 39th ACM International Conference on Supercomputing},
        pages={654--669},
        year={2025}
    }
    ```

The **[SC'23]** paper proposes the cuSZp compression framework with kernel fusion (a.k.a. cuSZp1, code see [Release](https://github.com/szcompressor/cuSZp/releases/tag/cuSZp-V1.1) or [Commit](https://github.com/szcompressor/cuSZp/tree/f4df2f1c5e9e529b05d344f2491a3fa7a5c2c0ed)). 
The **[SC'24]** paper includes new lossless encoding modes and several performance optimization (a.k.a. cuSZp2, code see [Release](https://github.com/szcompressor/cuSZp/releases/tag/cuSZp-V2.0) or [Commit](https://github.com/szcompressor/cuSZp/tree/16e164762fe67785f498a44bae7984058a7a6952)). 
The **[SC'25]** paper includes dimensionality support and versatility support. (a.k.a. cuSZp3, code see [Release]() or [Commit]()). 
The **[ICS'25]** paper includes a fast and high-ratio AaTrox mode for 1D processing manner.

## Copyright
(C) 2023 by Argonne National Laboratory and University of Iowa. For more details see [COPYRIGHT](https://github.com/szcompressor/cuSZp/blob/master/LICENSE).

## Acknowledgement
This research was supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem, including software, applications, hardware, advanced system engineering, and early testbed platforms, to support the nation’s exascale computing imperative. The material was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research (ASCR), under contract DE-AC02-06CH11357, and supported by the National Science Foundation under Grant OAC-2003709, OAC-2104023, and OAC-2311875. We acknowledge the computing resources provided on Bebop (operated by Laboratory Computing Resource Center at Argonne) and on Theta and JLSE (operated by Argonne Leadership Computing Facility). We acknowledge the support of ARAMCO.