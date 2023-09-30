# cuSZp
cuSZp is a user-friendly error-bounded lossy compression tool specifically designed for the compression of single- and double-precision floating-point data using NVIDIA GPUs. 
This tool fuses all compression or decompression computations into one single kernel, achieving ultra fast end-to-end throughput.
Specifically, the cuSZp framework is structured around four pivotal stages: Quantization and Prediction, Fixed-length Encoding, Global Synchronization, and Block Bit-shuffling. 
Noting that ongoing optimization efforts are being devoted to cuSZp, aimed at further improving its end-to-end performance.

- Developer: Yafan Huang
- Contributors: Sheng Di, Xiaodong Yu, Guanpeng Li, and Franck Cappello

## Environment Requirements
- Linux OS with NVIDIA GPUs
- Git >= 2.15
- CMake >= 3.21
- Cuda Toolkit >= 11.0
- GCC >= 7.3.0

## Compile and Run cuSZp Binary
You can compile and install cuSZp with following commands:
```shell
$ git clone https://github.com/szcompressor/cuSZp.git
$ cd cuSZp
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ ..
$ make -j
$ make install
```
# After compilation, you will see a executable binary ```cuSZpExample_gpu_api``` in path ```cuSZp/install/bin/```.
To use this binary, try following commands. We here use RTM pressure_1000 dataset (1.43 GB, 1008x1008x352) as an example.
```shell
# ./cuSZpExample_gpu_api TARGET_HPC_DATASET REL_ERROR_BOUND
$ ./cuSZpExample_gpu_api ./pressure_1000 1e-4
cuSZp finished!
cuSZp compression   end-to-end speed: 173.440116 GB/s
cuSZp decompression end-to-end speed: 267.704834 GB/s
cuSZp compression ratio: 65.041033

Pass error check!
```
More HPC dataset can be downloaded from [SDRBench](https://sdrbench.github.io/).

## Using cuSZp as an Internal API
cuSZp provides an example for using cuSZp compression and decompression, which can be found at ```cuSZp/examples/example_gpu_api.cpp```.
Assuming your original data, compressed data, and reconstructed data are all device pointers (allocated on GPU). The compression and decompression APIs can be called as below:
```C++
// cuSZp compression.
timer_GPU.StartCounter(); // set timer
SZp_compress_deviceptr(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, stream);
float cmpTime = timer_GPU.GetCounter();

// cuSZp decompression.
timer_GPU.StartCounter(); // set timer
SZp_decompress_deviceptr(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
float decTime = timer_GPU.GetCounter();
```
If your original data, compressed data, and reconstructed data are all host pointers (allocated on CPU). You can check ```cuSZp/examples/example_cpu_api.cpp``` for more details.

## Citation
```bibtex
@inproceedings{cuSZp2023huang,
      title = {cuSZp: An Ultra-Fast GPU Error-Bounded Lossy Compression Framework with Optimized End-to-End Performance}
     author = {Huang, Yafan and Di, Sheng and Yu, Xiaodong and Li, Guanpeng and Cappello, Franck},
       year = {2023},
       isbn = {979-8-4007-0109-2/23/11},
  publisher = {Association for Computing Machinery},
    address = {Denver, CO, USA},
        doi = {10.1145/3581784.3607048},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
   keywords = {Lossy compression; parallel computing; HPC; GPU},
     series = {SC'23}
}
```

## Acknowledgement
This research was supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem, including software, applications, hardware, advanced system engineering and early testbed platforms, to support the nation’s exascale computing imperative. The material was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research (ASCR), under contract DE-AC02-06CH11357, and supported by the National Science Foundation under Grant OAC-2003709 and OAC-2104023. We acknowledge the computing resources provided on Bebop (operated by Laboratory Computing Resource Center at Argonne) and on Theta and JLSE (operated by Argonne Leadership Computing Facility). We acknowledge the support of ARAMCO. 
