# cuSZp
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a> 

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

## Compile and Run cuSZp Prepared Executable Binary
You can compile and install cuSZp with following commands:
```shell
$ git clone https://github.com/szcompressor/cuSZp.git
$ cd cuSZp
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ ..
$ make -j
$ make install
```
After compilation, you will see a list of executable binaries ```cuSZp/install/bin/```:
- ```cuSZp_cpu_f32_api```: single-precision, host pointers (i.e. on CPU).
- ```cuSZp_gpu_f32_api```: single-precision, device pointers (i.e. on GPU).
- ```cuSZp_cpu_f64_api```: double-precision, host pointers (i.e. on CPU).
- ```cuSZp_gpu_f64_api```: double-precision, device pointers (i.e. on GPU).

To use those binaries, try following commands. 
We here use RTM pressure_2000 dataset (1.4 GB, 1008x1008x352) for single-precision example, and NWChem acd-tst.bin.d64 (6.0 GB) for double-precision example.
```shell
# Example for single-precision API
# ./cuSZp_gpu_f32_api TARGET_HPC_DATASET ERROR_MODE ERROR_BOUND
#                                        ABS or REL
$ ./cuSZp_gpu_f32_api ./pressure_2000 REL 1e-4
cuSZp finished!
cuSZp compression   end-to-end speed: 151.564649 GB/s
cuSZp decompression end-to-end speed: 232.503219 GB/s
cuSZp compression ratio: 13.003452

Pass error check!
$
# Example for double-precision API
# ./cuSZp_gpu_f64_api TARGET_HPC_DATASET ERROR_MODE ERROR_BOUND
#                                        ABS or REL
$ ./cuSZp_gpu_f64_api ./acd-tst.bin.d64 ABS 1E-8
cuSZp finished!
cuSZp compression   end-to-end speed: 110.117965 GB/s
cuSZp decompression end-to-end speed: 222.743097 GB/s
cuSZp compression ratio: 3.990585

Pass error check!
```
More HPC dataset can be downloaded from [SDRBench](https://sdrbench.github.io/).

## Using cuSZp as an Internal API
This repository provides several examples for using cuSZp compression and decompression for different scenarios (device pointer? host pointer? f32 or f64?).
The examples can be found in ```cuSZp/examples/```.
Assuming your original data, compressed data, and reconstructed data are all device pointers (allocated on GPU), and the data type is single-precision. The compression and decompression APIs can be called as below:
```C++
// For measuring the end-to-end throughput.
TimingGPU timer_GPU;

// cuSZp compression.
timer_GPU.StartCounter(); // set timer
SZp_compress_deviceptr_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, stream);
float cmpTime = timer_GPU.GetCounter();

// cuSZp decompression.
timer_GPU.StartCounter(); // set timer
SZp_decompress_deviceptr_f32(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
float decTime = timer_GPU.GetCounter();
```
More details can be checked in:
- **f32-hostptr**: ```cuSZp/examples/cuSZp_cpu_f32_api.cpp```.
- **f32-deviceptr**: ```cuSZp/examples/cuSZp_gpu_f32_api.cpp```.
- **f64-hostptr**: ```cuSZp/examples/cuSZp_cpu_f64_api.cpp```.
- **f64-deviceptr**: ```cuSZp/examples/cuSZp_gpu_f64_api.cpp```.

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

## Copyright
(C) 2023 by Argonne National Laboratory and University of Iowa. More details see [COPYRIGHT](https://github.com/szcompressor/cuSZp/blob/master/LICENSE).

## Acknowledgement
This research was supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem, including software, applications, hardware, advanced system engineering and early testbed platforms, to support the nation’s exascale computing imperative. The material was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research (ASCR), under contract DE-AC02-06CH11357, and supported by the National Science Foundation under Grant OAC-2003709 and OAC-2104023. We acknowledge the computing resources provided on Bebop (operated by Laboratory Computing Resource Center at Argonne) and on Theta and JLSE (operated by Argonne Leadership Computing Facility). We acknowledge the support of ARAMCO. 
