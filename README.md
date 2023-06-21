# cuSZp
cuSZp is a lossy error-bounded compression library for compression of floating-point data in GPU.

## Environment Requirements
### Common
- Git 2.15 or newer
- CMake 3.21 or newer
- Cuda Toolkit >= 11.0

## Installation
To build cuSZp type:
```
cd cuSZp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=path/to/your/cuszp/installation/directory ..
make -j
make install
```
