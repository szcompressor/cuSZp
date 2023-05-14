# cuSZp
cuSZp is a lossy error-bounded compression library for compression of floating-point data.

## Environment Requirements
### Common
- Git 2.15 or newer
- CMake 3.21 or newer
- Cuda Toolkit >= 11.0

## Installation
To build cuSZp type:
```
cd cuSZp
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
make install   
```
This installs the cuSZp library in the build/install/lib and the cuSZp 
command-line executable in the build/install/bin directory.
The CMAKE_INSTALL_PREFIX may be defined to set output installation path.