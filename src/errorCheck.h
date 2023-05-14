#ifndef CUSZP_SRC_ERRORCHECK_H
#define CUSZP_SRC_ERRORCHECK_H

#include <string>
#include <stdexcept>

#define CUCHK(call) {                                                       \
    cudaError_t error = call;                                               \
    if (cudaSuccess != error) {                                             \
        std::string msg = "Cuda error in file " + std::string(__FILE__) +   \
                          " in line " + std::to_string(__LINE__) + " : " +  \
                          std::string(cudaGetErrorString(error)) + "\n";    \
        throw std::runtime_error(msg);                                      \
    }}

#endif // CUSZP_SRC_ERRORCHECK_H
