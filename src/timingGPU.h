#ifndef CUSZP_SRC_TIMINGGPU_H
#define CUSZP_SRC_TIMINGGPU_H

#include <cuda.h>
#include <cuda_runtime.h>

struct PrivateTimingGPU {
    cudaEvent_t start;
    cudaEvent_t stop;
};

class TimingGPU
{
    private:
        PrivateTimingGPU *privateTimingGPU;

    public:

        TimingGPU();

        ~TimingGPU();

        void StartCounter();

        void StartCounterFlags();

        float GetCounter();

};

#endif // CUSZP_SRC_TIMINGGPU_H