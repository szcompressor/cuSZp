#ifndef CUSZP_INCLUDE_CUSZP_TIMER_H
#define CUSZP_INCLUDE_CUSZP_TIMER_H

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

#endif // CUSZP_INCLUDE_CUSZP_TIMER_H