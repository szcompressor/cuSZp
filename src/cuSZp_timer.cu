#include "cuSZp_timer.h"

/** ************************************************************************
 * @brief           CUDA event timer for measuring GPU kernel execution.
 * *********************************************************************** */
TimingGPU::TimingGPU() { privateTimingGPU = new PrivateTimingGPU;  }
TimingGPU::~TimingGPU() { }

/** ************************************************************************
 * @brief           Start timer.
 * *********************************************************************** */
void TimingGPU::StartCounter()
{
    cudaEventCreate(&((*privateTimingGPU).start));
    cudaEventCreate(&((*privateTimingGPU).stop));
    cudaEventRecord((*privateTimingGPU).start,0);
}

/** ************************************************************************
 * @brief           Start timer with flags.
 * *********************************************************************** */
void TimingGPU::StartCounterFlags()
{
    int eventflags = cudaEventBlockingSync;

    cudaEventCreateWithFlags(&((*privateTimingGPU).start),eventflags);
    cudaEventCreateWithFlags(&((*privateTimingGPU).stop),eventflags);
    cudaEventRecord((*privateTimingGPU).start,0);
}

/** ************************************************************************
 * @brief           End timer, get count in ms.
 * *********************************************************************** */
float TimingGPU::GetCounter()
{
    float time;
    cudaEventRecord((*privateTimingGPU).stop, 0);
    cudaEventSynchronize((*privateTimingGPU).stop);
    cudaEventElapsedTime(&time,(*privateTimingGPU).start,(*privateTimingGPU).stop);
    return time;
}
