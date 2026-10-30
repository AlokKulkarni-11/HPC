// ============================================================================
// EXPERIMENT 9 - TASK 1: Explore GPU Environment & Properties
// ============================================================================
// THEORY & CONCEPTS:
// 1. Streaming Multiprocessor (SM): The core computing engine of NVIDIA GPUs[cite: 13]. 
//    Contains CUDA cores, warp schedulers, registers, and shared memory/L1 cache[cite: 13].
// 2. Hardware Limits: Queries max threads per block (typically 1024), warp size (32), 
//    and grid dimensions to establish valid kernel launch bounds[cite: 13].
// ============================================================================

#include <stdio.h>
#include "cuda_check.h"

int main() {
    int n = 0, drv = 0, rt = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));

    printf("CUDA devices found : %d\n", n);
    printf("CUDA driver version: %d.%d\n", drv / 1000, (drv % 100) / 10);
    printf("CUDA runtime version: %d.%d\n", rt / 1000, (rt % 100) / 10);

    for (int d = 0; d < n; d++) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, d));
        printf("\n--- Device %d ---\n", d);
        printf("GPU name                 : %s\n", p.name);
        printf("Global memory            : %.2f GB\n", p.totalGlobalMem / 1073741824.0);
        printf("Compute capability       : %d.%d\n", p.major, p.minor);
        printf("Number of SMs            : %d\n", p.multiProcessorCount);
        printf("Max threads per block    : %d\n", p.maxThreadsPerBlock);
        printf("Max block dimensions     : %d x %d x %d\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf("Max grid dimensions      : %d x %d x %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
        printf("Warp size                : %d\n", p.warpSize);
        printf("Shared memory per block  : %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("Registers per block      : %d\n", p.regsPerBlock);
    }
    return 0;
}


// nvcc -O2 -arch=native gpu_info.cu -o gpu_info
// gpu_info