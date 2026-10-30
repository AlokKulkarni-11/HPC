// ============================================================================
// EXPERIMENT 9 - TASK 3: Block, Thread, and Global ID Calculation
// ============================================================================
// THEORY & CONCEPTS:
// 1. 1D Global Index Calculation: Each thread in a grid computes its unique 
//    global thread ID using: tid = blockIdx.x * blockDim.x + threadIdx.x[cite: 13].
// 2. Linear Work Mapping: Allows threads across different blocks to identify 
//    which element in a linear 1D array to process concurrently[cite: 13].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"

__global__ void ids() {
    int global = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block %d, Thread %d -> Global ID %d\n", blockIdx.x, threadIdx.x, global);
}

int main(int argc, char **argv) {
    int blocks  = argc > 1 ? atoi(argv[1]) : 2;
    int threads = argc > 2 ? atoi(argv[2]) : 4;

    ids<<<blocks, threads>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

// nvcc -O2 -arch=native t3_ids.cu -o t3_ids
// t3_ids 2 4