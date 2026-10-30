// ============================================================================
// EXPERIMENT 9 - TASK 2: First CUDA Program (Hello from GPU)
// ============================================================================
// THEORY & CONCEPTS:
// 1. __global__ Keyword: Marks a kernel function that executes on the device 
//    (GPU) and is callable from the host (CPU)[cite: 13].
// 2. threadIdx.x: Built-in 3D vector variable holding the thread's local index 
//    within its block[cite: 13].
// 3. Asynchronous Launches: Kernels run asynchronously to CPU code; 
//    `cudaDeviceSynchronize()` forces the host to wait for device completion[cite: 13].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"

__global__ void hello() {
    int tid = threadIdx.x; // Thread index within current block[cite: 13]
    printf("Hello from GPU Thread %d\n", tid);
}

int main(int argc, char **argv) {
    int blocks  = argc > 1 ? atoi(argv[1]) : 1;
    int threads = argc > 2 ? atoi(argv[2]) : 8;

    printf("Launching %d block(s) x %d thread(s) = %d threads\n", blocks, threads, blocks * threads);
    
    // Launch kernel on GPU[cite: 13]
    hello<<<blocks, threads>>>();
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel printf buffer to flush[cite: 13]

    return 0;
}

// nvcc -O2 -arch=native t2_hello.cu -o t2_hello
// t2_hello 2 8