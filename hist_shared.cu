// ============================================================================
// EXPERIMENT 10 - TASK 6: Shared-Memory Privatized Histogram
// ============================================================================
// THEORY & CONCEPTS:
// 1. Atomic Contention Bottleneck: Naive histograms issue global atomicAdd calls, 
//    serializing threads that hit identical bins[cite: 14].
// 2. Shared-Memory Privatization: Each block maintains a private histogram in fast 
//    shared memory[cite: 14]. Once the block completes, it performs a single merge step 
//    into the global histogram, slashing global atomic operations by orders of magnitude[cite: 14].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"

__global__ void hist_shared_kernel(const int *data, unsigned int *hist, int n, int bins) {
    extern __shared__ unsigned int sh[];

    // Initialize shared histogram to zero[cite: 14]
    for (int b = threadIdx.x; b < bins; b += blockDim.x) {
        sh[b] = 0;
    }
    __syncthreads();

    // Accumulate locally using fast shared atomics[cite: 14]
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        atomicAdd(&sh[data[i]], 1u);
    }
    __syncthreads();

    // Merge block-local histogram into global histogram[cite: 14]
    for (int b = threadIdx.x; b < bins; b += blockDim.x) {
        if (sh[b] > 0) {
            atomicAdd(&hist[b], sh[b]);
        }
    }
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int bins = 256;

    int *h_data = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_data[i] = rand() % bins;

    int *d_data;
    unsigned int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist, bins * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, bins * sizeof(unsigned int)));

    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    if (blocks > 1024) blocks = 1024;

    hist_shared_kernel<<<blocks, tpb, bins * sizeof(unsigned int)>>>(d_data, d_hist, N, bins);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned int *h_hist = (unsigned int*)malloc(bins * sizeof(unsigned int));
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, bins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    unsigned int total = 0;
    for (int b = 0; b < bins; b++) total += h_hist[b];

    printf("Privatized Histogram Completed! Total counts aggregated = %u (Expected %d)\n", total, N);

    cudaFree(d_data); cudaFree(d_hist);
    free(h_data); free(h_hist);
    return 0;
}

// nvcc -O3 -arch=native hist_shared.cu -o hist_shared
// hist_shared 1000000