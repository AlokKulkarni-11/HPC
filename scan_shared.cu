// ============================================================================
// EXPERIMENT 10 - TASK 5: Hillis-Steele Parallel Prefix Sum (Scan)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Loop-Carried Dependency: Unlike reduction, prefix sum produces N running totals[cite: 14].
// 2. Double-Buffering: To prevent race conditions where threads read elements being 
//    overwritten in the same step, two alternating shared memory buffers are used[cite: 14].
// 3. Step Efficiency: Completes inclusive scan in ceil(log2(B)) steps with a barrier 
//    after each offset doubling[cite: 14].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"

__global__ void block_scan_kernel(const int *in, int *out, int n) {
    extern __shared__ int sm[];
    int *b0 = sm;
    int *b1 = sm + blockDim.x;

    int tid = threadIdx.x;
    int pin = 0;

    b0[tid] = (tid < n) ? in[tid] : 0;
    __syncthreads();

    // Hillis-Steele scan: offset doubles every round[cite: 14]
    for (int off = 1; off < blockDim.x; off <<= 1) {
        int *src = pin ? b1 : b0;
        int *dst = pin ? b0 : b1;

        dst[tid] = (tid >= off) ? src[tid] + src[tid - off] : src[tid];
        pin ^= 1;
        __syncthreads(); // Barrier after each offset step[cite: 14]
    }

    if (tid < n) {
        out[tid] = (pin ? b1 : b0)[tid];
    }
}

int main() {
    const int N = 256;
    size_t bytes = N * sizeof(int);

    int *h_in = (int*)malloc(bytes);
    int *h_out = (int*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1;

    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Double-buffered shared memory allocated dynamically (2 * N * sizeof(int))[cite: 14]
    block_scan_kernel<<<1, N, 2 * N * sizeof(int)>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    printf("Prefix Sum Scan Completed:\n");
    printf("out[0]=%d, out[10]=%d, out[255]=%d (Expected 1, 11, 256)\n", 
           h_out[0], h_out[10], h_out[255]);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}

// nvcc -O3 -arch=native scan_shared.cu -o scan_shared
// scan_shared