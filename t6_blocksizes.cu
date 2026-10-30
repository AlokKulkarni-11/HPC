// ============================================================================
// EXPERIMENT 9 - TASK 6: Effect of Thread Block Size on GPU Execution Time
// ============================================================================
// THEORY & CONCEPTS:
// 1. Warp Multiples: Block sizes must be multiples of the warp size (32) to 
//    prevent under-utilized warp execution slots[cite: 13].
// 2. Memory-Bandwidth Saturation: Once enough resident warps hide memory latency, 
//    execution time becomes flat across block sizes (128, 256, 512)[cite: 13].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"

__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int reps = argc > 2 ? atoi(argv[2]) : 200;
    size_t bytes = (size_t)N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = (float)(i % 100); h_b[i] = 1.0f; }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int sizes[] = {32, 64, 128, 256, 512, 1024};
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    printf("%-10s %-14s %-10s %-18s %-8s\n", "N", "Threads/Block", "Blocks", "Kernel time (ms)", "Check");

    for (int k = 0; k < 6; k++) {
        int tpb = sizes[k];
        int blocks = (N + tpb - 1) / tpb;

        vecAdd<<<blocks, tpb>>>(d_a, d_b, d_c, N); // Warm-up launch[cite: 13]
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(s);
        for (int r = 0; r < reps; r++) {
            vecAdd<<<blocks, tpb>>>(d_a, d_b, d_c, N);
        }
        cudaEventRecord(e);
        CUDA_CHECK(cudaEventSynchronize(e));

        float ms;
        cudaEventElapsedTime(&ms, s, e);
        ms /= reps;

        CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < N; i++) if (h_c[i] != h_a[i] + 1.0f) bad++;

        printf("%-10d %-14d %-10d %-18.5f %-8s\n", N, tpb, blocks, ms, bad ? "WRONG" : "OK");
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}

// nvcc -O2 -arch=native t6_blocksizes.cu -o t6_blocksizes
// t6_blocksizes 1000000 200