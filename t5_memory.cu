// ============================================================================
// EXPERIMENT 9 - TASK 4 & 5: Memory Management and Vector Addition
// ============================================================================
// THEORY & CONCEPTS:
// 1. cudaMalloc & cudaFree: Allocates and frees memory in the GPU's global memory[cite: 13].
// 2. cudaMemcpy: Transfers data over the PCIe bus between Host and Device[cite: 13].
// 3. Boundary Guard (id < n): Essential because total threads (blocks * tpb) 
//    may exceed array size N, preventing memory corruption[cite: 13].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_check.h"

__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    size_t bytes = (size_t)N * sizeof(float);

    // Host memory allocation[cite: 13]
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(i % 100);
        h_b[i] = (float)((i * 7) % 100);
    }

    // Device memory allocation[cite: 13]
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);

    cudaEventRecord(e0);
    // Host to Device memory copy[cite: 13]
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(e1);

    // Launch CUDA Kernel[cite: 13]
    vecAdd<<<blocks, tpb>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(e2);

    // Device to Host memory copy[cite: 13]
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(e3);
    CUDA_CHECK(cudaEventSynchronize(e3));

    float h2d, ker, d2h;
    cudaEventElapsedTime(&h2d, e0, e1);
    cudaEventElapsedTime(&ker, e1, e2);
    cudaEventElapsedTime(&d2h, e2, e3);

    // Result Verification[cite: 13]
    int bad = 0;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) bad++;
    }

    printf("N=%d, %d blocks x %d threads, Mismatches = %d -> %s\n",
           N, blocks, tpb, bad, bad ? "WRONG" : "CORRECT");
    printf("Host->Device copy : %.4f ms\n", h2d);
    printf("Kernel execution  : %.4f ms\n", ker);
    printf("Device->Host copy : %.4f ms\n", d2h);
    printf("Total GPU phase   : %.4f ms\n", h2d + ker + d2h);

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c);
    return 0;
}

// nvcc -O2 -arch=native t5_memory.cu -o t5_memory
// t5_memory 1000000