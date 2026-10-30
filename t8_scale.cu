// ============================================================================
// EXPERIMENT 9 - TASK 8: Vector Scaling y[i] = alpha * x[i]
// ============================================================================
// THEORY & CONCEPTS:
// 1. Kernel vs. End-to-End Speedup: Low arithmetic intensity operations (1 multiply 
//    per 8 bytes moved) achieve high speedup during kernel execution, but PCIe 
//    host-to-device transfers dominate total wall-clock time[cite: 13].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "cuda_check.h"

__global__ void scale(const float *x, float *y, float alpha, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) y[id] = alpha * x[id];
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 10000000;
    float alpha = argc > 2 ? (float)atof(argv[2]) : 2.5f;
    size_t bytes = (size_t)N * sizeof(float);

    float *x = (float*)malloc(bytes);
    float *y_cpu = (float*)malloc(bytes);
    float *y_gpu = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) x[i] = (float)(i % 1000);

    // CPU execution baseline[cite: 13]
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) y_cpu[i] = alpha * x[i];
    double t_cpu = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

    // GPU allocation and execution[cite: 13]
    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));

    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);

    cudaEventRecord(e0);
    CUDA_CHECK(cudaMemcpy(dx, x, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(e1);

    scale<<<blocks, tpb>>>(dx, dy, alpha, N);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(e2);

    CUDA_CHECK(cudaMemcpy(y_gpu, dy, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(e3);
    CUDA_CHECK(cudaEventSynchronize(e3));

    float h2d, ker, d2h;
    cudaEventElapsedTime(&h2d, e0, e1);
    cudaEventElapsedTime(&ker, e1, e2);
    cudaEventElapsedTime(&d2h, e2, e3);

    double maxd = 0;
    for (int i = 0; i < N; i++) {
        maxd = fmax(maxd, fabs(y_cpu[i] - y_gpu[i]));
    }

    printf("N=%d, alpha=%.2f, %d blocks x %d threads\n", N, alpha, blocks, tpb);
    printf("Max element difference: %.1e -> %s\n", maxd, maxd == 0 ? "IDENTICAL" : "DIFFERENT");
    printf("CPU time: %.4f ms | GPU Kernel: %.4f ms | Total GPU: %.4f ms\n", t_cpu, ker, h2d + ker + d2h);
    printf("Speedup (kernel only): %.2f | Speedup (total incl transfers): %.2f\n", t_cpu / ker, t_cpu / (h2d + ker + d2h));

    cudaFree(dx); cudaFree(dy);
    free(x); free(y_cpu); free(y_gpu);
    return 0;
}

// nvcc -O2 -arch=native t8_scale.cu -o t8_scale
// t8_scale 10000000 2.5