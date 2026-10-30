// ============================================================================
// EXPERIMENT 11 - Matrix Multiplication: Basic vs Tiled vs cuBLAS SGEMM
// ============================================================================
// THEORY & CONCEPTS:
// 1. Vendor-Optimized Libraries: cuBLAS delivers near-peak throughput via register 
//    blocking, vectorized loads, pipeline scheduling, and Tensor Cores[cite: 15].
// 2. The Column-Major Trap: cuBLAS uses Fortran column-major order[cite: 15]. For C/C++ 
//    row-major matrices, computing C = A * B is achieved by passing swapped operands 
//    to compute C^T = B^T * A^T without any manual transpose overhead[cite: 15].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_check.h"

#define T 16

// 1. Basic Kernel (Global Memory Only)[cite: 15]
__global__ void mm_basic(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float s = 0.0f;
        for (int k = 0; k < n; ++k) s += A[row * n + k] * B[k * n + col];
        C[row * n + col] = s;
    }
}

// 2. Tiled Kernel (Shared Memory Reuse)[cite: 15]
__global__ void mm_tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float tA[T][T];
    __shared__ float tB[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.0f;

    for (int t = 0; t < (n + T - 1) / T; ++t) {
        int aCol = t * T + tx, bRow = t * T + ty;
        tA[ty][tx] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        tB[ty][tx] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < T; ++k) acc += tA[ty][k] * tB[k][tx];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = acc;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 512;
    size_t bytes = (size_t)n * n * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < n * n; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 blk(T, T);
    dim3 grd((n + T - 1) / T, (n + T - 1) / T);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Benchmark 1: Basic Global Kernel[cite: 15]
    cudaEventRecord(start);
    mm_basic<<<grd, blk>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float t_basic; cudaEventElapsedTime(&t_basic, start, stop);

    // Benchmark 2: Tiled Shared-Memory Kernel[cite: 15]
    cudaEventRecord(start);
    mm_tiled<<<grd, blk>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float t_tiled; cudaEventElapsedTime(&t_tiled, start, stop);

    // Benchmark 3: cuBLAS SGEMM[cite: 15]
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;

    // Passing (d_B, d_A) with leading dimensions of n addresses row/column-major layout[cite: 15]
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B, n, d_A, n, &beta, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float t_cublas; cudaEventElapsedTime(&t_cublas, start, stop);

    printf("=== Experiment 11 Performance Comparison (%dx%d) ===\n", n, n);
    printf("1. Basic Kernel Time  : %.4f ms (GFLOPS: %.2f)\n", t_basic, (2.0 * n * n * n) / (t_basic * 1e6));
    printf("2. Tiled Kernel Time  : %.4f ms (GFLOPS: %.2f)\n", t_tiled, (2.0 * n * n * n) / (t_tiled * 1e6));
    printf("3. cuBLAS SGEMM Time  : %.4f ms (GFLOPS: %.2f)\n", t_cublas, (2.0 * n * n * n) / (t_cublas * 1e6));

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}

// nvcc -O3 -arch=native exp11_bench.cu -o exp11_bench -lcublas
// exp11_bench 1024