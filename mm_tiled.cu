// ============================================================================
// EXPERIMENT 10 - TASK 1 & 2: Shared-Memory Tiled Matrix Multiplication
// ============================================================================
// THEORY & CONCEPTS:
// 1. Tiling & Shared Memory: On-chip SRAM (shared memory) is used to cache sub-blocks 
//    (tiles) of matrices A and B[cite: 14, 15].
// 2. Data Reuse: Each element loaded from slow off-chip global memory is reused 
//    T times by threads in the block, cutting global memory traffic by ~T[cite: 14, 15].
// 3. Dual Barriers (__syncthreads): The first ensures the tile is fully loaded 
//    before computation begins; the second ensures all threads finish computing 
//    before the next tile overwrites the shared memory buffer[cite: 14, 15].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_check.h"

#define TILE_DIM 16

__global__ void mm_tiled_kernel(const float *A, const float *B, float *C, int n) {
    __shared__ float tA[TILE_DIM][TILE_DIM];
    __shared__ float tB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;
    float acc = 0.0f;

    int numTiles = (n + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_DIM + tx;
        int bRow = t * TILE_DIM + ty;

        // Coalesced loads into shared memory with boundary checks[cite: 14, 15]
        tA[ty][tx] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        tB[ty][tx] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;

        __syncthreads(); // Wait until whole tile is loaded[cite: 14, 15]

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += tA[ty][k] * tB[k][tx];
        }

        __syncthreads(); // Wait until computation finishes before next load[cite: 14, 15]
    }

    if (row < n && col < n) {
        C[row * n + col] = acc;
    }
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 512;
    size_t bytes = (size_t)n * n * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < n * n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 blk(TILE_DIM, TILE_DIM);
    dim3 grd((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);

    mm_tiled_kernel<<<grd, blk>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    printf("Tiled Matrix Multiplication (%dx%d, Tile %d) Completed Successfully!\n", n, n, TILE_DIM);
    printf("Sample Check: C[0] = %.2f (Expected %.2f)\n", h_C[0], (float)n * 2.0f);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}

// nvcc -O3 -arch=native mm_tiled.cu -o mm_tiled
// mm_tiled 512