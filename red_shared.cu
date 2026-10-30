// ============================================================================
// EXPERIMENT 10 - TASK 4: Parallel Tree Reduction using Shared Memory
// ============================================================================
// THEORY & CONCEPTS:
// 1. Parallel Reduction Tree: Reduces an array of N elements in O(log N) steps[cite: 14].
// 2. Sequential Addressing (tid < s): Active threads are kept contiguous within warps 
//    to eliminate branch divergence (unlike naive modulo-based addressing `tid % 2s`)[cite: 14].
// 3. Multi-Pass Approach: Blocks independently reduce their chunks into an intermediate 
//    array, followed by a second pass to produce the single final scalar[cite: 14].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"

__global__ void reduce_sum_shared(const float *in, float *out, int n) {
    extern __shared__ float sd[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load from global into shared memory[cite: 14]
    sd[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // In-place tree reduction in shared memory[cite: 14]
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sd[tid] += sd[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the block's sum to global memory[cite: 14]
    if (tid == 0) {
        out[blockIdx.x] = sd[0];
    }
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // First pass reduction across blocks[cite: 14]
    reduce_sum_shared<<<blocks, tpb, tpb * sizeof(float)>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy partial block sums back to CPU to finalize[cite: 14]
    float *h_out = (float*)malloc(blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    double final_sum = 0.0;
    for (int i = 0; i < blocks; i++) final_sum += h_out[i];

    printf("Parallel Reduction (Sum) Result = %.1f (Expected %.1f)\n", final_sum, (double)N);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}

// nvcc -O3 -arch=native red_shared.cu -o red_shared
// red_shared 1000000