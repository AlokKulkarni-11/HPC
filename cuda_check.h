// ============================================================================
// CUDA ERROR CHECKING UTILITY (cuda_check.h)
// ============================================================================
#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,   \
                __LINE__, cudaGetErrorString(err_));                       \
        exit(EXIT_FAILURE);                                                \
    } } while (0)

#endif