// ============================================================================
// EXPERIMENT 8 - TASK 1B: OpenMP Matrix Multiplication
// ============================================================================
// THEORY & CONCEPTS:
// 1. Shared Memory Parallelism: Threads access matrices A and B in-place within 
//    the same address space, requiring zero data copying or network transfer[cite: 11].
// 2. Disjoint Output Writes: Partitioning the outer loop `i` gives each thread 
//    exclusive ownership of specific rows in C, avoiding data races[cite: 11].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "common.h"

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 500;
    long NN = (long)N * N;
    double *A = (double*)malloc(NN * sizeof(double));
    double *B = (double*)malloc(NN * sizeof(double));
    double *C0 = (double*)malloc(NN * sizeof(double));
    double *C = (double*)malloc(NN * sizeof(double));

    fill(A, NN, 1); fill(B, NN, 2);

    // Sequential baseline execution[cite: 11]
    double t0 = omp_get_wtime();
    matmul_seq(A, B, C0, N, N);
    double tseq = omp_get_wtime() - t0;
    printf("matmul N=%d sequential: %.4f s\n", N, tseq);

    // Run OpenMP parallel loop across specified thread counts[cite: 11]
    for (int a = 2; a < argc; a++) {
        int th = atoi(argv[a]);
        omp_set_num_threads(th);
        t0 = omp_get_wtime();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) C[(long)i * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                double aik = A[(long)i * N + k];
                for (int j = 0; j < N; j++) C[(long)i * N + j] += aik * B[(long)k * N + j];
            }
        }

        double tp = omp_get_wtime() - t0;
        double err = maxdiff(C, C0, NN);
        printf("matmul N=%d OpenMP threads=%d: %.4f s speedup=%.2f eff=%.2f maxerr=%.1e %s\n",
               N, th, tp, tseq / tp, tseq / tp / th, err, err < 1e-9 ? "OK" : "WRONG");
    }

    free(A); free(B); free(C0); free(C);
    return 0;
}

// gcc -O2 -fopenmp mm_omp.c -o mm_omp -lm
// mm_omp 1000 2 4 8