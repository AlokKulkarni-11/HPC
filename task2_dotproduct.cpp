// ============================================================================
// EXPERIMENT 3 - SECTION A TASK 2: Dot Product (Sequential vs Reduction)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Dot Product: Combines element-wise multiplication with a reduction sum 
//    ($\sum A[i] \cdot B[i]$). Foundational in neural network forward/backward passes, 
//    fully-connected layers, and attention mechanisms.
// 2. Parallelization: Since each product term $A[i] \cdot B[i]$ is independent, 
//    OpenMP's reduction clause allows threads to compute partial dot products 
//    privately and sum them safely at the end.
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    long N = (argc > 1) ? atol(argv[1]) : 1000000L;
    int nthreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    
    double *A = (double*)malloc(sizeof(double) * N);
    double *B = (double*)malloc(sizeof(double) * N);
    for (long i = 0; i < N; i++) {
        A[i] = 1.0 + (i % 5);
        B[i] = 2.0 - (i % 3);
    }
    omp_set_num_threads(nthreads);

    // Sequential Dot Product
    double t0 = omp_get_wtime();
    double dot_seq = 0.0;
    for (long i = 0; i < N; i++) {
        dot_seq += A[i] * B[i];
    }
    double t1 = omp_get_wtime();

    // Parallel Dot Product using Reduction
    double dot_par = 0.0;
    double t2 = omp_get_wtime();
    #pragma omp parallel for reduction(+:dot_par)
    for (long i = 0; i < N; i++) {
        dot_par += A[i] * B[i];
    }
    double t3 = omp_get_wtime();

    printf("N=%ld | Threads=%d | Seq Dot=%.1f | Par Dot=%.1f | Match: %s\n",
           N, nthreads, dot_seq, dot_par, (dot_seq == dot_par) ? "yes" : "no");
    printf("Seq Time: %.6fs | Par Time: %.6fs | Speedup: %.3f\n",
           t1 - t0, t3 - t2, (t3 - t2 > 0) ? (t1 - t0) / (t3 - t2) : 0.0);

    free(A); free(B);
    return 0;
}