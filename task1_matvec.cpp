// ============================================================================
// EXPERIMENT 4 - TASK 1: Matrix-Vector Multiplication (Input Decomposition)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Input Data Decomposition: The input matrix rows are partitioned contiguously 
//    across threads using `#pragma omp parallel for schedule(static)`.
// 2. Zero Synchronization: Because each thread computes a distinct, disjoint 
//    subset of output vector elements $y[i]$, no thread ever writes to the same 
//    memory location. Thus, no locks or critical sections are required.
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 2000;
    int nthreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();

    double *A = (double*)malloc(sizeof(double) * N * N);
    double *x = (double*)malloc(sizeof(double) * N);
    double *y_seq = (double*)malloc(sizeof(double) * N);
    double *y_par = (double*)malloc(sizeof(double) * N);

    for (int i = 0; i < N * N; i++) A[i] = (i % 9) + 1;
    for (int i = 0; i < N; i++) x[i] = 1.0;
    omp_set_num_threads(nthreads);

    // Sequential Matrix-Vector Multiplication
    double t0 = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        double s = 0;
        for (int j = 0; j < N; j++) s += A[i * N + j] * x[j];
        y_seq[i] = s;
    }
    double t1 = omp_get_wtime();

    // Parallel Matrix-Vector Multiplication (Row-wise Input Decomposition)
    double t2 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        double s = 0;
        for (int j = 0; j < N; j++) s += A[i * N + j] * x[j];
        y_par[i] = s;
    }
    double t3 = omp_get_wtime();

    // Correctness Check
    int match = 1;
    for (int i = 0; i < N; i++) {
        if (y_seq[i] != y_par[i]) { match = 0; break; }
    }

    printf("N=%d | Threads=%d | Seq Time=%.6fs | Par Time=%.6fs | Match: %s\n",
           N, nthreads, t1 - t0, t3 - t2, match ? "yes" : "no");

    free(A); free(x); free(y_seq); free(y_par);
    return 0;
}