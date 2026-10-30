// ============================================================================
// EXPERIMENT 4 - TASK 2: Matrix-Matrix Multiplication (Output Decomposition)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Output Data Decomposition: Work is partitioned based on the output matrix C. 
//    Each thread owns a distinct block of output rows.
// 2. Conflict Freedom: Since thread blocks write to disjoint memory locations in C, 
//    race conditions are eliminated without synchronization overhead.
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int N;
double *A, *B, *C;

void seq_mm() {
    memset(C, 0, sizeof(double) * N * N);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
    }
}

void par_mm() {
    memset(C, 0, sizeof(double) * N * N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
    }
}

int main(int argc, char *argv[]) {
    N = (argc > 1) ? atoi(argv[1]) : 300;
    int nthreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(nthreads);

    A = (double*)malloc(sizeof(double) * N * N);
    B = (double*)malloc(sizeof(double) * N * N);
    C = (double*)malloc(sizeof(double) * N * N);
    for (int i = 0; i < N * N; i++) { A[i] = (i % 7) + 1; B[i] = (i % 5) + 1; }

    double t0 = omp_get_wtime();
    seq_mm();
    double t1 = omp_get_wtime();

    double t2 = omp_get_wtime();
    par_mm();
    double t3 = omp_get_wtime();

    printf("N=%d | Threads=%d | Seq Time=%.6fs | Par Time=%.6fs | Speedup=%.3f\n",
           N, nthreads, t1 - t0, t3 - t2, (t3 - t2 > 0) ? (t1 - t0) / (t3 - t2) : 0.0);

    free(A); free(B); free(C);
    return 0;
}