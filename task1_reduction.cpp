// ============================================================================
// EXPERIMENT 3 - SECTION A TASK 1: Array Sum (Sequential vs Critical vs Reduction)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Critical Section Bottleneck (#pragma omp critical): Forces threads to acquire 
//    a mutual-exclusion lock on EVERY loop iteration. This serializes updates 
//    and incurs severe synchronization overhead (~10x-15x slower).
// 2. Reduction Clause (reduction(+:sum)): Avoids locking during the loop by 
//    allocating private local copies for each thread and combining them efficiently 
//    at the end. This is the optimal way to perform reductions in OpenMP.
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    long N = (argc > 1) ? atol(argv[1]) : 1000000L;
    int nthreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    double *A = (double*)malloc(sizeof(double) * N);
    for (long i = 0; i < N; i++) A[i] = 1.0;
    omp_set_num_threads(nthreads);

    // 1. Sequential Sum
    double t0 = omp_get_wtime();
    double sum_seq = 0.0;
    for (long i = 0; i < N; i++) sum_seq += A[i];
    double t1 = omp_get_wtime();

    // 2. Parallel Sum with Critical Section (Inefficient)
    double sum_crit = 0.0;
    double t2 = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        #pragma omp critical
        sum_crit += A[i]; // Serialization bottleneck!
    }
    double t3 = omp_get_wtime();

    // 3. Parallel Sum with Reduction Clause (Optimal)
    double sum_red = 0.0;
    double t4 = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum_red)
    for (long i = 0; i < N; i++) {
        sum_red += A[i];
    }
    double t5 = omp_get_wtime();

    printf("N=%ld threads=%d\n", N, nthreads);
    printf("  Sequential : %.1f (%.6fs)\n", sum_seq, t1 - t0);
    printf("  Critical   : %.1f (%.6fs) [Slow due to locks]\n", sum_crit, t3 - t2);
    printf("  Reduction  : %.1f (%.6fs) [Optimal]\n", sum_red, t5 - t4);

    free(A);
    return 0;
}