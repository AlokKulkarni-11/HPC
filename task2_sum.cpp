// ============================================================================
// EXPERIMENT 2 - TASK 2: Sum of N Numbers (Sequential vs OpenMP Parallel For)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Data Parallelism: Applies the same operation (addition) across partitioned 
//    partitions of data (array elements) concurrently across multiple threads.
// 2. Work-Sharing Construct (#pragma omp for): Divides the loop iteration space 
//    among the active thread team. 
// 3. Race Condition & Reduction: If threads concurrently update a shared sum variable 
//    (sum += A[i]), a race condition occurs. The reduction(+:sum_par) clause 
//    automatically creates private per-thread accumulators and combines them safely 
//    at the end without locking overhead.
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    // Parse command line arguments for array size N and thread count
    long N = (argc > 1) ? atol(argv[1]) : 1000000L;
    int nthreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();

    // Allocate memory and initialize array with 1.0
    double *A = (double*)malloc(sizeof(double) * N);
    for (long i = 0; i < N; i++) {
        A[i] = 1.0;
    }

    // ----------------------------------------------------
    // 1. Sequential Execution (Baseline)
    // ----------------------------------------------------
    double t0 = omp_get_wtime();
    double sum_seq = 0.0;
    for (long i = 0; i < N; i++) {
        sum_seq += A[i];
    }
    double t1 = omp_get_wtime();
    double seq_time = t1 - t0;

    // ----------------------------------------------------
    // 2. OpenMP Parallel Execution with Reduction
    // ----------------------------------------------------
    omp_set_num_threads(nthreads);
    double sum_par = 0.0;
    double t2 = omp_get_wtime();
    
    // #pragma omp parallel for combines parallel region and loop work-sharing.
    // reduction(+:sum_par) solves race conditions efficiently via private accumulators.
    #pragma omp parallel for reduction(+:sum_par)
    for (long i = 0; i < N; i++) {
        sum_par += A[i];
    }
    
    double t3 = omp_get_wtime();
    double par_time = t3 - t2;

    // Output performance metrics
    double speedup = (par_time > 0) ? seq_time / par_time : 0.0;
    printf("N=%ld | Threads=%d | Seq Sum=%.1f | Par Sum=%.1f | Seq Time=%.6fs | Par Time=%.6fs | Speedup=%.3f\n",
           N, nthreads, sum_seq, sum_par, seq_time, par_time, speedup);

    free(A);
    return 0;
}