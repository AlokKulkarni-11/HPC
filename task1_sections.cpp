// ============================================================================
// EXPERIMENT 5 - TASK 1 & 2: Independent Operations using OpenMP Sections
// ============================================================================
// THEORY & CONCEPTS:
// 1. Task Parallelism vs. Data Parallelism: Sections assigns different, independent 
//    code blocks to different threads simultaneously (different operations), 
//    whereas parallel for splits identical operations across loop iterations.
// 2. Sections Construct (#pragma omp sections): Used for coarse-grained task 
//    parallelism where a small, fixed set of tasks are known at compile time.
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    long N = 20000000L;
    double *A = (double*)malloc(sizeof(double) * N);
    for (long i = 0; i < N; i++) A[i] = (i % 1000) * 0.001;

    double sum = 0, max_val = -1e9, min_val = 1e9, avg = 0;

    double t0 = omp_get_wtime();
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            double s = 0;
            for (long i = 0; i < N; i++) s += A[i];
            sum = s;
        }
        #pragma omp section
        {
            double mx = A[0];
            for (long i = 1; i < N; i++) if (A[i] > mx) mx = A[i];
            max_val = mx;
        }
        #pragma omp section
        {
            double mn = A[0];
            for (long i = 1; i < N; i++) if (A[i] < mn) mn = A[i];
            min_val = mn;
        }
        #pragma omp section
        {
            double s = 0;
            for (long i = 0; i < N; i++) s += A[i];
            avg = s / N;
        }
    }

    double t1 = omp_get_wtime();

    printf("OpenMP Sections Executed in %.4fs\n", t1 - t0);
    printf("Results -> Sum: %.1f, Max: %.1f, Min: %.1f, Avg: %.2f\n", sum, max_val, min_val, avg);

    free(A);
    return 0;
}