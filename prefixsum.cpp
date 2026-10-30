// ============================================================================
// EXPERIMENT 3 - SECTION B TASK 1: Prefix Sum (Hillis-Steele Parallel Scan)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Prefix Sum (Scan): Computes cumulative running sums ($output[i] = \sum_{j=0}^{i} A[j]$). 
//    Unlike reduction (many-to-one), scan is many-to-many with inherent data dependencies.
// 2. Hillis-Steele Scan: A parallel scan algorithm where elements add values 
//    from $2^{k-1}$ positions behind in each of $\lceil \log_2 N \rceil$ rounds. 
//    It exposes high parallelism but performs $O(N \log N)$ total work compared to 
//    the sequential $O(N)$ algorithm.
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    long N = (argc > 1) ? atol(argv[1]) : 1000000L;
    int nthreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    
    double *A = (double*)malloc(sizeof(double) * N);
    double *seq_out = (double*)malloc(sizeof(double) * N);
    double *buf0 = (double*)malloc(sizeof(double) * N);
    double *buf1 = (double*)malloc(sizeof(double) * N);
    
    for (long i = 0; i < N; i++) A[i] = 1.0;
    omp_set_num_threads(nthreads);

    // Sequential Scan
    double t0 = omp_get_wtime();
    seq_out[0] = A[0];
    for (long i = 1; i < N; i++) {
        seq_out[i] = seq_out[i - 1] + A[i];
    }
    double t1 = omp_get_wtime();

    // Parallel Hillis-Steele Scan
    memcpy(buf0, A, sizeof(double) * N);
    double *cur = buf0, *nxt = buf1;
    
    double t2 = omp_get_wtime();
    for (long offset = 1; offset < N; offset *= 2) {
        #pragma omp parallel for
        for (long i = 0; i < N; i++) {
            nxt[i] = (i >= offset) ? cur[i] + cur[i - offset] : cur[i];
        }
        double *tmp = cur; cur = nxt; nxt = tmp;
    }
    double t3 = omp_get_wtime();

    // Verification
    int match = 1;
    for (long i = 0; i < N; i++) {
        if (cur[i] != seq_out[i]) { match = 0; break; }
    }

    printf("N=%ld | Threads=%d | Seq Time=%.6fs | Par Time=%.6fs | Match: %s\n",
           N, nthreads, t1 - t0, t3 - t2, match ? "yes" : "no");

    free(A); free(seq_out); free(buf0); free(buf1);
    return 0;
}