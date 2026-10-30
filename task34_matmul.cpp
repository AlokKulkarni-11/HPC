// ============================================================================
// EXPERIMENT 2 - TASK 3 & 4: Matrix Multiplication with OpenMP Scheduling
// ============================================================================
// THEORY & CONCEPTS:
// 1. Matrix Multiplication ($C = A \times B$): Involves triple nested loops. 
//    Reuses memory elements across inner loops, making cache utilization critical.
// 2. Scheduling Policies for Work-Sharing Loops:
//    - Static: Chunks of iterations are assigned to threads upfront in round-robin. 
//      Best for uniform workloads (zero runtime overhead).
//    - Dynamic: Threads fetch chunks dynamically from a queue as they finish. 
//      Best for irregular/unbalanced workloads at the cost of queue lock overhead.
//    - Guided: Chunk size starts large and decreases exponentially as threads finish, 
//      balancing overhead reduction with adaptability.
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

double *A, *B, *C;
int M, K, P;

void alloc_mats() {
    A = (double*)malloc(sizeof(double) * M * K);
    B = (double*)malloc(sizeof(double) * K * P);
    C = (double*)malloc(sizeof(double) * M * P);
    for (int i = 0; i < M * K; i++) A[i] = (i % 7) + 1;
    for (int i = 0; i < K * P; i++) B[i] = (i % 5) + 1;
}

// Sequential Matrix Multiplication Baseline
void matmul_seq() {
    memset(C, 0, sizeof(double) * M * P);
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            double a = A[i * K + k];
            for (int j = 0; j < P; j++) {
                C[i * P + j] += a * B[k * P + j];
            }
        }
    }
}

// Parallel Matrix Multiplication with configurable scheduling
void matmul_par(const char *sched, int chunk) {
    memset(C, 0, sizeof(double) * M * P);
    
    if (strcmp(sched, "static") == 0) {
        #pragma omp parallel for schedule(static, chunk)
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                double a = A[i * K + k];
                for (int j = 0; j < P; j++) C[i * P + j] += a * B[k * P + j];
            }
        }
    } else if (strcmp(sched, "dynamic") == 0) {
        #pragma omp parallel for schedule(dynamic, chunk)
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                double a = A[i * K + k];
                for (int j = 0; j < P; j++) C[i * P + j] += a * B[k * P + j];
            }
        }
    } else { // guided scheduling
        #pragma omp parallel for schedule(guided, chunk)
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                double a = A[i * K + k];
                for (int j = 0; j < P; j++) C[i * P + j] += a * B[k * P + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    M = K = P = (argc > 1) ? atoi(argv[1]) : 500;
    int nthreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(nthreads);
    alloc_mats();

    double t0 = omp_get_wtime();
    matmul_seq();
    double t1 = omp_get_wtime();
    double seq_time = t1 - t0;
    double checksum_seq = C[0] + C[M * P - 1];

    printf("Matrix Size: %dx%d | Threads: %d | Sequential Time: %.6fs\n", M, P, nthreads, seq_time);

    const char *scheds[3] = {"static", "dynamic", "guided"};
    for (int s = 0; s < 3; s++) {
        double t2 = omp_get_wtime();
        matmul_par(scheds[s], 4);
        double t3 = omp_get_wtime();
        double par_time = t3 - t2;
        double checksum_par = C[0] + C[M * P - 1];
        double speedup = (par_time > 0) ? seq_time / par_time : 0.0;

        printf("Schedule: %-8s | Time: %.6fs | Speedup: %.3f | Match: %s\n",
               scheds[s], par_time, speedup, (checksum_par == checksum_seq) ? "yes" : "no");
    }

    free(A); free(B); free(C);
    return 0;
}