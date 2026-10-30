// ============================================================================
// EXPERIMENT 8 - TASK 1C: MPI & Hybrid MPI+OpenMP Matrix Multiplication
// ============================================================================
// THEORY & CONCEPTS:
// 1. Distributed Memory Communication: Rank 0 broadcasts B to all processes 
//    via `MPI_Bcast` and scatters slices of A via `MPI_Scatter`[cite: 11].
// 2. Hybrid Parallelism: Inside each MPI process, threads share the local block 
//    of A and the full matrix B, parallelizing the local loop with OpenMP[cite: 11].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "common.h"

static void matmul_local(const double *A, const double *B, double *C, int rows, int n) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < n; j++) C[(long)i * n + j] = 0.0;
        for (int k = 0; k < n; k++) {
            double a = A[(long)i * n + k];
            for (int j = 0; j < n; j++) C[(long)i * n + j] += a * B[(long)k * n + j];
        }
    }
}

int main(int argc, char **argv) {
    int rank, p, N = argc > 1 ? atoi(argv[1]) : 500, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int rows = (N + p - 1) / p, Npad = rows * p;
    long NN = (long)N * N;
    double *A = NULL, *C = NULL, *C0 = NULL, *B = (double*)malloc(NN * sizeof(double));
    double *Aloc = (double*)malloc((long)rows * N * sizeof(double));
    double *Cloc = (double*)malloc((long)rows * N * sizeof(double));
    double tseq = 0;

    if (rank == 0) {
        A = (double*)calloc((long)Npad * N, sizeof(double));
        fill(A, NN, 1); fill(B, NN, 2);
        C = (double*)malloc((long)Npad * N * sizeof(double));
        C0 = (double*)malloc(NN * sizeof(double));
        double t0 = MPI_Wtime();
        matmul_seq(A, B, C0, N, N);
        tseq = MPI_Wtime() - t0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Broadcast matrix B and scatter rows of matrix A[cite: 11]
    MPI_Bcast(B, (int)NN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows * N, MPI_DOUBLE, Aloc, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double c0 = MPI_Wtime();
    matmul_local(Aloc, B, Cloc, rows, N);
    double tcomp = MPI_Wtime() - c0;

    // Gather output blocks back to root process[cite: 11]
    MPI_Gather(Cloc, rows * N, MPI_DOUBLE, C, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    double tpar = MPI_Wtime() - t0, cmax;
    MPI_Reduce(&tcomp, &cmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int th = 1;
        #ifdef _OPENMP
        th = omp_get_max_threads();
        #endif
        double err = maxdiff(C, C0, NN);
        printf("matmul N=%d MPI p=%d threads/proc=%d: %.4f s speedup=%.2f eff=%.2f | compute=%.4f comm=%.4f | maxerr=%.1e %s\n",
               N, p, th, tpar, tseq / tpar, tseq / tpar / (p * th), cmax, tpar - cmax, err, err < 1e-9 ? "OK" : "WRONG");
        free(A); free(C); free(C0);
    }

    free(B); free(Aloc); free(Cloc);
    MPI_Finalize();
    return 0;
}

// **Pure MPI:**

//  mpicc -O2 mm_mpi.c -o mm_mpi -lm
//  mpirun -np 4 ./mm_mpi 1000
  // **Hybrid MPI + OpenMP:**
//  mpicc -O2 -fopenmp mm_mpi.c -o mm_hybrid -lm
 // mpirun -np 2 ./mm_hybrid 1000