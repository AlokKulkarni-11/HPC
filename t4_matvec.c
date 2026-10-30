// ============================================================================
// EXPERIMENT 7 - TASK 4: MPI Matrix-Vector Multiplication (Y = A * X)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Row-Wise Decomposition: Matrix A is partitioned contiguously by rows across 
//    processes[cite: 12]. If N is not cleanly divisible by p, zero-padding rows are added[cite: 12].
// 2. Broadcast and Scatter: The vector X is broadcasted to all processes via 
//    `MPI_Bcast`, while rows of A are distributed using `MPI_Scatter`[cite: 12].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Local matrix-vector multiplication helper function[cite: 12]
static void matvec_local(const double *A, const double *x, double *y, int rows, int n) {
    for (int i = 0; i < rows; i++) {
        double s = 0;
        for (int k = 0; k < n; k++) {
            s += A[(long)i * n + k] * x[k];
        }
        y[i] = s;
    }
}

int main(int argc, char **argv) {
    int rank, p, N = argc > 1 ? atoi(argv[1]) : 1000;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = (N + p - 1) / p, Npad = rows * p;
    double *A = NULL, *Y = NULL, *x = (double*)malloc(N * sizeof(double));
    double *Aloc = (double*)malloc((long)rows * N * sizeof(double));
    double *yloc = (double*)malloc(rows * sizeof(double));

    if (rank == 0) {
        A = (double*)calloc((long)Npad * N, sizeof(double));
        Y = (double*)malloc(Npad * sizeof(double));
        for (long i = 0; i < (long)N * N; i++) A[i] = (double)(i % 10) / 10.0;
        for (int i = 0; i < N; i++) x[i] = 1.0;
    }

    // Broadcast vector X and scatter matrix A rows[cite: 12]
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows * N, MPI_DOUBLE, Aloc, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local matrix-vector product segment[cite: 12]
    matvec_local(Aloc, x, yloc, rows, N);

    // Gather partial results back to root[cite: 12]
    MPI_Gather(yloc, rows, MPI_DOUBLE, Y, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Matrix-Vector Multiplication completed successfully on root.\n");
        free(A); free(Y);
    }

    free(x); free(Aloc); free(yloc);
    MPI_Finalize();
    return 0;
}

// mpicc t4_matvec.c -o t4_matvec -lm
// mpirun -np 4 ./t4_matvec 1000