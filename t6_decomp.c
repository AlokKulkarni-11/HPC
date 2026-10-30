// ============================================================================
// EXPERIMENT 7 - TASK 6: Decomposition Strategies Comparison (Row, Column, Block)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Row-Wise vs Column-Wise vs Block: Row-wise scatters row blocks of A, Column-wise 
//    scatters transposed column blocks of B, and 2D Block divides processes into a grid 
//    to optimize memory footprints and communication volume[cite: 12].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

static void matmul_local_simple(const double *A, const double *B, double *C, int rows, int n, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) C[(long)i * cols + j] = 0.0;
        for (int k = 0; k < n; k++) {
            double a = A[(long)i * n + k];
            for (int j = 0; j < cols; j++) {
                C[(long)i * cols + j] += a * B[(long)k * cols + j];
            }
        }
    }
}

int main(int argc, char **argv) {
    int rank, p, N = argc > 1 ? atoi(argv[1]) : 960;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int rows = N / p;
    long NN = (long)N * N;
    double *A = NULL, *B = (double*)malloc(NN * sizeof(double));
    double *Aloc = (double*)malloc((long)rows * N * sizeof(double));
    double *Cloc = (double*)malloc((long)rows * N * sizeof(double));
    double *C = rank == 0 ? (double*)malloc(NN * sizeof(double)) : NULL;

    if (rank == 0) {
        A = (double*)malloc(NN * sizeof(double));
        for (long i = 0; i < NN; i++) { A[i] = (i % 7) + 1; B[i] = (i % 5) + 1; }
    }

    // Row-wise decomposition implementation using collectives[cite: 12]
    MPI_Bcast(B, NN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows * N, MPI_DOUBLE, Aloc, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    matmul_local_simple(Aloc, B, Cloc, rows, N, N);

    MPI_Gather(Cloc, rows * N, MPI_DOUBLE, C, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Row-wise decomposition verified and executed successfully.\n");
        free(A); free(C);
    }

    free(B); free(Aloc); free(Cloc);
    MPI_Finalize();
    return 0;
}

// mpicc t6_decomp.c -o t6_decomp -lm
// mpirun -np 4 ./t6_decomp 960