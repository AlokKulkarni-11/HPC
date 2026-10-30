// ============================================================================
// EXPERIMENT 7 - TASK 5: MPI Matrix-Matrix Multiplication (C = A * B)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Matrix-Matrix Partitioning: Matrix A is distributed row-wise across processes, 
//    while the entire matrix B is broadcasted to all nodes using `MPI_Bcast`[cite: 12].
// 2. Scalable Computations: Each process computes its designated block of output 
//    rows independently before gathering back to root[cite: 12].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

static void matmul_local(const double *A, const double *B, double *C, int rows, int n) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < n; j++) C[(long)i * n + j] = 0.0;
        for (int k = 0; k < n; k++) {
            double a = A[(long)i * n + k];
            for (int j = 0; j < n; j++) {
                C[(long)i * n + j] += a * B[(long)k * n + j];
            }
        }
    }
}

int main(int argc, char **argv) {
    int rank, p, N = argc > 1 ? atoi(argv[1]) : 500;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int rows = (N + p - 1) / p, Npad = rows * p;
    double *A = NULL, *C = NULL, *B = (double*)malloc((long)N * N * sizeof(double));
    double *Aloc = (double*)malloc((long)rows * N * sizeof(double));
    double *Cloc = (double*)malloc((long)rows * N * sizeof(double));

    if (rank == 0) {
        A = (double*)calloc((long)Npad * N, sizeof(double));
        C = (double*)malloc((long)Npad * N * sizeof(double));
        for (long i = 0; i < (long)N * N; i++) { A[i] = (i % 7) + 1; B[i] = (i % 5) + 1; }
    }

    // Broadcast B and scatter rows of A[cite: 12]
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows * N, MPI_DOUBLE, Aloc, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local matrix product subset[cite: 12]
    matmul_local(Aloc, B, Cloc, rows, N);

    // Gather final matrix blocks at root[cite: 12]
    MPI_Gather(Cloc, rows * N, MPI_DOUBLE, C, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Matrix-Matrix Multiplication completed successfully on root.\n");
        free(A); free(C);
    }

    free(B); free(Aloc); free(Cloc);
    MPI_Finalize();
    return 0;
}


// mpicc t5_matmul.c -o t5_matmul -lm
// mpirun -np 4 ./t5_matmul 500