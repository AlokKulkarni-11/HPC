// ============================================================================
// EXPERIMENT 7 - TASK 7: Non-Uniform Distribution with MPI_Scatterv
// ============================================================================
// THEORY & CONCEPTS:
// 1. MPI_Scatterv: Extends basic scatter by allowing variable data chunk sizes 
//    and displacements per process[cite: 12]. Essential when matrix dimensions 
//    are not evenly divisible by the total process count[cite: 12].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

static void matvec_local(const double *A, const double *x, double *y, int rows, int n) {
    for (int i = 0; i < rows; i++) {
        double s = 0;
        for (int k = 0; k < n; k++) s += A[(long)i * n + k] * x[k];
        y[i] = s;
    }
}

int main(int argc, char **argv) {
    int rank, p, N = argc > 1 ? atoi(argv[1]) : 10;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int *rcount = (int*)malloc(p * sizeof(int));
    int *rdispl = (int*)malloc(p * sizeof(int));
    int *scount = (int*)malloc(p * sizeof(int));
    int *sdispl = (int*)malloc(p * sizeof(int));

    int base = N / p, rem = N % p, off = 0;
    for (int i = 0; i < p; i++) {
        rcount[i] = base + (i < rem ? 1 : 0);
        rdispl[i] = off;
        scount[i] = rcount[i] * N;
        sdispl[i] = rdispl[i] * N;
        off += rcount[i];
    }

    int myrows = rcount[rank];
    double *A = NULL, *Y = NULL, *x = (double*)malloc(N * sizeof(double));
    double *Aloc = (double*)malloc((long)myrows * N * sizeof(double));
    double *yloc = (double*)malloc(myrows * sizeof(double));

    if (rank == 0) {
        A = (double*)malloc((long)N * N * sizeof(double));
        Y = (double*)malloc(N * sizeof(double));
        for (long i = 0; i < (long)N * N; i++) A[i] = 1.0;
        for (int i = 0; i < N; i++) x[i] = 1.0;
    }

    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Scatter non-uniform row blocks using Scatterv[cite: 12]
    MPI_Scatterv(A, scount, sdispl, MPI_DOUBLE, Aloc, myrows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matvec_local(Aloc, x, yloc, myrows, N);

    // Gather variable-sized chunks back using Gatherv[cite: 12]
    MPI_Gatherv(yloc, myrows, MPI_DOUBLE, Y, rcount, rdispl, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Scatterv / Gatherv non-uniform distribution test completed successfully.\n");
        free(A); free(Y);
    }

    free(x); free(Aloc); free(yloc);
    free(rcount); free(rdispl); free(scount); free(sdispl);
    MPI_Finalize();
    return 0;
}

// mpicc t7_scatterv.c -o t7_scatterv -lm
// mpirun -np 3 ./t7_scatterv 10