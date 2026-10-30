// ============================================================================
// EXPERIMENT 8 - TASK 2C: MPI 2D Convolution with Halo Exchange
// ============================================================================
// THEORY & CONCEPTS:
// 1. Halo / Ghost Rows: Border rows of neighbouring partitions needed to evaluate 
//    the 3x3 stencil kernel on boundary pixels[cite: 11].
// 2. Deadlock-Free Halo Exchange: `MPI_Sendrecv` concurrently shifts boundary 
//    rows upward and downward without triggering blocking deadlock[cite: 11].
// 3. MPI_PROC_NULL: Terminal boundary ranks (rank 0 and p-1) use `MPI_PROC_NULL` 
//    for nonexistent neighbors, turning boundary halo sends into no-ops[cite: 11].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "common.h"

int main(int argc, char **argv) {
    int rank, p, N = argc > 1 ? atoi(argv[1]) : 1000;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    double K[9]; for (int i = 0; i < 9; i++) K[i] = 1.0;
    long NN = (long)N * N;

    int *rc = (int*)malloc(p * sizeof(int)), *rd = (int*)malloc(p * sizeof(int));
    int *sc = (int*)malloc(p * sizeof(int)), *sd = (int*)malloc(p * sizeof(int));
    int base = N / p, rem = N % p, off = 0;
    for (int i = 0; i < p; i++) {
        rc[i] = base + (i < rem); rd[i] = off;
        sc[i] = rc[i] * N; sd[i] = rd[i] * N;
        off += rc[i];
    }

    int rows = rc[rank];
    double *I = NULL, *O = NULL, *O0 = NULL;
    double *loc = (double*)calloc((long)(rows + 2) * N, sizeof(double));
    double *oloc = (double*)malloc((long)rows * N * sizeof(double) + 1);
    double tseq = 0;

    if (rank == 0) {
        I = (double*)malloc(NN * sizeof(double));
        O = (double*)malloc(NN * sizeof(double));
        O0 = (double*)malloc(NN * sizeof(double));
        fill(I, NN, 3);
        double *pad = (double*)calloc((long)(N + 2) * N, sizeof(double));
        for (long i = 0; i < NN; i++) pad[N + i] = I[i];
        double t0 = MPI_Wtime();
        conv_rows(pad, O0, N, N, K);
        tseq = MPI_Wtime() - t0;
        free(pad);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Scatterv(I, sc, sd, MPI_DOUBLE, &loc[N], rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Halo Exchange with upper and lower neighbouring processes[cite: 11]
    int up = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    int down = rank < p - 1 ? rank + 1 : MPI_PROC_NULL;
    double h0 = MPI_Wtime();

    MPI_Sendrecv(&loc[N], N, MPI_DOUBLE, up, 0,
                 &loc[(long)(rows + 1) * N], N, MPI_DOUBLE, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&loc[(long)rows * N], N, MPI_DOUBLE, down, 1,
                 &loc[0], N, MPI_DOUBLE, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    double thalo = MPI_Wtime() - h0;

    double c0 = MPI_Wtime();
    conv_rows(loc, oloc, rows, N, K);
    double tcomp = MPI_Wtime() - c0;

    MPI_Gatherv(oloc, rows * N, MPI_DOUBLE, O, sc, sd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    double tpar = MPI_Wtime() - t0, cmax, hmax;
    MPI_Reduce(&tcomp, &cmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&thalo, &hmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double err = maxdiff(O, O0, NN);
        printf("conv N=%d MPI p=%d: %.4f s speedup=%.2f eff=%.2f | compute=%.4f halo=%.6f comm=%.4f | maxerr=%.1e %s\n",
               N, p, tpar, tseq / tpar, tseq / tpar / p, cmax, hmax, tpar - cmax, err, err < 1e-9 ? "OK" : "WRONG");
        free(I); free(O); free(O0);
    }

    free(loc); free(oloc); free(rc); free(rd); free(sc); free(sd);
    MPI_Finalize();
    return 0;
}

// mpicc -O2 conv_mpi.c -o conv_mpi -lm
// mpirun -np 4 ./conv_mpi 2000