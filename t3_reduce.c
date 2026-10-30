// ============================================================================
// EXPERIMENT 7 - TASK 3: Global Reduction Using MPI_Reduce
// ============================================================================
// THEORY & CONCEPTS:
// 1. MPI_Reduce: Combines values from all processes using specified operations 
//    (MPI_SUM, MPI_MAX, MPI_MIN)[cite: 12] and delivers the single result to the root.
// 2. Efficiency: Avoids routing all raw data elements to the root (unlike Gather), 
//    combining values hierarchically across nodes.
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 16

int main(int argc, char **argv) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) printf("N must be divisible by processes\n");
        MPI_Finalize();
        return 1;
    }

    int chunk = N / size;
    int *data = NULL, *local = (int*)malloc(chunk * sizeof(int));

    if (rank == 0) {
        data = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) data[i] = i + 1;
    }

    MPI_Scatter(data, chunk, MPI_INT, local, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    int ls = 0, lmax = local[0], lmin = local[0];
    for (int i = 0; i < chunk; i++) {
        ls += local[i];
        if (local[i] > lmax) lmax = local[i];
        if (local[i] < lmin) lmin = local[i];
    }

    int gsum, gmax, gmin;
    // Execute global reductions for sum, maximum, and minimum[cite: 12]
    MPI_Reduce(&ls,   &gsum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lmax, &gmax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lmin, &gmin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global Reductions -> Sum: %d, Max: %d, Min: %d\n", gsum, gmax, gmin);
        free(data);
    }

    free(local);
    MPI_Finalize();
    return 0;
}
// mpicc t3_reduce.c -o t3_reduce
// mpirun -np 4 ./t3_reduce