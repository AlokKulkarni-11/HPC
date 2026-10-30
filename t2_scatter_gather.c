// ============================================================================
// EXPERIMENT 7 - TASK 2: Scatter and Gather Operations
// ============================================================================
// THEORY & CONCEPTS:
// 1. MPI_Scatter: Distributes equal chunks of an array from the root process 
//    to each process in the communicator (one-to-all distribution)[cite: 12].
// 2. MPI_Gather: The reverse operation[cite: 12]. Each process sends its local 
//    computed piece back to the root, which assembles them in rank order[cite: 12].
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

    // Ensure N is cleanly divisible by the process count[cite: 12]
    if (N % size != 0) {
        if (rank == 0) printf("N=%d must be divisible by processes (%d)\n", N, size);
        MPI_Finalize();
        return 1;
    }

    int chunk = N / size;
    int *data = NULL, *local = (int*)malloc(chunk * sizeof(int)), *sums = NULL;

    if (rank == 0) {
        data = (int*)malloc(N * sizeof(int));
        sums = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < N; i++) data[i] = i + 1;
    }

    // Scatter chunks of 'data' to local buffers across processes[cite: 12]
    MPI_Scatter(data, chunk, MPI_INT, local, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    int ls = 0;
    for (int i = 0; i < chunk; i++) {
        ls += local[i];
    }

    printf("Process %d got local chunk sum = %d\n", rank, ls);

    // Gather local sums back into the 'sums' array at root[cite: 12]
    MPI_Gather(&ls, 1, MPI_INT, sums, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total = 0;
        printf("Gathered local sums:");
        for (int i = 0; i < size; i++) {
            printf(" %d", sums[i]);
            total += sums[i];
        }
        printf("\nFinal combined result at Process 0 = %d\n", total);
        free(data);
        free(sums);
    }

    free(local);
    MPI_Finalize();
    return 0;
}

// mpicc t2_scatter_gather.c -o t2_scatter_gather
// mpirun -np 4 ./t2_scatter_gather