// ============================================================================
// EXPERIMENT 7 - TASK 1: Broadcast Using MPI_Bcast
// ============================================================================
// THEORY & CONCEPTS:
// 1. Collective Communication: Involves all processes within a communicator 
//    in a single synchronized call[cite: 12]. Every process must call the routine 
//    with matching arguments[cite: 12].
// 2. MPI_Bcast: Copies data from a designated root process to every other process 
//    in the communicator[cite: 12] using optimized tree-based routing.
// ============================================================================

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size, value = 0;

    // Initialize MPI environment[cite: 12]
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process initializes the broadcast value[cite: 12]
    if (rank == 0) {
        value = 42;
        printf("Process 0 broadcasting value = %d\n", value);
    }

    // Broadcast the value from root (rank 0) to all processes[cite: 12]
    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Every process prints its rank and the received broadcast value[cite: 12]
    printf("Process %d of %d received value = %d\n", rank, size, value);

    // Finalize MPI environment[cite: 12]
    MPI_Finalize();
    return 0;
}


// mpicc t1_bcast.c -o t1_bcast
// mpirun -np 4 ./t1_bcast