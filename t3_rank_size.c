// ============================================================================
// EXPERIMENT 6 - TASK 3: MPI Rank and Size Identification
// ============================================================================
// THEORY & CONCEPTS:
// 1. MPI Rank: A unique integer ID (0 to size-1) assigned to each process within 
//    a communicator, obtained via `MPI_Comm_rank`[cite: 10].
// 2. MPI Size: The total number of processes participating in the communicator, 
//    obtained via `MPI_Comm_size`[cite: 10].
// 3. MPI_COMM_WORLD: The predefined default communicator encompassing all processes 
//    started by `mpirun`/`mpiexec`[cite: 10].
// ============================================================================

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);

    // Get the unique rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processes in the communicator
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Process %d of %d\n", rank, size);

    MPI_Finalize();
    return 0;
}