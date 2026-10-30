// ============================================================================
// EXPERIMENT 6 - TASK 6: Integer Data Communication
// ============================================================================
// THEORY & CONCEPTS:
// 1. Datatypes in MPI: MPI uses explicit datatypes (such as MPI_INT, MPI_CHAR, 
//    MPI_DOUBLE) so that heterogeneous systems can correctly translate 
//    binary formats if necessary.
// 2. Point-to-Point Integer Transfer: Demonstrates sending and receiving scalar 
//    integer values between rank 0 and rank 1 using MPI_Send and MPI_Recv[cite: 10].
// ============================================================================

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size, value;

    // Initialize MPI execution environment[cite: 10]
    MPI_Init(&argc, &argv);
    
    // Retrieve process rank and total communicator size[cite: 10]
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure at least 2 processes are launched[cite: 10]
    if (size < 2) {
        if (rank == 0) printf("Error: Need >= 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        value = 100;
        printf("Process 0 sending value = %d\n", value);
        // Send 1 integer value to Process 1[cite: 10]
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        // Receive 1 integer value from Process 0, ignoring status[cite: 10]
        MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 Received Value = %d  [%s]\n", value, value == 100 ? "VERIFIED" : "MISMATCH");
    }

    // Shut down MPI environment[cite: 10]
    MPI_Finalize();
    return 0;
}