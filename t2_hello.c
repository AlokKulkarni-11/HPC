// ============================================================================
// EXPERIMENT 6 - TASK 2: First MPI Program (Hello World)
// ============================================================================
// THEORY & CONCEPTS:
// 1. MPI_Init: Initializes the MPI execution environment. Must be called before 
//    any other MPI routine. Sets up `MPI_COMM_WORLD`.
// 2. MPI_Finalize: Cleanly shuts down the MPI execution environment. No MPI calls 
//    are allowed after this.
// ============================================================================

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    printf("Hello from MPI\n");

    // Shut down MPI environment
    MPI_Finalize();
    return 0;
}