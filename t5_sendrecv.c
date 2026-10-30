// ============================================================================
// EXPERIMENT 6 - TASK 5: MPI Send and Receive (String Message)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Distributed Memory & Explicit Messaging: In distributed memory systems, 
//    processes have private address spaces. Data sharing requires explicit 
//    message passing (`MPI_Send` and `MPI_Recv`)[cite: 10].
// 2. Point-to-Point Communication: Involves a specific sender and receiver. 
//    `MPI_Send` transmits data, while `MPI_Recv` blocks until the matching 
//    message arrives, returning status information (`MPI_Status`)[cite: 10].
// ============================================================================

#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size;
    char msg[64];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Error: Need at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        strcpy(msg, "Hello from Process 0");
        // Send message to Process 1 with tag 0
        MPI_Send(msg, strlen(msg) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Status st;
        // Receive message from Process 0
        MPI_Recv(msg, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &st);
        printf("Process 1 received: \"%s\" (source=%d, tag=%d)\n", msg, st.MPI_SOURCE, st.MPI_TAG);
    }

    MPI_Finalize();
    return 0;
}