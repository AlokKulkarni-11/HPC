// ============================================================================
// EXPERIMENT 6 - TASK 7: Array Communication & Computation
// ============================================================================
// THEORY & CONCEPTS:
// 1. Buffer Transmissions: Arrays are passed via contiguous memory pointers 
//    by specifying the count parameter (e.g., N elements of MPI_INT)[cite: 10].
// 2. Distributed Processing: Rank 0 sends an array chunk, while Rank 1 receives 
//    the complete block and performs local analytical calculations (Sum, Average, Max)[cite: 10].
// ============================================================================

#include <stdio.h>
#include <mpi.h>

#define N 5

int main(int argc, char **argv) {
    int rank, size;
    int a[N] = {10, 20, 30, 40, 50};

    // Initialize MPI environment[cite: 10]
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Error: Need >= 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        // Send array 'a' of N integers to Process 1[cite: 10]
        MPI_Send(a, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        int b[N], sum = 0, max;
        
        // Receive array into local buffer 'b'[cite: 10]
        MPI_Recv(b, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        max = b[0];
        for (int i = 0; i < N; i++) {
            sum += b[i];
            if (b[i] > max) max = b[i];
        }

        // Print received content and calculated aggregate statistics[cite: 10]
        printf("Process 1 received:");
        for (int i = 0; i < N; i++) printf(" %d", b[i]);
        printf("\nSum = %d, Average = %.2f, Maximum = %d\n", sum, (double)sum / N, max);
    }

    // Finalize MPI environment[cite: 10]
    MPI_Finalize();
    return 0;
}