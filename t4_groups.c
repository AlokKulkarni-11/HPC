// ============================================================================
// EXPERIMENT 6 - TASK 4: MPI Groups and Sub-Communicators
// ============================================================================
// THEORY & CONCEPTS:
// 1. MPI Group: An ordered set of processes. Groups can be manipulated locally 
//    (e.g., `MPI_Group_incl`) to select subsets of processes[cite: 10].
// 2. MPI Communicator: Combines a group with a communication context. Messages sent 
//    on a communicator are isolated from others. Subgroups are turned into new 
//    communicators via `MPI_Comm_create_group`[cite: 10].
// 3. MPI_COMM_NULL: Returned to processes that are not members of the newly created 
//    sub-communicator; non-members must not use it[cite: 10].
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int world_rank, world_size, new_rank = -1, new_size = -1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Extract the group from MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // Build a subgroup containing even global ranks (0, 2, 4, 6, ...)
    int n_even = (world_size + 1) / 2;
    int *ranks = (int*)malloc(n_even * sizeof(int));
    for (int i = 0; i < n_even; i++) {
        ranks[i] = 2 * i;
    }

    MPI_Group even_group;
    MPI_Group_incl(world_group, n_even, ranks, &even_group);

    // Create a new sub-communicator from the even group
    MPI_Comm even_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, even_group, 0, &even_comm);

    // Check membership and query ranks in the new communicator
    if (even_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(even_comm, &new_rank);
        MPI_Comm_size(even_comm, &new_size);
        printf("Global rank %d/%d -> rank %d/%d in even_comm\n",
               world_rank, world_size, new_rank, new_size);
        MPI_Comm_free(&even_comm);
    } else {
        printf("Global rank %d/%d -> NOT in even_comm\n", world_rank, world_size);
    }

    MPI_Group_free(&world_group);
    MPI_Group_free(&even_group);
    free(ranks);
    
    MPI_Finalize();
    return 0;
}
// mpicc t4_groups.c -o t4_groups
// mpirun -np 8 ./t4_groups
