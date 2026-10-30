// ============================================================================
// EXPERIMENT 5 - TASK 3: Pipeline Processing using OpenMP Tasks & Dependencies
// ============================================================================
// THEORY & CONCEPTS:
// 1. OpenMP Tasks (#pragma omp task): Enables dynamic task creation for irregular, 
//    dependent, or runtime-generated work items.
// 2. Task Dependencies (depend clause): Specifies input/output data dependencies 
//    between tasks. The OpenMP runtime automatically constructs a directed acyclic 
//    graph (DAG) and ensures tasks execute only when their data dependencies are met.
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void preprocess(double *data, int N) {
    for (int i = 0; i < N; i++) data[i] = data[i] * 2.0;
}

void compute(double *data, int N) {
    for (int i = 0; i < N; i++) data[i] = data[i] + 5.0;
}

int main() {
    int N = 1000000;
    double *data = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) data[i] = 1.0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Task 1: Preprocessing stage
            #pragma omp task depend(inout: data[0:N])
            {
                preprocess(data, N);
                printf("Preprocessing task completed.\n");
            }

            // Task 2: Computation stage (depends on Task 1)
            #pragma omp task depend(inout: data[0:N])
            {
                compute(data, N);
                printf("Computation task completed.\n");
            }
        }
    }

    printf("Pipeline finished successfully. Data[0] = %.1f\n", data[0]);
    free(data);
    return 0;
}