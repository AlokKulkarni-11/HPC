// Experiment 1 - Task 2: OpenMP Thread Creation
// Parallel region computing squares of first 1000 whole numbers
#include <stdio.h>
#include <stdlib.h>
#include <omp.h> 

int main(int argc, char *argv[]) {
    int requested_threads = (argc > 1) ? atoi(argv[1]) : omp_get_max_threads();
    int N = 1000;
    long long squares[1000]; 
    omp_set_num_threads(requested_threads); 

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int total = omp_get_num_threads(); 

        #pragma omp single
        printf("Total number of threads created: %d\n", total); 

        #pragma omp for
        for (int i = 0; i < N; i++) {
            squares[i] = (long long)i * i;
        } 

        #pragma omp critical
        printf("Thread ID: %d is active (team size = %d)\n", tid, total);
    } 

    printf("Sample check -> squares[10] = %lld, squares[999] = %lld\n",
           squares[10], squares[999]);
    return 0;
} 
```[cite: 1]

---

### Commands for Windows CMD / Terminal

**1. Create the file using Notepad:**
```cmd
notepad task2_threads.cpp