// ============================================================================
// EXPERIMENT 5 - CHALLENGE: Recursive Task Parallelism (Merge Sort)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Recursive Decomposition: Divide-and-conquer algorithms like Merge Sort create 
//    dynamically growing trees of subproblems well-suited for OpenMP tasks.
// 2. Task Granularity Threshold: Creating a task for every tiny sub-array creates 
//    massive scheduling overhead. A threshold switches to sequential execution 
//    for small sub-arrays to optimize performance.
// 3. Task Wait (#pragma omp taskwait): Forces the parent task to wait until both 
//    recursive child sorting tasks finish before merging the sorted halves.
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define THRESHOLD 1000

void merge(int *a, int lo, int mid, int hi, int *tmp) {
    int i = lo, j = mid + 1, k = lo;
    while (i <= mid && j <= hi) {
        if (a[i] <= a[j]) tmp[k++] = a[i++];
        else tmp[k++] = a[j++];
    }
    while (i <= mid) tmp[k++] = a[i++];
    while (j <= hi) tmp[k++] = a[j++];
    for (i = lo; i <= hi; i++) a[i] = tmp[i];
}

void mergeSortSeq(int *a, int lo, int hi, int *tmp) {
    if (lo >= hi) return;
    int mid = lo + (hi - lo) / 2;
    mergeSortSeq(a, lo, mid, tmp);
    mergeSortSeq(a, mid + 1, hi, tmp);
    merge(a, lo, mid, hi, tmp);
}

void mergeSortTask(int *a, int lo, int hi, int *tmp) {
    if (hi - lo + 1 <= THRESHOLD) {
        mergeSortSeq(a, lo, hi, tmp);
        return;
    }
    int mid = lo + (hi - lo) / 2;

    #pragma omp task shared(a, tmp) firstprivate(lo, mid)
    mergeSortTask(a, lo, mid, tmp);

    #pragma omp task shared(a, tmp) firstprivate(mid, hi)
    mergeSortTask(a, mid + 1, hi, tmp);

    #pragma omp taskwait
    merge(a, lo, mid, hi, tmp);
}

int main() {
    int N = 5000000;
    int *a = (int*)malloc(sizeof(int) * N);
    int *tmp = (int*)malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) a[i] = rand() % 100000;

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            mergeSortTask(a, 0, N - 1, tmp);
        }
    }
    double t1 = omp_get_wtime();

    printf("Parallel Merge Sort (N=%d) completed in %.4fs\n", N, t1 - t0);

    free(a); free(tmp);
    return 0;
}