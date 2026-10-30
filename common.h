// ============================================================================
// EXPERIMENT 8 - SHARED HELPER FUNCTIONS (common.h)
// ============================================================================
// THEORY & CONCEPTS:
// 1. Exact Verification: Uses integer-valued double data (0-9) so arithmetic 
//    results match bit-for-bit across sequential, OpenMP, and MPI implementations[cite: 11].
// 2. Cache-Friendly Loop Order: Matrix multiplication uses i-k-j loop ordering 
//    to ensure stride-1 contiguous access for matrices B and C[cite: 11].
// ============================================================================

#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <math.h>

static void fill(double *a, long n, unsigned seed) {
    srand(seed);
    for (long i = 0; i < n; i++) a[i] = (double)(rand() % 10);
}

static double maxdiff(const double *a, const double *b, long n) {
    double m = 0;
    for (long i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// C(rows x n) = A(rows x n) * B(n x n), i-k-j loop order
static void matmul_seq(const double *A, const double *B, double *C, int rows, int n) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < n; j++) C[(long)i * n + j] = 0.0;
        for (int k = 0; k < n; k++) {
            double a = A[(long)i * n + k];
            for (int j = 0; j < n; j++) C[(long)i * n + j] += a * B[(long)k * n + j];
        }
    }
}

// 3x3 convolution with zero padding. in has (rows+2) rows including halo rows[cite: 11]
static void conv_rows(const double *in, double *out, int rows, int n, const double *K) {
    for (int i = 1; i <= rows; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int u = -1; u <= 1; u++) {
                for (int v = -1; v <= 1; v++) {
                    int jj = j + v;
                    if (jj >= 0 && jj < n) {
                        s += in[(long)(i + u) * n + jj] * K[(u + 1) * 3 + (v + 1)];
                    }
                }
            }
            out[(long)(i - 1) * n + j] = s;
        }
    }
}

#endif