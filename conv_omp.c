// ============================================================================
// EXPERIMENT 8 - TASK 2B: OpenMP 2D Convolution
// ============================================================================
// THEORY & CONCEPTS:
// 1. Arithmetic Intensity: 2D convolution requires only ~9 multiply-adds per 
//    pixel loaded, making it memory-bandwidth bound compared to matrix multiplication[cite: 11].
// 2. Dynamic Schedule Choice: Demonstrates OpenMP `schedule(runtime)` where 
//    static, dynamic, or guided policies are configured via `omp_set_schedule`[cite: 11].
// ============================================================================

#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "common.h"

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1000;
    const char *sname = argc > 2 ? argv[2] : "static";
    int chunk = argc > 3 ? atoi(argv[3]) : 0;

    omp_sched_t kind = !strcmp(sname, "dynamic") ? omp_sched_dynamic : 
                       !strcmp(sname, "guided")  ? omp_sched_guided  : omp_sched_static;

    double K[9]; for (int i = 0; i < 9; i++) K[i] = 1.0;
    long NN = (long)N * N;

    // Padded input array: row 0 and row N+1 act as boundary halos[cite: 11]
    double *in = (double*)calloc((long)(N + 2) * N, sizeof(double));
    double *out0 = (double*)malloc(NN * sizeof(double));
    double *out = (double*)malloc(NN * sizeof(double));

    fill(&in[N], NN, 3);

    double t0 = omp_get_wtime();
    conv_rows(in, out0, N, N, K);
    double tseq = omp_get_wtime() - t0;
    printf("conv N=%d sequential: %.4f s\n", N, tseq);

    omp_set_schedule(kind, chunk);

    for (int a = 4; a < argc; a++) {
        int th = atoi(argv[a]);
        omp_set_num_threads(th);
        t0 = omp_get_wtime();

        #pragma omp parallel for schedule(runtime)
        for (int i = 1; i <= N; i++) {
            for (int j = 0; j < N; j++) {
                double s = 0;
                for (int u = -1; u <= 1; u++) {
                    for (int v = -1; v <= 1; v++) {
                        int jj = j + v;
                        if (jj >= 0 && jj < N) {
                            s += in[(long)(i + u) * N + jj] * K[(u + 1) * 3 + (v + 1)];
                        }
                    }
                }
                out[(long)(i - 1) * N + j] = s;
            }
        }

        double tp = omp_get_wtime() - t0;
        double err = maxdiff(out, out0, NN);
        printf("conv N=%d OpenMP schedule=%s,%d threads=%d: %.4f s speedup=%.2f eff=%.2f maxerr=%.1e %s\n",
               N, sname, chunk, th, tp, tseq / tp, tseq / tp / th, err, err < 1e-9 ? "OK" : "WRONG");
    }

    free(in); free(out0); free(out);
    return 0;
}

// gcc -O2 -fopenmp conv_omp.c -o conv_omp -lm
// conv_omp 2000 static 0 2 4 8