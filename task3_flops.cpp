// Experiment 1 - Task 3: Theoretical Peak FLOPS calculation
#include <stdio.h> 

int main() {
    int cores = 1;                 // physical cores detected in this environment (see lscpu)
    double freq_ghz = 2.80;        // base clock frequency in GHz (Intel Xeon @ 2.80GHz)
    int simd_width_sp = 16;        // AVX-512, single precision -> 16 floats per vector register
    int ops_per_instr_fma = 2;     // 1 FMA = 2 FLOPs (1 mul + 1 add) 

    double peak_flops_scalar = cores * (freq_ghz * 1e9) * 1;               // scalar, 1 FLOP/clock
    double peak_flops_avx512_fma = cores * (freq_ghz * 1e9) * simd_width_sp * ops_per_instr_fma; 

    printf("Cores                 : %d\n", cores);
    printf("Clock Frequency (GHz) : %.2f\n", freq_ghz);
    printf("Scalar Peak FLOPS      : %.3e FLOPS\n", peak_flops_scalar);
    printf("AVX-512 FMA Peak FLOPS : %.3e FLOPS (%.2f GFLOPS)\n",
           peak_flops_avx512_fma, peak_flops_avx512_fma / 1e9);
    return 0;
} 
```[cite: 1]