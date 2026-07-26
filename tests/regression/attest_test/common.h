#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// GEMM kernel args (same layout as tests/regression/checker_test).  C = A * B.
typedef struct {
    uint32_t grid_dim[2];
    uint32_t block_dim[2];
    uint32_t M;          // rows of A / rows of C   (== checker batch_size)
    uint32_t N;          // cols of B / cols of C
    uint32_t K;          // cols of A / rows of B   (== checker hidden_size)
    uint32_t tile_size;
    uint64_t A_addr;      // [M x K] FP32 activations — also the checker's tap
    uint64_t B_addr;      // [K x N] FP32 weights     — attested per-layer region(s)
    uint64_t C_addr;      // [M x N] FP32 output
} kernel_arg_t;

#endif
