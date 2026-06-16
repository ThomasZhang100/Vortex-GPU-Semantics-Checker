#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
    uint32_t grid_dim[2];
    uint32_t block_dim[2];
    uint32_t M;          // rows of A / rows of C        (== checker batch_size)
    uint32_t N;          // cols of B / cols of C
    uint32_t K;          // cols of A / rows of B        (== checker hidden_size)
    uint32_t tile_size;
    uint64_t A_addr;      // [M x K] FP32 "hidden states" tensor — also the checker's tap
                           // (checker narrows to FP16 itself in hardware; see VX_checker.sv)
    uint64_t B_addr;      // [K x N] FP32 weights
    uint64_t C_addr;      // [M x N] FP32 output
} kernel_arg_t;

#endif
