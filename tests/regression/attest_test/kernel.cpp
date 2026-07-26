#include <vx_spawn.h>
#include "common.h"

typedef float TYPE;

// Tiled/blocked sgemm (same algorithm as tests/regression/sgemm2). Matrix A
// is read in its native FP32 form — the exact tensor the semantic checker is
// independently tapping off the same L2 region via its own DCR registers
// (see main.cpp). The checker narrows to FP16 itself in hardware
// (VX_checker.sv's fp32_to_fp16), so the kernel never touches FP16 at all —
// it reads the same bytes in parallel, mirroring the real scenario where a
// downstream GEMM consumes the hidden-state tensor the checker is monitoring.
void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto A_ptr = reinterpret_cast<const TYPE*>(arg->A_addr);
    auto B_ptr = reinterpret_cast<const TYPE*>(arg->B_addr);
    auto C_ptr = reinterpret_cast<TYPE*>(arg->C_addr);

    // Local memory tile for A and B.
    auto local_ptr = __local_mem(2 * blockDim.x * blockDim.y * sizeof(TYPE));
    auto local_A = (TYPE*)local_ptr;
    auto local_B = (TYPE*)local_ptr + blockDim.x * blockDim.y;

    auto N = arg->N;
    auto K = arg->K;
    auto tile_size = arg->tile_size;

    // Global row/col of the output element this thread computes.
    auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
    auto g_col = blockIdx.y * blockDim.y + threadIdx.y;

    // Local row/col within the tile.
    auto l_row = threadIdx.x;
    auto l_col = threadIdx.y;

    TYPE sum(0);

    // Loop over tiles along K.
    for (uint32_t k = 0; k < K; k += tile_size) {
        // Load tile of A & B into local memory.
        local_A[l_row * tile_size + l_col] = A_ptr[g_row * K + (k + l_col)];
        local_B[l_row * tile_size + l_col] = B_ptr[(k + l_row) * N + g_col];

        __syncthreads();

        // Compute partial sum for the local tile.
        for (uint32_t j = 0; j < tile_size; ++j) {
            sum += local_A[l_row * tile_size + j] * local_B[j * tile_size + l_col];
        }

        __syncthreads();
    }

    C_ptr[g_row * N + g_col] = sum;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
