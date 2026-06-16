#include <vx_spawn.h>
#include <string.h>
#include "common.h"

typedef float TYPE;

// Exact FP16 -> FP32 widening (every FP16 value is exactly representable in
// FP32, so this is a pure bit-pattern conversion, no rounding involved).
static inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // Subnormal fp16 -> normalize into fp32's wider exponent range.
            int32_t e = -1;
            do {
                mant <<= 1;
                ++e;
            } while (!(mant & 0x400u));
            mant &= 0x3FFu;
            bits = sign | ((uint32_t)(112 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13); // inf / nan
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Tiled/blocked sgemm (same algorithm as tests/regression/sgemm2), except
// matrix A is read as FP16 "hidden states" — the exact tensor the semantic
// checker is independently tapping off the same L2 region via its own DCR
// registers (see main.cpp). This kernel and the checker read the same bytes
// in parallel, mirroring the real scenario where a downstream GEMM consumes
// the hidden-state tensor the checker is monitoring.
void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto A_ptr = reinterpret_cast<const uint16_t*>(arg->A_addr);
    auto B_ptr = reinterpret_cast<const TYPE*>(arg->B_addr);
    auto C_ptr = reinterpret_cast<TYPE*>(arg->C_addr);

    // Local memory tile for A (converted to float) and B.
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
        // Load tile of A (FP16 -> float) & B (float) into local memory.
        local_A[l_row * tile_size + l_col] = fp16_to_float(A_ptr[g_row * K + (k + l_col)]);
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
