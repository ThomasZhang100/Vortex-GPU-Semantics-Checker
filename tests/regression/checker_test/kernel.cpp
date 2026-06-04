#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto src = reinterpret_cast<float*>(arg->src_addr);
    auto dst = reinterpret_cast<float*>(arg->dst_addr);
    uint32_t N = arg->num_elems;

    // Read the entire src buffer. The checker is armed by the host via DCR
    // before kernel launch — no trigger instruction needed here.
    float sum = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        sum += src[i];
    }

    // Thread 0 of the single task writes the result.
    dst[blockIdx.x] = sum;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    // Single task: one thread sums the entire array.
    uint32_t grid_dim = 1;
    return vx_spawn_threads(1, &grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
