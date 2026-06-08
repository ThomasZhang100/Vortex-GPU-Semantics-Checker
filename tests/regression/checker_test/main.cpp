#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <vortex.h>
#include <VX_types.h>
#include "common.h"

// Checker parameters — must match VX_checker.sv B_TILE/hidden_size
#define B_TILE       4
#define HIDDEN_SIZE  64   // FP16 elements per token (small for simulation speed)

#define RT_CHECK(_expr)                                          \
   do {                                                          \
     int _ret = _expr;                                           \
     if (0 == _ret) break;                                       \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                  \
     exit(-1);                                                   \
   } while (false)

const char* kernel_file = "kernel.vxbin";
uint32_t num_elems = 16;

vx_device_h device      = nullptr;
vx_buffer_h act_buffer  = nullptr;   // FP16 activation tensor (B_TILE × HIDDEN_SIZE)
vx_buffer_h src_buffer  = nullptr;   // kernel float input
vx_buffer_h dst_buffer  = nullptr;   // kernel float output
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

// Minimal float→FP16 conversion (correct for integer-valued test data)
static uint16_t float_to_fp16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint16_t sign  = (bits >> 31) & 1;
    int32_t  exp32 = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant  = bits & 0x7FFFFF;
    if (exp32 < -24) return (uint16_t)(sign << 15);
    if (exp32 >  15) return (uint16_t)((sign << 15) | 0x7C00u);
    uint16_t exp16  = (uint16_t)(exp32 + 15);
    uint16_t mant16 = (uint16_t)(mant >> 13);
    return (uint16_t)((sign << 15) | (exp16 << 10) | mant16);
}

static void show_usage() {
    std::cout << "Vortex checker test." << std::endl;
    std::cout << "Usage: [-k kernel] [-n num_floats] [-h help]" << std::endl;
}

static void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:k:h")) != -1) {
        switch (c) {
        case 'n': num_elems = atoi(optarg);  break;
        case 'k': kernel_file = optarg;      break;
        case 'h': show_usage(); exit(0);
        default:  show_usage(); exit(-1);
        }
    }
}

void cleanup() {
    if (device) {
        vx_mem_free(act_buffer);
        vx_mem_free(src_buffer);
        vx_mem_free(dst_buffer);
        vx_mem_free(krnl_buffer);
        vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

int main(int argc, char* argv[]) {
    parse_args(argc, argv);

    RT_CHECK(vx_dev_open(&device));

    // --- Activation buffer: B_TILE tokens × HIDDEN_SIZE FP16 elements --------
    uint32_t act_size = B_TILE * HIDDEN_SIZE * sizeof(uint16_t);
    RT_CHECK(vx_mem_alloc(device, act_size, VX_MEM_READ, &act_buffer));
    uint64_t act_addr = 0;
    RT_CHECK(vx_mem_address(act_buffer, &act_addr));

    // Fill: token b, element k → fp16(b * HIDDEN_SIZE + k + 1)
    std::vector<uint16_t> h_act(B_TILE * HIDDEN_SIZE);
    for (int b = 0; b < B_TILE; ++b) {
        for (int k = 0; k < HIDDEN_SIZE; ++k) {
            float v = (float)(b * HIDDEN_SIZE + k + 1);
            h_act[b * HIDDEN_SIZE + k] = float_to_fp16(v);
        }
    }
    RT_CHECK(vx_copy_to_dev(act_buffer, h_act.data(), 0, act_size));

    printf("act_buffer: dev_addr=0x%lx  size=%u bytes  (%d tokens × %d FP16 elems)\n",
           (unsigned long)act_addr, act_size, B_TILE, HIDDEN_SIZE);

    // --- Kernel src/dst buffers (float sum kernel, unchanged) -----------------
    uint32_t src_size = num_elems * sizeof(float);
    uint32_t dst_size = sizeof(float);
    RT_CHECK(vx_mem_alloc(device, src_size, VX_MEM_READ,  &src_buffer));
    RT_CHECK(vx_mem_address(src_buffer, &kernel_arg.src_addr));
    RT_CHECK(vx_mem_alloc(device, dst_size, VX_MEM_WRITE, &dst_buffer));
    RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));
    kernel_arg.num_elems = num_elems;

    std::vector<float> h_src(num_elems);
    float expected_sum = 0.0f;
    for (uint32_t i = 0; i < num_elems; ++i) {
        h_src[i] = (float)(i + 1);
        expected_sum += h_src[i];
    }
    RT_CHECK(vx_copy_to_dev(src_buffer, h_src.data(), 0, src_size));
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

    // --- Arm the checker (trusted deployer window, before vx_start) -----------
    printf("Arming checker: act_addr=0x%lx  hidden_size=%d  batch_size=%d\n",
           (unsigned long)act_addr, HIDDEN_SIZE, B_TILE);
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TAP_ADDR0,
                          (uint32_t)(act_addr & 0xFFFFFFFFu)));
#ifdef XLEN_64
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TAP_ADDR1,
                          (uint32_t)(act_addr >> 32)));
#endif
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_HIDDEN_SIZE, HIDDEN_SIZE));
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_BATCH_SIZE,  B_TILE));
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_ENABLE, 1));

    printf("Launching kernel (num_elems=%u, expected_sum=%.1f)\n",
           num_elems, expected_sum);
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    float h_result = 0.0f;
    RT_CHECK(vx_copy_from_dev(&h_result, dst_buffer, 0, sizeof(float)));

    printf("GPU sum = %.1f  expected = %.1f\n", h_result, expected_sum);

    cleanup();

    if (fabsf(h_result - expected_sum) > 0.5f) {
        printf("FAILED!\n");
        return 1;
    }
    printf("PASSED!\n");
    return 0;
}
