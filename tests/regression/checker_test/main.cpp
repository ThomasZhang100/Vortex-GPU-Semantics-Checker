#include <iostream>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <vortex.h>
#include "common.h"

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
vx_buffer_h src_buffer  = nullptr;
vx_buffer_h dst_buffer  = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
    std::cout << "Vortex checker dummy test." << std::endl;
    std::cout << "Usage: [-k kernel] [-n num_floats] [-h help]" << std::endl;
}

static void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:k:h")) != -1) {
        switch (c) {
        case 'n': num_elems = atoi(optarg); break;
        case 'k': kernel_file = optarg; break;
        case 'h': show_usage(); exit(0);
        default:  show_usage(); exit(-1);
        }
    }
}

void cleanup() {
    if (device) {
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

    uint32_t src_size = num_elems * sizeof(float);
    uint32_t dst_size = sizeof(float); // single task writes one sum

    RT_CHECK(vx_mem_alloc(device, src_size, VX_MEM_READ,  &src_buffer));
    RT_CHECK(vx_mem_address(src_buffer, &kernel_arg.src_addr));
    RT_CHECK(vx_mem_alloc(device, dst_size, VX_MEM_WRITE, &dst_buffer));
    RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));
    kernel_arg.num_elems = num_elems;

    // Print the device address of the src buffer.
    // This is the value you need to set as TAP_ADDR in VX_core.sv (or via CONFIGS)
    // if it does not match the current hardcoded default (0x20000000).
    printf("src (hidden-state) device addr = 0x%lx  size = %u bytes\n",
           (unsigned long)kernel_arg.src_addr, src_size);
    printf("TAP_ADDR in RTL is hardcoded to 0x20000000 — update if these differ.\n");

    // Fill src with known values: src[i] = (float)(i+1)
    std::vector<float> h_src(num_elems);
    float expected_sum = 0.0f;
    for (uint32_t i = 0; i < num_elems; ++i) {
        h_src[i] = (float)(i + 1);
        expected_sum += h_src[i];
    }

    RT_CHECK(vx_copy_to_dev(src_buffer, h_src.data(), 0, src_size));
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

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
