#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <vortex.h>
#include <VX_types.h>
#include "common.h"

// Checker / GEMM parameters — overridable via CLI flags.
static int NUM_TOKENS   = 8;   // M: rows of A / rows of C  (VX_DCR_CHECKER_BATCH_SIZE)
static int NUM_FEATURES = 32;  // total SAE features (VX_DCR_CHECKER_NUM_FEATURES)
static int HIDDEN_SIZE  = 64;  // K: cols of A / rows of B  (VX_DCR_CHECKER_HIDDEN_SIZE)
static int OUT_WIDTH    = 16;  // N: cols of B / cols of C
static int TILE_SIZE    = 4;   // sgemm2-style local-memory tile size

#define RT_CHECK(_expr)                                          \
   do {                                                          \
     int _ret = _expr;                                           \
     if (0 == _ret) break;                                       \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                  \
     exit(-1);                                                   \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

// ULP distance is a relative metric: 1 ULP at magnitude 0.02 is ~2e-9, so a
// 1e-6 absolute error is ~500 ULPs even though it looks tiny in decimal.
// With signed random A (mean≈0), dot-product results can be near zero (due to
// cancellation), making any fixed ULP budget fail even for numerically sound
// computations.  Use relative+absolute tolerance instead.
//   rel=1e-3  covers accumulation-order FP32 divergence across K=64 terms.
//   abs=1e-5  floor prevents false failures when both values are near zero.
static bool compare_float(float a, float b, int index, int errors) {
    float diff  = std::abs(a - b);
    float scale = std::max(std::abs(a), std::abs(b));
    if (diff > 1e-3f * scale + 1e-5f) {
        if (errors < 100) {
            printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
        }
        return false;
    }
    return true;
}

// Reference sgemm: C[M x N] = A[M x K] * B[K x N]. A is native FP32, the same
// tensor the checker taps (it narrows to FP16 itself in hardware — see
// VX_checker.sv's fp32_to_fp16 — so this reference doesn't need to model that).
static void matmul_cpu(float* C, const float* A, const float* B,
                        uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

const char* kernel_file  = "kernel.vxbin";
bool ones_activation     = false;   // -o: fill A with 1.0f for ground-truth check
const char* act_file     = nullptr; // -A: load FP32 A from binary file (overrides -o)
const char* weight_file  = nullptr; // -W: FP16 weight binary  [hidden × MAX_FEATURES], DCR-loaded
const char* thresh_file  = nullptr; // -C: uint16 threshold binary [num_features+1],   DCR-loaded
// -e 1 (default): immediate arm on ENABLE DCR write (legacy/test mode).
// -e 3: address-range trigger mode — checker waits for first L2 read in B-matrix range.
static int ENABLE_MODE   = 1;

vx_device_h device      = nullptr;
vx_buffer_h A_buffer    = nullptr;   // [M x K] FP32 "hidden states" — also the checker's tap
vx_buffer_h B_buffer    = nullptr;   // [K x N] FP32 weights
vx_buffer_h C_buffer    = nullptr;   // [M x N] FP32 output
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
    std::cout << "Vortex checker+sgemm2 test." << std::endl;
    std::cout << "Usage: [-k kernel] [-o ones_activation] [-A act_file.bin]" << std::endl;
    std::cout << "       [-W weight_file.bin] [-C thresh_file.bin]" << std::endl;
    std::cout << "       [-e enable_mode (1=immediate, 3=addr-trigger)]" << std::endl;
    std::cout << "       [-T num_tokens] [-F num_features] [-H hidden_size]" << std::endl;
    std::cout << "       [-N out_width] [-t tile_size] [-h help]" << std::endl;
}

static void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "k:oA:W:C:e:T:F:H:N:t:h")) != -1) {
        switch (c) {
        case 'k': kernel_file   = optarg;        break;
        case 'o': ones_activation = true;        break;
        case 'A': act_file      = optarg;        break;
        case 'W': weight_file   = optarg;        break;
        case 'C': thresh_file   = optarg;        break;
        case 'e': ENABLE_MODE   = atoi(optarg);  break;
        case 'T': NUM_TOKENS    = atoi(optarg);  break;
        case 'F': NUM_FEATURES  = atoi(optarg);  break;
        case 'H': HIDDEN_SIZE   = atoi(optarg);  break;
        case 'N': OUT_WIDTH     = atoi(optarg);  break;
        case 't': TILE_SIZE     = atoi(optarg);  break;
        case 'h': show_usage(); exit(0);
        default:  show_usage(); exit(-1);
        }
    }
}

void cleanup() {
    if (device) {
        vx_mem_free(A_buffer);
        vx_mem_free(B_buffer);
        vx_mem_free(C_buffer);
        vx_mem_free(krnl_buffer);
        vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

// Stream SAE weights into the checker's private SRAM via VX_DCR_CHECKER_WEIGHT_DATA.
// File format (written by run_tests.py write_weight_bin):
//   [HIDDEN_SIZE × VX_CHECKER_MAX_FEATURES] FP16 values, row-major, little-endian.
// Each SRAM row is VX_CHECKER_MAX_FEATURES FP16 = VX_CHECKER_MAX_FEATURES/2 uint32 words.
// VX_cluster.sv assembles 32 words into a 1024-bit buffer then pulses the SRAM write.
static void load_checker_weights(vx_device_h dev, const char* path, int hidden_size) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Error: cannot open weight file '%s'\n", path); cleanup(); exit(-1); }
    const int words_per_row = VX_CHECKER_MAX_FEATURES / 2;  // 2 FP16 per uint32
    for (int k = 0; k < hidden_size; k++) {
        for (int w = 0; w < words_per_row; w++) {
            uint32_t word = 0;
            f.read(reinterpret_cast<char*>(&word), sizeof(word));
            if (!f) { fprintf(stderr, "Error: short read from weight file at row %d word %d\n", k, w); cleanup(); exit(-1); }
            RT_CHECK(vx_dcr_write(dev, VX_DCR_CHECKER_WEIGHT_DATA, word));
        }
    }
}

// Stream per-feature thresholds into the checker via VX_DCR_CHECKER_THRESH_DATA.
// File format (written by run_tests.py write_thresh_bin):
//   [num_features+1] uint16, little-endian.
//   [0] = count_k (uint16),  [1..N] = per-feature FP16 activation thresholds.
static void load_checker_thresholds(vx_device_h dev, const char* path, int num_features) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Error: cannot open threshold file '%s'\n", path); cleanup(); exit(-1); }
    for (int i = 0; i <= num_features; i++) {
        uint16_t val = 0;
        f.read(reinterpret_cast<char*>(&val), sizeof(val));
        if (!f) { fprintf(stderr, "Error: short read from threshold file at index %d\n", i); cleanup(); exit(-1); }
        RT_CHECK(vx_dcr_write(dev, VX_DCR_CHECKER_THRESH_DATA, static_cast<uint32_t>(val)));
    }
}

int main(int argc, char* argv[]) {
    parse_args(argc, argv);

    if ((NUM_TOKENS  % TILE_SIZE) != 0 ||
        (HIDDEN_SIZE % TILE_SIZE) != 0 ||
        (OUT_WIDTH   % TILE_SIZE) != 0) {
        fprintf(stderr, "Error: num_tokens (%d), hidden_size (%d), and out_width (%d) "
                         "must all be multiples of tile_size (%d)\n",
                NUM_TOKENS, HIDDEN_SIZE, OUT_WIDTH, TILE_SIZE);
        exit(-1);
    }

    RT_CHECK(vx_dev_open(&device));

    uint32_t M = NUM_TOKENS;
    uint32_t K = HIDDEN_SIZE;
    uint32_t N = OUT_WIDTH;

    uint32_t group_size = TILE_SIZE * TILE_SIZE;
    uint32_t local_mem   = 2 * group_size * sizeof(float);
    uint32_t max_localmem;
    RT_CHECK(vx_check_occupancy(device, group_size, &max_localmem));
    std::cout << "occupancy: max_localmem=" << max_localmem << " bytes (need " << local_mem << ")" << std::endl;
    if (local_mem > max_localmem) {
        fprintf(stderr, "Error: tile_size=%d needs %u bytes of local memory, device supports only %u\n",
                TILE_SIZE, local_mem, max_localmem);
        cleanup();
        exit(-1);
    }

    // --- A buffer: [M x K] FP32 "hidden states" — both the GEMM input and the
    // exact tensor the checker independently taps off L2 (see arming below).
    // The checker narrows each element to FP16 itself, in hardware, right at
    // the L2 response (VX_checker.sv's fp32_to_fp16) — nothing here or in
    // kernel.cpp ever produces or consumes FP16.
    uint32_t a_size = M * K * sizeof(float);
    RT_CHECK(vx_mem_alloc(device, a_size, VX_MEM_READ, &A_buffer));
    uint64_t A_addr = 0;
    RT_CHECK(vx_mem_address(A_buffer, &A_addr));

    // --- B buffer: [K x N] FP32 weights ----------------------------------
    uint32_t b_size = K * N * sizeof(float);
    RT_CHECK(vx_mem_alloc(device, b_size, VX_MEM_READ, &B_buffer));
    uint64_t B_addr = 0;
    RT_CHECK(vx_mem_address(B_buffer, &B_addr));

    // --- C buffer: [M x N] FP32 output -------------------------------------
    uint32_t c_size = M * N * sizeof(float);
    RT_CHECK(vx_mem_alloc(device, c_size, VX_MEM_WRITE, &C_buffer));
    uint64_t C_addr = 0;
    RT_CHECK(vx_mem_address(C_buffer, &C_addr));

    std::cout << "matrix A (hidden states): " << M << "x" << K << " (FP32)" << std::endl;
    std::cout << "matrix B (weights):       " << K << "x" << N << " (FP32)" << std::endl;
    std::cout << "matrix C (output):        " << M << "x" << N << " (FP32)" << std::endl;
    std::cout << "tile size: " << TILE_SIZE << "x" << TILE_SIZE << "  local memory: " << local_mem << " bytes" << std::endl;
    std::cout << "A_addr=0x" << std::hex << A_addr << "  B_addr=0x" << B_addr
               << "  C_addr=0x" << C_addr << std::dec << std::endl;

    // Fill A: -A file -> load FP32 binary; -o -> 1.0f; default -> ramp.
    std::vector<float> h_A(M * K);
    if (act_file) {
        std::ifstream f(act_file, std::ios::binary);
        if (!f) { fprintf(stderr, "Error: cannot open act_file '%s'\n", act_file); cleanup(); exit(-1); }
        f.read(reinterpret_cast<char*>(h_A.data()), a_size);
        if (!f) { fprintf(stderr, "Error: short read from '%s' (expected %u bytes)\n", act_file, a_size); cleanup(); exit(-1); }
    } else {
        for (uint32_t m = 0; m < M; ++m)
            for (uint32_t k = 0; k < K; ++k) {
                h_A[m * K + k] = ones_activation ? 1.0f : (float)(m * K + k + 1);
            }
    }

    // Fill B with random floats.
    std::srand(50);
    std::vector<float> h_B(K * N);
    for (uint32_t i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    std::cout << "upload matrix A (hidden states)" << std::endl;
    RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, a_size));

    std::cout << "upload matrix B (weights)" << std::endl;
    RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, b_size));

    kernel_arg.grid_dim[0]  = M / TILE_SIZE;
    kernel_arg.grid_dim[1]  = N / TILE_SIZE;
    kernel_arg.block_dim[0] = TILE_SIZE;
    kernel_arg.block_dim[1] = TILE_SIZE;
    kernel_arg.M = M;
    kernel_arg.N = N;
    kernel_arg.K = K;
    kernel_arg.tile_size = TILE_SIZE;
    kernel_arg.A_addr = A_addr;
    kernel_arg.B_addr = B_addr;
    kernel_arg.C_addr = C_addr;

    std::cout << "Upload kernel binary" << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

    std::cout << "upload kernel argument" << std::endl;
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

    // --- Load checker weights + thresholds via DCR (trusted deployer window) ---
    // DCR writes are always issued regardless of whether CHECKER_ENABLE is set in
    // the RTL build: without it the decoder silently drops writes to undefined
    // checker addresses, so the GEMM runs unchanged.  With CHECKER_ENABLE the
    // RTL acts on them to arm the checker.  This avoids a host/RTL CONFIGS skew
    // that would leave the checker instantiated but never armed.
    if (weight_file) {
        printf("Loading checker weights from '%s' (%d rows × %d features)\n",
               weight_file, K, VX_CHECKER_MAX_FEATURES);
        load_checker_weights(device, weight_file, K);
    }
    if (thresh_file) {
        printf("Loading checker thresholds from '%s' (%d entries)\n",
               thresh_file, NUM_FEATURES + 1);
        load_checker_thresholds(device, thresh_file, NUM_FEATURES);
    }

    // --- Arm the checker (trusted deployer window, before vx_start) -----------
    // Tap address = A (hidden states).  Trigger range = B (the unembedding proxy):
    // in address-trigger mode (ENABLE_MODE=3) the checker waits for the first
    // L2 read of any byte in [B_addr, B_addr+b_size) before starting the SAE matmul.
    // In immediate mode (ENABLE_MODE=1) the checker starts as soon as ENABLE is written.
    {
        uint64_t trig_lo = B_addr;
        uint64_t trig_hi = B_addr + b_size;
        printf("Arming checker: tap=0x%lx  hidden=%d  batch=%d  trig=[0x%lx,0x%lx)  mode=%d\n",
               (unsigned long)A_addr, K, M,
               (unsigned long)trig_lo, (unsigned long)trig_hi, ENABLE_MODE);

        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TAP_ADDR0,
                              (uint32_t)(A_addr & 0xFFFFFFFFu)));
#ifdef XLEN_64
        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TAP_ADDR1,
                              (uint32_t)(A_addr >> 32)));
#endif
        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TRIG_ADDR_LO,
                              (uint32_t)(trig_lo & 0xFFFFFFFFu)));
        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TRIG_ADDR_HI,
                              (uint32_t)(trig_hi & 0xFFFFFFFFu)));
#ifdef XLEN_64
        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TRIG_ADDR_LO1,
                              (uint32_t)(trig_lo >> 32)));
        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TRIG_ADDR_HI1,
                              (uint32_t)(trig_hi >> 32)));
#endif
        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_HIDDEN_SIZE,  K));
        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_BATCH_SIZE,   M));
        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_NUM_FEATURES, NUM_FEATURES));
        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_ENABLE, ENABLE_MODE));
    }

    std::cout << "start device" << std::endl;
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    std::vector<float> h_C(M * N);
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, c_size));

    // verify result
    std::cout << "verify result" << std::endl;
    int errors = 0;
    {
        std::vector<float> h_ref(M * N);
        matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), M, N, K);

        for (uint32_t i = 0; i < h_ref.size(); ++i) {
            if (!compare_float(h_C[i], h_ref[i], i, errors)) {
                ++errors;
            }
        }
    }

    cleanup();

    if (errors != 0) {
        std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return errors;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}
