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
const char* act_file     = nullptr; // -A: load FP32 A from binary file (overrides -o, mode 1 only)
const char* weight_file  = nullptr; // -W: FP16 SAE weight binary  [hidden × MAX_FEATURES], DCR-loaded
const char* thresh_file  = nullptr; // -C: uint16 threshold binary [num_features+1],   DCR-loaded
// -e 1 (default): immediate arm on ENABLE DCR write.
// -e 2: two-GEMM mode: GEMM1 (X×W1→A_hidden), then GEMM2 (A×B→C).
//        X and W1 are generated with fixed random seeds.
//        ENABLE DCR is written after GEMM1 completes so A is fully computed.
//        Checker trigger (e=3 sub-mode) fires on first B-matrix read in GEMM2.
// -e 3: address-range trigger mode.
static int ENABLE_MODE   = 1;
// -X: structural control.  Runs the GEMM(s) exactly as the chosen mode but skips
// ALL checker DCR writes (weights, thresholds, address, ENABLE), so with
// CHECKER_ENABLE the checker port exists in the L2 arbiter but never issues a
// request.  Isolates structural (port/arbiter-width) overhead from checker activity.
static bool NO_ARM       = false;
// -Z: cache-persistence verification (mode 2 only).  Points the checker tap at
// GEMM1's INPUT matrix X instead of the output A_hidden.  X is read by GEMM1 but
// NEVER read by GEMM2, so the only way X can be L2-resident when the checker
// reads it during GEMM2 is if it persisted from GEMM1.  Therefore:
//   CHK_L2_SUMMARY hit_pct high  -> persistence working
//   CHK_L2_SUMMARY hit_pct low   -> persistence not working (X cold-fetched)
// Removes the A_hidden confound where GEMM2's own core reads warm L2 regardless.
static bool TAP_X        = false;

vx_device_h device       = nullptr;
vx_buffer_h A_buffer     = nullptr;   // [M x K] FP32 "hidden states" — checker's tap
vx_buffer_h B_buffer     = nullptr;   // [K x N] FP32 GEMM2 weights
vx_buffer_h C_buffer     = nullptr;   // [M x N] FP32 output
vx_buffer_h X_buffer     = nullptr;   // [M x K] FP32 GEMM1 input  (mode 2 only)
vx_buffer_h W1_buffer    = nullptr;   // [K x K] FP32 GEMM1 weights (mode 2 only, random)
vx_buffer_h krnl_buffer  = nullptr;
vx_buffer_h args_buffer  = nullptr;   // kernel args for GEMM2 (or single GEMM)
vx_buffer_h args1_buffer = nullptr;   // kernel args for GEMM1 (mode 2 only)
kernel_arg_t kernel_arg  = {};

static void show_usage() {
    std::cout << "Vortex checker+sgemm2 test." << std::endl;
    std::cout << "Usage: [-k kernel] [-o ones_activation] [-A act_file.bin]" << std::endl;
    std::cout << "       [-W sae_weight_file.bin] [-C thresh_file.bin]" << std::endl;
    std::cout << "       [-e enable_mode]" << std::endl;
    std::cout << "       [-T num_tokens] [-F num_features] [-H hidden_size]" << std::endl;
    std::cout << "       [-N out_width] [-t tile_size] [-h help]" << std::endl;
    std::cout << "Enable modes:" << std::endl;
    std::cout << "  1: immediate arm (checker starts as soon as ENABLE DCR is written)" << std::endl;
    std::cout << "  2: two-GEMM + addr-range trigger:" << std::endl;
    std::cout << "       GEMM1: X(M*K) * W1(K*K) -> A_hidden  (random X and W1, fixed seeds)" << std::endl;
    std::cout << "       GEMM2: A_hidden(M*K) * B(K*N) -> C" << std::endl;
    std::cout << "       Checker taps A_hidden; ENABLE(3) written after GEMM1 so A is complete" << std::endl;
    std::cout << "       before the checker reads it; checker fires on first B-matrix L2 read" << std::endl;
    std::cout << "  3: addr-range trigger (single GEMM, checker waits for B-matrix L2 read)" << std::endl;
    std::cout << "  -X: structural control — configure the GEMM(s) exactly as the chosen" << std::endl;
    std::cout << "      mode but never write the checker ENABLE/config DCRs, so the checker" << std::endl;
    std::cout << "      hardware (and its L2 port) is present but fully idle.  Use with" << std::endl;
    std::cout << "      CHECKER_ENABLE to isolate the arbiter/port structural overhead from" << std::endl;
    std::cout << "      the checker's actual activity." << std::endl;
    std::cout << "  -Z: cache-persistence verification (mode 2 only) — tap GEMM1's input X" << std::endl;
    std::cout << "      instead of A_hidden.  X is read by GEMM1 but never by GEMM2, so a" << std::endl;
    std::cout << "      high CHK_L2 hit rate proves L2 persistence (no GEMM2-warming confound)." << std::endl;
}

static void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "k:oA:W:C:e:T:F:H:N:t:XZh")) != -1) {
        switch (c) {
        case 'k': kernel_file   = optarg;        break;
        case 'o': ones_activation = true;        break;
        case 'A': act_file      = optarg;        break;
        case 'W': weight_file   = optarg;        break;
        case 'C': thresh_file   = optarg;        break;
        case 'e': ENABLE_MODE   = atoi(optarg);  break;
        case 'X': NO_ARM        = true;          break;
        case 'Z': TAP_X         = true;          break;
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
        vx_mem_free(X_buffer);
        vx_mem_free(W1_buffer);
        vx_mem_free(krnl_buffer);
        vx_mem_free(args_buffer);
        vx_mem_free(args1_buffer);
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

    // -----------------------------------------------------------------------
    // Memory allocation: A (hidden states, checker tap), B (GEMM2 weights),
    // C (output), and for mode 2: X (GEMM1 input) and W1 (GEMM1 weights).
    // -----------------------------------------------------------------------
    uint32_t a_size = M * K * sizeof(float);
    // Mode 2: A is written by GEMM1 then read by GEMM2 + checker; needs both flags.
    // Modes 1/3: A is read-only from GPU's perspective (uploaded by host, read by kernel).
    int a_flags = (ENABLE_MODE == 2) ? (VX_MEM_READ | VX_MEM_WRITE) : VX_MEM_READ;
    RT_CHECK(vx_mem_alloc(device, a_size, a_flags, &A_buffer));
    uint64_t A_addr = 0;
    RT_CHECK(vx_mem_address(A_buffer, &A_addr));

    uint32_t b_size = K * N * sizeof(float);
    RT_CHECK(vx_mem_alloc(device, b_size, VX_MEM_READ, &B_buffer));
    uint64_t B_addr = 0;
    RT_CHECK(vx_mem_address(B_buffer, &B_addr));

    uint32_t c_size = M * N * sizeof(float);
    RT_CHECK(vx_mem_alloc(device, c_size, VX_MEM_WRITE, &C_buffer));
    uint64_t C_addr = 0;
    RT_CHECK(vx_mem_address(C_buffer, &C_addr));

    uint64_t X_addr  = 0;
    uint64_t W1_addr = 0;
    if (ENABLE_MODE == 2) {
        RT_CHECK(vx_mem_alloc(device, a_size, VX_MEM_READ, &X_buffer));
        RT_CHECK(vx_mem_address(X_buffer, &X_addr));
        uint32_t w1_size = K * K * sizeof(float);
        RT_CHECK(vx_mem_alloc(device, w1_size, VX_MEM_READ, &W1_buffer));
        RT_CHECK(vx_mem_address(W1_buffer, &W1_addr));
    }

    if (ENABLE_MODE == 2) {
        std::cout << "mode 2: two-GEMM sequence" << std::endl;
        std::cout << "  GEMM1: X(" << M << "x" << K << ") * W1(" << K << "x" << K
                  << ") -> A_hidden(" << M << "x" << K << ")" << std::endl;
        std::cout << "  GEMM2: A_hidden * B(" << K << "x" << N
                  << ") -> C(" << M << "x" << N << ")" << std::endl;
        std::cout << "  X_addr=0x"  << std::hex << X_addr
                  << "  W1_addr=0x" << W1_addr << std::dec << std::endl;
    } else {
        std::cout << "matrix A (hidden states): " << M << "x" << K << " (FP32)" << std::endl;
        std::cout << "matrix B (weights):       " << K << "x" << N << " (FP32)" << std::endl;
    }
    std::cout << "matrix C (output):   " << M << "x" << N << " (FP32)" << std::endl;
    std::cout << "tile size: " << TILE_SIZE << "x" << TILE_SIZE
              << "  local memory: " << local_mem << " bytes" << std::endl;
    std::cout << "A_addr=0x" << std::hex << A_addr << "  B_addr=0x" << B_addr
              << "  C_addr=0x" << C_addr << std::dec << std::endl;

    // -----------------------------------------------------------------------
    // Fill host-side matrices.
    // -----------------------------------------------------------------------

    // GEMM2 weights B (always random, seed 50).
    std::srand(50);
    std::vector<float> h_B(K * N);
    for (uint32_t i = 0; i < K * N; ++i)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // h_A: filled differently per mode.
    std::vector<float> h_A(M * K);
    std::vector<float> h_X, h_W1;

    if (ENABLE_MODE == 2) {
        // X: random (seed 51), W1: random (seed 52).
        // A = X * W1 is computed on GPU by GEMM1; CPU copy used for verification.
        h_X.resize(M * K);
        std::srand(51);
        for (uint32_t i = 0; i < M * K; ++i)
            h_X[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;

        h_W1.resize(K * K);
        std::srand(52);
        for (uint32_t i = 0; i < K * K; ++i)
            h_W1[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.1f;

        // CPU reference A = X * W1 (for verification and threshold calibration).
        matmul_cpu(h_A.data(), h_X.data(), h_W1.data(), M, K, K);
    } else {
        // Single GEMM: fill A from file / ones / ramp.
        if (act_file) {
            std::ifstream f(act_file, std::ios::binary);
            if (!f) { fprintf(stderr, "Error: cannot open act_file '%s'\n", act_file); cleanup(); exit(-1); }
            f.read(reinterpret_cast<char*>(h_A.data()), a_size);
            if (!f) { fprintf(stderr, "Error: short read from '%s'\n", act_file); cleanup(); exit(-1); }
        } else {
            for (uint32_t m = 0; m < M; ++m)
                for (uint32_t k = 0; k < K; ++k)
                    h_A[m * K + k] = ones_activation ? 1.0f : (float)(m * K + k + 1);
        }
    }

    // -----------------------------------------------------------------------
    // Upload data to device.
    // -----------------------------------------------------------------------
    if (ENABLE_MODE == 2) {
        std::cout << "upload GEMM1 input X" << std::endl;
        RT_CHECK(vx_copy_to_dev(X_buffer, h_X.data(), 0, a_size));
        std::cout << "upload GEMM1 weights W1" << std::endl;
        RT_CHECK(vx_copy_to_dev(W1_buffer, h_W1.data(), 0, K * K * sizeof(float)));
    } else {
        std::cout << "upload matrix A (hidden states)" << std::endl;
        RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, a_size));
    }

    std::cout << "upload matrix B (GEMM2 weights)" << std::endl;
    RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, b_size));

    // -----------------------------------------------------------------------
    // Upload kernel binary.
    // -----------------------------------------------------------------------
    std::cout << "Upload kernel binary" << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

    // -----------------------------------------------------------------------
    // Kernel args: one set for single-GEMM; two sets for mode 2.
    // -----------------------------------------------------------------------
    if (ENABLE_MODE == 2) {
        // GEMM1 args: X(M*K) * W1(K*K) -> A(M*K)
        kernel_arg_t arg1 = {};
        arg1.grid_dim[0]  = M / TILE_SIZE;
        arg1.grid_dim[1]  = K / TILE_SIZE;  // output cols = K (hidden dim)
        arg1.block_dim[0] = TILE_SIZE;
        arg1.block_dim[1] = TILE_SIZE;
        arg1.M = M;
        arg1.N = K;  // output width = hidden size (square W1)
        arg1.K = K;  // inner dim
        arg1.tile_size = TILE_SIZE;
        arg1.A_addr = X_addr;
        arg1.B_addr = W1_addr;
        arg1.C_addr = A_addr;  // write hidden states into A
        std::cout << "upload GEMM1 kernel args" << std::endl;
        RT_CHECK(vx_upload_bytes(device, &arg1, sizeof(kernel_arg_t), &args1_buffer));
    }

    // GEMM2 args (also the only args for modes 1/3).
    kernel_arg_t arg2 = {};
    arg2.grid_dim[0]  = M / TILE_SIZE;
    arg2.grid_dim[1]  = N / TILE_SIZE;
    arg2.block_dim[0] = TILE_SIZE;
    arg2.block_dim[1] = TILE_SIZE;
    arg2.M = M;
    arg2.N = N;
    arg2.K = K;
    arg2.tile_size = TILE_SIZE;
    arg2.A_addr = A_addr;
    arg2.B_addr = B_addr;
    arg2.C_addr = C_addr;
    std::cout << "upload GEMM2 kernel args" << std::endl;
    RT_CHECK(vx_upload_bytes(device, &arg2, sizeof(kernel_arg_t), &args_buffer));

    // -----------------------------------------------------------------------
    // Load checker SAE weights + thresholds (trusted deployer window).
    // DCR writes are always issued regardless of CHECKER_ENABLE in RTL:
    // without it the decoder silently drops them, so the GEMM runs unchanged.
    // -----------------------------------------------------------------------
    if (NO_ARM) {
        printf("NO_ARM (-X): skipping all checker DCR writes; checker stays idle "
               "(structural-overhead control)\n");
    }
    if (!NO_ARM && weight_file) {
        printf("Loading checker weights from '%s' (%d rows × %d features)\n",
               weight_file, K, VX_CHECKER_MAX_FEATURES);
        load_checker_weights(device, weight_file, K);
    }
    if (!NO_ARM && thresh_file) {
        printf("Loading checker thresholds from '%s' (%d entries)\n",
               thresh_file, NUM_FEATURES + 1);
        load_checker_thresholds(device, thresh_file, NUM_FEATURES);
    }

    // -----------------------------------------------------------------------
    // Configure checker address registers (tap=A, trigger=B range).
    // For modes 1 and 3 the ENABLE DCR is written now.
    // For mode 2 the ENABLE DCR is deferred until after GEMM1 completes so
    // the hidden states are fully written before the checker reads them.
    // Skipped entirely under -X (NO_ARM).
    // -----------------------------------------------------------------------
    if (!NO_ARM) {
        uint64_t trig_lo = B_addr;
        uint64_t trig_hi = B_addr + b_size;
        int arm_mode = (ENABLE_MODE == 2) ? 3 : ENABLE_MODE;  // mode 2 uses addr-trigger

        // Tap address: normally A_hidden (the monitored hidden states).
        // Under -Z (mode 2 verification): tap GEMM1's input X instead — X is
        // read by GEMM1 but never by GEMM2, so checker L2 hits there isolate
        // the persistence effect from GEMM2's own warming of A.
        uint64_t tap_addr = A_addr;
        if (TAP_X) {
            if (ENABLE_MODE != 2) {
                fprintf(stderr, "Error: -Z (tap X) is only valid in mode 2 (-e 2)\n");
                cleanup(); exit(-1);
            }
            tap_addr = X_addr;
        }

        printf("Arming checker: tap=0x%lx%s  hidden=%d  batch=%d  trig=[0x%lx,0x%lx)  mode=%d%s\n",
               (unsigned long)tap_addr, TAP_X ? " (X: persistence-verify)" : "",
               K, M,
               (unsigned long)trig_lo, (unsigned long)trig_hi, ENABLE_MODE,
               (ENABLE_MODE == 2) ? " (ENABLE deferred to after GEMM1)" : "");

        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TAP_ADDR0,
                              (uint32_t)(tap_addr & 0xFFFFFFFFu)));
#ifdef XLEN_64
        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TAP_ADDR1,
                              (uint32_t)(tap_addr >> 32)));
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

        if (ENABLE_MODE != 2) {
            // Modes 1 and 3: arm now.
            RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_ENABLE, arm_mode));
        }
    }

    // -----------------------------------------------------------------------
    // Execute kernel(s).
    // -----------------------------------------------------------------------
    if (ENABLE_MODE == 2) {
        std::cout << "start GEMM1 (X * W1 -> A_hidden)" << std::endl;
        RT_CHECK(vx_start(device, krnl_buffer, args1_buffer));
        std::cout << "wait for GEMM1 completion" << std::endl;
        RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

        // Arm checker with addr-trigger now that A_hidden is fully written.
        // Skipped under -X so the checker never fires (structural control).
        if (!NO_ARM) {
            printf("Arming checker after GEMM1: ENABLE=3 (addr-trigger on B reads)\n");
            RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_ENABLE, 3));
        }

        std::cout << "start GEMM2 (A_hidden * B -> C)" << std::endl;
        RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
        std::cout << "wait for GEMM2 completion" << std::endl;
        RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    } else {
        std::cout << "start device" << std::endl;
        RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
        std::cout << "wait for completion" << std::endl;
        RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    }

    // -----------------------------------------------------------------------
    // Download and verify C.
    // -----------------------------------------------------------------------
    std::vector<float> h_C(M * N);
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, c_size));

    std::cout << "verify result" << std::endl;
    int errors = 0;
    {
        // Reference: C_ref = A_ref * B where A_ref = X*W1 (mode 2) or h_A (mode 1/3).
        std::vector<float> h_ref(M * N);
        matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), M, N, K);
        for (uint32_t i = 0; i < h_ref.size(); ++i) {
            if (!compare_float(h_C[i], h_ref[i], i, errors))
                ++errors;
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
