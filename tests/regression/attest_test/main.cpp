// Boot-attestation end-to-end test (Task C).
//
// Flow: upload a GEMM (A*B->C) + its kernel/args + SAE weights/thresholds, build
// a SIGNED manifest over all of them (host computes the same SHA-256 the hardware
// verifier does), stream the manifest into the manifest SRAM, then let the boot
// verifier gate execution.  On a valid manifest the verifier releases the cores
// (fail-closed) and the GEMM runs; a tampered manifest keeps the cores held, so
// the GEMM never runs.
//
// The pass/fail signal is the GEMM output C, which is cache-independent: C is
// pre-filled with a sentinel written straight to DRAM, so "C == reference" proves
// the cores booted (attestation PASSED) and "C == sentinel" proves they stayed
// held (attestation FAILED).  The status word written to status_addr is read
// best-effort and printed.

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <vortex.h>
#include <VX_types.h>
#include "common.h"
#include "manifest.h"
#include "sha256.h"

static int M_TOKENS   = 8;    // M: rows of A / C   (checker batch_size)
static int HIDDEN     = 32;   // K: cols of A / rows of B (checker hidden_size)
static int OUT_WIDTH  = 16;   // N: cols of B / C
static int NUM_FEAT   = 8;    // SAE features
static int TILE_SIZE  = 4;
static const char* kernel_file = "kernel.vxbin";
static const char* tamper = "none"; // none|sig|magic|kernel|args|sae|thresh|weight

#define RT_CHECK(_expr) do { int _ret = _expr; if (0 == _ret) break; \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); cleanup(); exit(-1); } while(false)

vx_device_h device      = nullptr;
vx_buffer_h A_buffer     = nullptr, B_buffer = nullptr, C_buffer = nullptr;
vx_buffer_h status_buffer = nullptr, krnl_buffer = nullptr, args_buffer = nullptr;

void cleanup() {
    if (device) {
        vx_mem_free(A_buffer); vx_mem_free(B_buffer); vx_mem_free(C_buffer);
        vx_mem_free(status_buffer); vx_mem_free(krnl_buffer); vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

static void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "k:T:H:N:F:t:x:h")) != -1) {
        switch (c) {
        case 'k': kernel_file = optarg; break;
        case 'T': M_TOKENS   = atoi(optarg); break;
        case 'H': HIDDEN     = atoi(optarg); break;
        case 'N': OUT_WIDTH  = atoi(optarg); break;
        case 'F': NUM_FEAT   = atoi(optarg); break;
        case 't': TILE_SIZE  = atoi(optarg); break;
        case 'x': tamper     = optarg; break;
        case 'h': default:
            std::cout << "attest_test [-T tokens][-H hidden][-N out][-F feats][-t tile]\n"
                         "  -x none                                   valid manifest -> PASS, GEMM runs\n"
                         "  -x sig|magic|kernel|args|sae|thresh|weight  manifest-field tamper (caught by signature)\n"
                         "  -x weight0|weight1|argsdata|kerneldata     device-data tamper (caught by region hash)\n";
            exit(0);
        }
    }
}

static void matmul_cpu(float* C, const float* A, const float* B, int Mv, int Nv, int Kv) {
    for (int m = 0; m < Mv; ++m)
        for (int n = 0; n < Nv; ++n) {
            float s = 0.f;
            for (int k = 0; k < Kv; ++k) s += A[m*Kv+k] * B[k*Nv+n];
            C[m*Nv+n] = s;
        }
}

int main(int argc, char** argv) {
    parse_args(argc, argv);
    const int M = M_TOKENS, K = HIDDEN, N = OUT_WIDTH, F = NUM_FEAT;
    const int MAXF = VX_CHECKER_MAX_FEATURES;

    if ((M % TILE_SIZE) || (K % TILE_SIZE) || (N % TILE_SIZE)) {
        fprintf(stderr, "M,K,N must be multiples of tile_size %d\n", TILE_SIZE); return -1;
    }
    RT_CHECK(vx_dev_open(&device));

    // ---- allocate device buffers ----
    uint32_t a_sz = M*K*4, b_sz = K*N*4, c_sz = M*N*4;
    RT_CHECK(vx_mem_alloc(device, a_sz, VX_MEM_READ,  &A_buffer));
    RT_CHECK(vx_mem_alloc(device, b_sz, VX_MEM_READ,  &B_buffer));
    RT_CHECK(vx_mem_alloc(device, c_sz, VX_MEM_WRITE, &C_buffer));
    RT_CHECK(vx_mem_alloc(device, 4,    VX_MEM_WRITE, &status_buffer));
    uint64_t A_addr, B_addr, C_addr, status_addr;
    RT_CHECK(vx_mem_address(A_buffer, &A_addr));
    RT_CHECK(vx_mem_address(B_buffer, &B_addr));
    RT_CHECK(vx_mem_address(C_buffer, &C_addr));
    RT_CHECK(vx_mem_address(status_buffer, &status_addr));

    // ---- host matrices ----
    std::vector<float> h_A(M*K), h_B(K*N), h_ref(M*N);
    for (int i = 0; i < M*K; ++i) h_A[i] = (float)((i % 13) - 6) * 0.5f;
    std::srand(50);
    for (int i = 0; i < K*N; ++i) h_B[i] = (float)rand()/RAND_MAX;
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), M, N, K);

    RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, a_sz));
    RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, b_sz));

    // Sentinel-fill C so "unchanged" is detectable (goes straight to DRAM).
    std::vector<float> h_C(M*N, -123456.0f);
    RT_CHECK(vx_copy_to_dev(C_buffer, h_C.data(), 0, c_sz));

    // ---- SAE weights (K full rows × MAXF FP16) + thresholds (F+1 uint16) ----
    std::vector<uint8_t> sae_w((size_t)K*MAXF*2), sae_t((size_t)(F+1)*2);
    { uint32_t s = 0xC0FFEE; for (auto& b : sae_w) { s = s*1664525u+1013904223u; b = s>>24; } }
    { uint32_t s = 0x1234;   for (auto& b : sae_t) { s = s*22695477u+1u;       b = s>>24; } }
    // stream weights: K rows, MAXF/2 uint32 words per row
    for (int k = 0; k < K; ++k)
        for (int w = 0; w < MAXF/2; ++w) {
            uint32_t word; memcpy(&word, &sae_w[((size_t)k*MAXF/2 + w)*4], 4);
            RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_WEIGHT_DATA, word));
        }
    // stream thresholds
    for (int i = 0; i <= F; ++i) {
        uint16_t v; memcpy(&v, &sae_t[i*2], 2);
        RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_THRESH_DATA, v));
    }

    // ---- upload kernel; read .vxbin to learn load addr + code bytes ----
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    std::ifstream kf(kernel_file, std::ios::binary);
    if (!kf) { fprintf(stderr, "cannot open %s\n", kernel_file); cleanup(); return -1; }
    std::vector<uint8_t> kbytes((std::istreambuf_iterator<char>(kf)), {});
    if (kbytes.size() < 16) { fprintf(stderr, "kernel too small\n"); cleanup(); return -1; }
    uint64_t min_vma; memcpy(&min_vma, &kbytes[0], 8);
    uint64_t bin_size = kbytes.size() - 16;           // code region (excl. 16B header)
    const uint8_t* code = &kbytes[16];

    // ---- kernel args ----
    kernel_arg_t arg = {};
    arg.grid_dim[0]=M/TILE_SIZE; arg.grid_dim[1]=N/TILE_SIZE;
    arg.block_dim[0]=TILE_SIZE;  arg.block_dim[1]=TILE_SIZE;
    arg.M=M; arg.N=N; arg.K=K; arg.tile_size=TILE_SIZE;
    arg.A_addr=A_addr; arg.B_addr=B_addr; arg.C_addr=C_addr;
    RT_CHECK(vx_upload_bytes(device, &arg, sizeof(arg), &args_buffer));
    uint64_t args_addr; RT_CHECK(vx_mem_address(args_buffer, &args_addr));

    // ---- configure + arm the checker (addr-trigger on B reads; fires in-GEMM) ----
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TAP_ADDR0, (uint32_t)A_addr));
#ifdef XLEN_64
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TAP_ADDR1, (uint32_t)(A_addr>>32)));
#endif
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TRIG_ADDR_LO, (uint32_t)B_addr));
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TRIG_ADDR_HI, (uint32_t)(B_addr+b_sz)));
#ifdef XLEN_64
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TRIG_ADDR_LO1, (uint32_t)(B_addr>>32)));
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_TRIG_ADDR_HI1, (uint32_t)((B_addr+b_sz)>>32)));
#endif
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_HIDDEN_SIZE,  K));
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_BATCH_SIZE,   M));
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_NUM_FEATURES, F));
    RT_CHECK(vx_dcr_write(device, VX_DCR_CHECKER_ENABLE, 3)); // bit0=arm, bit1=addr-trigger

    // ---- build the signed manifest ----
    manifest_t mf; memset(&mf, 0, sizeof(mf));
    mf.magic = ATTEST_MAGIC; mf.version = ATTEST_VERSION;
    mf.startup_addr = min_vma; mf.status_addr = status_addr;
    mf.checker_tap_addr = A_addr; mf.checker_snoop_lo = B_addr; mf.checker_snoop_hi = B_addr+b_sz;
    mf.hidden_size = K; mf.num_features = F; mf.batch_size = M;
    mf.kernel_addr = min_vma; mf.kernel_len = bin_size;
    mf.args_addr = args_addr; mf.args_len = sizeof(arg);
    sha256(code, bin_size, mf.kernel_hash);
    sha256(&arg, sizeof(arg), mf.args_hash);
    sha256(sae_w.data(), sae_w.size(), mf.sae_weight_hash);
    sha256(sae_t.data(), sae_t.size(), mf.sae_thresh_hash);
    // two per-layer weight regions: split B by rows into halves
    int k_half = (K/2); // multiple of tile? not required for hashing, only addr split
    uint64_t len0 = (uint64_t)k_half*N*4, len1 = (uint64_t)(K-k_half)*N*4;
    mf.num_weight_regions = 2;
    mf.weight_region[0].base_addr = B_addr;        mf.weight_region[0].len = len0;
    sha256(&h_B[0], len0, mf.weight_region[0].hash);
    mf.weight_region[1].base_addr = B_addr + len0; mf.weight_region[1].len = len1;
    sha256(&h_B[(size_t)k_half*N], len1, mf.weight_region[1].hash);
    // Phase-0 signature = SHA256(body[0..992) || SECRET) into signature[0:32]
    {
        uint32_t sw[8] = {ATTEST_SECRET_W0,ATTEST_SECRET_W1,ATTEST_SECRET_W2,ATTEST_SECRET_W3,
                          ATTEST_SECRET_W4,ATTEST_SECRET_W5,ATTEST_SECRET_W6,ATTEST_SECRET_W7};
        SHA256 s; s.init();
        s.update((uint8_t*)&mf, ATTEST_SIGNED_BYTES);
        s.update((uint8_t*)sw, 32);
        s.fin(mf.signature);
    }

    // ---- optional tamper: two kinds ----
    //  (a) manifest-field tampers: flip a byte in the manifest. Because the Phase-0
    //      signature covers the whole manifest body, these are caught by the
    //      signature check (magic is caught even earlier at the header) — they test
    //      the outer seal, and all report FAIL(sig)/FAIL(magic).
    //  (b) data tampers: corrupt the DEVICE data (VRAM / uploaded kernel/args) AFTER
    //      the manifest is signed, so the signature stays valid but the region's
    //      recomputed hash mismatches — exercising the per-region hash checks and
    //      per-layer fail localization (weight0 -> layer 0, weight1 -> layer 1).
    bool expect_pass = true;
    if (strcmp(tamper,"none")) {
        expect_pass = false;
        // (a) manifest-field tampers
        if      (!strcmp(tamper,"sig"))    mf.signature[0]             ^= 0x5A;
        else if (!strcmp(tamper,"magic"))  mf.magic                     = 0xDEADBEEF;
        else if (!strcmp(tamper,"kernel")) mf.kernel_hash[0]           ^= 0xFF;
        else if (!strcmp(tamper,"args"))   mf.args_hash[0]             ^= 0xFF;
        else if (!strcmp(tamper,"sae"))    mf.sae_weight_hash[0]       ^= 0xFF;
        else if (!strcmp(tamper,"thresh")) mf.sae_thresh_hash[0]       ^= 0xFF;
        else if (!strcmp(tamper,"weight")) mf.weight_region[1].hash[0] ^= 0xFF;
        // (b) data tampers (manifest untouched; corrupt what's in the device)
        else if (!strcmp(tamper,"weight0")) {   // corrupt region 0 (first half of B)
            h_B[0] += 1.0f;
            RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, b_sz));
        } else if (!strcmp(tamper,"weight1")) { // corrupt region 1 (second half of B)
            h_B[(size_t)k_half * N] += 1.0f;
            RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, b_sz));
        } else if (!strcmp(tamper,"argsdata")) { // corrupt the uploaded kernel args
            kernel_arg_t bad = arg; bad.tile_size ^= 0xFFu;
            RT_CHECK(vx_copy_to_dev(args_buffer, &bad, 0, sizeof(bad)));
        } else if (!strcmp(tamper,"kerneldata")) { // corrupt the uploaded kernel code
            uint8_t bad = (uint8_t)(code[0] ^ 0xFF);
            RT_CHECK(vx_copy_to_dev(krnl_buffer, &bad, 0, 1));
        } else {
            fprintf(stderr, "unknown tamper '%s'\n", tamper); cleanup(); return -1;
        }
        printf("TAMPER: %s -> expecting attestation FAIL (GEMM must not run)\n", tamper);
    }

    // ---- stream the manifest into the manifest SRAM, then arm the verifier ----
    RT_CHECK(vx_dcr_write(device, VX_DCR_ATTEST_VERIFY_START, 0)); // rewind pointer
    const uint32_t* mw = reinterpret_cast<const uint32_t*>(&mf);
    for (int i = 0; i < ATTEST_MANIFEST_WORDS; ++i)
        RT_CHECK(vx_dcr_write(device, VX_DCR_ATTEST_MANIFEST_DATA, mw[i]));
    RT_CHECK(vx_dcr_write(device, VX_DCR_ATTEST_VERIFY_START, 1)); // arm

    printf("attest: kernel@0x%lx len=%lu  args@0x%lx len=%zu  B@0x%lx  status@0x%lx\n",
           (unsigned long)min_vma, (unsigned long)bin_size, (unsigned long)args_addr,
           sizeof(arg), (unsigned long)B_addr, (unsigned long)status_addr);

    // ---- launch: run() ticks through verification; on PASS cores run the GEMM,
    //      on FAIL busy drops (cores never boot) and run() returns. ----
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    // ---- read status (best-effort; write-back L2 may not have flushed on FAIL) ----
    uint32_t status = 0;
    vx_copy_from_dev(&status, status_buffer, 0, 4);

    // ---- primary signal: did the GEMM run? (C == reference vs sentinel) ----
    RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, c_sz));
    int diffs = 0;
    for (int i = 0; i < M*N; ++i) {
        float d = std::abs(h_C[i]-h_ref[i]), sc = std::max(std::abs(h_C[i]),std::abs(h_ref[i]));
        if (d > 1e-3f*sc + 1e-4f) ++diffs;
    }
    bool gemm_ran = (diffs == 0);

    printf("status=0x%08x  gemm_ran=%d  (expected %s)\n",
           status, gemm_ran, expect_pass ? "PASS/run" : "FAIL/blocked");

    cleanup();

    bool ok = expect_pass ? gemm_ran : !gemm_ran;
    if (ok) { printf("PASSED!\n"); return 0; }
    printf("FAILED!\n"); return 1;
}
