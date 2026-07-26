// Shared boot-attestation manifest layout (Task C).
//
// This is the single source of truth for the signed manifest that the host
// builds and the hardware boot verifier (VX_boot_verifier) parses.  The byte
// offsets here MUST stay in lockstep with hw/rtl/VX_attest_pkg.sv — the SV
// package mirrors every offset as a localparam.  Any change here requires the
// same change there.
//
// All multi-byte scalars are little-endian.  Addresses/lengths are 64-bit so the
// layout is identical for XLEN_32 and XLEN_64 (on XLEN_32 the host zero-fills the
// upper 32 bits).  The total size is a multiple of 4 bytes so the manifest can be
// streamed as 32-bit words over VX_DCR_ATTEST_MANIFEST_DATA.
//
// Hashes are SHA-256 (32 bytes, big-endian digest: byte 0 is the first hex byte).
// The signature is Ed25519 (64 bytes) over all preceding bytes (offset 0 ..
// offsetof(signature)).

#ifndef VX_ATTEST_MANIFEST_H
#define VX_ATTEST_MANIFEST_H

#include <stdint.h>

#define ATTEST_MAGIC        0x56544348u  // 'V''T''C''H' little-endian
#define ATTEST_VERSION      1u
#define ATTEST_HASH_BYTES   32
#define ATTEST_SIG_BYTES    64
#define ATTEST_MAX_LAYERS   16           // max per-layer weight regions

// -------------------------------------------------------------------------
// Phase-0 signature stand-in.  NOT real asymmetric crypto — a keyed hash used
// only to bring up the end-to-end verify mechanism.  Phase 1 replaces this with
// Ed25519 against a hardwired public key (VX_ed25519_verify.sv); the manifest
// layout and verifier FSM do not change.
//
//   signature[0:32] = SHA256( manifest_bytes[0 .. ATTEST_SIGNED_BYTES) ‖ SECRET )
//   signature[32:64] = 0 (unused in Phase 0)
//
// SECRET is a hardwired 32-byte constant shared by host (signer) and RTL
// (verifier).  Both must use the identical bytes.
#define ATTEST_SECRET_WORDS 8
#define ATTEST_SECRET_BYTES 32
// 8 little-endian 32-bit words = 32 secret bytes (byte 0 = 0x00 ... byte 31 = 0x1f
// when laid out as below, i.e. word w little-endian holds bytes 4w..4w+3).
#define ATTEST_SECRET_W0 0x03020100u
#define ATTEST_SECRET_W1 0x07060504u
#define ATTEST_SECRET_W2 0x0b0a0908u
#define ATTEST_SECRET_W3 0x0f0e0d0cu
#define ATTEST_SECRET_W4 0x13121110u
#define ATTEST_SECRET_W5 0x17161514u
#define ATTEST_SECRET_W6 0x1b1a1918u
#define ATTEST_SECRET_W7 0x1f1e1d1cu

// One attested memory region: base byte address, byte length, SHA-256 hash.
typedef struct {
    uint64_t base_addr;
    uint64_t len;
    uint8_t  hash[ATTEST_HASH_BYTES];
} attest_region_t;

// Fixed-layout signed manifest.  Field order == byte order (packed).
typedef struct {
    uint32_t magic;               // 0   : ATTEST_MAGIC
    uint32_t version;             // 4   : ATTEST_VERSION
    uint32_t num_weight_regions;  // 8   : N (<= ATTEST_MAX_LAYERS)
    uint32_t _pad0;               // 12  : reserved (keeps 64b fields 8-aligned)

    uint64_t startup_addr;        // 16  : verified kernel entry PC (released on PASS)
    uint64_t status_addr;         // 24  : VRAM addr the verifier writes status to
    uint64_t checker_tap_addr;    // 32  : hidden-state base addr for the checker
    uint64_t checker_snoop_lo;    // 40  : checker L2 snoop trigger range low (incl)
    uint64_t checker_snoop_hi;    // 48  : checker L2 snoop trigger range high (excl)

    uint16_t hidden_size;         // 56  : checker cfg — FP32 elems per token
    uint16_t num_features;        // 58  : checker cfg — SAE feature count
    uint16_t batch_size;          // 60  : checker cfg — token count
    uint16_t _pad1;               // 62  : reserved

    uint64_t kernel_addr;         // 64  : approved kernel binary region base
    uint64_t kernel_len;          // 72  : ... length
    uint64_t args_addr;           // 80  : approved kernel-args region base
    uint64_t args_len;            // 88  : ... length

    uint8_t sae_weight_hash[ATTEST_HASH_BYTES]; // 96
    uint8_t sae_thresh_hash[ATTEST_HASH_BYTES]; // 128
    uint8_t kernel_hash    [ATTEST_HASH_BYTES]; // 160
    uint8_t args_hash      [ATTEST_HASH_BYTES]; // 192

    attest_region_t weight_region[ATTEST_MAX_LAYERS]; // 224 (16 * 48 = 768)

    uint8_t signature[ATTEST_SIG_BYTES];        // 992 : Ed25519 over bytes [0..992)
} manifest_t;                                   // total = 1056 bytes = 264 words

// Byte offset where the signature begins == number of signed bytes.
#define ATTEST_SIGNED_BYTES   992
#define ATTEST_MANIFEST_BYTES 1056
#define ATTEST_MANIFEST_WORDS (ATTEST_MANIFEST_BYTES / 4)  // 264

// Status word values the verifier writes to status_addr (poll with vx_copy_from_dev).
#define ATTEST_STATUS_BUSY    0x00000000u
#define ATTEST_STATUS_PASS    0x50415353u  // 'PASS'
// FAIL packs a reason code in the low byte and (for weight/layer fails) the
// offending layer index in the next byte: 0x4641xxyy where yy=code, xx=layer.
#define ATTEST_STATUS_FAIL      0x46410000u // 'FA' << 16 | layer<<8 | code
#define ATTEST_FAIL_SIG       0x01u  // Ed25519 signature mismatch
#define ATTEST_FAIL_MAGIC     0x02u  // bad magic/version
#define ATTEST_FAIL_KERNEL    0x03u  // kernel hash mismatch
#define ATTEST_FAIL_ARGS      0x04u  // args hash mismatch
#define ATTEST_FAIL_SAE_W     0x05u  // SAE weight hash mismatch
#define ATTEST_FAIL_SAE_T     0x06u  // SAE threshold hash mismatch
#define ATTEST_FAIL_WEIGHT    0x07u  // a per-layer weight region hash mismatch

#endif // VX_ATTEST_MANIFEST_H
