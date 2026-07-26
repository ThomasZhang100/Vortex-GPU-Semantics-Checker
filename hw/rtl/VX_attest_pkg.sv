// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Boot-attestation manifest layout — SV mirror of tests/regression/attest_test/
// manifest.h.  Every byte offset here MUST match that header exactly; the boot
// verifier uses these offsets to index the DCR-streamed manifest SRAM.  All
// fields little-endian; addresses/lengths 64-bit; hashes SHA-256 (32B); signature
// Ed25519 (64B) over bytes [0, ATTEST_SIG_OFF).  Verified against the C header's
// _Static_asserts (offsets identical).

package VX_attest_pkg;

    // Offsets/sizes mirror the C header for a complete, checkable contract; the
    // RTL verifier uses only a subset, so silence unused-parameter lint here.
    /* verilator lint_off UNUSEDPARAM */

    localparam ATTEST_MAGIC       = 32'h56544348;  // 'VTCH'
    localparam ATTEST_VERSION     = 32'h1;
    localparam ATTEST_HASH_BYTES  = 32;
    localparam ATTEST_SIG_BYTES   = 64;
    localparam ATTEST_MAX_LAYERS  = 16;

    // Phase-0 keyed-hash signature stand-in (mirror manifest.h ATTEST_SECRET_*).
    localparam ATTEST_SECRET_WORDS = 8;

    // Per-layer region entry: {base_addr[8], len[8], hash[32]} = 48 bytes.
    localparam ATTEST_REGION_BYTES = 48;

    // Scalar/field byte offsets (see manifest.h).
    localparam ATTEST_OFF_MAGIC        = 0;
    localparam ATTEST_OFF_VERSION      = 4;
    localparam ATTEST_OFF_NUM_REGIONS  = 8;
    localparam ATTEST_OFF_STARTUP      = 16;
    localparam ATTEST_OFF_STATUS       = 24;
    localparam ATTEST_OFF_TAP_ADDR     = 32;
    localparam ATTEST_OFF_SNOOP_LO     = 40;
    localparam ATTEST_OFF_SNOOP_HI     = 48;
    localparam ATTEST_OFF_HIDDEN_SIZE  = 56;
    localparam ATTEST_OFF_NUM_FEATURES = 58;
    localparam ATTEST_OFF_BATCH_SIZE   = 60;
    localparam ATTEST_OFF_KERNEL_ADDR  = 64;
    localparam ATTEST_OFF_KERNEL_LEN   = 72;
    localparam ATTEST_OFF_ARGS_ADDR    = 80;
    localparam ATTEST_OFF_ARGS_LEN     = 88;
    localparam ATTEST_OFF_SAE_W_HASH   = 96;
    localparam ATTEST_OFF_SAE_T_HASH   = 128;
    localparam ATTEST_OFF_KERNEL_HASH  = 160;
    localparam ATTEST_OFF_ARGS_HASH    = 192;
    localparam ATTEST_OFF_REGIONS      = 224;
    localparam ATTEST_OFF_SIG          = 992;

    localparam ATTEST_SIGNED_BYTES     = 992;   // bytes covered by the signature
    localparam ATTEST_MANIFEST_BYTES   = 1056;
    localparam ATTEST_MANIFEST_WORDS   = 264;   // 1056 / 4

    // Offset of weight_region[i].{base,len,hash} in bytes.
    function automatic int region_base_off(input int i);
        return ATTEST_OFF_REGIONS + i*ATTEST_REGION_BYTES + 0;
    endfunction
    function automatic int region_len_off(input int i);
        return ATTEST_OFF_REGIONS + i*ATTEST_REGION_BYTES + 8;
    endfunction
    function automatic int region_hash_off(input int i);
        return ATTEST_OFF_REGIONS + i*ATTEST_REGION_BYTES + 16;
    endfunction

    // Status words written to status_addr (poll-able from the host).
    localparam ATTEST_STATUS_BUSY   = 32'h00000000;
    localparam ATTEST_STATUS_PASS   = 32'h50415353;  // 'PASS'
    localparam ATTEST_STATUS_FAIL   = 32'h46410000;  // 'FA' << 16 | {layer[15:8], code[7:0]}

    // Fail reason codes (low byte of the FAIL status word).
    localparam ATTEST_FAIL_SIG    = 8'h01;
    localparam ATTEST_FAIL_MAGIC  = 8'h02;
    localparam ATTEST_FAIL_KERNEL = 8'h03;
    localparam ATTEST_FAIL_ARGS   = 8'h04;
    localparam ATTEST_FAIL_SAE_W  = 8'h05;
    localparam ATTEST_FAIL_SAE_T  = 8'h06;
    localparam ATTEST_FAIL_WEIGHT = 8'h07;

endpackage
