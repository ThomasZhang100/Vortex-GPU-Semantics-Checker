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

// Boot verifier (Task C).  On `start`, walks the DCR-streamed manifest and, for
// each attested region, streams its bytes through a single SHA-256 core and
// compares the digest to the manifest's expected hash.  Regions checked, in
// order: manifest signature (Phase-0 keyed hash), kernel, args, SAE weights, SAE
// thresholds, then each per-layer weight region.  On the first mismatch it stops
// with FAIL(code[,layer]); if all pass it asserts boot_release (deasserts the
// core-reset hold) and reports PASS.  It writes the resulting status word to the
// manifest's status_addr in VRAM (host polls it).
//
// Memory access is abstracted into word-granular sources so the module is
// self-contained and unit-testable.  VX_cluster wires:
//   - mf_*   -> manifest SRAM (synchronous read, 1-cycle latency)
//   - mem_*  -> shared L2 port (VRAM reads for kernel/args/weights + status write)
//   - sae_*  -> checker weight-SRAM / threshold-table read ports (time-shared)
//
// Phase 0 uses a keyed-hash signature stand-in (ATTEST_SECRET_*); Phase 1 drops
// in VX_ed25519_verify without changing this FSM or the manifest layout.
//
// MEM_ADDRW is the VRAM byte-address width; VX_cluster instantiates this with
// `MEM_ADDR_WIDTH.  The module otherwise has no VX_define dependency, so it lints
// and unit-tests standalone (with VX_attest_pkg + VX_sha256).

module VX_boot_verifier import VX_attest_pkg::*; #(
    parameter MEM_ADDRW     = 32,
    // Checker weight-SRAM geometry (must match the checker's MAX_FEATURES).  The
    // SAE weight hash covers hidden_size full physical rows of MAX_FEATURES FP16,
    // so the verifier's word index maps to the SRAM by pure bit-slicing (requires
    // MAX_FEATURES a power of two).  Hashing full rows (incl. unused columns, which
    // the host zero-fills) attests the actual SRAM image the checker will read.
    parameter CHK_MAX_FEATURES = 256
) (
    input  wire        clk,
    input  wire        reset,

    // Level-armed start (survives the vx_start reset pulse, like the checker's
    // ENABLE): verification begins on the rising edge of verify_armed after reset
    // deasserts.  This lets the verifier run inside processor::run()'s clock domain
    // (the host arms via a DCR before vx_start; run() resets then ticks while
    // boot_hold holds the cores, giving the verifier its cycles).
    input  wire        verify_armed,

    // Manifest SRAM read (synchronous: present mf_raddr, mf_rdata valid next cycle)
    output logic [$clog2(ATTEST_MANIFEST_WORDS)-1:0] mf_raddr,
    input  wire  [31:0] mf_rdata,

    // VRAM port (one outstanding request; byte-addressed, word granular).
    output logic        mem_req_valid,
    output logic        mem_req_rw,     // 0=read, 1=write (status)
    output logic [MEM_ADDRW-1:0] mem_req_addr,   // byte address, word-aligned
    output logic [31:0] mem_req_wdata,
    input  wire         mem_req_ready,
    input  wire         mem_rsp_valid,
    input  wire  [31:0] mem_rsp_data,

    // Checker SAE memory read (word granular): sel 0=weight SRAM, 1=threshold table.
    output logic        sae_req_valid,
    output logic        sae_req_sel,
    output logic [31:0] sae_req_addr,   // word index within the selected source
    input  wire         sae_req_ready,
    input  wire         sae_rsp_valid,
    input  wire  [31:0] sae_rsp_data,

    // Results
    output logic        boot_release,   // level: 1 once PASS (deassert boot_hold)
    output logic        done,           // level: 1 once finished (PASS or FAIL)
    output logic        pass,           // valid when done
    output wire         verifying,      // level: 1 while a verification is in progress
    output logic [31:0] status_word     // ATTEST_STATUS_* (also written to status_addr)
);
    localparam MFW = $clog2(ATTEST_MANIFEST_WORDS);

    // Word offsets of manifest fields (all field byte-offsets are 4-aligned).
    localparam int W_MAGIC     = ATTEST_OFF_MAGIC       / 4; // 0
    localparam int W_VERSION   = ATTEST_OFF_VERSION     / 4; // 1
    localparam int W_NUMREG    = ATTEST_OFF_NUM_REGIONS / 4; // 2
    localparam int W_STATUS    = ATTEST_OFF_STATUS      / 4; // 6 (lo)
    localparam int W_HIDDEN    = ATTEST_OFF_HIDDEN_SIZE / 4; // 14 (hidden[15:0], numfeat[31:16])
    localparam int W_KERNEL_A  = ATTEST_OFF_KERNEL_ADDR / 4; // 16
    localparam int W_KERNEL_L  = ATTEST_OFF_KERNEL_LEN  / 4; // 18
    localparam int W_ARGS_A    = ATTEST_OFF_ARGS_ADDR   / 4; // 20
    localparam int W_ARGS_L    = ATTEST_OFF_ARGS_LEN    / 4; // 22
    localparam int W_SAEW_HASH = ATTEST_OFF_SAE_W_HASH  / 4; // 24
    localparam int W_SAET_HASH = ATTEST_OFF_SAE_T_HASH  / 4; // 32
    localparam int W_KERNEL_H  = ATTEST_OFF_KERNEL_HASH / 4; // 40
    localparam int W_ARGS_H    = ATTEST_OFF_ARGS_HASH   / 4; // 48
    localparam int W_REGIONS   = ATTEST_OFF_REGIONS     / 4; // 56
    localparam int W_SIG       = ATTEST_OFF_SIG         / 4; // 248
    localparam int SIG_BODY_WORDS = ATTEST_SIGNED_BYTES / 4; // 248
    localparam int REGION_WORDS   = ATTEST_REGION_BYTES / 4; // 12

    // Phase-0 keyed-hash secret (mirror of manifest.h ATTEST_SECRET_*).
    localparam logic [31:0] SECRET [ATTEST_SECRET_WORDS] = '{
        32'h03020100, 32'h07060504, 32'h0b0a0908, 32'h0f0e0d0c,
        32'h13121110, 32'h17161514, 32'h1b1a1918, 32'h1f1e1d1c
    };

    typedef enum logic [1:0] { SRC_MF, SRC_VRAM, SRC_SAEW, SRC_SAET } src_t;
    typedef enum logic [2:0] { CK_SIG, CK_KERNEL, CK_ARGS, CK_SAEW, CK_SAET, CK_REGION } check_t;

    // -------------------------------------------------------------------------
    // SHA-256 core
    // -------------------------------------------------------------------------
    logic         sha_start, sha_in_valid, sha_in_last, sha_in_ready, sha_out_valid;
    logic [31:0]  sha_in_word;
    logic [2:0]   sha_in_last_bytes;
    logic [255:0] sha_digest;

    /* verilator lint_off PINCONNECTEMPTY */
    VX_sha256 sha (
        .clk(clk), .reset(reset),
        .start(sha_start),
        .in_valid(sha_in_valid), .in_word(sha_in_word),
        .in_last(sha_in_last), .in_last_bytes(sha_in_last_bytes),
        .in_ready(sha_in_ready),
        .out_valid(sha_out_valid), .digest(sha_digest), .busy() // busy unused here
    );
    /* verilator lint_on PINCONNECTEMPTY */

    // -------------------------------------------------------------------------
    // Shared word-read mechanism (one outstanding, source-muxed).
    // -------------------------------------------------------------------------
    logic        rd_go, rd_done;
    src_t        rd_src;
    logic [31:0] rd_windex, rd_data;
    logic [MEM_ADDRW-1:0] rd_base;

    typedef enum logic [1:0] { RD_IDLE, RD_MFWAIT, RD_ISSUE, RD_WAITRSP } rdstate_t;
    rdstate_t    rd_state;
    src_t        rd_src_l;
    logic [31:0] rd_windex_l;
    logic [MEM_ADDRW-1:0] rd_base_l;

    // -------------------------------------------------------------------------
    // Sequencer
    // -------------------------------------------------------------------------
    typedef enum logic [4:0] {
        S_IDLE,
        S_HDR_MAGIC, S_HDR_VER, S_HDR_NREG, S_HDR_STAT, S_HDR_HID,
        S_DISPATCH, S_SIG_SETUP,
        S_VR_A0, S_VR_A1, S_VR_L0, S_VR_L1,
        S_SAEW_SETUP, S_SAET_SETUP,
        S_ENG_START, S_ENG_REQ, S_ENG_WAIT, S_ENG_FEED, S_ENG_FIN,
        S_CMP_WAIT, S_CMP_STEP,
        S_NEXT, S_STATUS_REQ, S_STATUS_WR, S_DONE
    } state_t;
    state_t state;

    check_t      check;
    logic [$clog2(ATTEST_MAX_LAYERS)-1:0] region_i;
    logic [31:0] num_regions;

    // Engine params.
    src_t        eng_src;
    logic [31:0] eng_words, eng_bodywords, eng_j;
    logic [2:0]  eng_lastbytes;
    logic [MEM_ADDRW-1:0] eng_base;
    logic [MFW-1:0] eng_exp_woff;
    logic        eng_secret;
    logic [31:0] eng_word_l;
    logic [2:0]  cmp_i;
    logic        cmp_ok;

    logic [31:0] fld_lo;
    logic [MEM_ADDRW-1:0] status_addr;
    logic [15:0] hidden_size, num_features;
    logic [7:0]  fail_code, fail_layer;

    // Manifest word offsets for the current VRAM check's addr/len/hash.
    logic [MFW-1:0] base_woff, len_woff, hash_woff;
    always_comb begin
        base_woff = MFW'(W_KERNEL_A); len_woff = MFW'(W_KERNEL_L); hash_woff = MFW'(W_KERNEL_H);
        unique case (check)
            CK_ARGS: begin
                base_woff = MFW'(W_ARGS_A); len_woff = MFW'(W_ARGS_L); hash_woff = MFW'(W_ARGS_H);
            end
            CK_REGION: begin
                base_woff = MFW'(W_REGIONS + int'(region_i)*REGION_WORDS);
                len_woff  = MFW'(W_REGIONS + int'(region_i)*REGION_WORDS + 2);
                hash_woff = MFW'(W_REGIONS + int'(region_i)*REGION_WORDS + 4);
            end
            default: ; // CK_KERNEL uses defaults
        endcase
    end

    function automatic logic [31:0] bswap(input logic [31:0] w);
        return {w[7:0], w[15:8], w[23:16], w[31:24]};
    endfunction

    wire [31:0] digest_word = sha_digest[255 - 32*cmp_i -: 32];
    wire [31:0] secret_word = SECRET[eng_j - eng_bodywords];
    wire        eng_is_last = (eng_j == eng_words - 32'd1);
    wire        wr_fire     = (state == S_STATUS_WR) && mem_req_ready;

    // -------------------------------------------------------------------------
    // Memory-port combinational drive (reads via rd mechanism, plus status write).
    // -------------------------------------------------------------------------
    always_comb begin
        mem_req_valid = 1'b0; mem_req_rw = 1'b0; mem_req_addr = '0; mem_req_wdata = '0;
        sae_req_valid = 1'b0; sae_req_sel = 1'b0; sae_req_addr = '0;
        mf_raddr = rd_go ? MFW'(rd_windex) : MFW'(rd_windex_l);

        if (state == S_STATUS_WR) begin
            mem_req_valid = 1'b1;
            mem_req_rw    = 1'b1;
            mem_req_addr  = status_addr;
            mem_req_wdata = status_word;
        end else if (rd_state == RD_ISSUE) begin
            unique case (rd_src_l)
                SRC_VRAM: begin
                    mem_req_valid = 1'b1;
                    mem_req_addr  = rd_base_l + MEM_ADDRW'(rd_windex_l) * MEM_ADDRW'(4);
                end
                SRC_SAEW: begin sae_req_valid = 1'b1; sae_req_sel = 1'b0; sae_req_addr = rd_windex_l; end
                SRC_SAET: begin sae_req_valid = 1'b1; sae_req_sel = 1'b1; sae_req_addr = rd_windex_l; end
                default: ;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Read mechanism FSM
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            rd_state <= RD_IDLE;  rd_done <= 1'b0;
        end else begin
            rd_done <= 1'b0;
            unique case (rd_state)
                RD_IDLE: if (rd_go) begin
                    rd_src_l <= rd_src; rd_windex_l <= rd_windex; rd_base_l <= rd_base;
                    
                    rd_state <= (rd_src == SRC_MF) ? RD_MFWAIT : RD_ISSUE;
                end
                RD_MFWAIT: begin
                    rd_data <= mf_rdata; rd_done <= 1'b1;  rd_state <= RD_IDLE;
                end
                RD_ISSUE: begin
                    if (rd_src_l == SRC_VRAM) begin
                        if (mem_req_ready) rd_state <= RD_WAITRSP;
                    end else begin
                        if (sae_req_ready) rd_state <= RD_WAITRSP;
                    end
                end
                RD_WAITRSP: begin
                    if ((rd_src_l == SRC_VRAM && mem_rsp_valid) ||
                        (rd_src_l != SRC_VRAM && sae_rsp_valid)) begin
                        rd_data  <= (rd_src_l == SRC_VRAM) ? mem_rsp_data : sae_rsp_data;
                        rd_done  <= 1'b1;  rd_state <= RD_IDLE;
                    end
                end
                default: rd_state <= RD_IDLE;
            endcase
        end
    end

    // Helper: compute (words, lastbytes) from a byte length.
    function automatic logic [34:0] len_split(input logic [31:0] nbytes);
        logic [31:0] w;
        logic [2:0]  lb;
        w  = (nbytes + 32'd3) >> 2;
        lb = 3'(nbytes - ((w - 32'd1) << 2));   // 1..4
        return {lb, w};
    endfunction

    // Rising-edge-after-reset start (mirrors VX_checker's rearm): armed_r resets
    // to 0, so if verify_armed is already 1 when reset deasserts, start_i pulses
    // once.  Gated with !reset so a mid-reset DCR apply can't spuriously fire it.
    logic armed_r;
    always_ff @(posedge clk) begin
        if (reset) armed_r <= 1'b0;
        else       armed_r <= verify_armed;
    end
    wire start_i = verify_armed && !armed_r && !reset;

    // High while a verification is running (between start and DONE).  The cluster
    // folds this into `busy` so processor::run() keeps ticking during verification
    // and returns cleanly on FAIL (cores never boot) instead of hanging.
    assign verifying = (state != S_IDLE) && (state != S_DONE);

    // -------------------------------------------------------------------------
    // Main sequencer
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= S_IDLE; boot_release <= 1'b0; done <= 1'b0; pass <= 1'b0;
            status_word <= ATTEST_STATUS_BUSY; rd_go <= 1'b0; sha_start <= 1'b0; sha_in_valid <= 1'b0;
        end else begin
            rd_go <= 1'b0; sha_start <= 1'b0; sha_in_valid <= 1'b0;

            unique case (state)
                // ---- start ------------------------------------------------
                S_IDLE: if (start_i) begin
                    boot_release <= 1'b0; done <= 1'b0; pass <= 1'b0;
                    status_word  <= ATTEST_STATUS_BUSY;
                    fail_code <= '0; fail_layer <= '0;
                    rd_src <= SRC_MF; rd_windex <= 32'(W_MAGIC); rd_go <= 1'b1;
                    state  <= S_HDR_MAGIC;
                end
                // ---- header ------------------------------------------------
                S_HDR_MAGIC: if (rd_done) begin
                    if (rd_data != ATTEST_MAGIC) begin fail_code <= ATTEST_FAIL_MAGIC; state <= S_STATUS_REQ; end
                    else begin rd_src <= SRC_MF; rd_windex <= 32'(W_VERSION); rd_go <= 1'b1; state <= S_HDR_VER; end
                end
                S_HDR_VER: if (rd_done) begin
                    if (rd_data != ATTEST_VERSION) begin fail_code <= ATTEST_FAIL_MAGIC; state <= S_STATUS_REQ; end
                    else begin rd_src <= SRC_MF; rd_windex <= 32'(W_NUMREG); rd_go <= 1'b1; state <= S_HDR_NREG; end
                end
                S_HDR_NREG: if (rd_done) begin
                    num_regions <= rd_data;
                    rd_src <= SRC_MF; rd_windex <= 32'(W_STATUS); rd_go <= 1'b1; state <= S_HDR_STAT;
                end
                S_HDR_STAT: if (rd_done) begin
                    status_addr <= MEM_ADDRW'(rd_data);
                    rd_src <= SRC_MF; rd_windex <= 32'(W_HIDDEN); rd_go <= 1'b1; state <= S_HDR_HID;
                end
                S_HDR_HID: if (rd_done) begin
                    hidden_size  <= rd_data[15:0];
                    num_features <= rd_data[31:16];
                    check <= CK_SIG; region_i <= '0; state <= S_DISPATCH;
                end
                // ---- dispatch per check -----------------------------------
                S_DISPATCH: begin
                    unique case (check)
                        CK_SIG:  state <= S_SIG_SETUP;
                        CK_SAEW: state <= S_SAEW_SETUP;
                        CK_SAET: state <= S_SAET_SETUP;
                        default: begin // CK_KERNEL / CK_ARGS / CK_REGION -> fetch addr/len
                            rd_src <= SRC_MF; rd_windex <= 32'(base_woff); rd_go <= 1'b1;
                            state  <= S_VR_A0;
                        end
                    endcase
                end
                S_SIG_SETUP: begin
                    eng_src <= SRC_MF; eng_base <= '0;
                    eng_bodywords <= 32'(SIG_BODY_WORDS);
                    eng_words     <= 32'(SIG_BODY_WORDS) + 32'(ATTEST_SECRET_WORDS);
                    eng_lastbytes <= 3'd4; eng_secret <= 1'b1;
                    eng_exp_woff  <= MFW'(W_SIG);
                    state <= S_ENG_START;
                end
                // ---- VRAM addr/len fetch ----------------------------------
                S_VR_A0: if (rd_done) begin
                    fld_lo <= rd_data;
                    rd_src <= SRC_MF; rd_windex <= 32'(base_woff) + 32'd1; rd_go <= 1'b1; state <= S_VR_A1;
                end
                S_VR_A1: if (rd_done) begin
                    
                    eng_base <= MEM_ADDRW'({rd_data, fld_lo});
                    rd_src <= SRC_MF; rd_windex <= 32'(len_woff); rd_go <= 1'b1; state <= S_VR_L0;
                end
                S_VR_L0: if (rd_done) begin
                    fld_lo <= rd_data;
                    rd_src <= SRC_MF; rd_windex <= 32'(len_woff) + 32'd1; rd_go <= 1'b1; state <= S_VR_L1;
                end
                S_VR_L1: if (rd_done) begin
                    {eng_lastbytes, eng_words} <= len_split(fld_lo);
                    eng_src <= SRC_VRAM; eng_secret <= 1'b0; eng_exp_woff <= hash_woff;
                    state <= S_ENG_START;
                end
                // ---- SAE setups -------------------------------------------
                S_SAEW_SETUP: begin
                    // Full physical rows: hidden_size × MAX_FEATURES FP16 (2 bytes each).
                    {eng_lastbytes, eng_words} <= len_split(32'(hidden_size) * 32'(CHK_MAX_FEATURES) * 32'd2);
                    eng_src <= SRC_SAEW; eng_base <= '0; eng_secret <= 1'b0;
                    eng_exp_woff <= MFW'(W_SAEW_HASH);
                    state <= S_ENG_START;
                end
                S_SAET_SETUP: begin
                    {eng_lastbytes, eng_words} <= len_split((32'(num_features) + 32'd1) * 32'd2);
                    eng_src <= SRC_SAET; eng_base <= '0; eng_secret <= 1'b0;
                    eng_exp_woff <= MFW'(W_SAET_HASH);
                    state <= S_ENG_START;
                end
                // ---- hash engine ------------------------------------------
                S_ENG_START: begin sha_start <= 1'b1; eng_j <= '0; state <= S_ENG_REQ; end
                S_ENG_REQ: begin
                    if (eng_secret && (eng_j >= eng_bodywords)) begin
                        eng_word_l <= secret_word; state <= S_ENG_FEED;
                    end else begin
                        rd_src <= eng_src; rd_windex <= eng_j; rd_base <= eng_base; rd_go <= 1'b1;
                        state  <= S_ENG_WAIT;
                    end
                end
                S_ENG_WAIT: if (rd_done) begin eng_word_l <= rd_data; state <= S_ENG_FEED; end
                S_ENG_FEED: if (sha_in_ready) begin
                    sha_in_valid      <= 1'b1;
                    // Memory/SRAM words are little-endian; SHA-256 hashes bytes in
                    // stream order (first byte = MSB).  Byte-swap so the digest is
                    // over the actual memory byte order, letting the host compute
                    // the expected hash over the raw bytes.
                    sha_in_word       <= bswap(eng_word_l);
                    sha_in_last       <= eng_is_last;
                    sha_in_last_bytes <= eng_is_last ? eng_lastbytes : 3'd4;
                    eng_j             <= eng_j + 32'd1;
                    state             <= eng_is_last ? S_ENG_FIN : S_ENG_REQ;
                end
                S_ENG_FIN: if (sha_out_valid) begin
                    cmp_i <= '0; cmp_ok <= 1'b1;
                    rd_src <= SRC_MF; rd_windex <= 32'(eng_exp_woff); rd_go <= 1'b1;
                    state <= S_CMP_WAIT;
                end
                // ---- compare digest to expected 8 words -------------------
                S_CMP_WAIT: if (rd_done) begin
                    if (bswap(rd_data) != digest_word) cmp_ok <= 1'b0;
                    if (cmp_i == 3'd7) state <= S_CMP_STEP;
                    else begin
                        cmp_i     <= cmp_i + 3'd1;
                        rd_src    <= SRC_MF;
                        rd_windex <= 32'(eng_exp_woff) + 32'(cmp_i) + 32'd1;
                        rd_go     <= 1'b1;
                    end
                end
                S_CMP_STEP: begin
                    if (!cmp_ok) begin
                        unique case (check)
                            CK_SIG:    fail_code <= ATTEST_FAIL_SIG;
                            CK_KERNEL: fail_code <= ATTEST_FAIL_KERNEL;
                            CK_ARGS:   fail_code <= ATTEST_FAIL_ARGS;
                            CK_SAEW:   fail_code <= ATTEST_FAIL_SAE_W;
                            CK_SAET:   fail_code <= ATTEST_FAIL_SAE_T;
                            default:   begin fail_code <= ATTEST_FAIL_WEIGHT; fail_layer <= 8'(region_i); end
                        endcase
                        state <= S_STATUS_REQ;
                    end else state <= S_NEXT;
                end
                // ---- advance ----------------------------------------------
                S_NEXT: begin
                    unique case (check)
                        CK_SIG:    begin check <= CK_KERNEL; state <= S_DISPATCH; end
                        CK_KERNEL: begin check <= CK_ARGS;   state <= S_DISPATCH; end
                        CK_ARGS:   begin check <= CK_SAEW;   state <= S_DISPATCH; end
                        CK_SAEW:   begin check <= CK_SAET;   state <= S_DISPATCH; end
                        CK_SAET:   begin check <= CK_REGION; region_i <= '0;
                                         state <= (num_regions == 0) ? S_STATUS_REQ : S_DISPATCH; end
                        default: begin // CK_REGION
                            if (32'(region_i) + 32'd1 >= num_regions) state <= S_STATUS_REQ;
                            else begin region_i <= region_i + 1'b1; state <= S_DISPATCH; end
                        end
                    endcase
                end
                // ---- status write + finish --------------------------------
                S_STATUS_REQ: begin
                    if (fail_code == 0) status_word <= ATTEST_STATUS_PASS;
                    else status_word <= ATTEST_STATUS_FAIL | (32'(fail_layer) << 8) | 32'(fail_code);
                    state <= S_STATUS_WR;
                end
                S_STATUS_WR: if (wr_fire) begin
                    pass         <= (fail_code == 0);
                    boot_release <= (fail_code == 0);
                    done         <= 1'b1;
                    state        <= S_DONE;
                end
                S_DONE: if (start_i) state <= S_IDLE; // re-verify only on a fresh arm edge
                default: state <= S_IDLE;
            endcase
        end
    end

`ifdef SIMULATION
    // Debug trace of the verification walk (region setups + compare results + outcome).
    logic boot_release_q;
    always @(posedge clk) begin
        if (reset) boot_release_q <= 1'b0;
        else begin
            boot_release_q <= boot_release;
            if (start_i)
                $display("%0t: [VERIFY] START", $time);
            if (state == S_HDR_HID && rd_done)
                $display("%0t: [VERIFY] hdr numreg=%0d status_addr=0x%0h hidden=%0d numfeat=%0d",
                         $time, num_regions, status_addr, rd_data[15:0], rd_data[31:16]);
            if (state == S_ENG_START)
                $display("%0t: [VERIFY] check=%0d src=%0d base=0x%0h words=%0d",
                         $time, check, eng_src, eng_base, eng_words);
            if (state == S_CMP_STEP)
                $display("%0t: [VERIFY] check=%0d cmp_ok=%0b", $time, check, cmp_ok);
            if (state == S_STATUS_REQ)
                $display("%0t: [VERIFY] STATUS fail_code=0x%0h layer=%0d", $time, fail_code, fail_layer);
            if (state == S_STATUS_WR && mem_req_ready)
                $display("%0t: [VERIFY] wrote status=0x%0h -> addr=0x%0h", $time, status_word, status_addr);
            if (boot_release && !boot_release_q)
                $display("%0t: [VERIFY] BOOT_RELEASE (PASS)", $time);
        end
    end
`endif

endmodule
