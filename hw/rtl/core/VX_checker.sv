// Semantic checker — tiled SAE matmul over a 4×16 output-stationary systolic array.
//
// Tiling: supports num_features > N_FEAT and batch_size > B_TILE via multiple passes.
//   feat_tiles  = ceil(num_features / N_FEAT)   — passes over feature dimension
//   batch_tiles = ceil(batch_size  / B_TILE)    — passes over batch dimension
// Total passes = feat_tiles × batch_tiles.  Outer loop = batch_tile, inner = feat_tile.
//
// Each pass: loads B_TILE hidden-state rows from L2 (same rows for all feat_tiles in a
// batch_tile), runs them through sa_array with the current N_FEAT weight column slice,
// then on scan_done compares outputs against per-feature thresholds on the fly.
//
// feat_count[b] accumulates per-batch_tile and ANDs across all feat_tile passes.
// flag_o[bt*B_TILE+b] is set after the last feat_tile if feat_count > threshold[0] (k).
//
// Weight SRAM: MAX_HIDDEN rows × (MAX_FEATURES × 16) bits.  Each row holds all features
// for one k value; feat_tile selects the N_FEAT-column slice before the hpipe.
// Thresholds: threshold[0] = count-k (uint16); threshold[1..F] = per-feature FP16 values.
//             Global feature index = feat_tile*N_FEAT + col_i → threshold index + 1.
//
// pass_reset  = rearm | next_pass        — restarts FIFOs/drain/scan each pass.
// batch_reset = rearm | (next_pass & last_feat_tile) — resets feat_count at batch boundary.
//
// Enable: -DCHECKER_ENABLE
// DCRs:   VX_DCR_CHECKER_ENABLE       (rising edge arms checker)
//         VX_DCR_CHECKER_TAP_ADDR0/1  (hidden-state base address)
//         VX_DCR_CHECKER_HIDDEN_SIZE  (FP32 elements per token)
//         VX_DCR_CHECKER_BATCH_SIZE   (total tokens, ≤ MAX_BATCH)
//         VX_DCR_CHECKER_NUM_FEATURES (total SAE features, ≤ MAX_FEATURES)
//         VX_DCR_CHECKER_WEIGHT_DATA  (stream 32b words into weight SRAM; 32 writes = one row)
//         VX_DCR_CHECKER_THRESH_DATA  (stream uint16 into threshold[]; auto-advance)
//
// Activations arrive in their native FP32 form (the core's FPU has no FP16
// datapath) and are narrowed to FP16 by fp32_to_fp16() right at the L2
// response, before anything is pushed into the per-row FIFOs. The systolic
// array and weight SRAM stay FP16 — only the ingest path widens to 4B/elem.
// This keeps the narrowing inside checker-owned hardware: the model's kernel
// is never trusted to downcast its own activations correctly.

`include "VX_define.vh"

module VX_checker import VX_gpu_pkg::*; #(
    parameter B_TILE        = 4,               // systolic array rows  (fixed)
    parameter N_FEAT        = 8,               // systolic array cols  (fixed)
    parameter MAX_HIDDEN    = 2048,            // max hidden_size (SRAM depth)
    parameter MAX_FEATURES  = 64,             // max total SAE features; must be multiple of N_FEAT
    parameter MAX_BATCH     = 16,             // max total batch size; must be multiple of B_TILE
    parameter FIFO_DEPTH    = 32,              // FP16 slots per activation FIFO row (2 cache lines; min to avoid refill overflow with 1 chunk in flight)
    parameter LINE_WORDS    = `L1_LINE_SIZE/4, // FP32 values per cache line (64B/4B=16)
    parameter `STRING WEIGHT_FILE    = "",     // $readmemh hex (MAX_HIDDEN × MAX_FEATURES FP16)
    parameter `STRING THRESHOLD_FILE = ""      // $readmemh hex (MAX_FEATURES FP16 thresholds)
) (
    input  wire clk,
    input  wire reset,

    // DCR-supplied config (latched in VX_cluster before vx_start)
    input  wire                         checker_armed,
    input  wire [`MEM_ADDR_WIDTH-1:0]   hidden_base_addr,
    input  wire [15:0]                  hidden_size,    // FP32 elements per token
    input  wire [15:0]                  num_features,   // actual number of SAE features
    input  wire [15:0]                  batch_size,     // total tokens across all batch tiles

    // Per-row flags: flag_o[bt*B_TILE+b]=1 if feat_count > threshold[0] (count-k).
    // Rows at or beyond batch_size are always 0.
    output wire [MAX_BATCH-1:0]         flag_o,

    // Dedicated L2 port for activation prefetch
    VX_mem_bus_if.master                act_bus_if,

    // Address-range trigger from VX_cluster.sv L2 snoop.
    // trigger_i: single-cycle pulse when a qualifying L2 read lands in [TRIG_LO, TRIG_HI).
    // addr_trig_en_i: selects trigger mode (ENABLE DCR bit 1).
    //   0 → immediate arm on rising edge of checker_armed (legacy / test mode)
    //   1 → arm only on trigger_i (paper mechanism: unembedding-read-triggered)
    input wire                                trigger_i,
    input wire                                addr_trig_en_i,

    // Weight SRAM write port — driven by VX_cluster.sv's DCR streaming state.
    // weight_we_i pulses for one cycle after the 32nd WEIGHT_DATA DCR word;
    // weight_waddr_i and weight_wdata_i are stable at that edge.
    input wire                                weight_we_i,
    input wire [`CLOG2(MAX_HIDDEN)-1:0]       weight_waddr_i,
    input wire [MAX_FEATURES*16-1:0]          weight_wdata_i,

    // Threshold write port — one uint16 per THRESH_DATA DCR write.
    input wire                                thresh_we_i,
    input wire [`CLOG2(MAX_FEATURES+2)-1:0]   thresh_waddr_i,
    input wire [15:0]                         thresh_wdata_i
);
    // -------------------------------------------------------------------------
    // Localparams
    // -------------------------------------------------------------------------
    localparam LINE_BYTES      = `L1_LINE_SIZE;
    localparam LINE_BITS       = `CLOG2(LINE_BYTES);
    localparam LOG_LW          = `CLOG2(LINE_WORDS);
    localparam FIFO_PTR_W      = `CLOG2(FIFO_DEPTH);
    localparam FIFO_CTR_W      = `CLOG2(FIFO_DEPTH + 1);
    localparam FIFO_HALF       = FIFO_DEPTH / 2;
    localparam CHUNKS_W        = 12;
    localparam ROW_ID_BITS     = `CLOG2(B_TILE);

    localparam TAG_VAL_W       = L2_TAG_WIDTH - `UP(UUID_WIDTH);
    localparam WEIGHT_DATAW    = MAX_FEATURES * 16;    // full SRAM row width
    localparam TILE_DATAW      = N_FEAT * 16;          // hpipe / col_in slice width
    localparam WEIGHT_ADDRW    = `CLOG2(MAX_HIDDEN);

    localparam HPIPE_DEPTH     = N_FEAT - 1;
    localparam VPIPE_DEPTH     = B_TILE - 1;
    localparam PE_LATENCY      = 1;
    localparam DRAIN_CYCLES    = HPIPE_DEPTH + VPIPE_DEPTH + PE_LATENCY;
    localparam DRAIN_CTR_W     = `CLOG2(DRAIN_CYCLES + 2);
    localparam ACC_W           = 16;

    localparam SCAN_INIT       = N_FEAT + B_TILE - 1;
    localparam SCAN_CTR_W      = `CLOG2(SCAN_INIT + 2);
    localparam FEAT_COUNT_W    = `CLOG2(MAX_FEATURES + 1); // counts 0..MAX_FEATURES

    localparam MAX_FEAT_TILES  = MAX_FEATURES / N_FEAT;
    localparam MAX_BATCH_TILES = MAX_BATCH    / B_TILE;
    localparam FEAT_TILE_W     = `CLOG2(MAX_FEAT_TILES  + 1);
    localparam BATCH_TILE_W    = `CLOG2(MAX_BATCH_TILES + 1);

    // COL_CNT_W: per-row scan column counter.  Values 0..N_FEAT-1 are valid;
    // N_FEAT is the sentinel meaning "this row's scan window has not started yet
    // or has already finished."  Needs one extra bit beyond log2(N_FEAT).
    localparam COL_CNT_W       = `CLOG2(N_FEAT + 1);  // 5 bits for N_FEAT=16
    // GF_W: global-feature-index width.  gf = {feat_tile, col_cnt[3:0]}; max = MAX_FEATURES-1.
    localparam GF_W            = FEAT_TILE_W + `CLOG2(N_FEAT);

    // -------------------------------------------------------------------------
    // Rising-edge detector for checker_armed
    // -------------------------------------------------------------------------
    logic armed_r;
    always_ff @(posedge clk) begin
        if (reset) armed_r <= 0;
        else       armed_r <= checker_armed;
    end
    // Immediate mode (addr_trig_en_i=0): rearm on rising edge of checker_armed DCR.
    // Address-trigger mode (addr_trig_en_i=1): rearm only when VX_cluster.sv's L2
    // snoop detects a read in [TRIG_LO, TRIG_HI) and pulses trigger_i.
    wire rearm = (!addr_trig_en_i && checker_armed && !armed_r) || trigger_i;

    // -------------------------------------------------------------------------
    // State machine: IDLE → ACTIVE (on rearm) → DONE (all passes complete)
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] { IDLE = 2'b00, ACTIVE = 2'b01, DONE = 2'b10 } state_t;
    state_t state;

    // -------------------------------------------------------------------------
    // Tile counters (only reset on rearm, not per-pass)
    // -------------------------------------------------------------------------
    logic [FEAT_TILE_W-1:0]  feat_tile,  feat_tiles;
    logic [BATCH_TILE_W-1:0] batch_tile, batch_tiles;

    wire last_feat_tile  = (feat_tile  == feat_tiles  - FEAT_TILE_W'(1));
    wire last_batch_tile = (batch_tile == batch_tiles - BATCH_TILE_W'(1));

    // Scan-done pulse is defined later; forward-declare for use in next_pass.
    logic scan_done_pulse;

    wire next_pass  = scan_done_pulse && !(last_feat_tile && last_batch_tile);
    wire pass_reset = rearm || next_pass;
    wire batch_reset = rearm || (next_pass && last_feat_tile);

    // Combinational next value of batch_tile: used to precompute row addresses
    // at the same posedge batch_reset fires, before the NBA commits batch_tile.
    wire [BATCH_TILE_W-1:0] batch_tile_next =
        rearm                         ? '0 :
        (next_pass && last_feat_tile) ? batch_tile + BATCH_TILE_W'(1) :
                                        batch_tile;

    // One-cycle delayed pass_reset: used to initialize req_addr_r and
    // row_addr_ready after every pass boundary (both feat_tile and batch_tile
    // transitions).  For batch transitions, row_start_r is committed by this
    // edge; for feat_tile transitions, row_start_r is unchanged and correct.
    logic pass_reset_r;
    always_ff @(posedge clk) pass_reset_r <= pass_reset;

    always_ff @(posedge clk) begin
        if (reset) begin
            state       <= IDLE;
            feat_tile   <= '0;
            batch_tile  <= '0;
            feat_tiles  <= '0;
            batch_tiles <= '0;
        end else if (rearm) begin
            state       <= ACTIVE;
            feat_tile   <= '0;
            batch_tile  <= '0;
            feat_tiles  <= FEAT_TILE_W'((32'(num_features) + N_FEAT  - 1) / N_FEAT);
            batch_tiles <= BATCH_TILE_W'((32'(batch_size)  + B_TILE  - 1) / B_TILE);
        end else if (next_pass) begin
            if (last_feat_tile) begin
                feat_tile  <= '0;
                batch_tile <= batch_tile + BATCH_TILE_W'(1);
            end else begin
                feat_tile  <= feat_tile + FEAT_TILE_W'(1);
            end
        end else if (scan_done_pulse && last_feat_tile && last_batch_tile) begin
            state <= DONE;
        end
    end

    // -------------------------------------------------------------------------
    // FP32 -> FP16 narrowing conversion (round-to-nearest-even), applied to
    // every activation element as it comes off the L2 response, before it's
    // pushed into a per-row FIFO. Real hidden states are produced by the
    // core's native FP32 FPU (Vortex has no FP16 datapath); the systolic
    // array stays FP16 for area, so this is the one place the narrowing
    // happens — inside checker-owned hardware, never in the model's kernel.
    // FP16-subnormal results are flushed to zero rather than rounded into
    // the subnormal range, trading a small amount of dynamic range at the
    // bottom of FP16 (< ~6e-5 in magnitude) for a much smaller, fixed-shift
    // conversion unit (no variable-width barrel shifter needed).
    // -------------------------------------------------------------------------
    function automatic logic [15:0] fp32_to_fp16(input logic [31:0] x);
        logic        sign   = x[31];
        logic [7:0]  exp32  = x[30:23];
        logic [22:0] mant32 = x[22:0];
        logic        round_up;
        logic [10:0] mant16_rnd;
        logic [8:0]  exp32_adj;  // exp32 + mantissa-carry, 9 bits

        if (exp32 == 8'hFF)
            return {sign, 5'h1F, (mant32 != 23'h0) ? 10'h200 : 10'h000};
        if (exp32 == 8'h00)
            return {sign, 15'h0000};

        // Round-to-nearest-even: guard=mant32[12], sticky=|mant32[11:0], tie=mant32[13]
        round_up   = mant32[12] && (mant32[11:0] != 12'h0 || mant32[13]);
        mant16_rnd = {1'b0, mant32[22:13]} + {10'b0, round_up};

        // Fold the mantissa carry into the exponent before range checks so there
        // is no int intermediate.  exp16 = exp32 - 112; we compare exp32_adj instead:
        //   exp16 >= 31  ↔  exp32_adj >= 143
        //   exp16 <= 0   ↔  exp32_adj <= 112
        exp32_adj = {1'b0, exp32} + {8'b0, mant16_rnd[10]};

        if (exp32_adj >= 9'd143)      return {sign, 5'h1F, 10'h000};
        else if (exp32_adj <= 9'd112) return {sign, 15'h0000};
        else begin
            // exp32_adj ∈ [113,142] → result ∈ [1,30]: bits [8:5] are always 0.
            /* verilator lint_off UNUSEDSIGNAL */
            automatic logic [8:0] exp16_v = exp32_adj - 9'd112;
            /* verilator lint_on UNUSEDSIGNAL */
            return {sign, exp16_v[4:0], mant16_rnd[9:0]};
        end
    endfunction

    // -------------------------------------------------------------------------
    // Per-row address precomputation (registered at each batch_tile boundary).
    //
    // row_start_r[b]: registered at batch_reset using batch_tile_next so the
    //   multiplier (batch_row_index × hidden_size × 4) runs once per batch
    //   boundary — off the active critical path (37+ idle cycles available).
    //
    // row_skip / per_row_chunks: derived combinationally from row_start_r with
    //   no multiplier: bit-extract for row_skip, one adder+shift for per_row_chunks.
    //
    // req_addr_r[b]: cache-line-aligned running address, initialized one cycle
    //   after batch_reset (when row_start_r is committed), incremented by
    //   LINE_BYTES on each issued request.  req_byte_addr = req_addr_r[issue_row]
    //   is a pure register read — 0 FO4 on the issue address critical path.
    //
    // row_addr_ready: gates issue_valid until req_addr_r is initialized.
    // -------------------------------------------------------------------------
    logic [`MEM_ADDR_WIDTH-1:0] row_start_r [B_TILE];

    always_ff @(posedge clk) begin
        if (batch_reset) begin
            for (int b = 0; b < B_TILE; b++)
                row_start_r[b] <= hidden_base_addr
                    + (`MEM_ADDR_WIDTH'(batch_tile_next) * B_TILE + `MEM_ADDR_WIDTH'(b))
                      * `MEM_ADDR_WIDTH'(hidden_size) * 4;
        end
    end

    // Combinational: bit-extract and one small adder — no multiply.
    wire [LOG_LW-1:0]   row_skip       [B_TILE];
    wire [CHUNKS_W-1:0] per_row_chunks [B_TILE];
    generate
        for (genvar b = 0; b < B_TILE; b++) begin : g_row_align
            assign row_skip[b]       = row_start_r[b][LINE_BITS-1:2];
            assign per_row_chunks[b] = CHUNKS_W'(
                (32'(hidden_size) + 32'(row_skip[b]) + LINE_WORDS - 1) >> LOG_LW);
        end
    endgenerate

    logic [`MEM_ADDR_WIDTH-1:0] req_addr_r       [B_TILE];
    logic [CHUNKS_W-1:0]        per_row_chunks_r [B_TILE]; // registered at pass_reset_r

    logic row_addr_ready;
    always_ff @(posedge clk) begin
        if (reset || pass_reset) row_addr_ready <= 1'b0;
        else if (pass_reset_r)   row_addr_ready <= 1'b1;
    end

    // first_chunk_done[b]: set after the first cache-line response for row b
    // arrives in a given pass; cleared on pass_reset.
    logic [B_TILE-1:0] first_chunk_done;

    // -------------------------------------------------------------------------
    // Per-row FIFO storage
    // -------------------------------------------------------------------------
    logic [B_TILE-1:0][FIFO_DEPTH-1:0][15:0] fifo;
    logic [B_TILE-1:0][FIFO_PTR_W-1:0]       rd_ptr, wr_ptr;
    logic [B_TILE-1:0][FIFO_CTR_W-1:0]       count;

    logic [B_TILE-1:0][CHUNKS_W-1:0] next_chunk;
    // Block chunk N+1 until chunk N's response arrives: DRAM misses can return
    // out of order, which would silently scramble the FIFO and corrupt MACs.
    logic [B_TILE-1:0] chunk_inflight;

    // -------------------------------------------------------------------------
    // Response routing
    // -------------------------------------------------------------------------
    wire rsp_fire = act_bus_if.rsp_valid && act_bus_if.rsp_ready;
    wire [ROW_ID_BITS-1:0] rsp_row = act_bus_if.rsp_data.tag.value[ROW_ID_BITS-1:0];

    // -------------------------------------------------------------------------
    // Per-row start flags, consumed-element counter, done flag
    // -------------------------------------------------------------------------
    logic [B_TILE-1:0] k_started;
    logic [B_TILE-1:0][15:0] k_count;

    logic [B_TILE-1:0] k_done;
    always_comb begin
        for (int b = 0; b < B_TILE; b++)
            k_done[b] = k_started[b] && (k_count[b] >= hidden_size);
    end

    logic k_stall;
    always_comb begin
        k_stall = 1'b0;
        for (int b = 0; b < B_TILE; b++)
            if (k_started[b] && !k_done[b] && (count[b] == '0))
                k_stall = 1'b1;
    end

    // -------------------------------------------------------------------------
    // Drain counter (resets each pass)
    // -------------------------------------------------------------------------
    logic [DRAIN_CTR_W-1:0] drain_cnt;
    logic                   drain_started;
    wire  drain_active = (drain_cnt != '0) || (k_done[0] && !drain_started);

    logic drain_active_r;
    always_ff @(posedge clk) begin
        if (reset || pass_reset) drain_active_r <= 1'b0;
        else                     drain_active_r <= drain_active;
    end
    wire mac_done = drain_active_r && !drain_active;

    always_ff @(posedge clk) begin
        if (reset || pass_reset) begin
            drain_cnt     <= '0;
            drain_started <= 1'b0;
        end else begin
            if (k_done[0] && !drain_started) begin
                drain_cnt     <= DRAIN_CTR_W'(DRAIN_CYCLES);
                drain_started <= 1'b1;
            end else if (drain_active) begin
                drain_cnt <= drain_cnt - DRAIN_CTR_W'(1);
            end
        end
    end

    // -------------------------------------------------------------------------
    // Push / pop helpers
    // -------------------------------------------------------------------------
    logic [B_TILE-1:0] row_push, row_pop;
    always_comb begin
        for (int b = 0; b < B_TILE; b++) begin
            row_push[b] = rsp_fire && (ROW_ID_BITS'(rsp_row) == ROW_ID_BITS'(b));
            row_pop[b]  = k_started[b] && !k_done[b] && !k_stall;
        end
    end

    // -------------------------------------------------------------------------
    // FIFO update (resets each pass)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset || pass_reset) begin
            for (int b = 0; b < B_TILE; b++) begin
                rd_ptr[b]  <= '0;
                wr_ptr[b]  <= '0;
                count[b]   <= '0;
                k_count[b] <= '0;
            end
            k_started        <= '0;
            first_chunk_done <= '0;
        end else if (state == ACTIVE) begin
            if (!k_started[0] && (count[0] > '0))
                k_started[0] <= 1'b1;
            for (int b = 1; b < B_TILE; b++)
                if (!k_started[b] && row_pop[b-1])
                    k_started[b] <= 1'b1;

            for (int b = 0; b < B_TILE; b++) begin
                if (row_push[b]) begin
                    first_chunk_done[b] <= 1'b1;
                    if (!first_chunk_done[b]) begin
                        // First cache-line response: skip row_skip[b] leading FP32
                        // elements that belong to the previous token's cache line.
                        // Each surviving element is narrowed to FP16 before storage.
                        for (int w = 0; w < LINE_WORDS; w++) begin
                            if (w >= int'(row_skip[b]))
                                fifo[b][FIFO_PTR_W'(wr_ptr[b] + FIFO_PTR_W'(w - int'(row_skip[b])))]
                                    <= fp32_to_fp16(act_bus_if.rsp_data.data[w*32 +: 32]);
                        end
                        wr_ptr[b] <= FIFO_PTR_W'(wr_ptr[b]
                                     + FIFO_PTR_W'(LINE_WORDS - int'(row_skip[b])));
                    end else begin
                        for (int w = 0; w < LINE_WORDS; w++)
                            fifo[b][FIFO_PTR_W'(wr_ptr[b] + FIFO_PTR_W'(w))]
                                <= fp32_to_fp16(act_bus_if.rsp_data.data[w*32 +: 32]);
                        wr_ptr[b] <= FIFO_PTR_W'(wr_ptr[b] + FIFO_PTR_W'(LINE_WORDS));
                    end
                end
                if (row_pop[b]) begin
                    rd_ptr[b]  <= rd_ptr[b] + FIFO_PTR_W'(1);
                    k_count[b] <= k_count[b] + 16'(1);
                end
                begin : count_update
                    // push_words: actual FP16 elements written this push (first chunk
                    // writes fewer if the row is not cache-line aligned).
                    automatic logic [FIFO_CTR_W-1:0] push_words =
                        (!first_chunk_done[b])
                            ? FIFO_CTR_W'(LINE_WORDS - int'(row_skip[b]))
                            : FIFO_CTR_W'(LINE_WORDS);
                    if (row_push[b] && row_pop[b])
                        count[b] <= FIFO_CTR_W'(count[b]) + push_words - FIFO_CTR_W'(1);
                    else if (row_push[b])
                        count[b] <= FIFO_CTR_W'(count[b]) + push_words;
                    else if (row_pop[b])
                        count[b] <= count[b] - FIFO_CTR_W'(1);
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Weight SRAM: MAX_HIDDEN rows × (MAX_FEATURES × 16) bits
    // Each row holds all features for one k; feat_tile selects the N_FEAT slice.
    // -------------------------------------------------------------------------
    wire [WEIGHT_DATAW-1:0] weight_row_out;

    VX_dp_ram #(
        .DATAW       (WEIGHT_DATAW),
        .SIZE        (MAX_HIDDEN),
        .OUT_REG     (0),
        .RDW_MODE    ("W"),
        .INIT_ENABLE (1),
        .INIT_FILE   (WEIGHT_FILE),
        .INIT_VALUE  (0)
    ) weight_sram (
        .clk   (clk),
        .reset (reset),
        .write (weight_we_i),
        .wren  (1'b1),
        .waddr (weight_waddr_i),
        .wdata (weight_wdata_i),
        .read  (1'b1),
        .raddr (k_count[0][WEIGHT_ADDRW-1:0]),
        .rdata (weight_row_out)
    );

    // threshold[0]:              count threshold k (raw uint16, not FP16).
    //                            global_flag fires when feat_count > k.
    // threshold[1..num_features]: per-feature FP16 activation thresholds.
    // Zero-initialized so entries beyond num_features are 0 (not X).
    logic [15:0] threshold [0:MAX_FEATURES];
    initial begin
        for (int n = 0; n <= MAX_FEATURES; n++) threshold[n] = '0;
        if (THRESHOLD_FILE != "") $readmemh(THRESHOLD_FILE, threshold);
    end
    // DCR write path: THRESH_DATA writes override the initial values at runtime.
    always_ff @(posedge clk) begin
        if (thresh_we_i)
            threshold[thresh_waddr_i] <= thresh_wdata_i;
    end

    // Extract the current feat_tile's N_FEAT column slice from the SRAM output row.
    // feat_tile is stable for the entire duration of a pass, so this mux is fine.
    wire [TILE_DATAW-1:0] weight_tile_flat;
    generate
        for (genvar n = 0; n < N_FEAT; n++) begin : g_wtile
            assign weight_tile_flat[n*16 +: 16] =
                weight_row_out[(feat_tile * N_FEAT + n) * 16 +: 16];
        end
    endgenerate

    // Horizontal weight pipe (HPIPE_DEPTH = N_FEAT-1 = 15 stages, TILE_DATAW bits per stage).
    // col_in[n] = W[k_count[0]-n][feat_tile*N_FEAT+n] via the delayed tile slice.
    logic [HPIPE_DEPTH-1:0][TILE_DATAW-1:0] w_hpipe;
    always_ff @(posedge clk) begin
        if (reset || pass_reset) begin
            for (int s = 0; s < HPIPE_DEPTH; s++) w_hpipe[s] <= '0;
        end else if (row_pop[0] || drain_active) begin
            w_hpipe[0] <= weight_tile_flat;
            for (int s = 1; s < HPIPE_DEPTH; s++)
                w_hpipe[s] <= w_hpipe[s-1];
        end
    end

    wire [N_FEAT-1:0][15:0] col_in;
    assign col_in[0] = weight_tile_flat[15:0];
    generate
        for (genvar n = 1; n < N_FEAT; n++) begin : g_colin
            assign col_in[n] = w_hpipe[n-1][n*16 +: 16];
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Systolic array inputs
    // -------------------------------------------------------------------------
    /* verilator lint_off ASCRANGE */
    wire [0:B_TILE-1][15:0] sa_a_in;
    /* verilator lint_on ASCRANGE */
    generate
        for (genvar b = 0; b < B_TILE; b++) begin : g_sa_ain
            assign sa_a_in[b] = (k_started[b] && !k_done[b]) ? fifo[b][rd_ptr[b]] : 16'h0;
        end
    endgenerate

    /* verilator lint_off ASCRANGE */
    wire [0:N_FEAT-1][15:0] sa_b_in;
    /* verilator lint_on ASCRANGE */
    generate
        for (genvar n = 0; n < N_FEAT; n++) begin : g_sa_bin
            assign sa_b_in[n] = col_in[n];
        end
    endgenerate

    wire sa_pipeline_en = row_pop[0] || drain_active;

    // -------------------------------------------------------------------------
    // Scan readout control (resets each pass)
    // -------------------------------------------------------------------------
    logic cswitch_pulse;
    logic [SCAN_CTR_W-1:0] scan_cnt;
    always_ff @(posedge clk) begin
        if (reset || pass_reset) begin
            cswitch_pulse <= 1'b0;
            scan_cnt      <= '0;
        end else begin
            cswitch_pulse <= mac_done;
            if (cswitch_pulse) scan_cnt <= SCAN_CTR_W'(SCAN_INIT);
            else if (scan_cnt > '0) scan_cnt <= scan_cnt - SCAN_CTR_W'(1);
        end
    end
    wire sa_cscan_en = (scan_cnt > '0);

    wire sa_pipeline_en_full = sa_pipeline_en || cswitch_pulse || sa_cscan_en;

    /* verilator lint_off ASCRANGE */
    wire [0:B_TILE-1][ACC_W-1:0] sa_c_out;
    /* verilator lint_on ASCRANGE */

    // -------------------------------------------------------------------------
    // SAURIA sa_array instantiation (4×16, FP16 FMA)
    // -------------------------------------------------------------------------
    sa_array #(
        .ARITHMETIC              (1),
        .MUL_TYPE                (0),
        .ADD_TYPE                (0),
        .M_APPROX                (0),
        .MM_APPROX               (0),
        .A_APPROX                (0),
        .AA_APPROX               (0),
        .X                       (N_FEAT),
        .Y                       (B_TILE),
        .IA_W                    (16),
        .IB_W                    (16),
        .OC_W                    (ACC_W),
        .TH_W                    (2),
        .STAGES_MUL              (0),
        .INTERMEDIATE_PIPELINE_STAGE (1),
        .ZERO_GATING_MULT        (0),
        .ZERO_GATING_ADD         (0),
        .ZD_LOOKAHEAD            (0),
        .EXTRA_CSREG             (0)
    ) mac_array (
        .i_clk          (clk),
        .i_rstn         (!reset),
        .i_a_arr        (sa_a_in),
        .i_b_arr        (sa_b_in),
        .i_c_arr        ('0),
        .i_reg_clear    (pass_reset),     // clear accumulators at the start of each pass
        .i_pipeline_en  (sa_pipeline_en_full),
        .i_cswitch_arr  (cswitch_pulse ? {N_FEAT{1'b1}} : {N_FEAT{1'b0}}),
        .i_cscan_en     (sa_cscan_en),
        .i_thres        ('0),
        .o_c_arr        (sa_c_out)
    );

    // -------------------------------------------------------------------------
    // FP16 greater-than: full IEEE 754 signed comparison including NaN.
    // Maps each value to an unsigned sort key that preserves total order:
    //   negative → ~x  (flips magnitude ordering for negative sign-magnitude)
    //   positive → x ^ 0x8000  (sets MSB so all positives sort above negatives)
    // NaN on either side returns false (IEEE unordered convention).
    // ±0 are treated as equal (both have zero exponent+mantissa).
    // -------------------------------------------------------------------------
    function automatic logic [15:0] fp16_order_key(input logic [15:0] x);
        if (x[15]) return ~x;
        else       return x ^ 16'h8000;
    endfunction

    /* verilator lint_off UNUSEDSIGNAL */
    function automatic logic fp16_is_nan(input logic [15:0] x);
        return (x[14:10] == 5'h1F) && (x[9:0] != 10'h000);  // sign bit irrelevant for NaN
    endfunction
    /* verilator lint_on UNUSEDSIGNAL */

    function automatic logic fp16_gt(input logic [15:0] a, input logic [15:0] th);
        if (fp16_is_nan(a) || fp16_is_nan(th)) return 1'b0;
        if (a[14:0] == 15'h0000 && th[14:0] == 15'h0000) return 1'b0;  // ±0 equal
        return fp16_order_key(a) > fp16_order_key(th);
    endfunction

    // -------------------------------------------------------------------------
    // Per-row scan column counter.
    // col_cnt[b] tracks which column of the current feat_tile row b is scanning.
    // Row b's window opens when scan_cnt == SCAN_INIT - b; col_cnt starts at 0
    // and increments each scan cycle, saturating at N_FEAT (done sentinel).
    // This replaces the combinational  scan_cnt → col_i → gf  chain with a
    // registered counter, shrinking the row_fired_now critical path.
    // -------------------------------------------------------------------------
    logic [COL_CNT_W-1:0] col_cnt [B_TILE];

    always_ff @(posedge clk) begin
        if (reset || pass_reset) begin
            for (int b = 0; b < B_TILE; b++)
                col_cnt[b] <= COL_CNT_W'(N_FEAT);
        end else begin
            for (int b = 0; b < B_TILE; b++) begin
                // Trigger one cycle BEFORE the row's first valid scan output so that
                // the NBA commits col_cnt[b]=0 in time to be read that first cycle.
                //   b=0: cswitch_pulse fires one cycle before scan_cnt reaches SCAN_INIT.
                //   b>0: scan_cnt == SCAN_INIT-b+1 is one cycle before SCAN_INIT-b.
                automatic logic start_now =
                    (b == 0) ? cswitch_pulse
                             : (sa_cscan_en && scan_cnt == SCAN_CTR_W'(SCAN_INIT - b + 1));
                if (start_now)
                    col_cnt[b] <= '0;
                else if (sa_cscan_en && col_cnt[b] < COL_CNT_W'(N_FEAT))
                    col_cnt[b] <= col_cnt[b] + COL_CNT_W'(1);
            end
        end
    end

    // -------------------------------------------------------------------------
    // row_fired_now[b]: the actual hardware firing decision for whatever
    // feature column b is scanning out this cycle (fp16_gt against its
    // threshold).  Factored out of the feat_count update so simulation-only
    // tracing can observe the exact bit the comparator produced, rather than
    // recomputing it in software from the dumped accumulator value.
    //
    // gf = {feat_tile, col_cnt[b][LOG_N_FEAT-1:0]}  — a concatenation, no arithmetic.
    // Validity: col_cnt[b] < N_FEAT  ↔  col_cnt[b][COL_CNT_W-1] == 0.
    // -------------------------------------------------------------------------
    logic [B_TILE-1:0] row_fired_now;
    always_comb begin
        for (int b = 0; b < B_TILE; b++) begin
            automatic logic [GF_W-1:0] gf =
                {feat_tile[FEAT_TILE_W-1:0], col_cnt[b][`CLOG2(N_FEAT)-1:0]};
            // col_cnt < N_FEAT means this row's scan window is open
            row_fired_now[b] = (col_cnt[b] < COL_CNT_W'(N_FEAT)) &&
                                (gf < GF_W'(num_features)) &&
                                fp16_gt(sa_c_out[b], threshold[gf + 1]);
        end
    end

    // -------------------------------------------------------------------------
    // Per-row feature-fire counter: counts how many features exceeded their
    // threshold.  Accumulates across feat_tiles; resets at batch boundaries.
    // global_flag fires when feat_count > threshold[0] (the count threshold k).
    // -------------------------------------------------------------------------
    logic [B_TILE-1:0][FEAT_COUNT_W-1:0] feat_count;
    always_ff @(posedge clk) begin
        if (reset || batch_reset) begin
            feat_count <= '0;
        end else if (sa_cscan_en) begin
            for (int b = 0; b < B_TILE; b++) begin
                if (row_fired_now[b])
                    feat_count[b] <= feat_count[b] + FEAT_COUNT_W'(1);
            end
        end
    end

    // -------------------------------------------------------------------------
    // scan_done_pulse: 1 cycle after the last scan cycle.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) scan_done_pulse <= (scan_cnt == SCAN_CTR_W'(1));

    // -------------------------------------------------------------------------
    // Global flag: MAX_BATCH wide. Written on last feat_tile of each batch_tile.
    // Fires when feat_count > threshold[0] (count threshold k, raw uint16).
    // -------------------------------------------------------------------------
    logic [MAX_BATCH-1:0] global_flag;
    always_ff @(posedge clk) begin
        if (reset || rearm) begin
            global_flag <= '0;
        end else if (scan_done_pulse && last_feat_tile) begin
            for (int b = 0; b < B_TILE; b++) begin
                // gb = batch_tile * B_TILE + b.  B_TILE=4 is a power of two,
                // so this is a concatenation: {batch_tile, 2-bit b}.
                // gb = batch_tile*B_TILE + b.  B_TILE=4 is a power of two so this
                // is a concatenation.  gb ≤ MAX_BATCH-1 = 15 (4 bits); the 5-bit
                // gb only exceeds 4 bits when batch_tile's MSB would be set, which
                // can't happen since batch_tile ≤ MAX_BATCH_TILES-1 = 3 (2 bits).
                automatic logic [BATCH_TILE_W+ROW_ID_BITS-1:0] gb =
                    {batch_tile, ROW_ID_BITS'(b)};
                if (gb < (BATCH_TILE_W+ROW_ID_BITS)'(MAX_BATCH))
                    /* verilator lint_off WIDTHTRUNC */
                    global_flag[gb] <= (gb < (BATCH_TILE_W+ROW_ID_BITS)'(batch_size)) &&
                                       (feat_count[b] > FEAT_COUNT_W'(threshold[0]));
                    /* verilator lint_on WIDTHTRUNC */
            end
        end
    end
    assign flag_o = global_flag;

    // -------------------------------------------------------------------------
    // Issue FSM: round-robin across rows, issue when FIFO has room and chunks remain.
    //
    // Split into two stages to shrink the critical path:
    //
    //   Stage 1 — can_issue[b]: evaluate all B_TILE eligibility conditions IN
    //     PARALLEL.  Each bit is independent; no chain through prior iterations.
    //     All inputs (count, next_chunk, per_row_chunks_r, chunk_inflight) are
    //     registered, so this is a single level of comparators.
    //
    //   Stage 2 — priority encode: scan can_issue[] starting from issue_rr to
    //     find the round-robin winner.  The chain now runs over B_TILE single-bit
    //     flags (~1 FO4 per step) rather than over full conditions (~12 FO4/step).
    //
    // per_row_chunks_r is registered at pass_reset_r alongside req_addr_r, so it
    // is a stable register output with 0 FO4 into the comparison.
    // -------------------------------------------------------------------------
    logic [ROW_ID_BITS-1:0] issue_rr;
    logic [ROW_ID_BITS-1:0] issue_row;
    logic                   issue_valid;

    // Stage 1: parallel eligibility — one bit per row, all computed simultaneously.
    logic [B_TILE-1:0] can_issue;
    always_comb begin
        for (int b = 0; b < B_TILE; b++)
            can_issue[b] = !rearm
                        && (state == ACTIVE)
                        && row_addr_ready
                        && (count[b] <= FIFO_CTR_W'(FIFO_HALF))
                        && (next_chunk[b] < per_row_chunks_r[b])
                        && !chunk_inflight[b];
    end

    // Stage 2: priority encode over single-bit can_issue flags, wrapped round-robin.
    always_comb begin
        issue_row   = '0;
        issue_valid = 1'b0;
        for (int i = 0; i < B_TILE; i++) begin
            automatic logic [ROW_ID_BITS-1:0] bi = issue_rr + ROW_ID_BITS'(i);
            if (!issue_valid && can_issue[bi]) begin
                issue_row   = bi;
                issue_valid = 1'b1;
            end
        end
    end

    wire req_fire = act_bus_if.req_valid && act_bus_if.req_ready;

    always_ff @(posedge clk) begin
        if (reset || pass_reset) begin
            issue_rr       <= '0;
            chunk_inflight <= '0;
            for (int b = 0; b < B_TILE; b++)
                next_chunk[b] <= '0;
        end else begin
            // Reinitialize req_addr_r and per_row_chunks_r one cycle after every
            // pass_reset (both feat_tile and batch_tile transitions).  For batch
            // transitions, row_start_r is committed by this edge.  For feat_tile
            // transitions, row_start_r is unchanged — reinitializing req_addr_r
            // from it resets the chunk pointer back to the row start for the new
            // feat_tile pass.  req_fire cannot overlap since row_addr_ready is 0.
            if (pass_reset_r) begin
                for (int b = 0; b < B_TILE; b++) begin
                    req_addr_r[b] <= {row_start_r[b][`MEM_ADDR_WIDTH-1:LINE_BITS],
                                      LINE_BITS'(0)};
                    per_row_chunks_r[b] <= CHUNKS_W'(
                        (32'(hidden_size) + 32'(row_skip[b]) + LINE_WORDS - 1) >> LOG_LW);
                end
            end
            if (req_fire) begin
                next_chunk[issue_row]     <= next_chunk[issue_row] + CHUNKS_W'(1);
                issue_rr                  <= issue_rr + ROW_ID_BITS'(1);
                chunk_inflight[issue_row] <= 1'b1;
                req_addr_r[issue_row]     <= req_addr_r[issue_row]
                                             + `MEM_ADDR_WIDTH'(LINE_BYTES);
            end
            if (rsp_fire)
                chunk_inflight[rsp_row] <= 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // Request address: includes batch_tile offset for the current pass.
    // -------------------------------------------------------------------------
    // req_byte_addr: pure register output — 0 FO4.  req_addr_r is cache-line
    // aligned (lower LINE_BITS = 0) so the addr field assignment below is exact.
    wire [`MEM_ADDR_WIDTH-1:0] req_byte_addr = req_addr_r[issue_row];

    // -------------------------------------------------------------------------
    // Drive act_bus_if
    // -------------------------------------------------------------------------
    assign act_bus_if.req_valid            = issue_valid;
    assign act_bus_if.req_data.rw          = 1'b0;
    assign act_bus_if.req_data.addr        = req_byte_addr[`MEM_ADDR_WIDTH-1:LINE_BITS];
    assign act_bus_if.req_data.data        = '0;
    assign act_bus_if.req_data.byteen      = '1;
    assign act_bus_if.req_data.flags       = '0;
    assign act_bus_if.req_data.tag.uuid    = '0;
    assign act_bus_if.req_data.tag.value   = TAG_VAL_W'(issue_row);

    assign act_bus_if.rsp_ready = 1'b1;

    // -------------------------------------------------------------------------
    // Simulation traces
    // -------------------------------------------------------------------------
`ifdef SIMULATION
    logic [B_TILE-1:0] k_done_r;
    always_ff @(posedge clk) begin
        if (reset || pass_reset) k_done_r <= '0;
        else                     k_done_r <= k_done;
    end

    // Delay ALL_DONE by 1 cycle so the full-matrix dump reads committed values.
    // full_matrix_capture[last_bt][last_ft] and global_flag[last_bt_rows] are both
    // written (NBA) at the posedge scan_done_pulse fires for the last pass; reading
    // them at that same posedge sees pre-update values.  all_done_r fires one cycle
    // later when all writes are committed.
    logic all_done_r;
    always_ff @(posedge clk)
        all_done_r <= scan_done_pulse && last_feat_tile && last_batch_tile;

    // Capture the full B_TILE×N_FEAT matmul output during the scan phase.
    // col_i for row b at scan_cnt = SCAN_INIT - scan_cnt - b (same as row_flag indexing).
    logic [B_TILE-1:0][N_FEAT-1:0][15:0] scan_capture;
    // Capture the actual per-feature fired bit (row_fired_now) alongside the
    // raw value, so tracing reflects the hardware comparator's own decision
    // rather than a value re-derived in software.
    logic [B_TILE-1:0][N_FEAT-1:0]       fired_capture;
    always_ff @(posedge clk) begin
        if (reset || pass_reset) begin
            scan_capture  <= '0;
            fired_capture <= '0;
        end else if (sa_cscan_en) begin
            for (int b = 0; b < B_TILE; b++) begin
                automatic int col_i = int'(SCAN_INIT) - int'(scan_cnt) - b;
                if (col_i >= 0 && col_i < N_FEAT) begin
                    scan_capture[b][col_i]  <= sa_c_out[b];
                    fired_capture[b][col_i] <= row_fired_now[b];
                end
            end
        end
    end

    // Per-token feature-fire count, captured at last feat_tile of each batch_tile.
    // feat_count is not reset after the last pass, so full_feat_count[last_bt_rows]
    // holds committed values when all_done_r fires.
    logic [MAX_BATCH-1:0][FEAT_COUNT_W-1:0] full_feat_count;
    always_ff @(posedge clk) begin
        if (reset || rearm) begin
            for (int gb = 0; gb < MAX_BATCH; gb++)
                full_feat_count[gb] <= '0;
        end else if (scan_done_pulse && last_feat_tile) begin
            for (int b = 0; b < B_TILE; b++) begin
                automatic int gb = int'(batch_tile) * B_TILE + b;
                if (gb < int'(batch_size))
                    full_feat_count[gb] <= feat_count[b];
            end
        end
    end

    // Accumulate completed passes into a full [MAX_BATCH × MAX_FEATURES] matrix.
    // Reads pre-reset scan_capture (NBA semantics guarantee pre-edge value).
    logic [MAX_BATCH-1:0][MAX_FEATURES-1:0][15:0] full_matrix_capture;
    // Same accumulation for the actual hardware fired bit, not a recomputation.
    logic [MAX_BATCH-1:0][MAX_FEATURES-1:0]       full_fired_capture;
    always_ff @(posedge clk) begin
        if (reset || rearm) begin
            // Element-wise clear avoids >8K-bit replication (Verilator WIDTHCONCAT).
            for (int gb = 0; gb < MAX_BATCH; gb++)
                for (int gf = 0; gf < MAX_FEATURES; gf++) begin
                    full_matrix_capture[gb][gf] <= '0;
                    full_fired_capture[gb][gf]  <= '0;
                end
        end else if (scan_done_pulse) begin
            for (int b = 0; b < B_TILE; b++) begin
                for (int n = 0; n < N_FEAT; n++) begin
                    automatic int gb = int'(batch_tile) * B_TILE + b;
                    automatic int gf = int'(feat_tile)  * N_FEAT + n;
                    if (gb < int'(batch_size) && gf < int'(num_features)) begin
                        full_matrix_capture[gb][gf] <= scan_capture[b][n];
                        full_fired_capture[gb][gf]  <= fired_capture[b][n];
                    end
                end
            end
        end
    end

    // Decode a raw FP16 bit-pattern to a SV real for display.
    function automatic real fp16_to_real(input logic [15:0] h);
        logic       sign   = h[15];
        logic [4:0] exp5   = h[14:10];
        logic [9:0] mant10 = h[9:0];
        int         e;
        real        mag;
        if (&exp5)         return 0.0;  // Inf / NaN → 0 for display
        if (exp5 == '0) begin           // subnormal: 0.mant × 2^−14
            mag = real'(mant10) / 1024.0 / 16384.0;
            return sign ? -mag : mag;
        end
        e   = int'(exp5) - 15;          // normal: 1.mant × 2^(exp−15)
        mag = (1.0 + real'(mant10) / 1024.0) * (2.0 ** e);
        return sign ? -mag : mag;
    endfunction

    always @(posedge clk) begin
        if (!reset && checker_armed) begin
            if (rearm)
                `TRACE(3, ("%t: [CHECKER] armed  base=0x%0h  hidden=%0d  features=%0d  batch=%0d  chunks[0]=%0d\n",
                    $time, hidden_base_addr, hidden_size, num_features, batch_size, per_row_chunks[0]))
            if (req_fire)
                `TRACE(3, ("%t: [CHECKER] req  bt=%0d ft=%0d row=%0d chunk=%0d addr=0x%0h\n",
                    $time, batch_tile, feat_tile, issue_row, next_chunk[issue_row], req_byte_addr))
            if (rsp_fire)
                `TRACE(3, ("%t: [CHECKER] rsp  row=%0d  data[31:0]=0x%0h\n",
                    $time, rsp_row, act_bus_if.rsp_data.data[31:0]))
            if (k_stall)
                `TRACE(3, ("%t: [CHECKER] stall  k_started=%0b  k_done=%0b\n",
                    $time, k_started, k_done))
            for (int b = 0; b < B_TILE; b++) begin
                if (k_done[b] && !k_done_r[b])
                    `TRACE(3, ("%t: [CHECKER] done  row=%0d  consumed=%0d\n",
                        $time, b, k_count[b]))
            end
            if (mac_done)
                `TRACE(3, ("%t: [CHECKER] mac_done  bt=%0d ft=%0d\n", $time, batch_tile, feat_tile))
            if (cswitch_pulse)
                `TRACE(3, ("%t: [CHECKER] cswitch\n", $time))
            if (sa_cscan_en)
                `TRACE(3, ("%t: [CHECKER] scan  cnt=%0d  sa_c_out[0]=0x%0h  sa_c_out[%0d]=0x%0h\n",
                    $time, scan_cnt, sa_c_out[0], B_TILE-1, sa_c_out[B_TILE-1]))
            if (scan_done_pulse) begin
                `TRACE(3, ("%t: [CHECKER] scan_done  bt=%0d ft=%0d  last_ft=%0b last_bt=%0b  feat_count:",
                    $time, batch_tile, feat_tile, last_feat_tile, last_batch_tile))
                for (int b = 0; b < B_TILE; b++)
                    `TRACE(3, (" %0d", feat_count[b]))
                `TRACE(3, ("\n"))
                // Full B_TILE×N_FEAT matmul output for this pass.
                for (int b = 0; b < B_TILE; b++) begin
                    automatic int gb = int'(batch_tile) * B_TILE + b;
                    `TRACE(3, ("%t: [CHECKER]   acc[tok=%0d] feat[%0d..%0d]:", $time, gb,
                               int'(feat_tile) * N_FEAT, int'(feat_tile) * N_FEAT + N_FEAT - 1))
                    for (int n = 0; n < N_FEAT; n++)
                        `TRACE(3, (" %04h", scan_capture[b][n]))
                    `TRACE(3, ("\n"))
                end
            end
            if (scan_done_pulse && last_feat_tile && last_batch_tile)
                `TRACE(3, ("%t: [CHECKER] ALL_DONE (flag vector + full matrix dump follow in 1 cycle)\n", $time))
            // Dump on all_done_r (1 cycle after ALL_DONE) so full_matrix_capture,
            // full_feat_count, and global_flag have their final committed values.
            if (all_done_r) begin
                `TRACE(3, ("%t: [CHECKER] ALL_DONE  count_thresh=%0d  global_flag=0x%0x\n",
                           $time, threshold[0], global_flag))
                `TRACE(3, ("%t: [CHECKER] === FLAG VECTOR (batch=%0d, features=%0d, k=%0d) ===\n",
                           $time, batch_size, num_features, threshold[0]))
                for (int gb = 0; gb < int'(batch_size); gb++)
                    `TRACE(3, ("%t: [CHECKER]   tok[%0d]: flag=%0b  fired=%0d/%0d\n",
                               $time, gb, global_flag[gb], full_feat_count[gb], num_features))
                `TRACE(3, ("%t: [CHECKER] === FULL MATMUL OUTPUT (batch=%0d, features=%0d) ===\n",
                           $time, batch_size, num_features))
                for (int gb = 0; gb < int'(batch_size); gb++) begin
                    `TRACE(3, ("%t: [CHECKER]   tok[%0d]:", $time, gb))
                    for (int gf = 0; gf < int'(num_features); gf++)
                        `TRACE(3, (" %g", fp16_to_real(full_matrix_capture[gb][gf])))
                    `TRACE(3, ("\n"))
                end
                `TRACE(3, ("%t: [CHECKER] ================================================\n", $time))
                // Per-feature fired bit as produced by the hardware comparator
                // (fp16_gt, via row_fired_now) — not re-derived from the dumped
                // matmul value.  Lets the test harness check the actual firing
                // decision rather than recomputing threshold comparisons in Python.
                `TRACE(3, ("%t: [CHECKER] === FIRED BITMAP (batch=%0d, features=%0d) ===\n",
                           $time, batch_size, num_features))
                for (int gb = 0; gb < int'(batch_size); gb++) begin
                    `TRACE(3, ("%t: [CHECKER]   tok[%0d]:", $time, gb))
                    for (int gf = 0; gf < int'(num_features); gf++)
                        `TRACE(3, (" %0d", full_fired_capture[gb][gf]))
                    `TRACE(3, ("\n"))
                end
                `TRACE(3, ("%t: [CHECKER] ================================================\n", $time))
            end
        end
    end
`endif

endmodule
