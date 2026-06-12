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
// row_flag[b] starts at 1 per batch_tile and ANDs across all feat_tile passes.
// flag_o[bt*B_TILE+b] is set after the last feat_tile if all N_FEAT comparisons passed.
//
// Weight SRAM: MAX_HIDDEN rows × (MAX_FEATURES × 16) bits.  Each row holds all features
// for one k value; feat_tile selects the N_FEAT-column slice before the hpipe.
// Thresholds: MAX_FEATURES FP16 values; global feature index = feat_tile*N_FEAT + col_i.
//
// pass_reset  = rearm | next_pass        — restarts FIFOs/drain/scan each pass.
// batch_reset = rearm | (next_pass & last_feat_tile) — resets row_flag at batch boundary.
//
// Enable: -DCHECKER_ENABLE
// DCRs:   VX_DCR_CHECKER_ENABLE       (rising edge arms checker)
//         VX_DCR_CHECKER_TAP_ADDR0/1  (hidden-state base address)
//         VX_DCR_CHECKER_HIDDEN_SIZE  (FP16 elements per token)
//         VX_DCR_CHECKER_BATCH_SIZE   (total tokens, ≤ MAX_BATCH)
//         VX_DCR_CHECKER_NUM_FEATURES (total SAE features, ≤ MAX_FEATURES)

`include "VX_define.vh"

module VX_checker import VX_gpu_pkg::*; #(
    parameter B_TILE        = 4,               // systolic array rows  (fixed)
    parameter N_FEAT        = 16,              // systolic array cols  (fixed)
    parameter MAX_HIDDEN    = 2048,            // max hidden_size (SRAM depth)
    parameter MAX_FEATURES  = 64,             // max total SAE features; must be multiple of N_FEAT
    parameter MAX_BATCH     = 16,             // max total batch size; must be multiple of B_TILE
    parameter FIFO_DEPTH    = 64,              // FP16 slots per activation FIFO row
    parameter LINE_WORDS    = `L1_LINE_SIZE/2, // FP16 values per cache line (64B/2B=32)
    parameter `STRING WEIGHT_FILE    = "",     // $readmemh hex (MAX_HIDDEN × MAX_FEATURES FP16)
    parameter `STRING THRESHOLD_FILE = ""      // $readmemh hex (MAX_FEATURES FP16 thresholds)
) (
    input  wire clk,
    input  wire reset,

    // DCR-supplied config (latched in VX_cluster before vx_start)
    input  wire                         checker_armed,
    input  wire [`MEM_ADDR_WIDTH-1:0]   hidden_base_addr,
    input  wire [15:0]                  hidden_size,    // FP16 elements per token
    input  wire [15:0]                  num_features,   // actual number of SAE features
    input  wire [15:0]                  batch_size,     // total tokens across all batch tiles

    // Per-row flags: flag_o[bt*B_TILE+b]=1 if all features of that token passed threshold.
    // Rows at or beyond batch_size are always 0.
    output wire [MAX_BATCH-1:0]         flag_o,

    // Dedicated L2 port for activation prefetch
    VX_mem_bus_if.master                act_bus_if
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

    localparam MAX_FEAT_TILES  = MAX_FEATURES / N_FEAT;
    localparam MAX_BATCH_TILES = MAX_BATCH    / B_TILE;
    localparam FEAT_TILE_W     = `CLOG2(MAX_FEAT_TILES  + 1);
    localparam BATCH_TILE_W    = `CLOG2(MAX_BATCH_TILES + 1);

    // -------------------------------------------------------------------------
    // Rising-edge detector for checker_armed
    // -------------------------------------------------------------------------
    logic armed_r;
    always_ff @(posedge clk) begin
        if (reset) armed_r <= 0;
        else       armed_r <= checker_armed;
    end
    wire rearm = checker_armed && !armed_r;

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
    // Configuration latched on rearm
    // -------------------------------------------------------------------------
    logic [CHUNKS_W-1:0] total_chunks;

    always_ff @(posedge clk) begin
        if (rearm)
            total_chunks <= CHUNKS_W'((32'(hidden_size) + LINE_WORDS - 1) >> LOG_LW);
    end

    // -------------------------------------------------------------------------
    // Per-row FIFO storage
    // -------------------------------------------------------------------------
    logic [B_TILE-1:0][FIFO_DEPTH-1:0][15:0] fifo;
    logic [B_TILE-1:0][FIFO_PTR_W-1:0]       rd_ptr, wr_ptr;
    logic [B_TILE-1:0][FIFO_CTR_W-1:0]       count;

    logic [B_TILE-1:0][CHUNKS_W-1:0] next_chunk;

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
            k_started <= '0;
        end else if (state == ACTIVE) begin
            if (!k_started[0] && (count[0] > '0))
                k_started[0] <= 1'b1;
            for (int b = 1; b < B_TILE; b++)
                if (!k_started[b] && row_pop[b-1])
                    k_started[b] <= 1'b1;

            for (int b = 0; b < B_TILE; b++) begin
                if (row_push[b]) begin
                    for (int w = 0; w < LINE_WORDS; w++)
                        fifo[b][FIFO_PTR_W'(wr_ptr[b] + FIFO_PTR_W'(w))]
                            <= act_bus_if.rsp_data.data[w*16 +: 16];
                    wr_ptr[b] <= FIFO_PTR_W'(wr_ptr[b] + FIFO_PTR_W'(LINE_WORDS));
                end
                if (row_pop[b]) begin
                    rd_ptr[b]  <= rd_ptr[b] + FIFO_PTR_W'(1);
                    k_count[b] <= k_count[b] + 16'(1);
                end
                if (row_push[b] && row_pop[b])
                    count[b] <= FIFO_CTR_W'(count[b]) + FIFO_CTR_W'(LINE_WORDS) - FIFO_CTR_W'(1);
                else if (row_push[b])
                    count[b] <= FIFO_CTR_W'(count[b] + FIFO_CTR_W'(LINE_WORDS));
                else if (row_pop[b])
                    count[b] <= count[b] - FIFO_CTR_W'(1);
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
        .write (1'b0),
        .wren  (1'b1),
        .waddr ({WEIGHT_ADDRW{1'b0}}),
        .wdata ({WEIGHT_DATAW{1'b0}}),
        .read  (1'b1),
        .raddr (k_count[0][WEIGHT_ADDRW-1:0]),
        .rdata (weight_row_out)
    );

    // Per-feature FP16 thresholds: MAX_FEATURES values loaded from THRESHOLD_FILE at init.
    // If THRESHOLD_FILE is empty, all thresholds default to +0 (every positive value passes).
    logic [15:0] threshold [0:MAX_FEATURES-1];
    initial begin
        if (THRESHOLD_FILE != "") $readmemh(THRESHOLD_FILE, threshold);
        else for (int n = 0; n < MAX_FEATURES; n++) threshold[n] = '0;
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
    // FP16 greater-than for non-negative thresholds.
    // Negative a (sign bit set) is always below any non-negative threshold.
    // Negative threshold: every non-negative activation trivially passes.
    // For two non-negative FP16 values, IEEE 754 ordering matches unsigned bit ordering.
    // -------------------------------------------------------------------------
    function automatic logic fp16_gt(input logic [15:0] a, input logic [15:0] th);
        if (a[15])  return 1'b0;
        if (th[15]) return 1'b1;
        return a[14:0] > th[14:0];
    endfunction

    // -------------------------------------------------------------------------
    // Per-row flag: ANDs comparisons across feat_tiles; resets at batch boundaries.
    // -------------------------------------------------------------------------
    logic [B_TILE-1:0] row_flag;
    always_ff @(posedge clk) begin
        if (reset || batch_reset) begin
            row_flag <= '1;
        end else if (sa_cscan_en) begin
            for (int b = 0; b < B_TILE; b++) begin
                automatic int col_i = int'(SCAN_INIT) - int'(scan_cnt) - b;
                if (col_i >= 0 && col_i < N_FEAT) begin
                    automatic int gf = int'(feat_tile) * N_FEAT + col_i;
                    if (gf < int'(num_features) && !fp16_gt(sa_c_out[b], threshold[gf]))
                        row_flag[b] <= 1'b0;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // scan_done_pulse: 1 cycle after the last scan cycle.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) scan_done_pulse <= (scan_cnt == SCAN_CTR_W'(1));

    // -------------------------------------------------------------------------
    // Global flag: MAX_BATCH wide. Written on last feat_tile of each batch_tile.
    // -------------------------------------------------------------------------
    logic [MAX_BATCH-1:0] global_flag;
    always_ff @(posedge clk) begin
        if (reset || rearm) begin
            global_flag <= '0;
        end else if (scan_done_pulse && last_feat_tile) begin
            for (int b = 0; b < B_TILE; b++) begin
                automatic int gb = int'(batch_tile) * B_TILE + b;
                if (gb < MAX_BATCH)
                    global_flag[gb] <= row_flag[b] && (gb < int'(batch_size));
            end
        end
    end
    assign flag_o = global_flag;

    // -------------------------------------------------------------------------
    // Issue FSM: round-robin across rows, issue when FIFO half-empty
    // -------------------------------------------------------------------------
    logic [ROW_ID_BITS-1:0] issue_rr;
    logic [ROW_ID_BITS-1:0] issue_row;
    logic                   issue_valid;

    always_comb begin
        issue_row   = '0;
        issue_valid = 1'b0;
        for (int i = 0; i < B_TILE; i++) begin
            automatic int bi = (int'(issue_rr) + i) % B_TILE;
            if (!issue_valid
                    && !rearm
                    && (state == ACTIVE)
                    && (count[bi] <= FIFO_CTR_W'(FIFO_HALF))
                    && (next_chunk[bi] < total_chunks)) begin
                issue_row   = ROW_ID_BITS'(bi);
                issue_valid = 1'b1;
            end
        end
    end

    wire req_fire = act_bus_if.req_valid && act_bus_if.req_ready;

    always_ff @(posedge clk) begin
        if (reset || pass_reset) begin
            issue_rr <= '0;
            for (int b = 0; b < B_TILE; b++)
                next_chunk[b] <= '0;
        end else if (req_fire) begin
            next_chunk[issue_row] <= next_chunk[issue_row] + CHUNKS_W'(1);
            issue_rr              <= issue_rr + ROW_ID_BITS'(1);
        end
    end

    // -------------------------------------------------------------------------
    // Request address: includes batch_tile offset for the current pass.
    // -------------------------------------------------------------------------
    wire [`MEM_ADDR_WIDTH-1:0] req_byte_addr =
          hidden_base_addr
        + ((`MEM_ADDR_WIDTH'(batch_tile) * B_TILE + `MEM_ADDR_WIDTH'(issue_row))
           * `MEM_ADDR_WIDTH'(hidden_size) * 2)
        + (`MEM_ADDR_WIDTH'(next_chunk[issue_row]) * LINE_BYTES);

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

    always @(posedge clk) begin
        if (!reset && checker_armed) begin
            if (rearm)
                `TRACE(3, ("%t: [CHECKER] armed  base=0x%0h  hidden=%0d  features=%0d  batch=%0d  chunks=%0d\n",
                    $time, hidden_base_addr, hidden_size, num_features, batch_size, total_chunks))
            if (req_fire)
                `TRACE(3, ("%t: [CHECKER] req  bt=%0d ft=%0d row=%0d chunk=%0d addr=0x%0h\n",
                    $time, batch_tile, feat_tile, issue_row, next_chunk[issue_row], req_byte_addr))
            if (rsp_fire)
                `TRACE(3, ("%t: [CHECKER] rsp  row=%0d  data[15:0]=0x%0h\n",
                    $time, rsp_row, act_bus_if.rsp_data.data[15:0]))
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
            if (scan_done_pulse)
                `TRACE(3, ("%t: [CHECKER] scan_done  bt=%0d ft=%0d  row_flag=0x%0x  last_ft=%0b last_bt=%0b\n",
                    $time, batch_tile, feat_tile, row_flag, last_feat_tile, last_batch_tile))
            if (scan_done_pulse && last_feat_tile && last_batch_tile)
                `TRACE(3, ("%t: [CHECKER] ALL_DONE  global_flag=0x%0x\n", $time, global_flag))
        end
    end
`endif

endmodule
