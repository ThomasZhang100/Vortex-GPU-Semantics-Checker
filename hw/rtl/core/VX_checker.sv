// Semantic checker — active-load SAE checker backed by a 4×16 output-stationary
// systolic array (SAURIA sa_array, ARITHMETIC=1 FP16) for the selected-feature SAE matmul.
//
// Activation loading: issues independent load requests through a dedicated L2
// port (act_bus_if).  Four rows (one per token in the batch) are prefetched in
// round-robin order; a new prefetch is issued for row b whenever its FIFO drops
// to half-empty and more chunks remain.
//
// FIFO: each row holds FIFO_DEPTH=64 FP16 elements (2 cache lines).  A cache-
// line response pushes 32 elements; the FIFO consumer pops 1 per cycle.
//
// Skewed/wavefront activation: row 0 starts consuming when its FIFO first has
// data; row b starts exactly 1 cycle after row b-1, establishing the inter-row
// skew required by the systolic array.
//
// Global stall: if ANY started row's FIFO runs empty, all started rows stall
// together so the fixed inter-row skew is never disturbed.
//
// Fully-systolic dataflow (output-stationary, via SAURIA sa_array):
//   Weight SRAM: private VX_dp_ram (MAX_HIDDEN rows × N_FEAT FP16 values).
//     Read at k_count[0] each active cycle.
//   Horizontal weight pipe (w_hpipe, HPIPE_DEPTH=15 stages):
//     col_in[n] = W[k_count[0]-n][n]  — injects weight at top of column n.
//   sa_array (4 rows × 16 cols):
//     i_a_arr[b] = fifo[b][rd_ptr[b]] (zero-masked until k_started[b])
//     i_b_arr[n] = col_in[n]
//     Internal a_q/b_q registers replace the old act_hpipe/w_col.
//     PE[b][n] accumulates A[b][k] * W[k][n] for all k.
//
// Drain: after row 0's last FIFO pop, DRAIN_CYCLES=21 more advances flush the
// systolic pipes (HPIPE_DEPTH=15 + VPIPE_DEPTH=3 + PE_LATENCY=3).
// mac_done pulses once on the falling edge of drain_active.
//
// Readout: one cswitch cycle copies all PE accumulators into the scan chain,
// then N_FEAT+B_TILE-1=19 cscan cycles shift them out via o_c_arr per row.
// cswitch propagates down via o_cswitch: row b's mac_sc_q is captured b cycles
// after cswitch_pulse.  During those same cycles, earlier rows' scan chains are
// already shifting.  Per-row offset: col captured for row b at global cscan cycle
// D (1-indexed) is col = D - b - 1; sa_cscan_en runs for SCAN_INIT = N_FEAT +
// B_TILE - 1 cycles to cover all rows.
// Per-row flagging: during scan, sa_c_out[b] for feature col_i is compared on the
// fly against threshold[col_i] (one FP16 threshold per feature, from THRESHOLD_FILE).
// row_flag[b] starts at 1 and ANDs down; cleared if any feature falls at or below
// its threshold.  flag_o[b] asserts after scan_done for rows b < batch_size that
// pass all N_FEAT feature comparisons.
//
// Enable: -DCHECKER_ENABLE
// DCRs:   VX_DCR_CHECKER_ENABLE      (rising edge = start streaming)
//         VX_DCR_CHECKER_TAP_ADDR0/1 (hidden-state base address)
//         VX_DCR_CHECKER_HIDDEN_SIZE  (FP16 elements per token)
//         VX_DCR_CHECKER_BATCH_SIZE   (tokens in batch, max B_TILE)

`include "VX_define.vh"

module VX_checker import VX_gpu_pkg::*; #(
    parameter B_TILE        = 4,               // rows  (tokens per batch, fixed)
    parameter N_FEAT        = 16,              // SAE feature columns (weight SRAM width)
    parameter MAX_HIDDEN    = 2048,            // max hidden_size supported (SRAM depth)
    parameter FIFO_DEPTH    = 64,              // FP16 slots per row  (2 cache lines)
    parameter LINE_WORDS    = `L1_LINE_SIZE/2, // FP16 values per cache line (64B/2B=32)
    parameter `STRING WEIGHT_FILE    = "",      // $readmemh hex path for SAE weights ("" → zeros)
    parameter `STRING THRESHOLD_FILE = ""      // $readmemh hex path for N_FEAT FP16 thresholds ("" → 0)
) (
    input  wire clk,
    input  wire reset,

    // DCR-supplied config (latched in VX_cluster before vx_start)
    input  wire                         checker_armed,
    input  wire [`MEM_ADDR_WIDTH-1:0]   hidden_base_addr,
    input  wire [15:0]                  hidden_size,   // FP16 elements per token
    input  wire [3:0]                   batch_size,    // tokens this batch (≤ B_TILE)

    // Per-row flags: flag_o[b]=1 after scan if ALL N_FEAT features of row b exceeded
    // their per-feature threshold.  Rows b >= batch_size are always 0.
    output wire [B_TILE-1:0]            flag_o,

    // Dedicated L2 port for activation prefetch
    VX_mem_bus_if.master                act_bus_if
);
    localparam LINE_BYTES   = `L1_LINE_SIZE;           // 64
    localparam LINE_BITS    = `CLOG2(LINE_BYTES);      // 6
    localparam LOG_LW       = `CLOG2(LINE_WORDS);      // log2(32) = 5
    localparam FIFO_PTR_W   = `CLOG2(FIFO_DEPTH);     // 6
    localparam FIFO_CTR_W   = `CLOG2(FIFO_DEPTH + 1); // 7
    localparam FIFO_HALF    = FIFO_DEPTH / 2;          // 32 = issue threshold
    localparam CHUNKS_W     = 12;                      // enough for ceil(65535/32)
    localparam ROW_ID_BITS  = `CLOG2(B_TILE);          // 2

    localparam TAG_VAL_W    = L2_TAG_WIDTH - `UP(UUID_WIDTH); // value field width
    localparam WEIGHT_DATAW = N_FEAT * 16;                   // bits per weight row (256b for N_FEAT=16)
    localparam WEIGHT_ADDRW = `CLOG2(MAX_HIDDEN);            // 11b for MAX_HIDDEN=2048
    // Horizontal pipe (HPIPE_DEPTH stages): injects W[k-n][n] at top of column n.
    // Vertical propagation is handled internally by sa_array's b_q registers (B_TILE-1 stages).
    localparam HPIPE_DEPTH  = N_FEAT - 1;  // 15 for N_FEAT=16 (column injection delay)
    localparam VPIPE_DEPTH  = B_TILE - 1;  // 3  for B_TILE=4  (internal to sa_array)
    localparam PE_LATENCY   = 1;           // STAGES_MUL=0 + INTERMEDIATE_PIPELINE_STAGE=1
    // DRAIN_CYCLES: after row 0's last FIFO pop, keep pipes advancing for this many
    // cycles so the last activation+weight reaches PE[B_TILE-1][N_FEAT-1] and the
    // multiplier pipeline fully drains into the accumulator.
    localparam DRAIN_CYCLES = HPIPE_DEPTH + VPIPE_DEPTH + PE_LATENCY; // 19
    localparam DRAIN_CTR_W  = `CLOG2(DRAIN_CYCLES + 2);  // 5 bits
    localparam ACC_W        = 16;          // sa_array OC_W: FP16 accumulator (matches IA_W=IB_W=16)
    // Scan readout: cswitch fires at τ=0 for row 0, τ=b for row b (propagates via
    // o_cswitch 1 cycle per row).  We run SCAN_INIT = N_FEAT + B_TILE - 1 cscan
    // cycles so every row has N_FEAT capture windows.
    localparam SCAN_INIT    = N_FEAT + B_TILE - 1;        // 19 for N_FEAT=16, B_TILE=4
    localparam SCAN_CTR_W   = `CLOG2(SCAN_INIT + 2);     // enough bits for 0..SCAN_INIT

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
    // State
    // -------------------------------------------------------------------------
    typedef enum logic { IDLE = 1'b0, ACTIVE = 1'b1 } state_t;
    state_t state;

    // -------------------------------------------------------------------------
    // Configuration latched on rearm
    // -------------------------------------------------------------------------
    logic [CHUNKS_W-1:0] total_chunks; // ceil(hidden_size / LINE_WORDS)

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

    // Per-row prefetch state
    logic [B_TILE-1:0][CHUNKS_W-1:0] next_chunk;   // next cache line to prefetch

    // -------------------------------------------------------------------------
    // Response routing
    // -------------------------------------------------------------------------
    wire rsp_fire = act_bus_if.rsp_valid && act_bus_if.rsp_ready;
    wire [ROW_ID_BITS-1:0] rsp_row = act_bus_if.rsp_data.tag.value[ROW_ID_BITS-1:0];

    // -------------------------------------------------------------------------
    // Per-row start flags, consumed-element counter, and done flag
    // -------------------------------------------------------------------------
    logic [B_TILE-1:0] k_started;
    logic [B_TILE-1:0][15:0] k_count;  // total FP16 elements consumed per row

    // Row b is done once it has consumed hidden_size elements.
    logic [B_TILE-1:0] k_done;
    always_comb begin
        for (int b = 0; b < B_TILE; b++)
            k_done[b] = k_started[b] && (k_count[b] >= hidden_size);
    end

    // Stall ALL non-done started rows if any non-done started row's FIFO is
    // empty.  Done rows are excluded so trailing rows can finish their
    // remaining elements after the first row completes.
    logic k_stall;
    always_comb begin
        k_stall = 1'b0;
        for (int b = 0; b < B_TILE; b++) begin
            if (k_started[b] && !k_done[b] && (count[b] == '0))
                k_stall = 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // Drain counter: after row 0's last pop, keep pipes advancing DRAIN_CYCLES
    // more cycles so the tail propagates through the entire sa_array.
    // Gap fix: drain_active is combinationally 1 on the same cycle k_done[0]
    // first rises (before drain_cnt is sequentially loaded), closing the 1-cycle
    // hole where the weight pipe would otherwise miss one advance.
    // -------------------------------------------------------------------------
    logic [DRAIN_CTR_W-1:0] drain_cnt;
    logic                   drain_started;
    wire  drain_active = (drain_cnt != '0) || (k_done[0] && !drain_started);
    // mac_done: 1-cycle pulse on falling edge of drain_active (all PEs complete).
    logic drain_active_r;
    always_ff @(posedge clk) begin
        if (reset || rearm) drain_active_r <= 1'b0;
        else                drain_active_r <= drain_active;
    end
    wire mac_done = drain_active_r && !drain_active;

    always_ff @(posedge clk) begin
        if (reset || rearm) begin
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
    // Push / pop combinational helpers
    // -------------------------------------------------------------------------
    logic [B_TILE-1:0] row_push, row_pop;
    always_comb begin
        for (int b = 0; b < B_TILE; b++) begin
            row_push[b] = rsp_fire && (ROW_ID_BITS'(rsp_row) == ROW_ID_BITS'(b));
            // Done rows stop draining; active rows stall together when any is empty
            row_pop[b]  = k_started[b] && !k_done[b] && !k_stall;
        end
    end

    // -------------------------------------------------------------------------
    // FIFO update (push 32 / pop 1 per row per cycle)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset || rearm) begin
            for (int b = 0; b < B_TILE; b++) begin
                rd_ptr[b]  <= '0;
                wr_ptr[b]  <= '0;
                count[b]   <= '0;
                k_count[b] <= '0;
            end
            k_started <= '0;
            state     <= IDLE;
            if (rearm) state <= ACTIVE;
        end else if (state == ACTIVE) begin
            // Row 0: start as soon as its FIFO has any data
            if (!k_started[0] && (count[0] > '0))
                k_started[0] <= 1'b1;
            // Row b: start the cycle row b-1 first pops (1-cycle activation skew).
            for (int b = 1; b < B_TILE; b++) begin
                if (!k_started[b] && row_pop[b-1])
                    k_started[b] <= 1'b1;
            end

            for (int b = 0; b < B_TILE; b++) begin
                // --- Push: write LINE_WORDS elements at wr_ptr ---
                if (row_push[b]) begin
                    for (int w = 0; w < LINE_WORDS; w++) begin
                        fifo[b][FIFO_PTR_W'(wr_ptr[b] + FIFO_PTR_W'(w))]
                            <= act_bus_if.rsp_data.data[w*16 +: 16];
                    end
                    wr_ptr[b] <= FIFO_PTR_W'(wr_ptr[b] + FIFO_PTR_W'(LINE_WORDS));
                end

                // --- Pop: advance read pointer ---
                if (row_pop[b]) begin
                    rd_ptr[b]  <= rd_ptr[b] + FIFO_PTR_W'(1);
                    k_count[b] <= k_count[b] + 16'(1);
                end

                // --- Count: handle simultaneous push + pop ---
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
    // Weight SRAM: private N_FEAT×MAX_HIDDEN FP16 storage
    // Each entry W[k] = N_FEAT FP16 values = WEIGHT_DATAW bits.
    // Read combinationally (OUT_REG=0) at raddr = k_count[0].
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

    // Per-feature FP16 thresholds: N_FEAT values loaded from THRESHOLD_FILE at init.
    // If THRESHOLD_FILE is empty all thresholds default to +0 (every positive value passes).
    logic [15:0] threshold [0:N_FEAT-1];
    initial begin
        if (THRESHOLD_FILE != "") $readmemh(THRESHOLD_FILE, threshold);
        else for (int n = 0; n < N_FEAT; n++) threshold[n] = '0;
    end

    // Horizontal weight pipe (HPIPE_DEPTH = N_FEAT-1 = 15 stages, each a full weight row).
    // col_in[n] = W[k_count[0]-n][n]: the weight entering at the TOP of column n.
    // Gate on row_pop[0] OR drain_active to freeze during stalls, drain after last pop.
    logic [HPIPE_DEPTH-1:0][WEIGHT_DATAW-1:0] w_hpipe;
    always_ff @(posedge clk) begin
        if (reset || rearm) begin
            for (int s = 0; s < HPIPE_DEPTH; s++) w_hpipe[s] <= '0;
        end else if (row_pop[0] || drain_active) begin
            w_hpipe[0] <= weight_row_out;
            for (int s = 1; s < HPIPE_DEPTH; s++)
                w_hpipe[s] <= w_hpipe[s-1];
        end
    end

    // Column injection: col_in[n] = feature n of weight row delayed n cycles.
    wire [N_FEAT-1:0][15:0] col_in;
    assign col_in[0] = weight_row_out[15:0];
    generate
        for (genvar n = 1; n < N_FEAT; n++) begin : g_colin
            assign col_in[n] = w_hpipe[n-1][n*16 +: 16];
        end
    endgenerate

    // -------------------------------------------------------------------------
    // SAURIA sa_array: 4×16 output-stationary systolic array (ARITHMETIC=0, integer)
    // -------------------------------------------------------------------------

    // Activation inputs: inject zero when row b hasn't started yet OR is finished.
    // The "done" mask is required for correctness during drain: after row b's last
    // pop, rd_ptr[b] points one past the valid data; without the mask, the stale
    // FIFO slot is injected and accumulates a garbage product at PE[b][n].
    /* verilator lint_off ASCRANGE */ // ascending dims match SAURIA port declarations
    wire [0:B_TILE-1][15:0] sa_a_in;
    /* verilator lint_on ASCRANGE */
    generate
        for (genvar b = 0; b < B_TILE; b++) begin : g_sa_ain
            assign sa_a_in[b] = (k_started[b] && !k_done[b]) ? fifo[b][rd_ptr[b]] : 16'h0;
        end
    endgenerate

    // Weight inputs: col_in[n] already encodes the n-cycle column skew via w_hpipe.
    /* verilator lint_off ASCRANGE */
    wire [0:N_FEAT-1][15:0] sa_b_in;
    /* verilator lint_on ASCRANGE */
    generate
        for (genvar n = 0; n < N_FEAT; n++) begin : g_sa_bin
            assign sa_b_in[n] = col_in[n];
        end
    endgenerate

    // Pipeline enable: advance sa_array during active pops and drain.
    wire sa_pipeline_en = row_pop[0] || drain_active;

    // Scan chain readout control.
    // With STAGES_MUL=0 and INTERMEDIATE_PIPELINE_STAGE=1, PE_LATENCY=1: the accumulator
    // mac_q has period 1 (single sub-accumulator), so one cswitch + one scan-out captures
    // the complete dot product. No multi-pass needed.
    logic cswitch_pulse;
    logic [SCAN_CTR_W-1:0] scan_cnt;
    always_ff @(posedge clk) begin
        if (reset || rearm) begin
            cswitch_pulse <= 1'b0;
            scan_cnt      <= '0;
        end else begin
            cswitch_pulse <= mac_done;
            if (cswitch_pulse) scan_cnt <= SCAN_CTR_W'(SCAN_INIT);
            else if (scan_cnt > '0) scan_cnt <= scan_cnt - SCAN_CTR_W'(1);
        end
    end
    wire sa_cscan_en = (scan_cnt > '0);

    // Keep i_pipeline_en high during cswitch and scan so sc_reg_en propagates.
    wire sa_pipeline_en_full = sa_pipeline_en || cswitch_pulse || sa_cscan_en;

    /* verilator lint_off ASCRANGE */
    wire [0:B_TILE-1][ACC_W-1:0] sa_c_out;
    /* verilator lint_on ASCRANGE */

    sa_array #(
        .ARITHMETIC              (1),   // FP16 FMA via sauria_fpnew_fma (renamed from fpnew_fma)
        .MUL_TYPE                (0),   // inferred exact multiplier
        .ADD_TYPE                (0),   // ideal adder
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
        .ZERO_GATING_MULT        (0),   // disabled; avoids zero_det_neg overhead
        .ZERO_GATING_ADD         (0),
        .ZD_LOOKAHEAD            (0),
        .EXTRA_CSREG             (0)
    ) mac_array (
        .i_clk          (clk),
        .i_rstn         (!reset),       // async active-low; i_reg_clear handles rearm
        .i_a_arr        (sa_a_in),
        .i_b_arr        (sa_b_in),
        .i_c_arr        ('0),           // scan chain preload = zero
        .i_reg_clear    (rearm),        // synchronous accumulator clear between batches
        .i_pipeline_en  (sa_pipeline_en_full),
        .i_cswitch_arr  (cswitch_pulse ? {N_FEAT{1'b1}} : {N_FEAT{1'b0}}),
        .i_cscan_en     (sa_cscan_en),
        .i_thres        ('0),
        .o_c_arr        (sa_c_out)
    );

    // FP16 greater-than for non-negative thresholds.
    // Negative a (sign bit set) is always below any non-negative threshold.
    // For two non-negative FP16 values, IEEE 754 ordering matches unsigned bit ordering.
    function automatic logic fp16_gt(input logic [15:0] a, input logic [15:0] th);
        if (a[15])  return 1'b0;   // a is negative → always below non-negative threshold
        if (th[15]) return 1'b1;   // threshold is negative → every non-negative a passes
        return a[14:0] > th[14:0]; // both non-negative: IEEE 754 ordering = unsigned bit ordering
    endfunction

    // Per-row flag: starts all-1 each batch; ANDs in per-feature comparisons as
    // values scan out.  row_flag[b] becomes 0 if any feature of row b falls at or
    // below its threshold.
    logic [B_TILE-1:0] row_flag;
    always_ff @(posedge clk) begin
        if (reset || rearm) begin
            row_flag <= '1;
        end else if (sa_cscan_en) begin
            for (int b = 0; b < B_TILE; b++) begin
                automatic int col_i = int'(SCAN_INIT) - int'(scan_cnt) - b;
                if (col_i >= 0 && col_i < N_FEAT) begin
                    if (!fp16_gt(sa_c_out[b], threshold[col_i]))
                        row_flag[b] <= 1'b0;
                end
            end
        end
    end

    // scan_done_pulse: 1 cycle after the last scan cycle completes.
    logic scan_done_pulse;
    always_ff @(posedge clk) scan_done_pulse <= (scan_cnt == SCAN_CTR_W'(1));

    // Latch per-row flags on scan_done; mask rows at or beyond batch_size to 0.
    logic [B_TILE-1:0] flag_r;
    always_ff @(posedge clk) begin
        if (reset || rearm) flag_r <= '0;
        else if (scan_done_pulse) begin
            for (int b = 0; b < B_TILE; b++)
                flag_r[b] <= row_flag[b] && (4'(b) < batch_size);
        end
    end
    assign flag_o = flag_r;

    // -------------------------------------------------------------------------
    // Issue FSM: round-robin across rows, issue when FIFO half-empty
    // -------------------------------------------------------------------------
    logic [ROW_ID_BITS-1:0] issue_rr;   // round-robin base

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
        if (reset || rearm) begin
            issue_rr <= '0;
            for (int b = 0; b < B_TILE; b++)
                next_chunk[b] <= '0;
        end else if (req_fire) begin
            next_chunk[issue_row] <= next_chunk[issue_row] + CHUNKS_W'(1);
            issue_rr              <= issue_rr + ROW_ID_BITS'(1);
        end
    end

    // -------------------------------------------------------------------------
    // Request address calculation
    // -------------------------------------------------------------------------
    wire [`MEM_ADDR_WIDTH-1:0] req_byte_addr =
          hidden_base_addr
        + (`MEM_ADDR_WIDTH'(issue_row)              * `MEM_ADDR_WIDTH'(hidden_size) * 2)
        + (`MEM_ADDR_WIDTH'(next_chunk[issue_row])  * LINE_BYTES);

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
        if (reset || rearm) k_done_r <= '0;
        else                k_done_r <= k_done;
    end

    always @(posedge clk) begin
        if (!reset && checker_armed) begin
            if (rearm)
                `TRACE(3, ("%t: [CHECKER] armed  base=0x%0h  hidden_size=%0d  batch=%0d  chunks=%0d\n",
                    $time, hidden_base_addr, hidden_size, batch_size, total_chunks))
            if (req_fire)
                `TRACE(3, ("%t: [CHECKER] req  row=%0d  chunk=%0d  addr=0x%0h\n",
                    $time, issue_row, next_chunk[issue_row],
                    req_byte_addr))
            if (rsp_fire)
                `TRACE(3, ("%t: [CHECKER] rsp  row=%0d  data[15:0]=0x%0h  count_after=%0d\n",
                    $time, rsp_row,
                    act_bus_if.rsp_data.data[15:0],
                    count[rsp_row] + (row_pop[rsp_row] ? FIFO_CTR_W'(LINE_WORDS)-1 : FIFO_CTR_W'(LINE_WORDS))))
            if (k_stall)
                `TRACE(3, ("%t: [CHECKER] stall  k_started=%0b  k_done=%0b\n",
                    $time, k_started, k_done))
            for (int b = 0; b < B_TILE; b++) begin
                if (k_done[b] && !k_done_r[b])
                    `TRACE(3, ("%t: [CHECKER] done  row=%0d  consumed=%0d\n",
                        $time, b, k_count[b]))
                if (row_pop[b])
                    `TRACE(3, ("%t: [CHECKER] pop  row=%0d  k=%0d  a_fifo=0x%0h  col_in[0]=0x%0h  col_in[1]=0x%0h\n",
                        $time, b, rd_ptr[b],
                        fifo[b][rd_ptr[b]], col_in[0], col_in[1]))
            end
            if (drain_cnt != '0)
                `TRACE(3, ("%t: [CHECKER] drain  cnt=%0d  col_in[0]=0x%0h  col_in[%0d]=0x%0h\n",
                    $time, drain_cnt, col_in[0], N_FEAT-1, col_in[N_FEAT-1]))
            if (mac_done)
                `TRACE(3, ("%t: [CHECKER] mac_done — starting cswitch+scan readout\n", $time))
            if (cswitch_pulse)
                `TRACE(3, ("%t: [CHECKER] cswitch — accumulators latched into scan chain\n", $time))
            if (sa_cscan_en)
                `TRACE(3, ("%t: [CHECKER] scan  cnt=%0d  col[0]=%0d  sa_c_out[0]=0x%0h  sa_c_out[%0d]=0x%0h\n",
                    $time, scan_cnt, int'(SCAN_INIT) - int'(scan_cnt),
                    sa_c_out[0], B_TILE-1, sa_c_out[B_TILE-1]))
            if (scan_done_pulse)
                `TRACE(3, ("%t: [CHECKER] scan_done  row_flag=0x%0x  flag[0]=%0b  flag[1]=%0b  flag[2]=%0b  flag[3]=%0b\n",
                    $time, row_flag,
                    row_flag[0], row_flag[1], row_flag[2], row_flag[3]))
        end
    end
`endif

endmodule
