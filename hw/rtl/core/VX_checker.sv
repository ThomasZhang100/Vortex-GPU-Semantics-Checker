// Semantic checker — active-load SAE checker.
//
// Activation loading: issues independent load requests through a dedicated L2
// port (act_bus_if).  Four rows (one per token in the batch) are prefetched in
// round-robin order; a new prefetch is issued for row b whenever its FIFO drops
// to half-empty and more chunks remain.
//
// FIFO: each row holds FIFO_DEPTH=64 FP16 elements (2 cache lines).  A cache-
// line response pushes 32 elements; the dummy consumer pops 1 per cycle.
//
// Skewed/wavefront activation: row 0 starts consuming when its FIFO first
// has data; row b starts exactly 1 cycle after row b-1.  This produces the
// 1-cycle activation stagger a systolic array requires: at cycle t, row b
// is processing k = t - b.
//
// Global stall: if ANY started row's FIFO runs empty, all started rows stall
// together so the fixed inter-row skew is never disturbed.
//
// Fully-systolic dataflow (output-stationary):
//   Activations flow rightward: A[b][k] enters PE[b][0] from the FIFO, shifts
//   right one column per cycle via act_hpipe[b][0..N_FEAT-2].
//   Weights flow downward: W[k][n] is injected at PE[0][n] and propagates down.
//   At time t, PE[b][n] sees A[b][k] and W[k][n] where k = t - b - n.
//
// Weight SRAM: private VX_dp_ram (MAX_HIDDEN × N_FEAT FP16, 1 row/cycle).
//   Read combinationally at raddr = k_count[0].
//   Horizontal pipe (w_hpipe, N_FEAT-1=15 stages): col_in[n] = W[k-n][n].
//   Vertical pipe (w_col[n], B_TILE-1=3 stages per column): weight flows downward.
//     PE[0][n] = col_in[n];  PE[b][n] = w_col[n][b-1]  (b >= 1)
//
// Drain: after row 0's last FIFO pop a DRAIN_CYCLES=18-cycle drain phase keeps
// all shift registers advancing so the tail propagates to PE[B_TILE-1][N_FEAT-1].
// mac_done pulses once the drain completes; accumulators (TODO) gate on row_pop||drain_active.
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
    parameter `STRING WEIGHT_FILE = ""         // $readmemh hex path for SAE weights ("" → zeros)
) (
    input  wire clk,
    input  wire reset,

    // DCR-supplied config (latched in VX_cluster before vx_start)
    input  wire                         checker_armed,
    input  wire [`MEM_ADDR_WIDTH-1:0]   hidden_base_addr,
    input  wire [15:0]                  hidden_size,   // FP16 elements per token
    input  wire [3:0]                   batch_size,    // tokens this batch (≤ B_TILE)

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
    // Fully-systolic weight distribution:
    //   Horizontal pipe (HPIPE_DEPTH stages): injects W[k-n][n] at top of column n.
    //   Vertical pipe (VPIPE_DEPTH stages per column): propagates weight downward through rows.
    localparam HPIPE_DEPTH  = N_FEAT - 1;  // 15 for N_FEAT=16 (column injection delay)
    localparam VPIPE_DEPTH  = B_TILE - 1;  // 3  for B_TILE=4  (row propagation depth)
    // After row 0's last FIFO pop the systolic pipes still hold tail data.
    // DRAIN_CYCLES = 18 ensures W[hidden_size-1] reaches w_pe[B_TILE-1][N_FEAT-1]
    // (15 hpipe stages + 3 vpipe stages) and the last activation for row B_TILE-1
    // reaches a_pe[B_TILE-1][N_FEAT-1] (B_TILE-1 row skew + N_FEAT-1 hpipe stages).
    localparam DRAIN_CYCLES = HPIPE_DEPTH + VPIPE_DEPTH; // 18
    localparam DRAIN_CTR_W  = `CLOG2(DRAIN_CYCLES + 2);  // 5 bits

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
    // Drain counter: after row 0's last pop, keep pipes advancing for
    // DRAIN_CYCLES more cycles so the tail propagates to PE[B_TILE-1][N_FEAT-1].
    // -------------------------------------------------------------------------
    logic [DRAIN_CTR_W-1:0] drain_cnt;
    logic                   drain_started;
    // drain_active covers the gap cycle: k_done[0] rises one cycle before drain_cnt
    // is loaded (sequential), so we OR in the combinational condition to ensure the
    // weight pipe does not miss its first advance.
    wire   drain_active = (drain_cnt != '0) || (k_done[0] && !drain_started);
    // mac_done: 1-cycle pulse on the falling edge of drain_active (drain complete).
    logic  drain_active_r;
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
            // Triggering on row_pop[b-1] (actual pop) rather than k_started[b-1]
            // (conceptual start) ensures k_started[b] never fires during a stall,
            // so the fixed inter-row offset is established correctly at first start.
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

                // --- Pop: consume 1 activation element; systolic array hook ---
                if (row_pop[b]) begin
                    // TODO: feed a_pe[b][n] * w_pe[b][n] into acc[b][n] for all n
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
    // Write port unused at runtime — weights loaded via $readmemh(WEIGHT_FILE).
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
        // write port: unused at runtime (deploy-time load only via INIT_FILE)
        .write (1'b0),
        .wren  (1'b1),
        .waddr ({WEIGHT_ADDRW{1'b0}}),
        .wdata ({WEIGHT_DATAW{1'b0}}),
        // read port: sequential scan of W[0..hidden_size-1], indexed by row 0's k
        .read  (1'b1),
        .raddr (k_count[0][WEIGHT_ADDRW-1:0]),
        .rdata (weight_row_out)
    );

    // Horizontal weight pipe (HPIPE_DEPTH = N_FEAT-1 = 15 stages, each a full weight row).
    // w_hpipe[0] = W[k] registered 1 cycle ago; w_hpipe[n-1] = W[k-n] (n cycles ago).
    // Gate on row_pop[0] OR drain_active: freezes during stalls but drains after last pop.
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

    // Column injection: col_in[n] = feature n delayed n cycles = W[k-n][n].
    // This is the weight entering at the TOP of column n (PE[0][n]).
    //   col_in[0] = weight_row_out[15:0]           (W[k][0], no delay)
    //   col_in[n] = w_hpipe[n-1][n*16 +: 16]      (W[k-n][n], n-cycle delayed)
    wire [N_FEAT-1:0][15:0] col_in;
    assign col_in[0] = weight_row_out[15:0];
    generate
        for (genvar n = 1; n < N_FEAT; n++) begin : g_colin
            assign col_in[n] = w_hpipe[n-1][n*16 +: 16];
        end
    endgenerate

    // Vertical weight pipe: each column has its own VPIPE_DEPTH=3-stage downward pipe.
    // w_col[n][0] = col_in[n] registered 1 cycle → feeds PE[1][n]
    // w_col[n][b-1]                               → feeds PE[b][n], for b >= 1
    logic [N_FEAT-1:0][VPIPE_DEPTH-1:0][15:0] w_col;
    always_ff @(posedge clk) begin
        if (reset || rearm) begin
            for (int n = 0; n < N_FEAT; n++)
                for (int b = 0; b < VPIPE_DEPTH; b++)
                    w_col[n][b] <= '0;
        end else if (row_pop[0] || drain_active) begin
            for (int n = 0; n < N_FEAT; n++) begin
                w_col[n][0] <= col_in[n];
                for (int b = 1; b < VPIPE_DEPTH; b++)
                    w_col[n][b] <= w_col[n][b-1];
            end
        end
    end

    // PE weight inputs: row 0 gets col_in (top of column); rows 1-3 get vertical pipe.
    wire [B_TILE-1:0][N_FEAT-1:0][15:0] w_pe;
    generate
        for (genvar n = 0; n < N_FEAT; n++) begin : g_wpe_col
            assign w_pe[0][n] = col_in[n];
            for (genvar b = 1; b < B_TILE; b++) begin : g_wpe_row
                assign w_pe[b][n] = w_col[n][b-1];
            end
        end
    endgenerate

    // Per-row activation horizontal shift register (N_FEAT-1 stages).
    // act_hpipe[b][0] = A[b][k] from 1 cycle ago (for PE[b][1]).
    // act_hpipe[b][n-1] feeds PE[b][n].  Gate on row_pop[b].
    logic [B_TILE-1:0][N_FEAT-2:0][15:0] act_hpipe;
    always_ff @(posedge clk) begin
        for (int b = 0; b < B_TILE; b++) begin
            if (reset || rearm) begin
                for (int n = 0; n < N_FEAT-1; n++) act_hpipe[b][n] <= '0;
            end else if (row_pop[b] || drain_active) begin
                act_hpipe[b][0] <= fifo[b][rd_ptr[b]];
                for (int n = 1; n < N_FEAT-1; n++)
                    act_hpipe[b][n] <= act_hpipe[b][n-1];
            end
        end
    end

    // PE activation inputs: a_pe[b][0] = FIFO head (combinational); a_pe[b][n] = n cycles ago.
    wire [B_TILE-1:0][N_FEAT-1:0][15:0] a_pe;
    generate
        for (genvar b = 0; b < B_TILE; b++) begin : g_ape_row
            assign a_pe[b][0] = fifo[b][rd_ptr[b]];
            for (genvar n = 1; n < N_FEAT; n++) begin : g_ape_col
                assign a_pe[b][n] = act_hpipe[b][n-1];
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Issue FSM: round-robin across rows, issue when FIFO half-empty
    // -------------------------------------------------------------------------
    logic [ROW_ID_BITS-1:0] issue_rr;   // round-robin base

    // Combinational: find next row needing a prefetch (starting from issue_rr)
    logic [ROW_ID_BITS-1:0] issue_row;
    logic                   issue_valid;

    always_comb begin
        issue_row   = '0;
        issue_valid = 1'b0;
        for (int i = 0; i < B_TILE; i++) begin
            automatic int bi = (int'(issue_rr) + i) % B_TILE;
            // Suppress during rearm: Verilator evaluates comb after NBA so
            // state=ACTIVE is already visible at the rearm cycle; without this
            // gate a phantom L2 request fires before next_chunk can increment.
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

    // Advance round-robin and chunk counter when a request fires
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
    // token b starts at: hidden_base_addr + b * hidden_size * 2
    // chunk c within token b: offset += c * LINE_BYTES
    // -------------------------------------------------------------------------
    wire [`MEM_ADDR_WIDTH-1:0] req_byte_addr =
          hidden_base_addr
        + (`MEM_ADDR_WIDTH'(issue_row)              * `MEM_ADDR_WIDTH'(hidden_size) * 2)
        + (`MEM_ADDR_WIDTH'(next_chunk[issue_row])  * LINE_BYTES);

    // -------------------------------------------------------------------------
    // Drive act_bus_if
    // -------------------------------------------------------------------------
    assign act_bus_if.req_valid            = issue_valid;
    assign act_bus_if.req_data.rw          = 1'b0;             // load
    assign act_bus_if.req_data.addr        = req_byte_addr[`MEM_ADDR_WIDTH-1:LINE_BITS];
    assign act_bus_if.req_data.data        = '0;
    assign act_bus_if.req_data.byteen      = '1;
    assign act_bus_if.req_data.flags       = '0;
    assign act_bus_if.req_data.tag.uuid    = '0;
    // Lower ROW_ID_BITS of value encode the row; upper bits zero
    assign act_bus_if.req_data.tag.value   = TAG_VAL_W'(issue_row);

    assign act_bus_if.rsp_ready = 1'b1;   // always accept responses

    // -------------------------------------------------------------------------
    // Simulation traces
    // -------------------------------------------------------------------------
`ifdef SIMULATION
    // One-cycle delayed k_done for rising-edge detection in the trace block.
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
                // Show col 0 and col 1 to verify both activation horizontal shift and weight skew.
                if (row_pop[b])
                    `TRACE(3, ("%t: [CHECKER] pop  row=%0d  k=%0d  a[col0]=0x%0h  a[col1]=0x%0h  w[col0]=0x%0h  w[col1]=0x%0h\n",
                        $time, b, rd_ptr[b],
                        a_pe[b][0], a_pe[b][1],
                        w_pe[b][0], w_pe[b][1]))
            end
            // Show drain only when cnt > 0 (skips the gap cycle where drain_active=1 but cnt=0).
            if (drain_cnt != '0)
                `TRACE(3, ("%t: [CHECKER] drain  cnt=%0d  w_pe[0][0]=0x%0h  w_pe[%0d][%0d]=0x%0h  a_pe[%0d][%0d]=0x%0h\n",
                    $time, drain_cnt,
                    w_pe[0][0],
                    B_TILE-1, N_FEAT-1, w_pe[B_TILE-1][N_FEAT-1],
                    B_TILE-1, N_FEAT-1, a_pe[B_TILE-1][N_FEAT-1]))
            if (mac_done)
                `TRACE(3, ("%t: [CHECKER] mac_done — all PEs complete\n", $time))
        end
    end

    // Skew verification trace (level 4) — fires every active cycle (not just on pop).
    //
    // With identity weights W[k][n] = float16(k) for all n, the expected values are:
    //   w_pe[b][n]  == float16(k_count[0] - b - n)   (combined row+col skew)
    //   a_pe[b][n]  == A[b][k_count[0] - b - n]       (activation from FIFO, same k)
    //
    // The 2×2 corner [0..1][0..1] covers both skew directions; [B_TILE-1][N_FEAT-1]
    // covers the maximum combined skew of (B_TILE-1)+(N_FEAT-1) cycles.
    always @(posedge clk) begin
        if (!reset && k_started[0] && !k_done[0]) begin
            `TRACE(4, ("%t: [CHECKER] skew  K=%0d\n", $time, k_count[0]))
            `TRACE(4, ("  w_pe[0][0]=0x%0h  w_pe[0][1]=0x%0h  (expected: K, K-1)\n",
                w_pe[0][0], w_pe[0][1]))
            `TRACE(4, ("  w_pe[1][0]=0x%0h  w_pe[1][1]=0x%0h  (expected: K-1, K-2)\n",
                w_pe[1][0], w_pe[1][1]))
            `TRACE(4, ("  w_pe[%0d][%0d]=0x%0h  (expected: K-%0d  max combined skew)\n",
                B_TILE-1, N_FEAT-1, w_pe[B_TILE-1][N_FEAT-1], (B_TILE-1)+(N_FEAT-1)))
            `TRACE(4, ("  a_pe[0][0]=0x%0h  a_pe[0][1]=0x%0h  a_pe[1][0]=0x%0h  a_pe[1][1]=0x%0h\n",
                a_pe[0][0], a_pe[0][1], a_pe[1][0], a_pe[1][1]))
        end
    end
`endif

endmodule
