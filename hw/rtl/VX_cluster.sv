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

`include "VX_define.vh"

module VX_cluster import VX_gpu_pkg::*; #(
    parameter CLUSTER_ID = 0,
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    // Clock
    input  wire                 clk,
    input  wire                 reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t         sysmem_perf,
`endif

    // DCRs
    VX_dcr_bus_if.slave         dcr_bus_if,

    // Memory
    VX_mem_bus_if.master        mem_bus_if [`L2_MEM_PORTS],

    // Status
    output wire                 busy
);

`ifdef SCOPE
    localparam scope_socket = 0;
    `SCOPE_IO_SWITCH (NUM_SOCKETS);
`endif

`ifdef PERF_ENABLE
    cache_perf_t l2_perf;
    sysmem_perf_t sysmem_perf_tmp;
    always @(*) begin
        sysmem_perf_tmp = sysmem_perf;
        sysmem_perf_tmp.l2cache = l2_perf;
    end
`endif

`ifdef GBAR_ENABLE

    VX_gbar_bus_if per_socket_gbar_bus_if[NUM_SOCKETS]();
    VX_gbar_bus_if gbar_bus_if();

    VX_gbar_arb #(
        .NUM_REQS (NUM_SOCKETS),
        .OUT_BUF  ((NUM_SOCKETS > 2) ? 1 : 0) // bgar_unit has no backpressure
    ) gbar_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_socket_gbar_bus_if),
        .bus_out_if (gbar_bus_if)
    );

    VX_gbar_unit #(
        .INSTANCE_ID (`SFORMATF(("gbar%0d", CLUSTER_ID)))
    ) gbar_unit (
        .clk         (clk),
        .reset       (reset),
        .gbar_bus_if (gbar_bus_if)
    );

`endif

    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) per_socket_mem_bus_if[NUM_SOCKETS * `L1_MEM_PORTS]();

    // Combined L2 core bus: socket ports [0..N-1], checker port [N] (CHECKER_ENABLE only).
    // L2_NUM_REQS == NUM_SOCKETS*L1_MEM_PORTS without CHECKER_ENABLE, so size is unchanged.
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) l2_core_bus_if[L2_NUM_REQS]();

    for (genvar p = 0; p < NUM_SOCKETS * `L1_MEM_PORTS; ++p) begin : g_l2_sock
        `ASSIGN_VX_MEM_BUS_IF (l2_core_bus_if[p], per_socket_mem_bus_if[p]);
    end

    `RESET_RELAY (l2_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    (`SFORMATF(("%s-l2cache", INSTANCE_ID))),
        .CACHE_SIZE     (`L2_CACHE_SIZE),
        .LINE_SIZE      (`L2_LINE_SIZE),
        .NUM_BANKS      (`L2_NUM_BANKS),
        .NUM_WAYS       (`L2_NUM_WAYS),
        .WORD_SIZE      (L2_WORD_SIZE),
        .NUM_REQS       (L2_NUM_REQS),
        .MEM_PORTS      (`L2_MEM_PORTS),
        .CRSQ_SIZE      (`L2_CRSQ_SIZE),
        .MSHR_SIZE      (`L2_MSHR_SIZE),
        .MRSQ_SIZE      (`L2_MRSQ_SIZE),
        .MREQ_SIZE      (`L2_WRITEBACK ? `L2_MSHR_SIZE : `L2_MREQ_SIZE),
        .TAG_WIDTH      (L2_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (`L2_WRITEBACK),
        .DIRTY_BYTES    (`L2_DIRTYBYTES),
        .REPL_POLICY    (`L2_REPL_POLICY),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`L2_ENABLED)
    ) l2cache (
        .clk            (clk),
        .reset          (l2_reset),
    `ifdef PERF_ENABLE
        .cache_perf     (l2_perf),
    `endif
        .core_bus_if    (l2_core_bus_if),
        .mem_bus_if     (mem_bus_if)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_SOCKETS-1:0] per_socket_busy;

    // Generate all sockets
    for (genvar socket_id = 0; socket_id < NUM_SOCKETS; ++socket_id) begin : g_sockets

        `RESET_RELAY (socket_reset, reset);

        VX_dcr_bus_if socket_dcr_bus_if();
        wire is_base_dcr_addr = (dcr_bus_if.write_addr >= `VX_DCR_BASE_STATE_BEGIN && dcr_bus_if.write_addr < `VX_DCR_BASE_STATE_END);
        `BUFFER_DCR_BUS_IF (socket_dcr_bus_if, dcr_bus_if, is_base_dcr_addr, (NUM_SOCKETS > 1))

        VX_socket #(
            .SOCKET_ID ((CLUSTER_ID * NUM_SOCKETS) + socket_id),
            .INSTANCE_ID (`SFORMATF(("%s-socket%0d", INSTANCE_ID, socket_id)))
        ) socket (
            `SCOPE_IO_BIND  (scope_socket+socket_id)

            .clk            (clk),
            .reset          (socket_reset),

        `ifdef PERF_ENABLE
            .sysmem_perf    (sysmem_perf_tmp),
        `endif

            .dcr_bus_if     (socket_dcr_bus_if),

            .mem_bus_if     (per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS +: `L1_MEM_PORTS]),

        `ifdef GBAR_ENABLE
            .gbar_bus_if    (per_socket_gbar_bus_if[socket_id]),
        `endif

            .busy           (per_socket_busy[socket_id])
        );
    end

    `BUFFER_EX(busy, (| per_socket_busy), 1'b1, 1, (NUM_SOCKETS > 1));

`ifdef CHECKER_ENABLE
    // DCR latch: captures VX_DCR_CHECKER_* writes from the host before vx_start.
    // Intentionally has no synchronous reset so values survive processor::run()'s
    // reset pulse (same pattern as VX_dcr_data.sv).
    logic                        checker_armed;
    logic                        checker_addr_trig_en;   // ENABLE DCR bit 1
    logic [`MEM_ADDR_WIDTH-1:0]  checker_hidden_base_addr;
    logic [15:0]                 checker_hidden_size;
    logic [15:0]                 checker_batch_size;
    logic [15:0]                 checker_num_features;
    logic [`MEM_ADDR_WIDTH-1:0]  checker_trigger_lo;     // trigger range low  (inclusive)
    logic [`MEM_ADDR_WIDTH-1:0]  checker_trigger_hi;     // trigger range high (exclusive)

    // Weight SRAM streaming state — must match VX_checker parameter defaults.
    localparam CHK_MAX_FEAT  = 64;
    localparam CHK_MAX_HIDN  = 2048;
    localparam CHK_W_DATAW   = CHK_MAX_FEAT * 16;      // 1024 bits per SRAM row
    localparam CHK_W_ADDRW   = $clog2(CHK_MAX_HIDN);   // 11
    localparam CHK_W_WORDS   = CHK_W_DATAW / 32;        // 32 uint32 words per row
    localparam CHK_W_WORD_W  = $clog2(CHK_W_WORDS);     // 5
    localparam CHK_T_CNT     = CHK_MAX_FEAT + 1;        // threshold[0..64]
    localparam CHK_T_IDX_W   = $clog2(CHK_T_CNT + 1);  // 7

    logic [CHK_W_ADDRW-1:0]  w_wrow;   // next SRAM row to write
    logic [CHK_W_WORD_W-1:0] w_wcol;   // word position within the row (0..CHK_W_WORDS-1)
    logic [CHK_W_DATAW-1:0]  w_wbuf;   // accumulation buffer for the current row
    logic                     w_we;    // single-cycle pulse: write w_wbuf to w_waddr
    logic [CHK_W_ADDRW-1:0]  w_waddr;  // SRAM row address latched when w_we fires
    logic [CHK_T_IDX_W-1:0]  t_widx;   // next threshold index to write
    logic                     t_we;    // single-cycle pulse: write t_wdata to t_waddr
    logic [CHK_T_IDX_W-1:0]  t_waddr;  // threshold index latched when t_we fires
    logic [15:0]              t_wdata;  // threshold value latched when t_we fires

    initial begin
        checker_armed            = 0;
        checker_addr_trig_en     = 0;
        checker_hidden_base_addr = 0;
        checker_hidden_size      = 0;
        checker_batch_size       = 0;
        checker_num_features     = 0;
        checker_trigger_lo       = '0;
        checker_trigger_hi       = '0;
        w_wrow  = '0;
        w_wcol  = '0;
        w_wbuf  = '0;
        w_we    = 1'b0;
        w_waddr = '0;
        t_widx  = '0;
        t_we    = 1'b0;
        t_waddr = '0;
        t_wdata = 16'h0;
    end

    always @(posedge clk) begin
        // Default: clear write-enable pulses each cycle.
        w_we <= 1'b0;
        t_we <= 1'b0;

        if (dcr_bus_if.write_valid) begin
            case (dcr_bus_if.write_addr)
                `VX_DCR_CHECKER_ENABLE: begin
                    checker_armed        <= dcr_bus_if.write_data[0];
                    checker_addr_trig_en <= dcr_bus_if.write_data[1];
                end
                `VX_DCR_CHECKER_TAP_ADDR0:
                    checker_hidden_base_addr[31:0]              <= dcr_bus_if.write_data;
            `ifdef XLEN_64
                `VX_DCR_CHECKER_TAP_ADDR1:
                    checker_hidden_base_addr[`MEM_ADDR_WIDTH-1:32] <= (`MEM_ADDR_WIDTH-32)'(dcr_bus_if.write_data);
            `endif
                `VX_DCR_CHECKER_HIDDEN_SIZE:
                    checker_hidden_size                         <= dcr_bus_if.write_data[15:0];
                `VX_DCR_CHECKER_BATCH_SIZE:
                    checker_batch_size                          <= dcr_bus_if.write_data[15:0];
                `VX_DCR_CHECKER_NUM_FEATURES:
                    checker_num_features                        <= dcr_bus_if.write_data[15:0];
                `VX_DCR_CHECKER_TRIG_ADDR_LO:
                    checker_trigger_lo[31:0]                    <= dcr_bus_if.write_data;
                `VX_DCR_CHECKER_TRIG_ADDR_HI:
                    checker_trigger_hi[31:0]                    <= dcr_bus_if.write_data;
            `ifdef XLEN_64
                `VX_DCR_CHECKER_TRIG_ADDR_LO1:
                    checker_trigger_lo[`MEM_ADDR_WIDTH-1:32]    <= (`MEM_ADDR_WIDTH-32)'(dcr_bus_if.write_data);
                `VX_DCR_CHECKER_TRIG_ADDR_HI1:
                    checker_trigger_hi[`MEM_ADDR_WIDTH-1:32]    <= (`MEM_ADDR_WIDTH-32)'(dcr_bus_if.write_data);
            `endif

                // Weight SRAM streaming: accumulate 32 bits at a time into w_wbuf.
                // After CHK_W_WORDS (32) writes, latch address and pulse w_we for one
                // cycle so VX_checker's weight_sram sees a valid write at the next edge.
                `VX_DCR_CHECKER_WEIGHT_DATA: begin
                    w_wbuf[w_wcol * 32 +: 32] <= dcr_bus_if.write_data;
                    if (w_wcol == CHK_W_WORD_W'(CHK_W_WORDS - 1)) begin
                        w_we    <= 1'b1;
                        w_waddr <= w_wrow;
                        w_wrow  <= w_wrow + CHK_W_ADDRW'(1);
                        w_wcol  <= '0;
                    end else begin
                        w_wcol  <= w_wcol + CHK_W_WORD_W'(1);
                    end
                end

                // Threshold streaming: one uint16 per write, auto-advance index.
                // Saturates at CHK_T_CNT-1 so excess writes don't wrap into threshold[0].
                `VX_DCR_CHECKER_THRESH_DATA: begin
                    t_we    <= 1'b1;
                    t_waddr <= t_widx;
                    t_wdata <= dcr_bus_if.write_data[15:0];
                    if (t_widx != CHK_T_IDX_W'(CHK_T_CNT - 1))
                        t_widx <= t_widx + CHK_T_IDX_W'(1);
                end

                default:;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // L2 read-address snoop: trigger the checker when any socket's L1 cache
    // issues a read miss whose byte address falls in [checker_trigger_lo, checker_trigger_hi).
    // Only the core-side ports (slots 0..NUM_SOCKETS*L1_MEM_PORTS-1) are snooped;
    // the checker's own port (last slot) is excluded.
    // addr field on the bus is the cache-line address (byte_addr >> LINE_BITS),
    // so reconstruct full byte addr before comparing against the DCR range.
    // -------------------------------------------------------------------------
    localparam CHK_LINE_BITS = $clog2(`L1_LINE_SIZE);  // 6 for 64-byte lines
    localparam CHK_SNOOP_N   = NUM_SOCKETS * `L1_MEM_PORTS;
    localparam CHK_ADDR_W    = `MEM_ADDR_WIDTH - CHK_LINE_BITS;

    // Interface arrays must be indexed by a compile-time constant; use a generate
    // loop (genvar) to extract signals into plain wire arrays that the always_comb
    // loop can then index with a variable integer.
    wire                      snoop_valid [CHK_SNOOP_N];
    wire                      snoop_ready [CHK_SNOOP_N]; // req_ready from L2 back to core port
    wire                      snoop_rw    [CHK_SNOOP_N];
    wire [CHK_ADDR_W-1:0]     snoop_laddr [CHK_SNOOP_N];

    generate
        for (genvar si = 0; si < CHK_SNOOP_N; si++) begin : g_snoop_flat
            assign snoop_valid[si] = l2_core_bus_if[si].req_valid;
            assign snoop_ready[si] = l2_core_bus_if[si].req_ready;
            assign snoop_rw[si]    = l2_core_bus_if[si].req_data.rw;
            assign snoop_laddr[si] = l2_core_bus_if[si].req_data.addr;
        end
    endgenerate

    logic addr_snoop;
    always_comb begin
        addr_snoop = 1'b0;
        for (int i = 0; i < CHK_SNOOP_N; i++) begin
            automatic logic [`MEM_ADDR_WIDTH-1:0] req_byte =
                `MEM_ADDR_WIDTH'(snoop_laddr[i]) << CHK_LINE_BITS;
            if (snoop_valid[i] && !snoop_rw[i]
                    && (req_byte >= checker_trigger_lo)
                    && (req_byte <  checker_trigger_hi))
                addr_snoop = 1'b1;
        end
    end

    // One-shot: emit a single-cycle trigger_i pulse on the first qualifying snoop
    // hit after arming.  triggered_r suppresses re-triggers within the same arm
    // cycle so the checker doesn't rearm mid-run.  Cleared when checker_armed
    // deasserts (deployer writes ENABLE=0 between runs).
    logic triggered_r;
    initial triggered_r = 1'b0;

    wire addr_trigger = addr_snoop && checker_armed && !triggered_r;

    always @(posedge clk) begin
        if (!checker_armed)
            triggered_r <= 1'b0;
        else if (addr_trigger)
            triggered_r <= 1'b1;
    end

    // Checker's dedicated L2 port (wired into l2_core_bus_if at slot N)
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) chk_act_bus_if();

    // Lower-priority L2 access: checker only issues a request when no core port
    // has a pending request.  snoop_valid[] already captures every core port's
    // req_valid, so reuse it here.  This gives cores 100% of L2 bandwidth under
    // load and lets the checker consume only idle cycles — overhead → near zero.
    logic any_core_req;
    always_comb begin
        any_core_req = 1'b0;
        for (int i = 0; i < CHK_SNOOP_N; i++)
            any_core_req = any_core_req | snoop_valid[i];
    end

    // Expand ASSIGN_VX_MEM_BUS_IF with req_valid gated on !any_core_req.
    // Responses (rsp) are never gated — in-flight responses always complete.
    localparam CHK_L2_PORT = NUM_SOCKETS * `L1_MEM_PORTS;
    assign l2_core_bus_if[CHK_L2_PORT].req_valid = chk_act_bus_if.req_valid && !any_core_req;
    assign l2_core_bus_if[CHK_L2_PORT].req_data  = chk_act_bus_if.req_data;
    assign chk_act_bus_if.req_ready = l2_core_bus_if[CHK_L2_PORT].req_ready && !any_core_req;
    assign chk_act_bus_if.rsp_valid = l2_core_bus_if[CHK_L2_PORT].rsp_valid;
    assign chk_act_bus_if.rsp_data  = l2_core_bus_if[CHK_L2_PORT].rsp_data;
    assign l2_core_bus_if[CHK_L2_PORT].rsp_ready = chk_act_bus_if.rsp_ready;

    wire [15:0] checker_flag;
    wire        chk_all_done;
    VX_checker sem_checker (
        .clk              (clk),
        .reset            (reset),
        .checker_armed    (checker_armed),
        .hidden_base_addr (checker_hidden_base_addr),
        .hidden_size      (checker_hidden_size),
        .num_features     (checker_num_features),
        .batch_size       (checker_batch_size),
        .flag_o           (checker_flag),
        .all_done_o       (chk_all_done),
        .act_bus_if       (chk_act_bus_if),
        .trigger_i        (addr_trigger),
        .addr_trig_en_i   (checker_addr_trig_en),
        .weight_we_i      (w_we),
        .weight_waddr_i   (w_waddr),
        .weight_wdata_i   (w_wbuf),
        .thresh_we_i      (t_we),
        .thresh_waddr_i   (t_waddr),
        .thresh_wdata_i   (t_wdata)
    );
    `UNUSED_VAR (checker_flag);

`ifdef SIMULATION
    // -------------------------------------------------------------------------
    // Core L2 contention trace: emitted whenever the checker has a pending
    // request that is being held back by !any_core_req.  Shows every core port
    // active at that cycle, its address, rw flag, and whether it falls in the
    // checker's own tap range [hidden_base_addr, hidden_base_addr+batch*hidden*4).
    // Enable with TRACE_LEVEL >= 3.  Filter in output: grep "CONTENTION".
    // -------------------------------------------------------------------------
    // CONTENTION trace A: checker has a pending request but is blocked by core traffic.
    // Shows what the cores are reading/writing that cycle.
    always @(posedge clk) begin
        if (checker_armed && chk_act_bus_if.req_valid && any_core_req) begin
            `TRACE(3, ("%t: [CONTENTION] checker blocked by cores:\n", $time))
            for (int ci = 0; ci < CHK_SNOOP_N; ci++) begin
                if (snoop_valid[ci]) begin
                    automatic logic [`MEM_ADDR_WIDTH-1:0] core_byte =
                        `MEM_ADDR_WIDTH'(snoop_laddr[ci]) << CHK_LINE_BITS;
                    automatic logic in_tap =
                        (core_byte >= checker_hidden_base_addr) &&
                        (core_byte <  checker_hidden_base_addr
                                      + `MEM_ADDR_WIDTH'(checker_batch_size)
                                        * `MEM_ADDR_WIDTH'(checker_hidden_size) * 4);
                    `TRACE(3, ("%t: [CONTENTION]   port=%0d  addr=0x%0h  %s  %s\n",
                        $time, ci, core_byte,
                        snoop_rw[ci] ? "WRITE" : "READ",
                        in_tap       ? "<-- TAP RANGE (A-matrix)" : ""))
                end
            end
        end
    end

    // CONTENTION trace B2: checker has a request in-flight in the L2 (sent but not
    // yet responded to) and a core port simultaneously has req_valid=1 but
    // req_ready=0 — the core is stalled because the L2 bank or MSHR is still
    // occupied by the checker's request.
    // chk_inflight tracks from chk_req_fire until the response arrives.
    logic chk_inflight_r;
    always_ff @(posedge clk) begin
        if (reset)
            chk_inflight_r <= 1'b0;
        else if (chk_req_fire)
            chk_inflight_r <= 1'b1;
        else if (chk_act_bus_if.rsp_valid)
            chk_inflight_r <= 1'b0;
    end

    always @(posedge clk) begin
        if (checker_armed && chk_inflight_r) begin
            for (int ci = 0; ci < CHK_SNOOP_N; ci++) begin
                if (snoop_valid[ci] && !snoop_ready[ci]) begin
                    automatic logic [`MEM_ADDR_WIDTH-1:0] core_byte =
                        `MEM_ADDR_WIDTH'(snoop_laddr[ci]) << CHK_LINE_BITS;
                    `TRACE(3, ("%t: [CHK_INFLIGHT_BLOCK] core port=%0d stalled  addr=0x%0h  %s  (checker rsp pending)\n",
                        $time, ci, core_byte,
                        snoop_rw[ci] ? "WRITE" : "READ"))
                end
            end
        end
    end

    // CONTENTION trace B: checker's request IS going out to the L2 this cycle.
    // Log any core ports simultaneously holding a pending request — these are the
    // cores displaced by the checker in an equal-priority arbiter (with the current
    // low-priority gating they should always be empty, confirming zero interference).
    wire chk_req_fire = l2_core_bus_if[CHK_L2_PORT].req_valid
                     && l2_core_bus_if[CHK_L2_PORT].req_ready;

    always @(posedge clk) begin
        if (checker_armed && chk_req_fire) begin
            automatic logic [`MEM_ADDR_WIDTH-1:0] chk_byte =
                {chk_act_bus_if.req_data.addr, CHK_LINE_BITS'(0)};
            automatic int n_waiting = 0;
            for (int cj = 0; cj < CHK_SNOOP_N; cj++)
                if (snoop_valid[cj]) n_waiting++;
            `TRACE(3, ("%t: [CHK_FIRE] checker L2 req accepted  addr=0x%0h  waiting_cores=%0d\n",
                $time, chk_byte, n_waiting))
            for (int ci = 0; ci < CHK_SNOOP_N; ci++) begin
                if (snoop_valid[ci]) begin
                    automatic logic [`MEM_ADDR_WIDTH-1:0] core_byte =
                        `MEM_ADDR_WIDTH'(snoop_laddr[ci]) << CHK_LINE_BITS;
                    `TRACE(3, ("%t: [CHK_FIRE]   displaced port=%0d  addr=0x%0h  %s\n",
                        $time, ci, core_byte,
                        snoop_rw[ci] ? "WRITE" : "READ"))
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Prefetch-effect trace (PREFETCH).
    //
    // Answers: when a core port first requests an L2 line, did the checker
    // already load it into L2?  The checker's A-matrix reads run ahead of
    // GEMM's A-matrix reads with e=1 (or even e=3 when it arms early enough).
    //
    // PREFETCH_LOADED  → checker fetched this line BEFORE the core's first
    //                    request → core gets an L2 hit instead of DRAM miss.
    // COLD_MISS        → core is first; checker hadn't loaded this line.
    //
    // Runs at TRACE level 2 so it fires even in standard debug runs.
    // Filter: grep "PREFETCH"
    // -------------------------------------------------------------------------
    // Associative arrays indexed by longint: Verilator-safe.
    int unsigned prefetch_time [longint unsigned]; // laddr → $time when checker fetched
    bit          core_seen     [longint unsigned]; // laddr → true once a core has logged

    /* verilator lint_off BLKSEQ */
    /* verilator lint_off WIDTHTRUNC */
    always @(posedge clk) begin
        // Record every unique line the checker fetches.
        if (checker_armed && chk_req_fire) begin
            automatic longint unsigned laddr = longint'(chk_act_bus_if.req_data.addr);
            if (prefetch_time.exists(laddr) == 0) begin
                prefetch_time[laddr] = int'($time);
                `TRACE(2, ("%t: [PREFETCH_LOAD] checker fetched laddr=0x%0h  byte=0x%0h\n",
                    $time, laddr, laddr << CHK_LINE_BITS))
            end
        end

        // First time any core port requests a line: log whether checker had it.
        for (int ci = 0; ci < CHK_SNOOP_N; ci++) begin
            if (snoop_valid[ci] && snoop_ready[ci] && !snoop_rw[ci]) begin
                automatic longint unsigned laddr = longint'(snoop_laddr[ci]);
                if (core_seen.exists(laddr) == 0) begin
                    core_seen[laddr] = 1;
                    if (prefetch_time.exists(laddr) != 0) begin
                        `TRACE(2, ("%t: [PREFETCH_HIT] core_port=%0d  laddr=0x%0h  byte=0x%0h  checker_loaded_at=%0t  lag=%0d_cyc\n",
                            $time, ci,
                            laddr, laddr << CHK_LINE_BITS,
                            prefetch_time[laddr],
                            (int'($time) - int'(prefetch_time[laddr])) / 2))
                    end else begin
                        `TRACE(2, ("%t: [COLD_MISS] core_port=%0d  laddr=0x%0h  byte=0x%0h\n",
                            $time, ci, laddr, laddr << CHK_LINE_BITS))
                    end
                end
            end
        end
    end
    /* verilator lint_on WIDTHTRUNC */
    /* verilator lint_on BLKSEQ */

    // -------------------------------------------------------------------------
    // Checker-window stall summary:
    //   chk_window_active: high from first chk_req_fire (after arm) to all_done
    //   chk_window_stall_cnt: cycles where ANY core port had req_valid=1 but
    //     req_ready=0 at the L2 cluster input (backpressure) during checker window
    //   chk_window_core_cycles: cycles where ANY core port had req_valid=1
    //   chk_window_chk_fires: number of checker L2 requests accepted
    //
    // Printed when the checker finishes.  If stall_cnt == 0, L2-cluster
    // backpressure is NOT the overhead source — look deeper (xbar, response path).
    // Filter: grep "CHK_WINDOW_SUMMARY"
    // -------------------------------------------------------------------------
    logic chk_window_active;
    logic [31:0] chk_window_stall_cnt;
    logic [31:0] chk_window_core_cycles;
    logic [31:0] chk_window_chk_fires;

    // Combinational reduction: any core L2 cluster-input stall / any core request.
    logic win_any_stall, win_any_req;
    always_comb begin
        win_any_stall = 1'b0;
        win_any_req   = 1'b0;
        for (int ci = 0; ci < CHK_SNOOP_N; ci++) begin
            if (snoop_valid[ci] && !snoop_ready[ci]) win_any_stall = 1'b1;
            if (snoop_valid[ci])                     win_any_req   = 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            chk_window_active      <= 1'b0;
            chk_window_stall_cnt   <= '0;
            chk_window_core_cycles <= '0;
            chk_window_chk_fires   <= '0;
        end else begin
            if (chk_req_fire && !chk_window_active) chk_window_active <= 1'b1;
            if (chk_all_done)                       chk_window_active <= 1'b0;
            if (chk_window_active) begin
                if (win_any_stall) chk_window_stall_cnt   <= chk_window_stall_cnt   + 32'd1;
                if (win_any_req)   chk_window_core_cycles <= chk_window_core_cycles + 32'd1;
                if (chk_req_fire)  chk_window_chk_fires   <= chk_window_chk_fires   + 32'd1;
            end
        end
    end

    always @(posedge clk) begin
        if (chk_all_done) begin
            `TRACE(3, ("%t: [CHK_WINDOW_SUMMARY] cluster_input_stalls=%0d  core_req_cycles=%0d  chk_fires=%0d\n",
                       $time, chk_window_stall_cnt, chk_window_core_cycles, chk_window_chk_fires))
            `TRACE(3, ("%t: [CHK_WINDOW_SUMMARY]   stalls>0 → L2-cluster backpressure; 0 → interference is deeper (xbar/response path)\n", $time))
        end
    end
`endif

`endif  // CHECKER_ENABLE

endmodule
