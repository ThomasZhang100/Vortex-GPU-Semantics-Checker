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

    // Source-resolved L2 miss tracking (SIMULATION-only).  The checker shares the
    // L2 through a dedicated input port at index L2_NUM_REQS-1; passing that index
    // as CHK_SRC lets the L2 bank tag each DRAM-read-causing miss with its true
    // origin (core vs checker) at the point hit/miss is actually determined.
`ifdef CHECKER_ENABLE
    localparam L2_CHK_SRC = L2_NUM_REQS - 1;
`else
    localparam L2_CHK_SRC = -1;
`endif
`ifdef SIMULATION
    wire [`CLOG2(`L2_NUM_BANKS+1)-1:0] l2_core_miss_inc; // core misses this cycle
    wire [`CLOG2(`L2_NUM_BANKS+1)-1:0] l2_chk_miss_inc;  // checker misses this cycle
`endif

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
        .CHK_SRC        (L2_CHK_SRC),
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
    `ifdef SIMULATION
        .perf_core_miss (l2_core_miss_inc),
        .perf_chk_miss  (l2_chk_miss_inc),
    `endif
        .core_bus_if    (l2_core_bus_if),
        .mem_bus_if     (mem_bus_if)
    );

    ///////////////////////////////////////////////////////////////////////////

    // Boot-attestation gate: hold all cores (sockets) in reset until the boot
    // verifier passes.  The L2 is NOT gated — the verifier needs it to read VRAM.
    // ATTEST_ENABLE requires CHECKER_ENABLE (the verifier reuses the checker's L2
    // port and SAE SRAM read ports).  boot_hold survives the vx_start reset pulse
    // (no sync reset) so a PASS latched before vx_start keeps the cores released.
`ifdef ATTEST_ENABLE
    logic boot_hold;
    initial boot_hold = 1'b1;              // fail-closed: cores held until PASS
    logic attest_busy;                     // driven in the ATTEST block below
    wire cluster_core_reset = reset | boot_hold;
`else
    wire cluster_core_reset = reset;
`endif

    wire [NUM_SOCKETS-1:0] per_socket_busy;

    // Generate all sockets
    for (genvar socket_id = 0; socket_id < NUM_SOCKETS; ++socket_id) begin : g_sockets

        `RESET_RELAY (socket_reset, cluster_core_reset);

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

    // Fold attestation activity into `busy` so processor::run() keeps ticking
    // through verification (PASS: cores then take over; FAIL: busy drops so run()
    // returns instead of spinning forever on cores that never boot).
`ifdef ATTEST_ENABLE
    wire cluster_busy_src = (| per_socket_busy) | attest_busy;
`else
    wire cluster_busy_src = (| per_socket_busy);
`endif
    `BUFFER_EX(busy, cluster_busy_src, 1'b1, 1, (NUM_SOCKETS > 1));

`ifdef SIMULATION
    // -------------------------------------------------------------------------
    // Global cache miss-rate counters.
    //
    // l1_miss_cnt : total L2 requests from core sockets (one per L1 cache miss).
    // l2_miss_cnt : DRAM READ requests from L2 (fills) — one per L2 read miss.
    //               Writebacks (DRAM writes) are excluded so this equals the
    //               total fill count, i.e. core+chk source-resolved misses.
    // L2 miss rate = l2_miss_cnt / l1_miss_cnt
    //
    // Printed when busy falls (all warps have retired).
    // Compare with/without CHECKER_ENABLE or DEAD_CYCLE to see checker impact.
    // Filter: grep "MISS_RATE"
    // -------------------------------------------------------------------------
    localparam NUM_CORE_PORTS = NUM_SOCKETS * `L1_MEM_PORTS;

    // Extract core port handshake signals into plain logic arrays so they can
    // be iterated with variable indices in always_comb without Verilator issues.
    logic core_port_fire [NUM_CORE_PORTS];
    generate
        for (genvar cp = 0; cp < NUM_CORE_PORTS; ++cp) begin : g_core_port_fire
            assign core_port_fire[cp] = per_socket_mem_bus_if[cp].req_valid
                                     && per_socket_mem_bus_if[cp].req_ready;
        end
    endgenerate

    // Count only DRAM READ requests (fills) — i.e. true L2 misses.  Writebacks
    // (rw=1) are dirty-line evictions/flushes, not misses, so they are excluded;
    // counting them here is what previously made l2_dram_reads exceed core+chk.
    logic mem_port_fire [`L2_MEM_PORTS];
    generate
        for (genvar mp = 0; mp < `L2_MEM_PORTS; ++mp) begin : g_mem_port_fire
            assign mem_port_fire[mp] = mem_bus_if[mp].req_valid
                                    && mem_bus_if[mp].req_ready
                                    && !mem_bus_if[mp].req_data.rw;
        end
    endgenerate

    // Combinational reductions (counts per cycle).
    logic [3:0] l1_fires;   // max NUM_CORE_PORTS
    logic [2:0] l2_fires;   // max L2_MEM_PORTS
    always_comb begin
        l1_fires = '0;
        for (int cp = 0; cp < NUM_CORE_PORTS; cp++)
            if (core_port_fire[cp]) l1_fires++;
        l2_fires = '0;
        for (int mp = 0; mp < `L2_MEM_PORTS; mp++)
            if (mem_port_fire[mp]) l2_fires++;
    end

    // Per-kernel counters: cleared on each vx_start reset pulse (same pattern as
    // cyc_cnt below), so the print at each busy falling edge reports only that
    // kernel's misses.  In mode 2 the two GEMMs each get their own reset, so GEMM1
    // and GEMM2 print independent counts rather than a running cumulative total.
    logic [63:0] l1_miss_cnt, l2_miss_cnt;
    initial begin
        l1_miss_cnt = '0;
        l2_miss_cnt = '0;
    end
    always_ff @(posedge clk) begin
        if (reset) begin
            l1_miss_cnt <= '0;
            l2_miss_cnt <= '0;
        end else begin
            l1_miss_cnt <= l1_miss_cnt + 64'(l1_fires);
            l2_miss_cnt <= l2_miss_cnt + 64'(l2_fires);
        end
    end

    // Print on every falling edge of busy (once per vx_start/vx_ready_wait pair).
    // Mode 1/3: one print at kernel end.
    // Mode 2: two prints — GEMM1 then GEMM2, each showing that kernel's own counts.
    // initial=0 avoids a spurious X-driven edge at time 0.
    logic busy_prev;
    initial busy_prev = 1'b0;
    always_ff @(posedge clk) busy_prev <= busy;

    // -------------------------------------------------------------------------
    // Per-kernel cycle counter + start/end traces.
    //
    // cyc_cnt resets to 0 on each vx_start's reset pulse, so it measures cycles
    // within one kernel launch.  kernel_idx counts launches (1=GEMM1, 2=GEMM2,
    // ... in mode 2).  Captures cyc_cnt at the busy rising edge (KERNEL_START)
    // and reports the elapsed count at the falling edge (KERNEL_END).
    // Filter: grep "KERNEL_START\|KERNEL_END"
    // -------------------------------------------------------------------------
    logic [63:0] cyc_cnt;
    logic [31:0] kernel_idx;
    logic [63:0] kernel_start_cyc;
    initial begin
        cyc_cnt          = '0;
        kernel_idx       = '0;
        kernel_start_cyc = '0;
    end
    always_ff @(posedge clk) begin
        if (reset) cyc_cnt <= '0;          // restart per-kernel counter each launch
        else       cyc_cnt <= cyc_cnt + 64'd1;
    end

    always @(posedge clk) begin
        // Rising edge of busy: kernel begins executing.
        if (!reset && !busy_prev && busy) begin
            kernel_start_cyc <= cyc_cnt;
            kernel_idx       <= kernel_idx + 32'd1;
            `TRACE(1, ("%t: [KERNEL_START] kernel=%0d  start_cycle=%0d\n",
                $time, kernel_idx + 32'd1, cyc_cnt))
        end
        // Falling edge of busy: kernel finished.
        if (!reset && busy_prev && !busy) begin
            `TRACE(1, ("%t: [KERNEL_END] kernel=%0d  end_cycle=%0d  duration=%0d cycles\n",
                $time, kernel_idx, cyc_cnt, cyc_cnt - kernel_start_cyc))
        end
    end

    // Source-resolved L2 miss counts, taken DIRECTLY from the L2 bank.
    //
    // l2_core_miss_inc / l2_chk_miss_inc are this-cycle counts of DRAM-read-causing
    // misses, classified inside the bank by the missing request's originating input
    // port (req_idx == checker port => checker, else core).  This is the actual
    // hit/miss decision at the actual miss point — no address residency model, no
    // subtraction.  A checker request that hits a core-loaded line is correctly a
    // hit (no miss pulse), so it is never mis-attributed.
    //
    // When CHECKER_ENABLE is off, L2_CHK_SRC=-1 so chk is always 0 and core counts
    // every L2 miss.  Per-kernel: cleared on each vx_start reset pulse (like the
    // l1/l2 counters above), so each busy-falling-edge print is that kernel only.
    logic [63:0] core_l2_miss_cnt, chk_l2_miss_cnt;
    initial begin
        core_l2_miss_cnt = '0;
        chk_l2_miss_cnt  = '0;
    end
    always_ff @(posedge clk) begin
        if (reset) begin
            core_l2_miss_cnt <= '0;
            chk_l2_miss_cnt  <= '0;
        end else begin
            core_l2_miss_cnt <= core_l2_miss_cnt + 64'(l2_core_miss_inc);
            chk_l2_miss_cnt  <= chk_l2_miss_cnt  + 64'(l2_chk_miss_inc);
        end
    end

    // l2_dram_reads is the actual DRAM-read count at the L2 mem port (ground truth).
    // It must equal core + chk: a cross-check that the bank-level attribution is
    // complete (every DRAM-read-causing miss was tagged to exactly one source).
    always @(posedge clk) begin
        if (!reset && busy_prev && !busy) begin
            `TRACE(1, ("%t: [MISS_RATE] l1_misses=%0d  l2_misses_core=%0d  l2_misses_chk=%0d  l2_dram_reads=%0d  l2_miss_pct=%0d\n",
                $time, l1_miss_cnt, core_l2_miss_cnt, chk_l2_miss_cnt, l2_miss_cnt,
                (l1_miss_cnt > 0) ? (core_l2_miss_cnt * 100 / l1_miss_cnt) : 0))
        end
    end
`endif // SIMULATION

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

    // Weight SRAM streaming state — sized from the shared VX_CHECKER_MAX_FEATURES
    // macro (VX_types.vh) and passed to VX_checker below so they cannot drift.
    localparam CHK_MAX_FEAT  = `VX_CHECKER_MAX_FEATURES;
    localparam CHK_MAX_HIDN  = `VX_CHECKER_MAX_HIDDEN;
    localparam CHK_W_DATAW   = CHK_MAX_FEAT * 16;      // bits per SRAM row (256*16=4096)
    localparam CHK_W_ADDRW   = $clog2(CHK_MAX_HIDN);   // 11
    localparam CHK_W_WORDS   = CHK_W_DATAW / 32;        // uint32 words per row (4096/32=128)
    localparam CHK_W_WORD_W  = $clog2(CHK_W_WORDS);     // 7
    localparam CHK_T_CNT     = CHK_MAX_FEAT + 1;        // threshold[0..MAX_FEAT]
    localparam CHK_T_IDX_W   = $clog2(CHK_T_CNT + 1);  // 9

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

    // DEAD_CYCLE (define via CONFIGS="-DDEAD_CYCLE"): checker only issues L2
    // requests during cycles when no core has a pending L2 request, giving
    // cores 100% of L2 bandwidth.  Without DEAD_CYCLE the checker enters the
    // L2 round-robin at equal priority, incurring potential contention.
    localparam CHK_L2_PORT = NUM_SOCKETS * `L1_MEM_PORTS;

    // DEAD_CYCLE gating (checker side).
`ifdef DEAD_CYCLE
    wire chk_req_valid_eff  = chk_act_bus_if.req_valid && !any_core_req;
    wire chk_req_ready_gate = !any_core_req;
`else
    wire chk_req_valid_eff  = chk_act_bus_if.req_valid;
    wire chk_req_ready_gate = 1'b1;
`endif

`ifdef ATTEST_ENABLE
    // =====================================================================
    // Boot attestation (Task C): manifest SRAM + boot verifier.  Time-shares
    // the checker's L2 port (this port) and SAE SRAM reads: the verifier owns
    // them while boot_hold=1 (cores held, checker idle); the checker owns them
    // after PASS.  Requires CHECKER_ENABLE.
    // =====================================================================
    localparam ATT_WORDS = VX_attest_pkg::ATTEST_MANIFEST_WORDS; // 264
    localparam ATT_MFW   = $clog2(ATT_WORDS);

    // ---- manifest SRAM + DCR streaming ----
    // verify_armed is a latch (no sync reset, initial 0) so it survives the
    // vx_start reset pulse — like the checker's ENABLE.  The verifier self-starts
    // on the rising edge of verify_armed after reset deasserts inside run().
    logic [ATT_MFW-1:0] mf_wptr;
    logic               mf_we;
    logic [ATT_MFW-1:0] mf_waddr;
    logic [31:0]        mf_wdata;
    logic               verify_armed;
    initial begin mf_wptr = '0; verify_armed = 1'b0; end
    always @(posedge clk) begin
        mf_we <= 1'b0;
        if (dcr_bus_if.write_valid) begin
            case (dcr_bus_if.write_addr)
                `VX_DCR_ATTEST_MANIFEST_DATA: begin
                    mf_we    <= 1'b1;
                    mf_waddr <= mf_wptr;
                    mf_wdata <= dcr_bus_if.write_data;
                    mf_wptr  <= mf_wptr + ATT_MFW'(1);
                end
                `VX_DCR_ATTEST_VERIFY_START: begin
                    if (dcr_bus_if.write_data[0]) verify_armed <= 1'b1;
                    else begin verify_armed <= 1'b0; mf_wptr <= '0; end // rewind for a fresh stream
                end
                default:;
            endcase
        end
    end

    wire [ATT_MFW-1:0] mf_raddr;
    wire [31:0]        mf_rdata;
    VX_dp_ram #(
        .DATAW (32), .SIZE (ATT_WORDS), .OUT_REG (0), .RDW_MODE ("W")
    ) manifest_sram (
        .clk(clk), .reset(reset),
        .write(mf_we), .wren(1'b1), .waddr(mf_waddr), .wdata(mf_wdata),
        .read(1'b1), .raddr(mf_raddr), .rdata(mf_rdata)
    );

    // ---- boot_hold: released on PASS, survives vx_start reset (no sync reset) ----
    wire        boot_release, verify_done, verify_pass, verify_verifying;
    wire [31:0] verify_status;
    always @(posedge clk) if (boot_release) boot_hold <= 1'b0;

    // Keep `busy` asserted through verification, and after a PASS hold it until the
    // cores actually take over — their cold instruction fetch can exceed any fixed
    // bridge, so a latch (cleared once per_socket_busy first rises) is used instead.
    // This stops processor::run() from returning in the handoff gap.  On FAIL,
    // boot_release never fires, so busy falls when verification ends and run()
    // returns cleanly (cores never boot).
    logic booted_seen;
    always @(posedge clk) begin
        if (reset)                  booted_seen <= 1'b0;
        else if (| per_socket_busy) booted_seen <= 1'b1;
    end
    assign attest_busy = verify_verifying | (boot_release & ~booted_seen);

    // ---- verifier <-> adapter/SAE wiring ----
    wire        vmem_req_valid, vmem_req_rw, vmem_req_ready, vmem_rsp_valid;
    wire [`MEM_ADDR_WIDTH-1:0] vmem_req_addr;
    wire [31:0] vmem_req_wdata, vmem_rsp_data;
    wire        vsae_req_valid, vsae_req_sel, vsae_req_ready, vsae_rsp_valid;
    wire [31:0] vsae_req_addr, vsae_rsp_data;

    // Checker SAE read ports (driven by the SAE adapter below).
    wire        verify_active = boot_hold;
    wire [31:0] verify_w_widx, verify_t_widx;
    wire [31:0] verify_w_word, verify_t_word;

    VX_boot_verifier #(
        .MEM_ADDRW        (`MEM_ADDR_WIDTH),
        .CHK_MAX_FEATURES (CHK_MAX_FEAT)
    ) boot_verifier (
        .clk(clk), .reset(reset),
        .verify_armed(verify_armed),
        .mf_raddr(mf_raddr), .mf_rdata(mf_rdata),
        .mem_req_valid(vmem_req_valid), .mem_req_rw(vmem_req_rw),
        .mem_req_addr(vmem_req_addr), .mem_req_wdata(vmem_req_wdata),
        .mem_req_ready(vmem_req_ready), .mem_rsp_valid(vmem_rsp_valid), .mem_rsp_data(vmem_rsp_data),
        .sae_req_valid(vsae_req_valid), .sae_req_sel(vsae_req_sel), .sae_req_addr(vsae_req_addr),
        .sae_req_ready(vsae_req_ready), .sae_rsp_valid(vsae_rsp_valid), .sae_rsp_data(vsae_rsp_data),
        .boot_release(boot_release), .done(verify_done), .pass(verify_pass),
        .verifying(verify_verifying), .status_word(verify_status)
    );
    `UNUSED_VAR (verify_done);
    `UNUSED_VAR (verify_pass);
    `UNUSED_VAR (verify_status);

    // Word<->line adapter driving the verifier's own L2 master bus.
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE), .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) ver_mem_bus_if();
    VX_boot_mem_adapter #(
        .LINE_SIZE (`L1_LINE_SIZE), .MEM_ADDRW (`MEM_ADDR_WIDTH)
    ) boot_mem_adapter (
        .clk(clk), .reset(reset),
        .mem_req_valid(vmem_req_valid), .mem_req_rw(vmem_req_rw), .mem_req_addr(vmem_req_addr),
        .mem_req_wdata(vmem_req_wdata), .mem_req_ready(vmem_req_ready),
        .mem_rsp_valid(vmem_rsp_valid), .mem_rsp_data(vmem_rsp_data),
        .bus_if(ver_mem_bus_if)
    );

    // SAE read adapter: verifier word request -> checker combinational SRAM read.
    logic        sae_busy, sae_sel_l;
    logic [31:0] sae_addr_l;
    logic        vsae_rsp_valid_r;
    logic [31:0] vsae_rsp_data_r;
    assign vsae_req_ready = !sae_busy;
    assign verify_w_widx  = sae_addr_l;
    assign verify_t_widx  = sae_addr_l;
    assign vsae_rsp_valid = vsae_rsp_valid_r;
    assign vsae_rsp_data  = vsae_rsp_data_r;
    always @(posedge clk) begin
        if (reset) begin
            sae_busy <= 1'b0; vsae_rsp_valid_r <= 1'b0;
        end else begin
            vsae_rsp_valid_r <= 1'b0;
            if (vsae_req_valid && !sae_busy) begin
                sae_sel_l <= vsae_req_sel; sae_addr_l <= vsae_req_addr; sae_busy <= 1'b1;
            end else if (sae_busy) begin
                vsae_rsp_valid_r <= 1'b1;
                vsae_rsp_data_r  <= sae_sel_l ? verify_t_word : verify_w_word;
                sae_busy         <= 1'b0;
            end
        end
    end

    // ---- shared L2 port mux: verifier owns during boot, checker after PASS ----
    wire chk_owns_l2 = !boot_hold;
    assign l2_core_bus_if[CHK_L2_PORT].req_valid = chk_owns_l2 ? chk_req_valid_eff : ver_mem_bus_if.req_valid;
    assign l2_core_bus_if[CHK_L2_PORT].req_data  = chk_owns_l2 ? chk_act_bus_if.req_data : ver_mem_bus_if.req_data;
    assign l2_core_bus_if[CHK_L2_PORT].rsp_ready = chk_owns_l2 ? chk_act_bus_if.rsp_ready : ver_mem_bus_if.rsp_ready;
    assign chk_act_bus_if.req_ready = chk_owns_l2 ? (l2_core_bus_if[CHK_L2_PORT].req_ready && chk_req_ready_gate) : 1'b0;
    assign chk_act_bus_if.rsp_valid = chk_owns_l2 ? l2_core_bus_if[CHK_L2_PORT].rsp_valid : 1'b0;
    assign chk_act_bus_if.rsp_data  = l2_core_bus_if[CHK_L2_PORT].rsp_data;
    assign ver_mem_bus_if.req_ready = chk_owns_l2 ? 1'b0 : l2_core_bus_if[CHK_L2_PORT].req_ready;
    assign ver_mem_bus_if.rsp_valid = chk_owns_l2 ? 1'b0 : l2_core_bus_if[CHK_L2_PORT].rsp_valid;
    assign ver_mem_bus_if.rsp_data  = l2_core_bus_if[CHK_L2_PORT].rsp_data;
`else
    assign l2_core_bus_if[CHK_L2_PORT].req_valid = chk_req_valid_eff;
    assign chk_act_bus_if.req_ready = l2_core_bus_if[CHK_L2_PORT].req_ready && chk_req_ready_gate;
    assign l2_core_bus_if[CHK_L2_PORT].req_data  = chk_act_bus_if.req_data;
    assign chk_act_bus_if.rsp_valid = l2_core_bus_if[CHK_L2_PORT].rsp_valid;
    assign chk_act_bus_if.rsp_data  = l2_core_bus_if[CHK_L2_PORT].rsp_data;
    assign l2_core_bus_if[CHK_L2_PORT].rsp_ready = chk_act_bus_if.rsp_ready;
`endif

    wire [15:0] checker_flag;
    wire        chk_all_done;
    VX_checker #(
        .MAX_FEATURES (CHK_MAX_FEAT),
        .MAX_HIDDEN   (CHK_MAX_HIDN)
    ) sem_checker (
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
`ifdef ATTEST_ENABLE
        ,
        .verify_active    (verify_active),
        .verify_w_widx    (verify_w_widx),
        .verify_w_word    (verify_w_word),
        .verify_t_widx    (verify_t_widx),
        .verify_t_word    (verify_t_word)
`endif
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
    // DRAM-range read counter (DRAM_TAP) — ground-truth L2 hit/miss for the tap.
    //
    // Counts actual DRAM read requests (at the cluster's mem_bus_if, the real
    // L2->DRAM port) whose byte address falls in the checker's tap range
    // [hidden_base, hidden_base + batch*hidden*4).  A DRAM read in that range
    // means the line was NOT L2-resident (an L2 miss); zero such reads means the
    // tap data was served entirely from L2 (hits).  Unlike latency, this directly
    // observes whether the bytes came from DRAM — no threshold guessing.
    //
    // Reset per kernel launch (like cyc_cnt) and printed at the busy falling edge.
    //
    // Verification with -Z (tap = X, which GEMM2 never reads):
    //   GEMM2 tap_range_reads == 0  -> X persisted in L2 from GEMM1 (persistence OK)
    //   GEMM2 tap_range_reads  > 0  -> X cold-fetched from DRAM (no persistence)
    // Filter: grep "DRAM_TAP"
    // -------------------------------------------------------------------------
    localparam CHK_MEM_LINE_BITS = $clog2(`L2_LINE_SIZE);

    // Flatten mem_bus_if (interface array needs genvar indexing).
    wire                      dram_rd_fire [`L2_MEM_PORTS];
    wire                      dram_wr_fire [`L2_MEM_PORTS];
    wire [`MEM_ADDR_WIDTH-1:0] dram_byte    [`L2_MEM_PORTS];
    generate
        for (genvar mi = 0; mi < `L2_MEM_PORTS; mi++) begin : g_dram_flat
            wire fire = mem_bus_if[mi].req_valid && mem_bus_if[mi].req_ready;
            assign dram_rd_fire[mi] = fire && !mem_bus_if[mi].req_data.rw;
            assign dram_wr_fire[mi] = fire &&  mem_bus_if[mi].req_data.rw;
            assign dram_byte[mi] =
                `MEM_ADDR_WIDTH'(mem_bus_if[mi].req_data.addr) << CHK_MEM_LINE_BITS;
        end
    endgenerate

    // Tap byte range (same expression as the CONTENTION trace's in_tap).
    wire [`MEM_ADDR_WIDTH-1:0] tap_lo = checker_hidden_base_addr;
    wire [`MEM_ADDR_WIDTH-1:0] tap_hi = checker_hidden_base_addr
        + `MEM_ADDR_WIDTH'(checker_batch_size)
          * `MEM_ADDR_WIDTH'(checker_hidden_size) * 4;

    // Per-cycle reductions: reads/writes in tap range, plus all reads.
    logic [2:0] tap_rd_fires, tap_wr_fires, all_rd_fires;
    always_comb begin
        tap_rd_fires = '0;
        tap_wr_fires = '0;
        all_rd_fires = '0;
        for (int mi = 0; mi < `L2_MEM_PORTS; mi++) begin
            automatic logic in_tap =
                (dram_byte[mi] >= tap_lo) && (dram_byte[mi] < tap_hi);
            if (dram_rd_fire[mi]) begin
                all_rd_fires = all_rd_fires + 3'd1;
                if (in_tap) tap_rd_fires = tap_rd_fires + 3'd1;
            end
            if (dram_wr_fire[mi] && in_tap)
                tap_wr_fires = tap_wr_fires + 3'd1;
        end
    end

    // Per-kernel counters (reset each vx_start, like cyc_cnt).
    logic [63:0] dram_rd_tap_cnt, dram_wr_tap_cnt, dram_rd_all_cnt;
    always_ff @(posedge clk) begin
        if (reset) begin
            dram_rd_tap_cnt <= '0;
            dram_wr_tap_cnt <= '0;
            dram_rd_all_cnt <= '0;
        end else begin
            dram_rd_tap_cnt <= dram_rd_tap_cnt + 64'(tap_rd_fires);
            dram_wr_tap_cnt <= dram_wr_tap_cnt + 64'(tap_wr_fires);
            dram_rd_all_cnt <= dram_rd_all_cnt + 64'(all_rd_fires);
        end
    end

    always @(posedge clk) begin
        if (!reset && busy_prev && !busy) begin
            `TRACE(1, ("%t: [DRAM_TAP] tap_reads=%0d  tap_writebacks=%0d  total_dram_reads=%0d  tap=[0x%0h,0x%0h)\n",
                $time, dram_rd_tap_cnt, dram_wr_tap_cnt, dram_rd_all_cnt, tap_lo, tap_hi))
        end
    end

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
