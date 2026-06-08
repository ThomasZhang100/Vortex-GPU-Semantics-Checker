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
    logic [`MEM_ADDR_WIDTH-1:0]  checker_hidden_base_addr;
    logic [15:0]                 checker_hidden_size;
    logic [3:0]                  checker_batch_size;

    initial begin
        checker_armed            = 0;
        checker_hidden_base_addr = 0;
        checker_hidden_size      = 0;
        checker_batch_size       = 0;
    end

    always @(posedge clk) begin
        if (dcr_bus_if.write_valid) begin
            case (dcr_bus_if.write_addr)
                `VX_DCR_CHECKER_ENABLE:
                    checker_armed                               <= dcr_bus_if.write_data[0];
                `VX_DCR_CHECKER_TAP_ADDR0:
                    checker_hidden_base_addr[31:0]              <= dcr_bus_if.write_data;
            `ifdef XLEN_64
                `VX_DCR_CHECKER_TAP_ADDR1:
                    checker_hidden_base_addr[`MEM_ADDR_WIDTH-1:32] <= (`MEM_ADDR_WIDTH-32)'(dcr_bus_if.write_data);
            `endif
                `VX_DCR_CHECKER_HIDDEN_SIZE:
                    checker_hidden_size                         <= dcr_bus_if.write_data[15:0];
                `VX_DCR_CHECKER_BATCH_SIZE:
                    checker_batch_size                          <= dcr_bus_if.write_data[3:0];
                default:;
            endcase
        end
    end

    // Checker's dedicated L2 port (wired into l2_core_bus_if at slot N)
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) chk_act_bus_if();

    `ASSIGN_VX_MEM_BUS_IF (l2_core_bus_if[NUM_SOCKETS * `L1_MEM_PORTS], chk_act_bus_if);

    VX_checker sem_checker (
        .clk              (clk),
        .reset            (reset),
        .checker_armed    (checker_armed),
        .hidden_base_addr (checker_hidden_base_addr),
        .hidden_size      (checker_hidden_size),
        .batch_size       (checker_batch_size),
        .act_bus_if       (chk_act_bus_if)
    );
`endif

endmodule
