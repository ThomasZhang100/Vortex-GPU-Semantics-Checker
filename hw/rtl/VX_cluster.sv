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
        .core_bus_if    (per_socket_mem_bus_if),
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
    // dcr_bus_if is visible here (unfiltered) before being forwarded to sockets.
    logic                       checker_armed;
    logic [`MEM_ADDR_WIDTH-1:0] checker_tap_addr;
    logic [`MEM_ADDR_WIDTH-1:0] checker_tap_len;

    always @(posedge clk) begin
        if (reset) begin
            checker_armed    <= 0;
            checker_tap_addr <= 0;
            checker_tap_len  <= 0;
        end else if (dcr_bus_if.write_valid) begin
            case (dcr_bus_if.write_addr)
                `VX_DCR_CHECKER_ENABLE:
                    checker_armed            <= dcr_bus_if.write_data[0];
                `VX_DCR_CHECKER_TAP_ADDR0:
                    checker_tap_addr[31:0]   <= dcr_bus_if.write_data;
            `ifdef XLEN_64
                `VX_DCR_CHECKER_TAP_ADDR1:
                    checker_tap_addr[63:32]  <= dcr_bus_if.write_data;
            `endif
                `VX_DCR_CHECKER_TAP_LEN:
                    checker_tap_len          <= dcr_bus_if.write_data;
                default:;
            endcase
        end
    end

    // Flatten all L1→L2 bus signals for the checker.
    localparam integer CHK_NUM_PORTS = NUM_SOCKETS * `L1_MEM_PORTS;
    localparam integer CHK_LINE_BITS = `CLOG2(`L1_LINE_SIZE);
    localparam integer CHK_ADDR_W   = `MEM_ADDR_WIDTH - CHK_LINE_BITS;

    wire [CHK_NUM_PORTS-1:0]                      chk_req_valid;
    wire [CHK_NUM_PORTS-1:0]                      chk_req_rw;
    wire [CHK_NUM_PORTS-1:0][CHK_ADDR_W-1:0]     chk_req_addr;
    wire [CHK_NUM_PORTS-1:0]                      chk_rsp_valid;
    wire [CHK_NUM_PORTS-1:0][`L1_LINE_SIZE*8-1:0] chk_rsp_data;

    for (genvar p = 0; p < CHK_NUM_PORTS; ++p) begin : g_chk_ports
        assign chk_req_valid[p] = per_socket_mem_bus_if[p].req_valid;
        assign chk_req_rw[p]    = per_socket_mem_bus_if[p].req_data.rw;
        assign chk_req_addr[p]  = per_socket_mem_bus_if[p].req_data.addr;
        assign chk_rsp_valid[p] = per_socket_mem_bus_if[p].rsp_valid;
        assign chk_rsp_data[p]  = per_socket_mem_bus_if[p].rsp_data.data;
    end

    VX_checker sem_checker (
        .clk           (clk),
        .reset         (reset),
        .checker_armed (checker_armed),
        .tap_addr      (checker_tap_addr),
        .tap_len       (checker_tap_len),
        .mem_req_valid (chk_req_valid),
        .mem_req_rw    (chk_req_rw),
        .mem_req_addr  (chk_req_addr),
        .mem_rsp_valid (chk_rsp_valid),
        .mem_rsp_data  (chk_rsp_data)
    );
`endif

endmodule
