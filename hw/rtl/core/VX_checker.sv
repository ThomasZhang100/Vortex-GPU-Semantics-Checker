// Semantic checker — snoops the L1→L2 bus (per_socket_mem_bus_if in VX_cluster).
//
// Activation: deployer writes VX_DCR_CHECKER_ENABLE=1 before vx_start.
//             checker_armed is a level signal; no kernel modification needed.
//
// Memory tap: assembles all L1→L2 read responses whose cache-line address falls
//             in [tap_addr, tap_addr+tap_len) into hidden_layer, indexed by
//             byte offset within the range.  hidden_layer_complete pulses for
//             one cycle when every expected cache line has arrived.
//
// Sizing: MAX_TAP_LINES controls the maximum hidden-state size in cache lines.
//         Default 128 covers 128 × 64 B = 8 KB (e.g. 2048-dim FP32 hidden state).
//
// Enable:      -DCHECKER_ENABLE
// View output: --debug=3 passed to blackbox.sh

`include "VX_define.vh"

module VX_checker import VX_gpu_pkg::*; #(
    parameter integer MAX_TAP_LINES = 128   // max cache lines to capture
) (
    input wire clk,
    input wire reset,

    // Deployer-controlled level signal (from VX_DCR_CHECKER_ENABLE DCR).
    // High = checker is armed.  Rising edge resets the capture state.
    input wire                         checker_armed,

    // Runtime tap window (from VX_DCR_CHECKER_TAP_ADDR / TAP_LEN DCRs).
    input wire [`MEM_ADDR_WIDTH-1:0]   tap_addr,
    input wire [`MEM_ADDR_WIDTH-1:0]   tap_len,

    // L1→L2 bus snoop: NUM_SOCKETS * L1_MEM_PORTS parallel ports.
    // Requests carry a cache-line word address (1 unit = L1_LINE_SIZE bytes).
    // Responses carry L1_LINE_SIZE bytes of data (tag only, no address).
    input wire [NUM_SOCKETS*`L1_MEM_PORTS-1:0]                        mem_req_valid,
    input wire [NUM_SOCKETS*`L1_MEM_PORTS-1:0]                        mem_req_rw,
    input wire [NUM_SOCKETS*`L1_MEM_PORTS-1:0][L1_ADDR_WIDTH-1:0]    mem_req_addr,
    input wire [NUM_SOCKETS*`L1_MEM_PORTS-1:0]                        mem_rsp_valid,
    input wire [NUM_SOCKETS*`L1_MEM_PORTS-1:0][`L1_LINE_SIZE*8-1:0]  mem_rsp_data
);
    localparam integer LINE_BITS     = `CLOG2(`L1_LINE_SIZE);
    localparam integer L1_ADDR_WIDTH = `MEM_ADDR_WIDTH - LINE_BITS;
    localparam integer NUM_PORTS     = NUM_SOCKETS * `L1_MEM_PORTS;
    localparam integer LINE_CTR_W    = `CLOG2(MAX_TAP_LINES + 1);

    // Dynamic cache-line address range derived from DCR-supplied byte addresses.
    wire [L1_ADDR_WIDTH-1:0] tap_line_lo = L1_ADDR_WIDTH'(tap_addr >> LINE_BITS);
    wire [L1_ADDR_WIDTH-1:0] tap_line_hi = L1_ADDR_WIDTH'((tap_addr + tap_len - 1) >> LINE_BITS);
    wire [LINE_CTR_W-1:0]    tap_lines   = LINE_CTR_W'(tap_line_hi - tap_line_lo) + 1'b1;

    // Detect rising edge of checker_armed to reset capture state between tokens.
    logic checker_armed_r;
    always @(posedge clk) begin
        if (reset) checker_armed_r <= 0;
        else       checker_armed_r <= checker_armed;
    end
    wire rearm = checker_armed && !checker_armed_r;

    // Assembled hidden layer — MAX_TAP_LINES × L1_LINE_SIZE bytes.
    // Slot k holds the cache line at byte offset k*L1_LINE_SIZE from tap_addr.
    /* verilator lint_off UNUSEDSIGNAL */
    logic [MAX_TAP_LINES*`L1_LINE_SIZE*8-1:0] hidden_layer;
    /* verilator lint_on UNUSEDSIGNAL */

    // Counter of distinct cache lines received so far this capture.
    logic [LINE_CTR_W-1:0] lines_received_count;

    wire hidden_layer_full = checker_armed && (lines_received_count == tap_lines);

    // Edge-detect: pulse for exactly one cycle when the last line arrives.
    logic hidden_layer_full_r;
    always @(posedge clk) begin
        if (reset) hidden_layer_full_r <= 0;
        else       hidden_layer_full_r <= hidden_layer_full;
    end
    wire hidden_layer_complete = hidden_layer_full && !hidden_layer_full_r;

    // Per-port: remember the address of an outstanding tap-range read so that
    // when the response arrives (tag only, no address) we can slot it correctly.
    logic [NUM_PORTS-1:0]                    pending;
    logic [NUM_PORTS-1:0][L1_ADDR_WIDTH-1:0] pending_addr;

    always @(posedge clk) begin
        if (reset || rearm) begin
            /* verilator lint_off WIDTHCONCAT */
            hidden_layer         <= '0;
            /* verilator lint_on WIDTHCONCAT */
            lines_received_count <= '0;
            pending              <= '0;
            pending_addr         <= '0;
        end else begin
            for (integer i = 0; i < NUM_PORTS; ++i) begin
                // Arm pending when a read request to the tap range is issued.
                if (mem_req_valid[i] && !mem_req_rw[i]
                        && mem_req_addr[i] >= tap_line_lo
                        && mem_req_addr[i] <= tap_line_hi) begin
                    pending[i]      <= 1;
                    pending_addr[i] <= mem_req_addr[i];
                end
                // On response: write into the correct hidden_layer slot.
                if (mem_rsp_valid[i] && pending[i]) begin
                    automatic integer line_idx;
                    line_idx = integer'(pending_addr[i]) - integer'(tap_line_lo);
                    hidden_layer[line_idx*`L1_LINE_SIZE*8 +: `L1_LINE_SIZE*8]
                        <= mem_rsp_data[i];
                    lines_received_count <= lines_received_count + 1'b1;
                    pending[i] <= 0;
                end
            end
        end
    end

    always @(posedge clk) begin
        if (!reset) begin
            if (rearm) begin
                `TRACE(3, ("%t: [CHECKER] armed  tap=[0x%0h, 0x%0h)  lines=%0d\n",
                    $time, tap_addr, tap_addr + tap_len, tap_lines))
            end
            for (integer i = 0; i < NUM_PORTS; ++i) begin
                if (checker_armed && mem_req_valid[i] && !mem_req_rw[i]
                        && mem_req_addr[i] >= tap_line_lo
                        && mem_req_addr[i] <= tap_line_hi) begin
                    `TRACE(3, ("%t: [CHECKER] tap req  port=%0d line_idx=%0d byte=0x%0h\n",
                        $time, i,
                        integer'(mem_req_addr[i]) - integer'(tap_line_lo),
                        `MEM_ADDR_WIDTH'(mem_req_addr[i]) << LINE_BITS))
                end
                if (checker_armed && mem_rsp_valid[i] && pending[i]) begin
                    automatic integer line_idx;
                    line_idx = integer'(pending_addr[i]) - integer'(tap_line_lo);
                    `TRACE(3, ("%t: [CHECKER] tap rsp  port=%0d line_idx=%0d data[63:0]=0x%0h\n",
                        $time, i, line_idx, mem_rsp_data[i][63:0]))
                end
            end
            if (hidden_layer_complete) begin
                `TRACE(3, ("%t: [CHECKER] hidden_layer complete  lines=%0d  [63:0]=0x%0h\n",
                    $time, lines_received_count, hidden_layer[63:0]))
            end
        end
    end

endmodule
