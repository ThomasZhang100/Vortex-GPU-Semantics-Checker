// Semantic checker — snoops the L1→L2 bus (per_socket_mem_bus_if in VX_cluster).
// Trigger:     1-bit pulse from core 0's fetch stage (ADDI x0,x0,2047).
// Memory tap:  assembles all L1→L2 read responses in [TAP_ADDR, TAP_ADDR+TAP_LEN)
//              into a single hidden_layer vector, ordered by byte offset.
//              hidden_layer_valid goes high when every cache line has arrived.
// Enable:      -DCHECKER_ENABLE -DTAP_ADDR=<decimal> -DTAP_LEN=<decimal>
// View output: --debug=3 passed to blackbox.sh

`include "VX_define.vh"

module VX_checker import VX_gpu_pkg::*; #(
    parameter [`MEM_ADDR_WIDTH-1:0] TAP_ADDR = 65536,
    parameter [`MEM_ADDR_WIDTH-1:0] TAP_LEN  = 64
) (
    input wire clk,
    input wire reset,

    // 1-bit trigger: pulsed for one cycle when the trigger instruction is
    // fetched on core 0.
    input wire trigger_in,

    // L1→L2 bus snoop: NUM_SOCKETS * L1_MEM_PORTS parallel ports.
    // Requests carry a cache-line word address (1 unit = L1_LINE_SIZE bytes).
    // Responses carry L1_LINE_SIZE bytes of data (no address — tag only).
    input wire [NUM_SOCKETS*`L1_MEM_PORTS-1:0]                        mem_req_valid,
    input wire [NUM_SOCKETS*`L1_MEM_PORTS-1:0]                        mem_req_rw,
    input wire [NUM_SOCKETS*`L1_MEM_PORTS-1:0][L1_ADDR_WIDTH-1:0]    mem_req_addr,
    input wire [NUM_SOCKETS*`L1_MEM_PORTS-1:0]                        mem_rsp_valid,
    input wire [NUM_SOCKETS*`L1_MEM_PORTS-1:0][`L1_LINE_SIZE*8-1:0]  mem_rsp_data
);
    localparam integer LINE_BITS     = `CLOG2(`L1_LINE_SIZE);
    localparam integer L1_ADDR_WIDTH = `MEM_ADDR_WIDTH - LINE_BITS;
    localparam integer NUM_PORTS     = NUM_SOCKETS * `L1_MEM_PORTS;

    // Cache-line word addresses covering [TAP_ADDR, TAP_ADDR+TAP_LEN).
    localparam [L1_ADDR_WIDTH-1:0] TAP_LINE_LO =
        L1_ADDR_WIDTH'(TAP_ADDR >> LINE_BITS);
    localparam [L1_ADDR_WIDTH-1:0] TAP_LINE_HI =
        L1_ADDR_WIDTH'((TAP_ADDR + TAP_LEN - 1) >> LINE_BITS);

    // Number of cache lines spanning the tap range.
    localparam integer TAP_LINES =
        integer'(TAP_LINE_HI) - integer'(TAP_LINE_LO) + 1;

    // The assembled hidden layer: TAP_LINES × L1_LINE_SIZE bytes, packed LSB-first
    // by ascending cache-line address.
    // hidden_layer[line_idx * L1_LINE_SIZE*8 +: L1_LINE_SIZE*8] = cache line at
    //   byte address (TAP_LINE_LO + line_idx) << LINE_BITS.
    // Suppress the "bits not read" warning — upper bits are assigned for future
    // SAE compute use; only [63:0] is printed in the current trace.
    /* verilator lint_off UNUSEDSIGNAL */
    logic [TAP_LINES*`L1_LINE_SIZE*8-1:0] hidden_layer;
    /* verilator lint_on UNUSEDSIGNAL */

    // One bit per cache line — set when that line has been latched.
    logic [TAP_LINES-1:0] line_received;

    // Combinational: high whenever all lines are in.
    wire all_lines_received = triggered && (&line_received);

    // Edge-detect: pulse for exactly one cycle on the 0→1 transition.
    logic all_lines_received_r;
    always @(posedge clk) begin
        if (reset) all_lines_received_r <= 0;
        else       all_lines_received_r <= all_lines_received;
    end
    wire hidden_layer_valid = all_lines_received && !all_lines_received_r;

    logic triggered;

    always @(posedge clk) begin
        if (reset) begin
            triggered     <= 0;
            hidden_layer  <= '0;
            line_received <= '0;
        end else begin
            if (trigger_in && !triggered) begin
                triggered     <= 1;
                // Reset collection state so each trigger starts a clean capture.
                hidden_layer  <= '0;
                line_received <= '0;
            end
        end
    end

    // Per-port pending: tracks an outstanding tap-range read request so that
    // when the response arrives (no address in response — tag only) we can
    // derive which cache line it belongs to.
    logic [NUM_PORTS-1:0]                    pending;
    logic [NUM_PORTS-1:0][L1_ADDR_WIDTH-1:0] pending_addr;

    always @(posedge clk) begin
        if (reset) begin
            pending      <= '0;
            pending_addr <= '0;
        end else begin
            for (integer i = 0; i < NUM_PORTS; ++i) begin
                // Arm pending when a read request to the tap range is issued.
                if (mem_req_valid[i] && !mem_req_rw[i]
                        && mem_req_addr[i] >= TAP_LINE_LO
                        && mem_req_addr[i] <= TAP_LINE_HI) begin
                    pending[i]      <= 1;
                    pending_addr[i] <= mem_req_addr[i];
                end

                // When the response arrives, write the data into the correct
                // slot of hidden_layer using pending_addr as the line index.
                if (mem_rsp_valid[i] && pending[i]) begin
                    automatic integer line_idx;
                    line_idx = integer'(pending_addr[i]) - integer'(TAP_LINE_LO);
                    hidden_layer[line_idx*`L1_LINE_SIZE*8 +: `L1_LINE_SIZE*8]
                        <= mem_rsp_data[i];
                    line_received[line_idx] <= 1;
                    pending[i] <= 0;
                end
            end
        end
    end

    // Trace output.
    always @(posedge clk) begin
        if (!reset) begin
            if (trigger_in && !triggered) begin
                `TRACE(3, ("%t: [CHECKER] trigger armed  tap=[0x%0h, 0x%0h)  lines=%0d\n",
                    $time, TAP_ADDR, TAP_ADDR + TAP_LEN, TAP_LINES))
            end
            for (integer i = 0; i < NUM_PORTS; ++i) begin
                if (triggered && mem_req_valid[i] && !mem_req_rw[i]
                        && mem_req_addr[i] >= TAP_LINE_LO
                        && mem_req_addr[i] <= TAP_LINE_HI) begin
                    `TRACE(3, ("%t: [CHECKER] tap req  port=%0d line_idx=%0d byte=0x%0h\n",
                        $time, i,
                        integer'(mem_req_addr[i]) - integer'(TAP_LINE_LO),
                        `MEM_ADDR_WIDTH'(mem_req_addr[i]) << LINE_BITS))
                end
                if (triggered && mem_rsp_valid[i] && pending[i]) begin
                    automatic integer line_idx;
                    line_idx = integer'(pending_addr[i]) - integer'(TAP_LINE_LO);
                    `TRACE(3, ("%t: [CHECKER] tap rsp  port=%0d line_idx=%0d data[63:0]=0x%0h\n",
                        $time, i, line_idx, mem_rsp_data[i][63:0]))
                end
            end
            if (hidden_layer_valid) begin
                `TRACE(3, ("%t: [CHECKER] hidden_layer complete  [63:0]=0x%0h\n",
                    $time, hidden_layer[63:0]))
            end
        end
    end

endmodule
