// Dummy semantic checker module.
// Passively taps the fetch bus for a trigger instruction, then watches the
// dcache request bus for loads that fall within [TAP_ADDR, TAP_ADDR+TAP_LEN).
// No pipeline signals are driven — purely observational.
// Enable at build time with -DCHECKER_ENABLE.
// View output with --debug=3 passed to blackbox.sh.

`include "VX_define.vh"

module VX_checker import VX_gpu_pkg::*; #(
    // Instruction encoding that arms the checker.
    // Default: ADDI x0, x0, 2047  (32'h7FF00013) — a true NOP with a unique immediate.
    parameter [31:0]                TRIGGER_INSTR = 32'h7FF00013,
    // Byte address of the start of the monitored memory region.
    parameter [`MEM_ADDR_WIDTH-1:0] TAP_ADDR      = `MEM_ADDR_WIDTH'h20000000,
    // Length in bytes of the monitored region.
    parameter [`MEM_ADDR_WIDTH-1:0] TAP_LEN       = `MEM_ADDR_WIDTH'h00001000
) (
    input wire clk,
    input wire reset,

    // Tap point 1: fetch→decode bus (carries raw 32-bit instruction word).
    input wire                                               fetch_valid,
    input wire                                               fetch_ready,
    input wire [31:0]                                        fetch_instr,

    // Tap point 2: core→dcache request bus (DCACHE_NUM_REQS parallel ports).
    // addr is a word address: byte_addr = addr << log2(DCACHE_WORD_SIZE).
    input wire [DCACHE_NUM_REQS-1:0]                         dcache_req_valid,
    input wire [DCACHE_NUM_REQS-1:0]                         dcache_req_rw,
    input wire [DCACHE_NUM_REQS-1:0][DCACHE_ADDR_WIDTH-1:0]  dcache_req_addr
);
    // Convert the byte-addressed TAP parameters to word addresses so they can
    // be directly compared against dcache_req_addr.
    localparam integer WORD_BITS  = `CLOG2(DCACHE_WORD_SIZE);
    localparam [DCACHE_ADDR_WIDTH-1:0] TAP_WORD_LO =
        DCACHE_ADDR_WIDTH'(TAP_ADDR >> WORD_BITS);
    localparam [DCACHE_ADDR_WIDTH-1:0] TAP_WORD_HI =
        DCACHE_ADDR_WIDTH'((TAP_ADDR + TAP_LEN - 1) >> WORD_BITS);

    logic triggered;

    wire trigger_fire = fetch_valid && fetch_ready && (fetch_instr == TRIGGER_INSTR);

    always @(posedge clk) begin
        if (reset) begin
            triggered <= 0;
        end else if (trigger_fire && !triggered) begin
            triggered <= 1;
        end
    end

    always @(posedge clk) begin
        if (!reset) begin
            if (trigger_fire && !triggered) begin
                `TRACE(3, ("%t: [CHECKER] trigger armed (instr=0x%08h, tap_byte_range=[0x%0h, 0x%0h))\n",
                    $time, fetch_instr, TAP_ADDR, TAP_ADDR + TAP_LEN))
            end
            if (triggered) begin
                for (integer i = 0; i < DCACHE_NUM_REQS; ++i) begin
                    if (dcache_req_valid[i]
                            && !dcache_req_rw[i]
                            && dcache_req_addr[i] >= TAP_WORD_LO
                            && dcache_req_addr[i] <= TAP_WORD_HI) begin
                        `TRACE(3, ("%t: [CHECKER] tap load  port=%0d word_addr=0x%0h (byte 0x%0h)\n",
                            $time, i,
                            dcache_req_addr[i],
                            (`MEM_ADDR_WIDTH'(dcache_req_addr[i]) << WORD_BITS)))
                    end
                end
            end
        end
    end

endmodule
