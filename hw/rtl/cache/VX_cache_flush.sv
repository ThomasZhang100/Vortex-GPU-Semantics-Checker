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

`include "VX_cache_define.vh"

module VX_cache_flush import VX_gpu_pkg::*; #(
    parameter BANK_ID    = 0,
    // Size of cache in bytes
    parameter CACHE_SIZE = 1024,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE  = 64,
    // Number of banks
    parameter NUM_BANKS  = 1,
    // Number of associative ways
    parameter NUM_WAYS   = 1,
    // Enable cache writeback
    parameter WRITEBACK  = 0
) (
    input  wire clk,
    input  wire reset,
    input  wire flush_begin,
    output wire flush_end,
    output wire flush_init,
    output wire flush_valid,
    output wire [`CS_LINE_SEL_BITS-1:0] flush_line,
    output wire [`CS_WAY_SEL_WIDTH-1:0] flush_way,
    input  wire flush_ready,
    input  wire mshr_empty,
    input  wire bank_empty
);
    // ways interation is only needed when eviction is enabled
    localparam CTR_WIDTH = `CS_LINE_SEL_BITS + (WRITEBACK ? `CS_WAY_SEL_BITS : 0);

    localparam STATE_IDLE  = 0;
    localparam STATE_INIT  = 1;
    localparam STATE_WAIT1 = 2;
    localparam STATE_FLUSH = 3;
    localparam STATE_WAIT2 = 4;
    localparam STATE_DONE  = 5;

    reg [2:0] state, state_n;

    reg [CTR_WIDTH-1:0] counter;

`ifdef CACHE_PERSIST
    // Cache-persistence across kernel launches (opt-in via CONFIGS="-DCACHE_PERSIST").
    // On a normal reset the sequencer walks every line in STATE_INIT writing valid=0,
    // clearing the cache.  The tag SRAM (VX_dp_ram, RESET_RAM=0) is NOT cleared by the
    // reset signal itself, so skipping STATE_INIT preserves cached contents.
    //
    // cold_done has NO reset, so it survives the per-kernel-launch reset pulse and (in
    // the Verilator model) persists between processor::run() invocations:
    //   first reset:  cold_done=0 -> run STATE_INIT once (initialize SRAM valid bits)
    //   later resets: cold_done=1 -> skip to STATE_IDLE, retaining cached lines
    reg cold_done;
    initial cold_done = 1'b0;
    wire do_cold_init = !cold_done;

`ifdef SIMULATION
    // Track reset edge so the diagnostic trace fires once per launch.
    reg reset_d;
    initial reset_d = 1'b0;
    always @(posedge clk) reset_d <= reset;
`endif
`else
    wire do_cold_init = 1'b1;
`endif

    always @(*) begin
        state_n = state;
        case (state)
            // STATE_IDLE:
            default : begin
                if (flush_begin) begin
                    state_n = STATE_WAIT1;
                end
            end
            STATE_INIT: begin
                if (counter == ((2 ** `CS_LINE_SEL_BITS)-1)) begin
                    state_n = STATE_IDLE;
                end
            end
            STATE_WAIT1: begin
                // wait for pending requests to complete
                if (mshr_empty) begin
                    state_n = STATE_FLUSH;
                end
            end
            STATE_FLUSH: begin
                if (counter == ((2 ** CTR_WIDTH)-1) && flush_ready) begin
                    state_n = (BANK_ID == 0) ? STATE_DONE : STATE_WAIT2;
                end
            end
            STATE_WAIT2: begin
                // ensure the bank is empty before notifying the cache flush unit,
                // because the flush request to lower caches only goes through bank0
                // and it is important that request gets send out last.
                if (bank_empty) begin
                    state_n = STATE_DONE;
                end
            end
            STATE_DONE: begin
                // generate a completion pulse
                state_n = STATE_IDLE;
            end
        endcase
    end

    always @(posedge clk) begin
        if (reset) begin
            // Cold reset runs STATE_INIT to clear valid bits; warm reset (CACHE_PERSIST
            // with cold_done already set) skips straight to IDLE, preserving the SRAM.
            // NOTE: cold_done is NOT set here.  Reset is asserted for several cycles;
            // setting it here would flip do_cold_init mid-pulse and abort the init
            // walk on the cold launch.  It is set below only when STATE_INIT finishes.
            state   <= 3'(do_cold_init ? STATE_INIT : STATE_IDLE);
            counter <= '0;
`ifdef CACHE_PERSIST
`ifdef SIMULATION
            // Diagnostic: fire ONCE on the rising edge of reset (reset_d=0).
            // do_cold_init=1 -> clearing cache; 0 -> preserving (persistence on).
            if (BANK_ID == 0 && !reset_d)
                `TRACE(1, ("%t: [CACHE_PERSIST] cache_size=%0d reset: do_cold_init=%0b (%s)\n",
                    $time, CACHE_SIZE, do_cold_init,
                    do_cold_init ? "CLEARING cache" : "PRESERVING cache"))
`endif
`endif
        end else begin
            state <= state_n;
            if (state != STATE_IDLE) begin
                if ((state == STATE_INIT)
                || ((state == STATE_FLUSH) && flush_ready)) begin
                    counter <= counter + CTR_WIDTH'(1);
                end
            end else begin
                counter <= '0;
            end
`ifdef CACHE_PERSIST
            // Mark the one-time cold init consumed when the INIT walk completes.
            // After this, subsequent (warm) resets skip STATE_INIT and preserve SRAM.
            if (state == STATE_INIT && counter == CTR_WIDTH'((2 ** `CS_LINE_SEL_BITS)-1))
                cold_done <= 1'b1;
`endif
        end
    end

    assign flush_end   = (state == STATE_DONE);
    assign flush_init  = (state == STATE_INIT);
    assign flush_valid = (state == STATE_FLUSH);
    assign flush_line  = counter[`CS_LINE_SEL_BITS-1:0];

    if (WRITEBACK && (NUM_WAYS > 1)) begin : g_flush_way
        assign flush_way = counter[`CS_LINE_SEL_BITS +: `CS_WAY_SEL_BITS];
    end else begin : g_flush_way_all
        assign flush_way = '0;
    end

endmodule
