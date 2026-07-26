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

// Word<->line adapter for the boot verifier's VRAM port.  The verifier issues
// 32-bit word reads (and one 32-bit status write) on a simple req/rsp interface;
// the L2 bus is cache-line granular.  This adapter turns each word access into a
// single line transaction: for a read it issues a line read and extracts the
// addressed 32-bit word from the response; for a write it issues a line write
// with byte-enables covering only the addressed word (a partial-line write, which
// the write-through L1/L2 path already supports).  One outstanding transaction.
//
// NOTE: a single-line read buffer (serving all 16 words of a fetched line without
// re-issuing to L2) was attempted to cut verify latency, but destabilized the
// rtlsim build in the constrained test environment; left as future work.  Reads
// already benefit from L2 residency (words 1..15 of a line hit L2 after word 0),
// so this per-word version is correct and simply not bandwidth-optimal.
//
// It drives one VX_mem_bus_if master; in VX_cluster this master is muxed onto the
// checker's shared L2 port (verifier owns it during boot, checker after PASS).

`include "VX_define.vh"

module VX_boot_mem_adapter #(
    parameter LINE_SIZE = `L1_LINE_SIZE,
    parameter MEM_ADDRW = `MEM_ADDR_WIDTH
) (
    input  wire        clk,
    input  wire        reset,

    // Verifier-side word interface
    input  wire        mem_req_valid,
    input  wire        mem_req_rw,
    input  wire [MEM_ADDRW-1:0] mem_req_addr,   // byte address, word-aligned
    input  wire [31:0] mem_req_wdata,
    output wire        mem_req_ready,
    output logic       mem_rsp_valid,
    output logic [31:0] mem_rsp_data,

    // L2-side line interface
    VX_mem_bus_if.master bus_if
);
    localparam LINE_BITS  = `CLOG2(LINE_SIZE);        // 6 for 64B lines
    localparam LINE_WORDS = LINE_SIZE / 4;            // 16
    localparam WOFF_W     = `CLOG2(LINE_WORDS);       // 4
    localparam LADDR_W    = MEM_ADDRW - LINE_BITS;

    typedef enum logic [1:0] { A_IDLE, A_REQ, A_RSP } astate_t;
    astate_t state;

    logic                 req_rw_l;
    logic [LADDR_W-1:0]   laddr_l;
    logic [WOFF_W-1:0]    woff_l;
    logic [31:0]          wdata_l;

    wire [WOFF_W-1:0]  woff_in  = mem_req_addr[LINE_BITS-1:2];
    wire [LADDR_W-1:0] laddr_in = mem_req_addr[MEM_ADDRW-1:LINE_BITS];
    `UNUSED_VAR (mem_req_addr[1:0]) // word-aligned: low 2 bits intentionally unused

    // Request accepted (both read and write) when the line request fires.
    assign mem_req_ready = (state == A_REQ) && bus_if.req_ready;

    // Line request payload.
    assign bus_if.req_valid       = (state == A_REQ);
    assign bus_if.req_data.rw     = req_rw_l;
    assign bus_if.req_data.addr   = laddr_l;
    assign bus_if.req_data.flags  = '0;
    assign bus_if.req_data.tag    = '0;
    always_comb begin
        bus_if.req_data.data   = '0;
        bus_if.req_data.byteen = '0;
        if (req_rw_l) begin
            bus_if.req_data.data[woff_l*32 +: 32]  = wdata_l;
            bus_if.req_data.byteen[woff_l*4 +: 4]  = 4'hF;
        end else begin
            bus_if.req_data.byteen = '1; // read: byteen don't-care, set all
        end
    end

    assign bus_if.rsp_ready = (state == A_RSP);

    always_ff @(posedge clk) begin
        if (reset) begin
            state         <= A_IDLE;
            mem_rsp_valid <= 1'b0;
        end else begin
            mem_rsp_valid <= 1'b0;
            unique case (state)
                A_IDLE: if (mem_req_valid) begin
                    req_rw_l <= mem_req_rw;
                    laddr_l  <= laddr_in;
                    woff_l   <= woff_in;
                    wdata_l  <= mem_req_wdata;
                    state    <= A_REQ;
                end
                A_REQ: if (bus_if.req_ready) begin
                    state <= req_rw_l ? A_IDLE : A_RSP; // writes need no response
                end
                A_RSP: if (bus_if.rsp_valid) begin
                    mem_rsp_valid <= 1'b1;
                    mem_rsp_data  <= bus_if.rsp_data.data[woff_l*32 +: 32];
                    state         <= A_IDLE;
                end
                default: state <= A_IDLE;
            endcase
        end
    end

endmodule
