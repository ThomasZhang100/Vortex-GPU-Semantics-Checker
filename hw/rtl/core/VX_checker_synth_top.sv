// Synthesis-only wrapper for VX_checker.
//
// sv2v / Yosys cannot use a module with SystemVerilog *interface* ports as a
// synthesis top.  VX_checker has exactly one interface port (act_bus_if, its
// dedicated L2 activation-prefetch port).  This wrapper materializes that
// interface internally and flattens its master-modport signals to plain ports,
// so VX_checker_synth_top can be used as the top-level entity for area/power.
//
// Faithful for area/power: every internal path is kept live — the request
// address reaches a top output (preserves the prefetch address-gen FSM) and
// the response data+tag come from top inputs (preserves the per-row FIFO
// ingest and tag-based row routing).  The unused request payload fields
// (data/byteen/flags on a read-only port) are the only things left to prune.
//
// Build (Yosys flow):
//   make CONFIGS="-DCHECKER_ENABLE -DCHECKER_SYNTH" TOP_LEVEL_ENTITY=VX_checker_synth_top
//
// Guarded by CHECKER_SYNTH so it is inert (empty compile unit) in normal
// simulation/FPGA builds.

`include "VX_define.vh"

`ifdef CHECKER_SYNTH

module VX_checker_synth_top import VX_gpu_pkg::*; #(
    parameter MAX_HIDDEN   = `VX_CHECKER_MAX_HIDDEN,
    parameter MAX_FEATURES = `VX_CHECKER_MAX_FEATURES,
    parameter N_FEAT       = `VX_CHECKER_N_FEAT,
    parameter B_TILE       = `VX_CHECKER_B_TILE,
    parameter MAX_BATCH    = 16
) (
    input  wire clk,
    input  wire reset,

    // DCR-supplied config
    input  wire                              checker_armed,
    input  wire [`MEM_ADDR_WIDTH-1:0]        hidden_base_addr,
    input  wire [15:0]                       hidden_size,
    input  wire [15:0]                       num_features,
    input  wire [15:0]                       batch_size,

    // Outputs
    output wire [MAX_BATCH-1:0]              flag_o,
    output wire                              all_done_o,

    // Flattened L2 activation port (master modport of act_bus_if)
    output wire                              act_req_valid,
    output wire [`MEM_ADDR_WIDTH-1:0]        act_req_addr,
    input  wire                              act_req_ready,
    input  wire                              act_rsp_valid,
    input  wire [`L1_LINE_SIZE*8-1:0]        act_rsp_data,
    input  wire [L1_MEM_ARB_TAG_WIDTH-1:0]   act_rsp_tag,
    output wire                              act_rsp_ready,

    // Triggers
    input  wire                              trigger_i,
    input  wire                              addr_trig_en_i,

    // Weight SRAM write port
    input  wire                              weight_we_i,
    input  wire [`CLOG2(MAX_HIDDEN)-1:0]     weight_waddr_i,
    input  wire [MAX_FEATURES*16-1:0]        weight_wdata_i,

    // Threshold write port
    input  wire                              thresh_we_i,
    input  wire [`CLOG2(MAX_FEATURES+2)-1:0] thresh_waddr_i,
    input  wire [15:0]                       thresh_wdata_i
);

    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) act_bus_if();

    // Drive flat ports from the interface (and vice-versa)
    assign act_req_valid            = act_bus_if.req_valid;
    assign act_req_addr             = act_bus_if.req_data.addr;
    assign act_bus_if.req_ready     = act_req_ready;
    assign act_bus_if.rsp_valid     = act_rsp_valid;
    assign act_bus_if.rsp_data.data = act_rsp_data;
    assign act_bus_if.rsp_data.tag  = act_rsp_tag;
    assign act_rsp_ready            = act_bus_if.rsp_ready;

    VX_checker #(
        .MAX_HIDDEN   (MAX_HIDDEN),
        .MAX_FEATURES (MAX_FEATURES),
        .N_FEAT       (N_FEAT),
        .B_TILE       (B_TILE),
        .MAX_BATCH    (MAX_BATCH)
    ) sem_checker (
        .clk              (clk),
        .reset            (reset),
        .checker_armed    (checker_armed),
        .hidden_base_addr (hidden_base_addr),
        .hidden_size      (hidden_size),
        .num_features     (num_features),
        .batch_size       (batch_size),
        .flag_o           (flag_o),
        .all_done_o       (all_done_o),
        .act_bus_if       (act_bus_if),
        .trigger_i        (trigger_i),
        .addr_trig_en_i   (addr_trig_en_i),
        .weight_we_i      (weight_we_i),
        .weight_waddr_i   (weight_waddr_i),
        .weight_wdata_i   (weight_wdata_i),
        .thresh_we_i      (thresh_we_i),
        .thresh_waddr_i   (thresh_waddr_i),
        .thresh_wdata_i   (thresh_wdata_i)
    );

endmodule

`endif
