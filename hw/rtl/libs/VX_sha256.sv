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

// Streaming SHA-256 core (FIPS 180-4).  Compact rolled datapath: one message
// block (512 bits) is compressed in 64 cycles (one round/cycle) using a 16-word
// sliding message-schedule window, so area is ~a handful of 32-bit adders and
// registers rather than a fully-unrolled 64-round pipe.
//
// This is the content-hash primitive for the boot verifier (Task C): it hashes
// per-layer weight regions, the kernel, args, and the SAE weight/threshold SRAM,
// and each digest is compared against the signed manifest.
//
// Interface (word-streaming, big-endian bytes — byte 0 of the message is
// in_word[31:24]):
//   start          : pulse (1 cycle) to begin a new message.  Loads the IV.
//   in_valid       : a message word is present this cycle (only sampled when in_ready).
//   in_word        : the 32-bit message word (big-endian byte order).
//   in_last        : this is the final word of the message.
//   in_last_bytes  : number of valid bytes in the final word, 0..4 (only used
//                    when in_last).  0 => the message ends with no bytes in this
//                    beat (e.g. empty message, or a length that is an exact
//                    multiple of 4 already delivered — send a 0-byte terminator).
//   in_ready       : core can accept a word this cycle (deasserts during
//                    compression and internal padding).
//   out_valid      : digest is valid (held until the next start).
//   digest         : 256-bit result, digest[255:224]=H0 ... digest[31:0]=H7
//                    (big-endian: byte 0 of the hex digest is digest[255:248]).
//   busy           : high from start until out_valid.
//
// Padding (the '1' bit, zero fill, and 64-bit length) is generated internally,
// so the caller only streams message words plus the final-word byte count.

module VX_sha256 (
    input  wire         clk,
    input  wire         reset,

    input  wire         start,
    input  wire         in_valid,
    input  wire [31:0]  in_word,
    input  wire         in_last,
    input  wire [2:0]   in_last_bytes,   // 0..4
    output wire         in_ready,

    output wire         out_valid,
    output wire [255:0] digest,
    output wire         busy
);
    // Round constants (first 32 bits of the fractional parts of the cube roots
    // of the first 64 primes).
    localparam logic [31:0] K [64] = '{
        32'h428a2f98, 32'h71374491, 32'hb5c0fbcf, 32'he9b5dba5,
        32'h3956c25b, 32'h59f111f1, 32'h923f82a4, 32'hab1c5ed5,
        32'hd807aa98, 32'h12835b01, 32'h243185be, 32'h550c7dc3,
        32'h72be5d74, 32'h80deb1fe, 32'h9bdc06a7, 32'hc19bf174,
        32'he49b69c1, 32'hefbe4786, 32'h0fc19dc6, 32'h240ca1cc,
        32'h2de92c6f, 32'h4a7484aa, 32'h5cb0a9dc, 32'h76f988da,
        32'h983e5152, 32'ha831c66d, 32'hb00327c8, 32'hbf597fc7,
        32'hc6e00bf3, 32'hd5a79147, 32'h06ca6351, 32'h14292967,
        32'h27b70a85, 32'h2e1b2138, 32'h4d2c6dfc, 32'h53380d13,
        32'h650a7354, 32'h766a0abb, 32'h81c2c92e, 32'h92722c85,
        32'ha2bfe8a1, 32'ha81a664b, 32'hc24b8b70, 32'hc76c51a3,
        32'hd192e819, 32'hd6990624, 32'hf40e3585, 32'h106aa070,
        32'h19a4c116, 32'h1e376c08, 32'h2748774c, 32'h34b0bcb5,
        32'h391c0cb3, 32'h4ed8aa4a, 32'h5b9cca4f, 32'h682e6ff3,
        32'h748f82ee, 32'h78a5636f, 32'h84c87814, 32'h8cc70208,
        32'h90befffa, 32'ha4506ceb, 32'hbef9a3f7, 32'hc67178f2
    };

    // Initial hash value (fractional parts of the square roots of the first 8 primes).
    localparam logic [31:0] IV [8] = '{
        32'h6a09e667, 32'hbb67ae85, 32'h3c6ef372, 32'ha54ff53a,
        32'h510e527f, 32'h9b05688c, 32'h1f83d9ab, 32'h5be0cd19
    };

    function automatic logic [31:0] ror(input logic [31:0] x, input int n);
        return (x >> n) | (x << (32 - n));
    endfunction
    function automatic logic [31:0] bsig0(input logic [31:0] x); // Σ0
        return ror(x,2) ^ ror(x,13) ^ ror(x,22);
    endfunction
    function automatic logic [31:0] bsig1(input logic [31:0] x); // Σ1
        return ror(x,6) ^ ror(x,11) ^ ror(x,25);
    endfunction
    function automatic logic [31:0] ssig0(input logic [31:0] x); // σ0
        return ror(x,7) ^ ror(x,18) ^ (x >> 3);
    endfunction
    function automatic logic [31:0] ssig1(input logic [31:0] x); // σ1
        return ror(x,17) ^ ror(x,19) ^ (x >> 10);
    endfunction
    function automatic logic [31:0] ch(input logic [31:0] x, y, z);
        return (x & y) ^ (~x & z);
    endfunction
    function automatic logic [31:0] maj(input logic [31:0] x, y, z);
        return (x & y) ^ (x & z) ^ (y & z);
    endfunction

    // Merge the final data word with the 0x80 pad bit for a partial (<4-byte) word.
    function automatic logic [31:0] padword(input logic [31:0] w, input logic [2:0] vb);
        case (vb)
            3'd1:    return {w[31:24], 8'h80, 16'h0000};
            3'd2:    return {w[31:16], 8'h80, 8'h00};
            3'd3:    return {w[31:8],  8'h80};
            default: return w; // vb==4: no room, pad bit is emitted as a separate word
        endcase
    endfunction

    typedef enum logic [2:0] {
        S_IDLE, S_ABSORB, S_PAD, S_LOAD, S_COMPRESS, S_DONE
    } state_t;
    state_t state;

    logic [31:0] H   [8];
    logic [31:0] msg [16];   // current block being assembled (big-endian words)
    logic [31:0] sch [16];   // sliding message-schedule window during compression
    logic [31:0] a,b,c,d,e,f,g,h;
    logic [4:0]  widx;       // next free slot in msg (0..16)
    logic [6:0]  round;      // 0..63
    logic [63:0] bitlen;     // total message length in bits
    logic        pad_done;   // the 0x80 pad bit has been placed
    logic        len_hi;     // length high word has been placed in this run
    logic        last_seen;  // final data word has been absorbed
    logic        len_done;   // both length words placed -> hashing complete after final block

    assign in_ready  = (state == S_ABSORB);
    assign out_valid = (state == S_DONE);
    assign busy      = (state != S_IDLE) && (state != S_DONE);

    // Compression round combinational values (use sch[0] as this round's W_t).
    wire [31:0] wt   = sch[0];
    wire [31:0] t1   = h + bsig1(e) + ch(e,f,g) + K[round[5:0]] + wt;
    wire [31:0] t2   = bsig0(a) + maj(a,b,c);
    // Next message-schedule word for the sliding window.
    wire [31:0] wnew = ssig1(sch[14]) + sch[9] + ssig0(sch[1]) + sch[0];

    // Padding word for the current widx while in S_PAD.
    logic [31:0] pad_w;
    always_comb begin
        if (!pad_done)          pad_w = 32'h80000000;
        else if (widx < 5'd14)  pad_w = 32'h0;
        else if (widx == 5'd14) pad_w = bitlen[63:32];
        else                    pad_w = len_hi ? bitlen[31:0] : 32'h0; // widx==15
    end

    integer i;
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= S_IDLE;
        end else begin
            case (state)
                // ---------------------------------------------------------
                S_IDLE: begin
                    if (start) begin
                        for (i = 0; i < 8; i++) H[i] <= IV[i];
                        widx      <= '0;
                        bitlen    <= '0;
                        pad_done  <= 1'b0;
                        len_hi    <= 1'b0;
                        last_seen <= 1'b0;
                        len_done  <= 1'b0;
                        state     <= S_ABSORB;
                    end
                end
                // ---------------------------------------------------------
                S_ABSORB: begin
                    if (in_valid) begin
                        if (in_last) begin
                            last_seen <= 1'b1;
                            bitlen    <= bitlen + 64'(in_last_bytes) * 64'd8;
                            if (in_last_bytes == 3'd0) begin
                                // No data byte in this beat: go straight to padding.
                                pad_done <= 1'b0;
                                state    <= S_PAD;
                            end else begin
                                msg[widx[3:0]] <= padword(in_word, in_last_bytes);
                                pad_done  <= (in_last_bytes != 3'd4); // <4 => pad bit merged
                                if (widx == 5'd15) begin
                                    widx  <= '0;
                                    state <= S_LOAD;   // block full; pad block follows
                                end else begin
                                    widx  <= widx + 5'd1;
                                    state <= S_PAD;
                                end
                            end
                        end else begin
                            msg[widx[3:0]] <= in_word;
                            bitlen    <= bitlen + 64'd32;
                            if (widx == 5'd15) begin
                                widx  <= '0;
                                state <= S_LOAD;
                            end else begin
                                widx <= widx + 5'd1;
                            end
                        end
                    end
                end
                // ---------------------------------------------------------
                S_PAD: begin
                    msg[widx[3:0]] <= pad_w;
                    if (!pad_done)          pad_done <= 1'b1;
                    else if (widx == 5'd14) len_hi   <= 1'b1;
                    else if (widx == 5'd15 && len_hi) len_done <= 1'b1;
                    if (widx == 5'd15) begin
                        widx  <= '0;
                        state <= S_LOAD;
                    end else begin
                        widx <= widx + 5'd1;
                    end
                end
                // ---------------------------------------------------------
                S_LOAD: begin
                    for (i = 0; i < 16; i++) sch[i] <= msg[i];
                    a <= H[0]; b <= H[1]; c <= H[2]; d <= H[3];
                    e <= H[4]; f <= H[5]; g <= H[6]; h <= H[7];
                    round <= '0;
                    state <= S_COMPRESS;
                end
                // ---------------------------------------------------------
                S_COMPRESS: begin
                    // one round
                    h <= g; g <= f; f <= e; e <= d + t1;
                    d <= c; c <= b; b <= a; a <= t1 + t2;
                    // advance sliding schedule window
                    for (i = 0; i < 15; i++) sch[i] <= sch[i+1];
                    sch[15] <= wnew;

                    if (round == 7'd63) begin
                        H[0] <= H[0] + (t1 + t2);
                        H[1] <= H[1] + a;
                        H[2] <= H[2] + b;
                        H[3] <= H[3] + c;
                        H[4] <= H[4] + (d + t1);
                        H[5] <= H[5] + e;
                        H[6] <= H[6] + f;
                        H[7] <= H[7] + g;
                        // Decide what follows this block.
                        if (len_done)
                            state <= S_DONE;
                        else if (last_seen)
                            state <= S_PAD;   // data filled a full block; append pad block
                        else
                            state <= S_ABSORB;
                    end else begin
                        round <= round + 7'd1;
                    end
                end
                // ---------------------------------------------------------
                S_DONE: begin
                    if (start) begin
                        for (i = 0; i < 8; i++) H[i] <= IV[i];
                        widx      <= '0;
                        bitlen    <= '0;
                        pad_done  <= 1'b0;
                        len_hi    <= 1'b0;
                        last_seen <= 1'b0;
                        len_done  <= 1'b0;
                        state     <= S_ABSORB;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    assign digest = {H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7]};

endmodule
