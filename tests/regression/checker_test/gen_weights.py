#!/usr/bin/env python3
"""
Generate a SAE weight hex file for VX_checker simulation.

The weight SRAM is MAX_HIDDEN rows × N_FEAT FP16 values.  Each hex line is one
SRAM row (WEIGHT_DATAW = N_FEAT*16 bits wide), written big-endian so that
feature 0 occupies the LSB of the 256-bit word:

    hex line k:  W[k][N_FEAT-1] ... W[k][1] W[k][0]   (MSB → LSB)

Two modes:
  --mode identity   W[k][n] = float16(k) for all n.
                    Used to verify the skew: at cycle t, w_pe[b][n] == float16(k_count[0]-b-n).
  --mode saedec     Load real SAE decoder weights from a .npy file (shape [hidden, n_feat]).
                    Pass the file with --weights <path>.

Usage:
  python3 gen_weights.py --mode identity --hidden 64  --nfeat 16  --out sae_weights_test.hex
  python3 gen_weights.py --mode saedec  --weights W.npy            --out sae_weights.hex
"""

import argparse
import struct
import sys
import numpy as np


# ── FP16 helpers ──────────────────────────────────────────────────────────────

def to_fp16_bits(v: float) -> int:
    """Return the 16-bit integer representation of float16(v)."""
    return int(np.float16(v).view(np.uint16))


def pack_row(features_fp16_bits: list[int], n_feat: int) -> int:
    """Pack N_FEAT FP16 bit-patterns into one WEIGHT_DATAW-bit integer.

    Feature 0 goes to bits [15:0], feature n to bits [n*16+15 : n*16].
    This matches the RTL extraction:  w_pe[b][n] = w_hpipe[...][n*16 +: 16]
    and the $readmemh big-endian convention (MSB digit first in the hex string).
    """
    assert len(features_fp16_bits) == n_feat
    word = 0
    for n, bits in enumerate(features_fp16_bits):
        word |= (bits & 0xFFFF) << (n * 16)
    return word


# ── Modes ─────────────────────────────────────────────────────────────────────

def gen_identity(hidden_size: int, n_feat: int, max_hidden: int) -> list[int]:
    """W[k][n] = float16(k) for all n.

    With this pattern, w_pe[b][n] at cycle t should equal float16(k_count[0]-b-n).
    All 16 features carry the same k-index, so any misalignment between features
    or rows is immediately visible in the trace.
    """
    rows = []
    for k in range(max_hidden):
        val = to_fp16_bits(float(k)) if k < hidden_size else 0
        rows.append(pack_row([val] * n_feat, n_feat))
    return rows


def gen_from_npy(weights_path: str, n_feat: int, max_hidden: int) -> list[int]:
    """Load real SAE decoder weights from a .npy file.

    Expected shape: [hidden_size, n_feat] (float32 or float16).
    Will be cast to float16 before packing.
    """
    W = np.load(weights_path)
    if W.ndim != 2:
        sys.exit(f"Error: weight array must be 2-D [hidden, n_feat], got shape {W.shape}")
    hidden_size, nf = W.shape
    if nf != n_feat:
        sys.exit(f"Error: weight array has {nf} features but --nfeat={n_feat}")

    W16 = W.astype(np.float16).view(np.uint16)  # shape [hidden, n_feat], uint16

    rows = []
    for k in range(max_hidden):
        if k < hidden_size:
            bits = [int(W16[k, n]) for n in range(n_feat)]
        else:
            bits = [0] * n_feat
        rows.append(pack_row(bits, n_feat))
    return rows


# ── Writer ────────────────────────────────────────────────────────────────────

def write_hex(rows: list[int], n_feat: int, out_path: str) -> None:
    hex_chars = n_feat * 4  # n_feat * 16 bits / 4 bits per hex char
    with open(out_path, "w") as f:
        for word in rows:
            f.write(f"{word:0{hex_chars}x}\n")
    print(f"Wrote {len(rows)} rows ({hex_chars} hex chars each) → {out_path}")


# ── Verify (round-trip sanity check) ─────────────────────────────────────────

def verify(rows: list[int], n_feat: int, n_check: int = 4) -> None:
    """Print first n_check rows decoded back to float16 for visual inspection."""
    print(f"\nFirst {n_check} SRAM rows (decoded):")
    for k, word in enumerate(rows[:n_check]):
        features = []
        for n in range(n_feat):
            bits = (word >> (n * 16)) & 0xFFFF
            val = np.array(bits, dtype=np.uint16).view(np.float16)
            features.append(float(val))
        print(f"  W[{k:4d}][0..{n_feat-1}] = {features}")

    print("\nExpected skew at steady state (identity mode, k_count[0]=K):")
    print("  w_pe[0][0] = W[K][0]     = float16(K)")
    print("  w_pe[0][1] = W[K-1][1]   = float16(K-1)   (1-cycle column skew)")
    print("  w_pe[1][0] = W[K-1][0]   = float16(K-1)   (1-cycle row skew)")
    print("  w_pe[1][1] = W[K-2][1]   = float16(K-2)   (combined b+n=2)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",    choices=["identity", "saedec"], default="identity")
    p.add_argument("--hidden",  type=int, default=64,
                   help="hidden_size used in checker_test (default: 64)")
    p.add_argument("--nfeat",   type=int, default=16,
                   help="N_FEAT columns (default: 16)")
    p.add_argument("--maxhidden", type=int, default=2048,
                   help="MAX_HIDDEN SRAM depth (default: 2048)")
    p.add_argument("--weights", default=None,
                   help="Path to .npy weight file (required for --mode saedec)")
    p.add_argument("--out",     default="sae_weights_test.hex",
                   help="Output hex file path (default: sae_weights_test.hex)")
    args = p.parse_args()

    if args.mode == "identity":
        rows = gen_identity(args.hidden, args.nfeat, args.maxhidden)
    else:
        if args.weights is None:
            sys.exit("Error: --weights <path.npy> required for --mode saedec")
        rows = gen_from_npy(args.weights, args.nfeat, args.maxhidden)

    write_hex(rows, args.nfeat, args.out)
    verify(rows, args.nfeat)


if __name__ == "__main__":
    main()
