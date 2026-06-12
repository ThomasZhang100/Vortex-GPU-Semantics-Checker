#!/usr/bin/env python3
"""
Generate a per-feature FP16 threshold hex file for VX_checker simulation.

The threshold file has N_FEAT lines, one FP16 value per line (4 hex chars).
threshold[n] is compared against the SAE matmul output for feature n.
A batch row is flagged only if ALL features exceed their threshold.

Modes:
  --mode zeros    All thresholds = 0.0 (every positive activation passes; useful for testing)
  --mode value    All thresholds = --value (single constant applied to every feature)
  --mode file     Load per-feature thresholds from a .npy file (shape [n_feat], float32 or float16)

Usage:
  python3 gen_thresholds.py --mode zeros  --nfeat 16  --out thresholds.hex
  python3 gen_thresholds.py --mode value  --value 0.5 --nfeat 16  --out thresholds.hex
  python3 gen_thresholds.py --mode file   --weights thresholds.npy  --out thresholds.hex
"""

import argparse
import sys
import numpy as np


def to_fp16_bits(v: float) -> int:
    return int(np.float16(v).view(np.uint16))


def gen_zeros(n_feat: int) -> list[int]:
    return [to_fp16_bits(0.0)] * n_feat


def gen_constant(value: float, n_feat: int) -> list[int]:
    return [to_fp16_bits(value)] * n_feat


def gen_from_npy(path: str, n_feat: int) -> list[int]:
    arr = np.load(path)
    if arr.ndim != 1:
        sys.exit(f"Error: threshold array must be 1-D [n_feat], got shape {arr.shape}")
    if arr.shape[0] != n_feat:
        sys.exit(f"Error: array has {arr.shape[0]} entries but --nfeat={n_feat}")
    return [int(np.float16(v).view(np.uint16)) for v in arr]


def write_hex(thresholds: list[int], out_path: str) -> None:
    with open(out_path, "w") as f:
        for bits in thresholds:
            f.write(f"{bits:04x}\n")
    print(f"Wrote {len(thresholds)} thresholds → {out_path}")


def verify(thresholds: list[int]) -> None:
    print("Thresholds (decoded):")
    for n, bits in enumerate(thresholds):
        val = np.array(bits, dtype=np.uint16).view(np.float16)
        print(f"  threshold[{n:2d}] = 0x{bits:04x}  ({float(val):.6g})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",    choices=["zeros", "value", "file"], default="zeros")
    p.add_argument("--value",   type=float, default=0.0,
                   help="Threshold value for --mode value (default: 0.0)")
    p.add_argument("--nfeat",   type=int, default=16,
                   help="N_FEAT columns (default: 16)")
    p.add_argument("--weights", default=None,
                   help="Path to .npy file for --mode file (shape [n_feat])")
    p.add_argument("--out",     default="thresholds.hex",
                   help="Output hex file (default: thresholds.hex)")
    args = p.parse_args()

    if args.mode == "zeros":
        thresholds = gen_zeros(args.nfeat)
    elif args.mode == "value":
        thresholds = gen_constant(args.value, args.nfeat)
    else:
        if args.weights is None:
            sys.exit("Error: --weights <path.npy> required for --mode file")
        thresholds = gen_from_npy(args.weights, args.nfeat)

    write_hex(thresholds, args.out)
    verify(thresholds)


if __name__ == "__main__":
    main()
