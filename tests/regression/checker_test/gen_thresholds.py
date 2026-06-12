#!/usr/bin/env python3
"""
Generate a per-feature FP16 threshold hex file for VX_checker simulation.

The threshold file has num_features lines, one FP16 value per line (4 hex chars).
threshold[n] is compared against the SAE matmul output for feature n.
A batch row is flagged only if ALL features exceed their threshold.

num_features  — total number of SAE features; must equal VX_DCR_CHECKER_NUM_FEATURES
                and be ≤ VX_checker MAX_FEATURES parameter.

Modes:
  --mode zeros    All thresholds = 0.0 (every positive activation passes; useful for testing)
  --mode value    All thresholds = --value (single constant applied to every feature)
  --mode file     Load per-feature thresholds from a .npy file (shape [num_features])

Usage:
  python3 gen_thresholds.py --mode zeros  --num-features 16  --out thresholds.hex
  python3 gen_thresholds.py --mode value  --value 0.5 --num-features 64  --out thresholds.hex
  python3 gen_thresholds.py --mode file   --weights thresholds.npy  --out thresholds.hex
"""

import argparse
import sys
import numpy as np


def to_fp16_bits(v: float) -> int:
    return int(np.float16(v).view(np.uint16))


def gen_zeros(num_features: int) -> list[int]:
    return [to_fp16_bits(0.0)] * num_features


def gen_constant(value: float, num_features: int) -> list[int]:
    return [to_fp16_bits(value)] * num_features


def gen_from_npy(path: str, num_features: int) -> list[int]:
    arr = np.load(path)
    if arr.ndim != 1:
        sys.exit(f"Error: threshold array must be 1-D [num_features], got shape {arr.shape}")
    if arr.shape[0] != num_features:
        sys.exit(f"Error: array has {arr.shape[0]} entries but --num-features={num_features}")
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
    p.add_argument("--mode",         choices=["zeros", "value", "file"], default="zeros")
    p.add_argument("--value",        type=float, default=0.0,
                   help="Threshold value for --mode value (default: 0.0)")
    p.add_argument("--num-features", type=int, default=16,
                   help="Total number of SAE features; must equal VX_DCR_CHECKER_NUM_FEATURES "
                        "and be ≤ VX_checker MAX_FEATURES (default: 16)")
    p.add_argument("--weights",      default=None,
                   help="Path to .npy file for --mode file (shape [num_features])")
    p.add_argument("--out",          default="thresholds.hex",
                   help="Output hex file (default: thresholds.hex)")
    args = p.parse_args()

    num_features = args.num_features

    if args.mode == "zeros":
        thresholds = gen_zeros(num_features)
    elif args.mode == "value":
        thresholds = gen_constant(args.value, num_features)
    else:
        if args.weights is None:
            sys.exit("Error: --weights <path.npy> required for --mode file")
        thresholds = gen_from_npy(args.weights, num_features)

    write_hex(thresholds, args.out)
    verify(thresholds)


if __name__ == "__main__":
    main()
