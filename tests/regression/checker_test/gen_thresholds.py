#!/usr/bin/env python3
"""
Generate the threshold hex file for VX_checker simulation.

File format (num_features + 1 lines, one uint16 per line as 4 hex chars):
  line 0:          count threshold k (raw uint16) — global_flag fires when
                   feat_count > k, i.e. more than k features exceeded their
                   per-feature threshold.
  lines 1..N:      per-feature FP16 activation thresholds for features 0..N-1.

k=0  → flag whenever at least 1 feature fires
k=N  → flag only when all N features fire (strictest; equivalent to old AND logic)

Modes:
  --mode zeros    All per-feature thresholds = 0.0
  --mode value    All per-feature thresholds = --value (constant)
  --mode file     Load per-feature thresholds from a .npy file (shape [num_features])

Usage:
  python3 gen_thresholds.py --mode zeros  --num-features 16 --count-threshold 0 --out t.hex
  python3 gen_thresholds.py --mode value  --value 0.5 --num-features 64 --count-threshold 8 --out t.hex
  python3 gen_thresholds.py --mode file   --weights thresholds.npy --count-threshold 4 --out t.hex
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


def write_hex(count_k: int, feature_thresholds: list[int], out_path: str) -> None:
    entries = [count_k] + feature_thresholds
    with open(out_path, "w") as f:
        for bits in entries:
            f.write(f"{bits:04x}\n")
    print(f"Wrote {len(entries)} entries (1 count header + {len(feature_thresholds)} features) → {out_path}")


def verify(count_k: int, feature_thresholds: list[int]) -> None:
    print(f"  threshold[0] = {count_k}  (count threshold k — flag when fired > {count_k})")
    print("Per-feature thresholds (FP16):")
    for n, bits in enumerate(feature_thresholds):
        val = np.array(bits, dtype=np.uint16).view(np.float16)
        print(f"  threshold[{n + 1:2d}]  feat[{n:2d}] = 0x{bits:04x}  ({float(val):.6g})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",            choices=["zeros", "value", "file"], default="zeros")
    p.add_argument("--value",           type=float, default=0.0,
                   help="Per-feature threshold for --mode value (default: 0.0)")
    p.add_argument("--num-features",    type=int, default=16,
                   help="Total SAE features; must equal VX_DCR_CHECKER_NUM_FEATURES "
                        "and be ≤ VX_checker MAX_FEATURES (default: 16)")
    p.add_argument("--count-threshold", type=int, default=0,
                   help="Count threshold k: flag fires when feat_count > k "
                        "(0 = flag on any hit; num_features = require all; default: 0)")
    p.add_argument("--weights",         default=None,
                   help="Path to .npy file for --mode file (shape [num_features])")
    p.add_argument("--out",             default="thresholds.hex",
                   help="Output hex file (default: thresholds.hex)")
    args = p.parse_args()

    num_features = args.num_features
    count_k      = args.count_threshold

    if count_k < 0 or count_k > num_features:
        sys.exit(f"Error: --count-threshold {count_k} out of range [0, {num_features}]")

    if args.mode == "zeros":
        feature_thresholds = gen_zeros(num_features)
    elif args.mode == "value":
        feature_thresholds = gen_constant(args.value, num_features)
    else:
        if args.weights is None:
            sys.exit("Error: --weights <path.npy> required for --mode file")
        feature_thresholds = gen_from_npy(args.weights, num_features)

    write_hex(count_k, feature_thresholds, args.out)
    verify(count_k, feature_thresholds)


if __name__ == "__main__":
    main()
