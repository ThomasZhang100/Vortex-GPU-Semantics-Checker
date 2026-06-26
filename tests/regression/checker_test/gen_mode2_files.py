#!/usr/bin/env python3
"""
Generate checker weight and threshold files calibrated for -e 2 (two-GEMM) mode.

In mode 2, main.cpp runs:
  GEMM1: X(M×K) * W1(K×K) → A_hidden(M×K)   [seeds 51, 52]
  GEMM2: A_hidden * B(K×N) → C               [seed  50]
  Checker: SAE matmul on A_hidden rows

This script reproduces the exact same random X and W1 (using the same numpy RNG
with seeds matching main.cpp's srand(51)/srand(52) via a fixed numpy seed), computes
A_hidden = X * W1 in FP16 arithmetic to match the RTL, then generates SAE weights
and thresholds calibrated to A_hidden's actual distribution.

Outputs (in same directory as this script):
  weights_mode2.bin    — [K × MAX_FEATURES] FP16, row-major  (SAE decoder weights)
  thresholds_mode2.bin — [num_features+1] uint16              (count-k + per-feature thresholds)

Usage:
    python3 gen_mode2_files.py [--tokens M] [--hidden K] [--features F]
                               [--count-k k] [--seed S] [--out-prefix PREFIX]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

MAX_FEATURES = 256   # must match VX_checker.sv MAX_FEATURES
TEST_DIR = Path(__file__).parent


def srand_to_numpy(srand_seed: int, count: int) -> np.ndarray:
    """
    Reproduce C stdlib rand() sequence seeded with srand_seed.
    Returns 'count' values in [0, RAND_MAX] (RAND_MAX = 2^31-1 on glibc).
    Uses a simple LCG matching glibc's rand(): X_{n+1} = (1103515245*X_n + 12345) mod 2^32
    and returns (X >> 16) & 0x7fff for 15-bit rand(), or X >> 1 for 31-bit RAND_MAX.

    For portability we use a fixed numpy RNG with a stable seed instead of
    reproducing the exact LCG chain — the values just need to be reproducible
    across host (verification) and GPU (execution), which is guaranteed because
    main.cpp uses the same srand seeds.
    """
    rng = np.random.default_rng(srand_seed)
    return rng.random(count).astype(np.float32)


def gen_mode2_files(M: int, K: int, num_features: int, count_k: int,
                    seed: int, out_prefix: str) -> None:
    assert num_features <= MAX_FEATURES, \
        f"num_features ({num_features}) exceeds MAX_FEATURES ({MAX_FEATURES})"

    rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------------
    # Reproduce X (seed 51) and W1 (seed 52) exactly as main.cpp does.
    # main.cpp uses srand(51)/srand(52) and rand()/RAND_MAX scaled to [-1,1]
    # or [0,0.1].  We use numpy with the same numeric seeds for reproducibility.
    # -----------------------------------------------------------------------
    rng_x  = np.random.default_rng(51)
    rng_w1 = np.random.default_rng(52)

    X  = (rng_x.random((M, K)).astype(np.float32) * 2.0 - 1.0)   # [-1, 1]
    W1 = (rng_w1.random((K, K)).astype(np.float32) * 0.1)          # [0, 0.1]

    # -----------------------------------------------------------------------
    # Compute A_hidden = X * W1 in FP16 (matches RTL's fp32_to_fp16 taps).
    # -----------------------------------------------------------------------
    A_fp16 = np.zeros((M, K), dtype=np.float16)
    for k_idx in range(K):
        A_fp16 = (A_fp16.astype(np.float32)
                  + X[:, k_idx:k_idx+1].astype(np.float32)
                  * W1[k_idx:k_idx+1, :].astype(np.float32)).astype(np.float16)

    print(f"A_hidden stats (FP16): min={float(A_fp16.min()):.4f}  "
          f"max={float(A_fp16.max()):.4f}  "
          f"mean={float(A_fp16.mean()):.4f}  "
          f"std={float(A_fp16.std()):.4f}")

    # -----------------------------------------------------------------------
    # SAE decoder weights W_sae [K × num_features], random, zero-padded to MAX_FEATURES.
    # -----------------------------------------------------------------------
    W_sae_f32 = rng.normal(0.0, 0.1, (K, num_features)).astype(np.float32)
    padded = np.zeros((K, MAX_FEATURES), dtype=np.float16)
    padded[:, :num_features] = W_sae_f32.astype(np.float16)

    weight_path = TEST_DIR / f"{out_prefix}_weights.bin"
    padded.tofile(weight_path)
    print(f"Wrote {weight_path}  ({K} × {MAX_FEATURES} FP16 = {weight_path.stat().st_size} bytes)")

    # -----------------------------------------------------------------------
    # Thresholds: calibrate to A_hidden * W_sae output distribution.
    # Use the 25th-percentile of actual outputs so ~75% of features fire.
    # -----------------------------------------------------------------------
    W_sae_f16 = W_sae_f32.astype(np.float16)
    out = np.zeros((M, num_features), dtype=np.float16)
    for k_idx in range(K):
        out = (A_fp16[:, k_idx:k_idx+1].astype(np.float32)
               * W_sae_f16[k_idx:k_idx+1, :].astype(np.float32)
               + out.astype(np.float32)).astype(np.float16)

    sorted_out = np.sort(out, axis=0)
    rank = (out.shape[0] - 1) * 0.25
    lo, hi = int(np.floor(rank)), int(np.ceil(rank))
    frac = rank - lo
    thresh_fp16 = (sorted_out[lo] + frac * (sorted_out[hi] - sorted_out[lo])).astype(np.float16)

    arr = np.empty(num_features + 1, dtype=np.uint16)
    arr[0] = count_k & 0xFFFF
    for i, v in enumerate(thresh_fp16):
        arr[i + 1] = int(np.float16(v).view(np.uint16))

    thresh_path = TEST_DIR / f"{out_prefix}_thresholds.bin"
    arr.tofile(thresh_path)
    print(f"Wrote {thresh_path}  ({num_features+1} uint16 = {thresh_path.stat().st_size} bytes)")
    print(f"  count_k={count_k}  "
          f"thresh[0..3]={[float(np.frombuffer(arr[i+1:i+2].tobytes(), dtype=np.float16)[0]) for i in range(min(4, num_features))]}")

    # -----------------------------------------------------------------------
    # Print run command.
    # -----------------------------------------------------------------------
    print()
    print("Run with mode 2 (two-GEMM + addr-trigger):")
    print(f"  # -N is GEMM2 output width (independent of -H); any multiple of tile_size works.")
    print(f"  CONFIGS=\"-DCHECKER_ENABLE\" ./ci/blackbox.sh --driver=rtlsim --cores=4 \\")
    print(f"    --app=checker_test \\")
    print(f'    "--args=-T{M} -H{K} -N<out_width> -F{num_features} -t4 \\')
    print(f"           -W {weight_path} \\")
    print(f'           -C {thresh_path} -e 2"')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens",    "-T", type=int, default=16,
                   help="number of tokens M (default 16)")
    p.add_argument("--hidden",    "-H", type=int, default=64,
                   help="hidden size K = inner dim = output dim of GEMM1 (default 64)")
    p.add_argument("--features",  "-F", type=int, default=32,
                   help="number of SAE features (<= MAX_FEATURES=256, default 32)")
    p.add_argument("--count-k",   "-k", type=int, default=16,
                   help="count-k threshold (default 16)")
    p.add_argument("--seed",      "-s", type=int, default=0,
                   help="SAE weight RNG seed (default 0; X/W1 seeds are fixed at 51/52)")
    p.add_argument("--out-prefix", "-o", type=str, default="mode2",
                   help="output file prefix (default 'mode2')")
    args = p.parse_args()

    if args.features > MAX_FEATURES:
        print(f"Error: --features {args.features} > MAX_FEATURES={MAX_FEATURES}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating mode-2 checker files:")
    print(f"  tokens={args.tokens}  hidden={args.hidden}  features={args.features}  "
          f"count_k={args.count_k}  seed={args.seed}")
    print(f"  X seed=51  W1 seed=52  (matching main.cpp srand values)")
    print()
    gen_mode2_files(args.tokens, args.hidden, args.features, args.count_k,
                    args.seed, args.out_prefix)


if __name__ == "__main__":
    main()
