#!/usr/bin/env python3
"""
Generate weight and threshold binary files for large-scale checker_test perf runs.

Usage:
    python3 gen_perf_files.py [--hidden H] [--features F] [--seed S]

Outputs (in the same directory as this script):
    weights_perf.bin    — [H × MAX_FEATURES] FP16, row-major
    thresholds_perf.bin — [F+1] uint16  (threshold[0]=k, threshold[1..F]=FP16)

These are passed to the checker_test binary via -W and -C flags.
Weights and thresholds are random but fixed-seed so results are reproducible.
MAX_FEATURES=256 is the RTL compile-time constant (VX_checker MAX_FEATURES param).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

MAX_FEATURES = 256   # must match VX_checker.sv MAX_FEATURES parameter
TEST_DIR     = Path(__file__).parent


def gen_perf_files(hidden: int, num_features: int, count_k: int, seed: int) -> None:
    assert num_features <= MAX_FEATURES, \
        f"num_features ({num_features}) exceeds MAX_FEATURES ({MAX_FEATURES})"
    assert hidden > 0 and num_features > 0

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Weight binary: [hidden × MAX_FEATURES] FP16, zero-padded per row.
    # Features beyond num_features are zero so the checker ignores them.
    # ------------------------------------------------------------------
    weights_f32 = rng.normal(0.0, 0.1, (hidden, num_features)).astype(np.float32)
    padded = np.zeros((hidden, MAX_FEATURES), dtype=np.float16)
    padded[:, :num_features] = weights_f32.astype(np.float16)

    weight_path = TEST_DIR / "weights_perf.bin"
    padded.tofile(weight_path)
    print(f"Wrote {weight_path}  ({hidden} × {MAX_FEATURES} FP16 = {weight_path.stat().st_size} bytes)")

    # ------------------------------------------------------------------
    # Threshold binary: [num_features+1] uint16.
    #   [0]      = count_k  (uint16: global flag fires when fired_count > k)
    #   [1..F]   = per-feature FP16 thresholds
    #
    # Default: thresholds at the 25th percentile of expected outputs so
    # roughly 75% of features fire on random activations — same policy as
    # run_tests.py build_random().  count_k set so ~half the tokens flag.
    # ------------------------------------------------------------------
    # Estimate expected outputs using fp16 arithmetic (matches RTL FMA chain)
    sample_acts = rng.normal(0.0, 1.0, (16, hidden)).astype(np.float32)
    a16 = sample_acts.astype(np.float16)
    w16 = weights_f32.astype(np.float16)[:, :num_features]
    out = np.zeros((16, num_features), dtype=np.float16)
    for k in range(hidden):
        out = (a16[:, k:k+1].astype(np.float32) * w16[k:k+1, :].astype(np.float32)
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

    thresh_path = TEST_DIR / "thresholds_perf.bin"
    arr.tofile(thresh_path)
    print(f"Wrote {thresh_path}  ({num_features+1} uint16 = {thresh_path.stat().st_size} bytes)")
    print(f"  count_k={count_k}  thresh[0..3]={[float(np.frombuffer(arr[i+1:i+2].tobytes(), dtype=np.float16)[0]) for i in range(min(4,num_features))]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden",    "-H", type=int, default=512,
                   help="hidden_size (rows of weight SRAM, == -H in checker_test)")
    p.add_argument("--features",  "-F", type=int, default=64,
                   help="num_features (<= MAX_FEATURES=256)")
    p.add_argument("--count-k",   "-k", type=int, default=32,
                   help="count threshold k (flag fires when fired_count > k)")
    p.add_argument("--seed",      "-s", type=int, default=0,
                   help="RNG seed for reproducibility")
    args = p.parse_args()

    if args.features > MAX_FEATURES:
        print(f"Error: --features {args.features} exceeds MAX_FEATURES={MAX_FEATURES}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating perf files: hidden={args.hidden}  features={args.features}  "
          f"count_k={args.count_k}  seed={args.seed}")
    gen_perf_files(args.hidden, args.features, args.count_k, args.seed)
    print()
    print("Run with checker:")
    print(f"  ./ci/blackbox.sh --driver=rtlsim --cores=2 --app=checker_test \\")
    print(f'  "--args=-T16 -H{args.hidden} -N512 -F{args.features} -t4 \\')
    print(f"         -W {TEST_DIR}/weights_perf.bin \\")
    print(f'         -C {TEST_DIR}/thresholds_perf.bin -e 1"')
    print()
    print("Run without checker (rebuild with CONFIGS=\"\" first):")
    print(f"  ./ci/blackbox.sh --driver=rtlsim --cores=2 --app=checker_test \\")
    print(f'  "--args=-T16 -H{args.hidden} -N512 -F{args.features} -t4"')


if __name__ == "__main__":
    main()
