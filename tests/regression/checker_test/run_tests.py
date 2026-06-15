#!/usr/bin/env python3
"""
Automated sweep harness for VX_checker RTL simulation.

Generates weight/threshold hex files, runs blackbox.sh with rtlsim, parses
the [CHECKER] FLAG VECTOR trace, and compares against a Python reference.

One-time build (before running this script):
  CONFIGS="-DCHECKER_ENABLE" make -s -j4

Then sweep without rebuilding:
  python3 run_tests.py

The harness only rebuilds the hex files between cases; the Verilator binary is
reused. RTL compile-time parameters (B_TILE=4, N_FEAT=16, MAX_BATCH, MAX_FEATURES)
are fixed at build time and must match the ranges tested here.
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths — adjust if your layout differs
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parents[3]          # vortex-research/
BUILD_DIR  = REPO_ROOT / "build"
TEST_DIR   = Path(__file__).parent
WEIGHT_HEX = TEST_DIR / "sae_weights_test.hex"
THRESH_HEX = TEST_DIR / "thresholds.hex"
GEN_W      = TEST_DIR / "gen_weights.py"
GEN_T      = TEST_DIR / "gen_thresholds.py"
BLACKBOX   = BUILD_DIR / "ci" / "blackbox.sh"


# ---------------------------------------------------------------------------
# Reference computation (mirrors RTL FP16 matmul + threshold logic)
# ---------------------------------------------------------------------------

def fp16_matmul(activations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    activations: [batch, hidden]  float32 (will be cast to fp16 internally)
    weights:     [hidden, nfeat]  float32
    Returns:     [batch, nfeat]   float32 (fp16 precision)
    """
    a16 = activations.astype(np.float16)
    w16 = weights.astype(np.float16)
    # Accumulate in fp16 to match RTL (each MAC is fp16)
    out = np.zeros((a16.shape[0], w16.shape[1]), dtype=np.float32)
    for k in range(a16.shape[1]):
        out += (a16[:, k:k+1] * w16[k:k+1, :]).astype(np.float32)
    return out.astype(np.float16)


def reference_flags(activations: np.ndarray, weights: np.ndarray,
                    thresholds: np.ndarray, count_k: int) -> np.ndarray:
    """
    Returns boolean array [batch] — True if token should be flagged.
    thresholds: [nfeat] float32 per-feature thresholds
    count_k:    integer count threshold (flag when fired > count_k)
    """
    out   = fp16_matmul(activations, weights)           # [batch, nfeat]
    fired = (out > thresholds.astype(np.float16))      # [batch, nfeat] bool
    counts = fired.sum(axis=1)                          # [batch]
    return counts > count_k


# ---------------------------------------------------------------------------
# Hex file generation helpers (delegates to gen_weights.py / gen_thresholds.py)
# ---------------------------------------------------------------------------

def write_weight_hex(weights: np.ndarray, path: Path) -> None:
    """weights: [hidden, nfeat] float32 → write as FP16 hex (row-major, all features per row)."""
    hidden, nfeat = weights.shape
    with open(path, "w") as f:
        for k in range(hidden):
            for n in range(nfeat):
                bits = int(np.float16(weights[k, n]).view(np.uint16))
                f.write(f"{bits:04x}\n")


def write_threshold_hex(count_k: int, thresholds: np.ndarray, path: Path) -> None:
    """thresholds: [nfeat] float32 → threshold[0]=k, threshold[1..N]=fp16 per-feature."""
    with open(path, "w") as f:
        f.write(f"{count_k & 0xFFFF:04x}\n")
        for v in thresholds:
            bits = int(np.float16(v).view(np.uint16))
            f.write(f"{bits:04x}\n")


# ---------------------------------------------------------------------------
# Simulation runner + trace parser
# ---------------------------------------------------------------------------

def run_sim(num_tokens: int, num_features: int, hidden_size: int,
            cores: int = 2, extra_app_args: str = "") -> tuple[int, str]:
    """
    Run blackbox.sh and return (returncode, combined_stdout_stderr).
    blackbox.sh is invoked from BUILD_DIR so toolchain_env.sh is already
    sourced in the environment (caller must ensure that, or add it here).
    """
    app_args = f"-T {num_tokens} -F {num_features} -H {hidden_size}"
    if extra_app_args:
        app_args += " " + extra_app_args

    cmd = [
        str(BLACKBOX),
        "--driver=rtlsim",
        f"--cores={cores}",
        "--app=checker_test",
        f"--args={app_args}",
    ]
    env = os.environ.copy()
    env["CONFIGS"] = "-DCHECKER_ENABLE"
    result = subprocess.run(
        cmd,
        cwd=BUILD_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = result.stdout + result.stderr
    return result.returncode, combined


def parse_flag_vector(output: str, batch_size: int) -> Optional[list[bool]]:
    """
    Extract per-token flags from the [CHECKER] FLAG VECTOR trace block.
    Returns list of bools length batch_size, or None if block not found.
    """
    # Match lines like:  tok[3]: flag=1  fired=29/32
    pattern = re.compile(r'\[CHECKER\]\s+tok\[(\d+)\]:\s+flag=(\d)')
    flags: dict[int, bool] = {}
    for m in pattern.finditer(output):
        flags[int(m.group(1))] = bool(int(m.group(2)))
    if not flags:
        return None
    return [flags.get(i, False) for i in range(batch_size)]


def parse_fired_counts(output: str, batch_size: int) -> Optional[list[int]]:
    """Extract per-token fired feature counts: fired=N/F."""
    pattern = re.compile(r'\[CHECKER\]\s+tok\[(\d+)\]:\s+flag=\d\s+fired=(\d+)/\d+')
    counts: dict[int, int] = {}
    for m in pattern.finditer(output):
        counts[int(m.group(1))] = int(m.group(2))
    if not counts:
        return None
    return [counts.get(i, 0) for i in range(batch_size)]


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    name:         str
    num_tokens:   int
    num_features: int
    hidden_size:  int
    count_k:      int
    # Numpy arrays set up before running
    activations:  Optional[np.ndarray] = field(default=None, repr=False)
    weights:      Optional[np.ndarray] = field(default=None, repr=False)
    thresholds:   Optional[np.ndarray] = field(default=None, repr=False)

    def build_ones(self) -> "TestCase":
        """All-ones activations, all-ones weights, zero thresholds."""
        B, H, F = self.num_tokens, self.hidden_size, self.num_features
        self.activations = np.ones((B, H), dtype=np.float32)
        self.weights     = np.ones((H, F), dtype=np.float32)
        self.thresholds  = np.zeros(F, dtype=np.float32)
        return self

    def build_random(self, seed: int = 0) -> "TestCase":
        rng = np.random.default_rng(seed)
        B, H, F = self.num_tokens, self.hidden_size, self.num_features
        self.activations = rng.normal(0, 1, (B, H)).astype(np.float32)
        self.weights     = rng.normal(0, 0.1, (H, F)).astype(np.float32)
        # Thresholds at 25th percentile of expected outputs → ~75% of features fire
        out = fp16_matmul(self.activations, self.weights)
        self.thresholds  = np.percentile(out, 25, axis=0).astype(np.float32)
        return self

    def build_negative(self, seed: int = 42) -> "TestCase":
        """Mix of positive and negative activations/weights to exercise signed FP16."""
        rng = np.random.default_rng(seed)
        B, H, F = self.num_tokens, self.hidden_size, self.num_features
        self.activations = rng.uniform(-2, 2, (B, H)).astype(np.float32)
        self.weights     = rng.uniform(-0.5, 0.5, (H, F)).astype(np.float32)
        out = fp16_matmul(self.activations, self.weights)
        self.thresholds  = np.percentile(out, 50, axis=0).astype(np.float32)
        return self


# ---------------------------------------------------------------------------
# Run one test case
# ---------------------------------------------------------------------------

def run_case(tc: TestCase, verbose: bool = False) -> bool:
    print(f"\n{'='*60}")
    print(f"TEST: {tc.name}")
    print(f"  tokens={tc.num_tokens}  features={tc.num_features}  "
          f"hidden={tc.hidden_size}  k={tc.count_k}")

    # Write hex files
    write_weight_hex(tc.weights, WEIGHT_HEX)
    write_threshold_hex(tc.count_k, tc.thresholds, THRESH_HEX)

    # Compute reference
    expected = reference_flags(tc.activations, tc.weights, tc.thresholds, tc.count_k)
    print(f"  expected flags: {expected.astype(int).tolist()}")

    # Run simulation
    try:
        rc, output = run_sim(tc.num_tokens, tc.num_features, tc.hidden_size)
    except subprocess.TimeoutExpired:
        print("  FAIL: simulation timed out")
        return False

    if verbose:
        # Print only CHECKER trace lines
        for line in output.splitlines():
            if "[CHECKER]" in line:
                print("  " + line)

    # Check host program passed (kernel sum correctness)
    if rc != 0:
        print(f"  FAIL: blackbox.sh exited with code {rc}")
        if not verbose:
            for line in output.splitlines()[-30:]:
                print("  " + line)
        return False

    # Parse checker output
    rtl_flags = parse_flag_vector(output, tc.num_tokens)
    if rtl_flags is None:
        print("  FAIL: [CHECKER] FLAG VECTOR block not found in output")
        return False

    fired = parse_fired_counts(output, tc.num_tokens)
    print(f"  rtl    flags: {[int(f) for f in rtl_flags]}")
    if fired:
        print(f"  fired counts: {fired}")

    # Compare
    mismatches = [i for i in range(tc.num_tokens) if rtl_flags[i] != bool(expected[i])]
    if mismatches:
        print(f"  FAIL: mismatch at token(s) {mismatches}")
        for i in mismatches:
            print(f"    tok[{i}]: rtl={int(rtl_flags[i])} expected={int(expected[i])}")
        return False

    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Test suite definition
# ---------------------------------------------------------------------------

def build_suite() -> list[TestCase]:
    return [
        # --- ones: accumulate = hidden_size per PE; every feature fires ---
        TestCase("ones_8tok_32feat_64hidden_k0",
                 num_tokens=8, num_features=32, hidden_size=64, count_k=0
                 ).build_ones(),
        TestCase("ones_8tok_32feat_64hidden_k31",
                 num_tokens=8, num_features=32, hidden_size=64, count_k=31
                 ).build_ones(),
        TestCase("ones_8tok_32feat_64hidden_k32_noflag",
                 num_tokens=8, num_features=32, hidden_size=64, count_k=32  # strict > never fires
                 ).build_ones(),

        # --- tiling: batch > B_TILE (4) and features > N_FEAT (16) ---
        TestCase("ones_4tok_16feat_32hidden_k0",
                 num_tokens=4, num_features=16, hidden_size=32, count_k=0
                 ).build_ones(),
        TestCase("ones_12tok_32feat_64hidden_k16",
                 num_tokens=12, num_features=32, hidden_size=64, count_k=16
                 ).build_ones(),

        # --- random with positive/negative values ---
        TestCase("rand_8tok_32feat_64hidden_k8_seed0",
                 num_tokens=8, num_features=32, hidden_size=64, count_k=8
                 ).build_random(seed=0),
        TestCase("rand_8tok_32feat_64hidden_k8_seed1",
                 num_tokens=8, num_features=32, hidden_size=64, count_k=8
                 ).build_random(seed=1),

        # --- negative activations and thresholds ---
        TestCase("neg_8tok_32feat_64hidden_k4",
                 num_tokens=8, num_features=32, hidden_size=64, count_k=4
                 ).build_negative(seed=42),
        TestCase("neg_4tok_16feat_32hidden_k2",
                 num_tokens=4, num_features=16, hidden_size=32, count_k=2
                 ).build_negative(seed=7),

        # --- non-power-of-2 counts (exercises padding zeros in last tile) ---
        TestCase("rand_5tok_20feat_48hidden_k5",
                 num_tokens=5, num_features=20, hidden_size=48, count_k=5
                 ).build_random(seed=3),
        TestCase("rand_6tok_24feat_40hidden_k12",
                 num_tokens=6, num_features=24, hidden_size=40, count_k=12
                 ).build_random(seed=99),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print all [CHECKER] trace lines for each test")
    p.add_argument("--filter", "-f", default=None,
                   help="Only run tests whose name contains this substring")
    args = p.parse_args()

    suite = build_suite()
    if args.filter:
        suite = [tc for tc in suite if args.filter in tc.name]
        if not suite:
            sys.exit(f"No tests match filter '{args.filter}'")

    passed, failed = 0, 0
    for tc in suite:
        ok = run_case(tc, verbose=args.verbose)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
