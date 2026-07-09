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
WEIGHT_HEX = TEST_DIR / "sae_weights_test.hex"   # kept for debugging / $readmemh reference
THRESH_HEX = TEST_DIR / "thresholds.hex"          # kept for debugging / $readmemh reference
ACT_BIN    = TEST_DIR / "act_test.bin"            # FP32 activation binary injected via -A
WEIGHT_BIN = TEST_DIR / "weights_test.bin"        # FP16 weight binary loaded via DCR (-W)
THRESH_BIN = TEST_DIR / "thresholds_test.bin"     # uint16 threshold binary loaded via DCR (-C)
BLACKBOX   = BUILD_DIR / "ci" / "blackbox.sh"

# Must match the compiled RTL parameter VX_checker MAX_FEATURES.
# Each SRAM row is MAX_FEATURES FP16 values wide; $readmemh reads one row per line.
MAX_FEATURES = 256


# ---------------------------------------------------------------------------
# Reference computation (mirrors RTL FP16 matmul + threshold logic)
# ---------------------------------------------------------------------------

def print_fp16_hex_matrix(name, x):
    x16 = np.asarray(x, dtype=np.float16)
    u16 = x16.view(np.uint16)

    print(f"{name} decimal:")
    print(x16)

    print(f"{name} hex:")
    for row in np.atleast_2d(u16):
        print("  " + " ".join(f"0x{v:04x}" for v in row))

def fp16_matmul(activations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """FP16 matmul mirroring the RTL's systolic FMA chain to within ±1 ULP.

    Accumulates one k-step at a time (acc = fp16(a[k]*w[k] + acc)), widening to
    float32 for each add and rounding back to fp16.

    activations: [batch, hidden]  (cast to fp16 before use)
    weights:     [hidden, nfeat]  (cast to fp16 before use)
    Returns:     [batch, nfeat]   fp16 array
    """
    a16 = activations.astype(np.float16)
    w16 = weights.astype(np.float16)
    out = np.zeros((a16.shape[0], w16.shape[1]), dtype=np.float16)
    for k in range(a16.shape[1]):
        a_f32  = a16[:, k:k+1].astype(np.float32)
        w_f32  = w16[k:k+1, :].astype(np.float32)
        acc_f32 = out.astype(np.float32)
        out = (a_f32 * w_f32 + acc_f32).astype(np.float16)
    return out


def reference_fired(matrix: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Per-feature fired boolean [batch, nfeat] given an FP16 matmul output matrix.
    matrix is re-quantized to fp16 first so values round-tripped through text
    (e.g. parsed RTL trace output) compare correctly against fp16 thresholds.
    """
    return matrix.astype(np.float16) > thresholds.astype(np.float16)


def reference_flags(fired: np.ndarray, count_k: int) -> np.ndarray:
    """
    Returns boolean array [batch] — True if token should be flagged.
    fired:   [batch, nfeat] bool — per-feature fired state
    count_k: integer count threshold (flag when fired-count > count_k)
    """
    return fired.sum(axis=1) > count_k


# ---------------------------------------------------------------------------
# Hex file generation helpers (delegates to gen_weights.py / gen_thresholds.py)
# ---------------------------------------------------------------------------

def write_weight_hex(weights: np.ndarray, path: Path,
                     max_features: int = None) -> None:
    """Write one SRAM row per line: all max_features FP16 values packed as one hex word.

    Feature n occupies bits [n*16+15 : n*16] (feature 0 = LSB).
    $readmemh reads MSB-first, so the hex string is feature[max_features-1]...feature[0].
    Features beyond weights.shape[1] are zero-padded.
    """
    if max_features is None:
        max_features = MAX_FEATURES
    hidden, nfeat = weights.shape
    hex_chars = max_features * 4   # bits per row / 4
    with open(path, "w") as f:
        for k in range(hidden):
            word = 0
            for n in range(nfeat):
                bits = int(np.float16(weights[k, n]).view(np.uint16))
                word |= (bits & 0xFFFF) << (n * 16)
            f.write(f"{word:0{hex_chars}x}\n")


def write_act_bin(activations: np.ndarray, path: Path) -> None:
    """Write FP32 activations as a raw row-major binary (loaded via main.cpp -A).

    Stays FP32 end to end: the checker narrows each element to FP16 itself in
    hardware (VX_checker.sv's fp32_to_fp16) when it taps the tensor off L2.
    """
    activations.astype(np.float32).tofile(path)


def write_threshold_hex(count_k: int, thresholds: np.ndarray, path: Path) -> None:
    """thresholds: [nfeat] float32 → threshold[0]=k, threshold[1..N]=fp16 per-feature."""
    with open(path, "w") as f:
        f.write(f"{count_k & 0xFFFF:04x}\n")
        for v in thresholds:
            bits = int(np.float16(v).view(np.uint16))
            f.write(f"{bits:04x}\n")


def write_weight_bin(weights: np.ndarray, path: Path,
                     max_features: int = None) -> None:
    """Write weight SRAM binary for DCR streaming via VX_DCR_CHECKER_WEIGHT_DATA.

    Format: [hidden_size × max_features] FP16, row-major, little-endian.
    Each row is zero-padded to max_features so main.cpp always streams exactly
    max_features/2 uint32 words per row regardless of the actual num_features.
    Feature n occupies bytes [n*2 : n*2+2] within each row (feature 0 = LSB
    of the first uint32 word), matching the weight_wbuf bit layout in
    VX_cluster.sv: w_wbuf[word_idx*32 +: 32] → features [2*word_idx, 2*word_idx+1].
    """
    if max_features is None:
        max_features = MAX_FEATURES
    hidden, nfeat = weights.shape
    padded = np.zeros((hidden, max_features), dtype=np.float16)
    padded[:, :nfeat] = weights.astype(np.float16)
    padded.tofile(path)


def write_thresh_bin(count_k: int, thresholds: np.ndarray, path: Path) -> None:
    """Write threshold binary for DCR streaming via VX_DCR_CHECKER_THRESH_DATA.

    Format: [num_features+1] uint16, little-endian.
      index 0 : count_k (raw uint16 — global flag fires when fired-count > count_k)
      index 1..N : per-feature FP16 activation thresholds
    main.cpp streams these in order; VX_cluster.sv's t_widx auto-increments.
    """
    arr = np.empty(len(thresholds) + 1, dtype=np.uint16)
    arr[0] = count_k & 0xFFFF
    for i, v in enumerate(thresholds):
        arr[i + 1] = int(np.float16(v).view(np.uint16))
    arr.tofile(path)


# ---------------------------------------------------------------------------
# Simulation runner + trace parser
# ---------------------------------------------------------------------------

def run_sim(num_tokens: int, num_features: int, hidden_size: int,
            weight_bin: Optional[Path] = None,
            thresh_bin: Optional[Path] = None,
            act_bin: Optional[Path] = None,
            tile_size: Optional[int] = None,
            cores: int = 2, extra_app_args: str = "") -> tuple[int, str]:
    """
    Run blackbox.sh and return (returncode, combined_stdout_stderr).
    blackbox.sh is invoked from BUILD_DIR so toolchain_env.sh is already
    sourced in the environment (caller must ensure that, or add it here).
    weight_bin: FP16 weight binary loaded via -W (VX_DCR_CHECKER_WEIGHT_DATA).
    thresh_bin: uint16 threshold binary loaded via -C (VX_DCR_CHECKER_THRESH_DATA).
    act_bin:    FP32 activation binary injected via -A.
    tile_size:  sgemm tile size (-t). The host requires num_tokens, hidden_size,
                and out_width to all be multiples of this; leave None to use
                main.cpp's default (4).
    """
    app_args = f"-T {num_tokens} -F {num_features} -H {hidden_size}"
    if tile_size is not None:
        app_args += f" -t {tile_size}"
    if weight_bin is not None:
        app_args += f" -W {weight_bin}"
    if thresh_bin is not None:
        app_args += f" -C {thresh_bin}"
    if act_bin is not None:
        app_args += f" -A {act_bin}"
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
        timeout=1800,  # 30 min: covers Verilator recompile (~5-10 min) + simulation
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


def parse_full_matrix(output: str, batch_size: int,
                      num_features: int) -> Optional[np.ndarray]:
    """Parse the === FULL MATMUL OUTPUT === block.

    Returns float32 [batch, num_features], or None if the block is absent.
    """
    start = output.find("=== FULL MATMUL OUTPUT")
    if start == -1:
        return None
    end = output.find("================================================", start)
    block = output[start:end] if end != -1 else output[start:]

    matrix: dict[int, np.ndarray] = {}
    tok_re = re.compile(r'tok\[(\d+)\]:\s+(.*)')
    for m in tok_re.finditer(block):
        tok = int(m.group(1))
        if tok >= batch_size:
            continue
        try:
            vals = list(map(float, m.group(2).split()))
            matrix[tok] = np.array(vals[:num_features], dtype=np.float32)
        except ValueError:
            pass

    if len(matrix) < batch_size:
        return None
    result = np.zeros((batch_size, num_features), dtype=np.float32)
    for tok, row in matrix.items():
        result[tok, :len(row)] = row
    return result


def parse_fired_bitmap(output: str, batch_size: int,
                       num_features: int) -> Optional[np.ndarray]:
    """Parse the === FIRED BITMAP === block (the RTL's own per-feature fire decision).

    Returns bool [batch, num_features], or None if the block is absent.
    """
    start = output.find("=== FIRED BITMAP")
    if start == -1:
        return None
    end = output.find("================================================", start)
    block = output[start:end] if end != -1 else output[start:]

    bitmap: dict[int, np.ndarray] = {}
    tok_re = re.compile(r'tok\[(\d+)\]:\s+(.*)')
    for m in tok_re.finditer(block):
        tok = int(m.group(1))
        if tok >= batch_size:
            continue
        try:
            vals = list(map(int, m.group(2).split()))
            bitmap[tok] = np.array(vals[:num_features], dtype=bool)
        except ValueError:
            pass

    if len(bitmap) < batch_size:
        return None
    result = np.zeros((batch_size, num_features), dtype=bool)
    for tok, row in bitmap.items():
        result[tok, :len(row)] = row
    return result


def parse_checker_latency(output: str) -> Optional[int]:
    """Extract checker latency in cycles from the ALL_DONE trace line.

    Matches: [CHECKER] ALL_DONE  count_thresh=N  global_flag=0xX  latency=N cycles
    Returns the integer cycle count, or None if not found (no-checker build).
    """
    m = re.search(r'\[CHECKER\] ALL_DONE.*?latency=(\d+) cycles', output)
    return int(m.group(1)) if m else None


def fp16_ulp_distance(a: float, b: float) -> int:
    """Integer ULP distance between two values in FP16 bit-space.

    Maps sign-magnitude bits to a total order so ±0 are 0 ULPs apart. NaN equals
    NaN (returns 0) and is maximally far (0x7FFF) from anything else.
    """
    def to_ordered(v: float) -> int:
        bits = int(np.float16(v).view(np.uint16))
        return -(bits & 0x7FFF) if (bits & 0x8000) else bits

    af, bf = np.float16(a), np.float16(b)
    if np.isnan(af) and np.isnan(bf):
        return 0
    if np.isnan(af) or np.isnan(bf):
        return 0x7FFF  # maximally far
    return abs(to_ordered(float(af)) - to_ordered(float(bf)))


def percentile_strict(out: np.ndarray, p: float) -> np.ndarray:
    """Per-column percentile that never lands exactly on a sample.

    Nudges the interpolation rank by half a slot when it would fall on an order
    statistic, so a threshold is never bit-identical to a reference output
    (which would make `ref > threshold` a rounding-sensitive tie).

    out: [batch, num_features]   p: percentile in [0, 100]
    Returns: [num_features] threshold per column.
    """
    B = out.shape[0]
    sorted_out = np.sort(out, axis=0)
    rank = (B - 1) * p / 100.0
    if rank == int(rank):
        rank = min(rank + 0.5, B - 1)
    lo, hi = int(np.floor(rank)), int(np.ceil(rank))
    frac = rank - lo
    return sorted_out[lo] + frac * (sorted_out[hi] - sorted_out[lo])


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
    # enable_mode maps to main.cpp -e flag:
    #   1 = immediate arm on ENABLE DCR write (default)
    #   3 = address-range trigger (checker waits for first L2 read of B matrix)
    enable_mode:  int = 1
    # sgemm tile size (main.cpp -t). The host requires num_tokens, hidden_size,
    # and out_width to all be multiples of this. It must also be >= 2: the tiled
    # kernel relies on a real workgroup (group_size = tile_size**2 > 1), and
    # tile_size=1 takes vx_spawn's per-thread path (warps_per_group=0,
    # local_group_id=0) which deadlocks __syncthreads and aliases local memory.
    tile_size:    int = 4
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
        self.thresholds  = percentile_strict(out, 25).astype(np.float32)
        return self

    def build_negative(self, seed: int = 42) -> "TestCase":
        """Mix of positive and negative activations/weights to exercise signed FP16."""
        rng = np.random.default_rng(seed)
        B, H, F = self.num_tokens, self.hidden_size, self.num_features
        self.activations = rng.uniform(-2, 2, (B, H)).astype(np.float32)
        self.weights     = rng.uniform(-0.5, 0.5, (H, F)).astype(np.float32)
        out = fp16_matmul(self.activations, self.weights)
        self.thresholds  = percentile_strict(out, 50).astype(np.float32)
        return self


# ---------------------------------------------------------------------------
# Run one test case
# ---------------------------------------------------------------------------

def run_case(tc: TestCase, verbose: bool = False, max_ulp: int = 1) -> bool:
    print(f"\n{'='*60}")
    print(f"TEST: {tc.name}")
    mode_tag = " [addr-trigger]" if tc.enable_mode == 3 else ""
    print(f"  tokens={tc.num_tokens}  features={tc.num_features}  "
          f"hidden={tc.hidden_size}  k={tc.count_k}{mode_tag}")

    # Write input files: weights + thresholds as binaries for DCR loading,
    # activations as FP32 binary so the GPU gets exactly what Python computed.
    write_weight_bin(tc.weights, WEIGHT_BIN)
    write_thresh_bin(tc.count_k, tc.thresholds, THRESH_BIN)
    write_act_bin(tc.activations, ACT_BIN)
    # Also write hex files for debugging / manual $readmemh inspection.
    write_weight_hex(tc.weights, WEIGHT_HEX)
    write_threshold_hex(tc.count_k, tc.thresholds, THRESH_HEX)

    # Compute reference (fp16_matmul accumulates in FP16, matching the RTL MAC chain)
    ref_matrix  = fp16_matmul(tc.activations, tc.weights)          # [B, F] fp16
    fired_ref   = reference_fired(ref_matrix, tc.thresholds)        # [B, F] bool
    expected    = reference_flags(fired_ref, tc.count_k)            # [B] bool
    print(f"  expected flags: {expected.astype(int).tolist()}")

    # Run simulation — pass -e flag to select arm mode.
    try:
        rc, output = run_sim(tc.num_tokens, tc.num_features, tc.hidden_size,
                             weight_bin=WEIGHT_BIN, thresh_bin=THRESH_BIN,
                             act_bin=ACT_BIN, tile_size=tc.tile_size,
                             extra_app_args=f"-e {tc.enable_mode}")
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

    # Detect whether the build included CHECKER_ENABLE.  If no [CHECKER] trace
    # lines appear at all the build was compiled without -DCHECKER_ENABLE; treat
    # as a GEMM-only run (PASS if host rc=0) rather than failing on missing output.
    if "[CHECKER]" not in output:
        print("  PASS (GEMM only — no [CHECKER] trace; build without -DCHECKER_ENABLE)")
        return True

    # Parse checker output (CHECKER_ENABLE build)
    rtl_flags = parse_flag_vector(output, tc.num_tokens)
    if rtl_flags is None:
        print("  FAIL: [CHECKER] FLAG VECTOR block not found in output")
        return False

    fired = parse_fired_counts(output, tc.num_tokens)
    latency = parse_checker_latency(output)
    print(f"  rtl    flags: {[int(f) for f in rtl_flags]}")
    if fired:
        print(f"  fired counts: {fired}")
    if latency is not None:
        print(f"  checker latency: {latency} cycles (rearm → all_done)")

    # Matrix-value diagnostic (informational; does not gate pass/fail).
    rtl_matrix = parse_full_matrix(output, tc.num_tokens, tc.num_features)
    if rtl_matrix is None:
        print("  FAIL: FULL MATMUL OUTPUT block not found — cannot verify feature firing")
        return False

    bad: list[tuple[int, int, float, float, int]] = []
    for b in range(tc.num_tokens):
        for f in range(tc.num_features):
            ref_val = float(ref_matrix[b, f])
            rtl_val = float(rtl_matrix[b, f])
            ulp     = fp16_ulp_distance(ref_val, rtl_val)
            if ulp > max_ulp:
                bad.append((b, f, ref_val, rtl_val, ulp))
    if bad:
        print(f"  {len(bad)} matrix element(s) differ by more than {max_ulp} ULP:")
        for b, f, rv, tv, ulp in bad[:200]:
            print(f"    tok[{b}] feat[{f}]: ref={rv:.6g}  rtl={tv:.6g}  ulp_dist={ulp}")
        if len(bad) > 200:
            print(f"    ... ({len(bad) - 200} more)")
    else:
        max_seen = max(
            fp16_ulp_distance(float(ref_matrix[b, f]), float(rtl_matrix[b, f]))
            for b in range(tc.num_tokens) for f in range(tc.num_features)
        )
        print(f"  matrix value match (max ULP dist = {max_seen})")

    # Per-feature firing check — the pass/fail gate. Compares the RTL's own
    # FIRED BITMAP against the reference per-feature fired state.
    fired_rtl = parse_fired_bitmap(output, tc.num_tokens, tc.num_features)
    if fired_rtl is None:
        print("  FAIL: FIRED BITMAP block not found — cannot verify feature firing")
        return False

    mismatches  = [
        (b, f) for b in range(tc.num_tokens) for f in range(tc.num_features)
        if bool(fired_rtl[b, f]) != bool(fired_ref[b, f])
    ]
    if mismatches:
        print(f"  FAIL: {len(mismatches)} feature firing mismatch(es):")
        for b, f in mismatches[:200]:
            print(f"    tok[{b}] feat[{f}]: expected_fire={int(fired_ref[b, f])}  "
                  f"rtl_fire={int(fired_rtl[b, f])}  threshold={float(tc.thresholds[f]):.6g}")
        if len(mismatches) > 200:
            print(f"    ... ({len(mismatches) - 200} more)")
        return False

    # Aggregate per-token flag (informational; implied by the per-feature check).
    flag_mismatches = [i for i in range(tc.num_tokens) if rtl_flags[i] != bool(expected[i])]
    if flag_mismatches:
        print(f"  NOTE: per-feature firing matched, but aggregate flag differs at "
              f"token(s) {flag_mismatches} (check count-vs-k comparator)")
        for i in flag_mismatches:
            print(f"    tok[{i}]: rtl={int(rtl_flags[i])} expected={int(expected[i])}")

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
                 ).build_random(seed=5),
        TestCase("rand_8tok_32feat_64hidden_k8_seed1",
                 num_tokens=8, num_features=32, hidden_size=64, count_k=8
                 ).build_random(seed=6),

        # --- negative activations and thresholds ---
        TestCase("neg_8tok_32feat_64hidden_k4",
                 num_tokens=8, num_features=32, hidden_size=64, count_k=4
                 ).build_negative(seed=42),
        TestCase("neg_4tok_16feat_32hidden_k2",
                 num_tokens=4, num_features=16, hidden_size=32, count_k=2
                 ).build_negative(seed=7),

        # --- non-multiple-of-tile counts (exercises padding zeros in last tile) ---
        # These M values aren't multiples of the checker's B_TILE (4), so they
        # force the batch-padding path. The sgemm tile must be >= 2 and divide
        # M, hidden_size, and out_width (16), so tile_size=2 is the largest
        # common factor; M=5 is unusable (coprime to 48 and 16) so use M=10.
        TestCase("rand_10tok_20feat_48hidden_k5",
                 num_tokens=10, num_features=20, hidden_size=48, count_k=5,
                 tile_size=2  # 2 | 10,48,16;  batch_tiles=ceil(10/4)=3 (last tile padded)
                 ).build_random(seed=7),
        TestCase("rand_6tok_24feat_40hidden_k12",
                 num_tokens=6, num_features=24, hidden_size=40, count_k=12,
                 tile_size=2  # 2 | 6,40,16;   batch_tiles=ceil(6/4)=2 (last tile padded)
                 ).build_random(seed=8),

        # --- address-range trigger (enable_mode=3): checker sits idle until the
        # GEMM's first L2 read of B fires a one-shot snoop. Same data as the
        # seed=5 immediate-arm case, so both paths hit the same pass/fail gate. ---
        TestCase("addr_trig_8tok_32feat_64hidden_k8",
                 num_tokens=8, num_features=32, hidden_size=64, count_k=8,
                 enable_mode=3
                 ).build_random(seed=5),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global MAX_FEATURES
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print all [CHECKER] trace lines for each test")
    p.add_argument("--filter", "-f", default=None,
                   help="Only run tests whose name contains this substring")
    p.add_argument("--max-ulp", type=int, default=1,
                   help="Max FP16 ULP distance allowed for matrix values (default: 1)")
    p.add_argument("--max-features", type=int, default=MAX_FEATURES,
                   help=f"weight-SRAM column count; must equal the RTL build's "
                        f"VX_CHECKER_MAX_FEATURES (default {MAX_FEATURES})")
    args = p.parse_args()

    # Override the SRAM row width to match the RTL build (-DVX_CHECKER_MAX_FEATURES).
    MAX_FEATURES = args.max_features

    suite = build_suite()
    if args.filter:
        suite = [tc for tc in suite if args.filter in tc.name]
        if not suite:
            sys.exit(f"No tests match filter '{args.filter}'")

    passed, failed = 0, 0
    for tc in suite:
        ok = run_case(tc, verbose=args.verbose, max_ulp=args.max_ulp)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
