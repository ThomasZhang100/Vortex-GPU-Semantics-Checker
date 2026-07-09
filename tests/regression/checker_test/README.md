# checker_test

RTL test for `VX_checker` — the non-bypassable semantic checker. A tiled sgemm
kernel writes a hidden-state tensor to memory; the checker independently taps
that tensor off L2, runs the selected-feature SAE matmul + threshold in its
systolic array, and raises per-token flags. This directory holds the host/kernel
program and the Python tooling that feeds it inputs and checks its output.

## Typical workflow

```bash
# 1. Build the RTL with the checker enabled (once).
cd build && source ./ci/toolchain_env.sh
CONFIGS="-DCHECKER_ENABLE" make -s -j4

# 2. Run the correctness sweep.
python3 ../tests/regression/checker_test/run_tests.py

# 3. For a perf run, generate inputs and use the printed command.
python3 ../tests/regression/checker_test/gen_perf_files.py --hidden 512 --features 64
```

## Build

The checker is gated behind a compile-time flag. Build once from the repo `build/`
directory before running anything here:

```bash
CONFIGS="-DCHECKER_ENABLE" make -s -j4
```

The Verilator binary is reused across runs; only the input files change between
cases. RTL compile-time parameters (`B_TILE`, `N_FEAT`, `MAX_BATCH`,
`MAX_FEATURES`, `MAX_HIDDEN`) are fixed at build time and must match the ranges
the scripts generate for.

## The test program

`main.cpp` (host) + `kernel.cpp` (GPU) build into the `checker_test` app, run via
`ci/blackbox.sh`. It is invoked directly by `run_tests.py`, or manually:

```bash
CONFIGS="-DCHECKER_ENABLE" ./ci/blackbox.sh --driver=rtlsim --cores=2 \
  --app=checker_test --args="-T8 -H64 -N16 -F32 -t4 -e1 -W weights.bin -C thresh.bin -A act.bin"
```

### `main.cpp` arguments

| Flag | Meaning |
|------|---------|
| `-T <n>` | num_tokens (M): rows of A / checker batch size |
| `-F <n>` | num_features: total SAE features (≤ RTL `MAX_FEATURES`) |
| `-H <n>` | hidden_size (K): cols of A / rows of B / checker hidden dim |
| `-N <n>` | out_width: cols of B / cols of C (GEMM output width) |
| `-t <n>` | sgemm tile size. Must be ≥ 2 and divide `-T`, `-H`, and `-N`. |
| `-e <m>` | enable/trigger mode (see below) |
| `-W <f>` | SAE weight binary (FP16, `[hidden × MAX_FEATURES]`) loaded via DCR |
| `-C <f>` | threshold binary (`uint16`: `[0]`=count-k, `[1..F]`=FP16 per-feature) |
| `-A <f>` | activation binary (FP32, row-major) injected as matrix A |
| `-o` | all-ones activations (instead of `-A`) |
| `-X` | structural control: build the GEMM(s) but never arm the checker |
| `-Z` | mode-2 cache-persistence check: tap GEMM1's input X instead of A_hidden |

**`-t` constraint:** `num_tokens`, `hidden_size`, and `out_width` must all be
multiples of the tile size, and the tile must be ≥ 2 (the tiled kernel needs a
real workgroup — `group_size = tile_size² > 1`). A tile of 1 deadlocks
`__syncthreads` and aliases local memory.

### Enable modes (`-e`)

| Mode | Behavior |
|------|----------|
| `1` | immediate arm — checker starts as soon as the ENABLE DCR is written |
| `2` | two-GEMM + addr-trigger: GEMM1 `X·W1→A_hidden`, GEMM2 `A_hidden·B→C`; checker taps A_hidden, armed after GEMM1, fires on the first B-matrix L2 read |
| `3` | addr-trigger, single GEMM — checker waits for the first B-matrix L2 read |

## Loading weights & thresholds: streamed binary vs hard-coded hex

The checker's weight SRAM and threshold table can be filled two ways, and each
gen script targets exactly one of them. This is the main thing to understand
before picking a script.

**1. Streamed binary — runtime, no rebuild. This is the path actually used.**
`main.cpp` streams a `.bin` file word-by-word into the checker at runtime through
the `VX_DCR_CHECKER_WEIGHT_DATA` / `VX_DCR_CHECKER_THRESH_DATA` DCR registers
(the `-W` / `-C` flags), and activations through `-A`. Nothing is baked into the
Verilator image, so you swap inputs between runs without recompiling.
`gen_perf_files.py`, `gen_mode2_files.py`, and `run_tests.py`'s own writers all
produce these binaries.

**2. Hard-coded hex — elaboration-time `$readmemh`. Currently NOT wired up.**
`VX_checker` has `WEIGHT_FILE` / `THRESHOLD_FILE` parameters that, when
non-empty, `$readmemh` a hex file into the SRAM/threshold table at time 0 — the
values become part of the compiled image. **`VX_cluster.sv` instantiates the
checker without setting these params, so they default to `""` and this path is
inactive in the current build.** `gen_weights.py` / `gen_thresholds.py` emit this
hex format; today it is only a `$readmemh` reference and a way to eyeball values.
Making it live requires editing the `VX_checker` instantiation in `VX_cluster.sv`
to set the file params, then rebuilding — there is no `blackbox.sh` flag for it.

| | Streamed binary (`.bin`) | Hard-coded hex (`.hex`) |
|---|---|---|
| Loaded | runtime, DCR streaming (`-W`/`-C`/`-A`) | elaboration, `$readmemh` at time 0 |
| Change inputs | swap the file, rerun | edit RTL param + rebuild Verilator |
| Wired up today | **yes** | **no** (params default `""`) |
| Weights ↔ thresholds | calibrated together (`gen_perf`/`gen_mode2`) | generated independently |
| Produced by | `gen_perf_files.py`, `gen_mode2_files.py`, `run_tests.py` | `gen_weights.py`, `gen_thresholds.py` |

## Scripts

All scripts live here and write their output files into this directory. Run them
with `python3 <script>.py`; each takes `--help`. Every example below assumes you
are in the `build/` directory with `source ./ci/toolchain_env.sh` already done
and the checker build in place (`CONFIGS="-DCHECKER_ENABLE" make -s -j4`).

The `blackbox.sh` examples reference input files as `$VORTEX_HOME/tests/...`, so
set `VORTEX_HOME` to your Vortex project root once per shell

### `run_tests.py` — the sweep harness

Runs a suite of `TestCase`s through `blackbox.sh` (rtlsim), parses the
`[CHECKER]` trace, and compares against a NumPy FP16 reference. Regenerates the
weight/threshold/activation input files per case; does not rebuild the RTL.

```bash
python3 run_tests.py                 # run the whole suite
python3 run_tests.py -f rand_6tok    # only cases whose name contains this
python3 run_tests.py -v              # print all [CHECKER] trace lines
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-f, --filter <s>` | none | run only cases whose name contains `<s>` |
| `-v, --verbose` | off | print every `[CHECKER]` trace line per case |
| `--max-ulp <n>` | 1 | max FP16 ULP distance allowed on dumped matrix values |
| `--max-features <n>` | 256 | weight-SRAM column count; must equal the RTL build's `MAX_FEATURES` |

Exit code 0 if all cases pass, 1 otherwise. The pass/fail gate is per-feature
firing (RTL `FIRED BITMAP` vs reference); matrix-value ULP is diagnostic only.

### `gen_perf_files.py` — SAE weight/threshold binaries for perf runs (for streaming through control register)

Generates large-scale random (fixed-seed) weight/threshold binaries
(`weights_perf.bin`, `thresholds_perf.bin`, the **streamed binary** path) for
performance measurement, then prints run commands with and without the checker.

```bash
python3 gen_perf_files.py --hidden 128 --features 64
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-H, --hidden <n>` | 512 | hidden_size (weight-SRAM rows) |
| `-F, --features <n>` | 64 | num_features (≤ `--max-features`) |
| `--max-features <n>` | 256 | weight-SRAM column count; must equal the RTL param |
| `-k, --count-k <n>` | 32 | count threshold k |
| `-s, --seed <n>` | 0 | RNG seed |

**Then run** (mode 1 taps matrix A directly; with `-A` omitted A is a
deterministic host-side ramp, `-o` fills it with all-ones, or `-A <file>` injects
your own activations):

```bash
# With the checker active:
CONFIGS="-DCHECKER_ENABLE" ./ci/blackbox.sh \
    --driver=rtlsim \
    --cores=4 \
    --app=checker_test \
    "--args=-T16 -H128 -N128 -F64 -t4 \
           -W $VORTEX_HOME/tests/regression/checker_test/weights_perf.bin \
           -C $VORTEX_HOME/tests/regression/checker_test/thresholds_perf.bin \
           -e 1"

# Baseline without the checker (rebuild with CONFIGS="" first):
./ci/blackbox.sh \
    --driver=rtlsim \
    --cores=4 \
    --app=checker_test \
    "--args=-T16 -H128 -N128 -F64 -t4"
```

### `gen_mode2_files.py` — SAE weight/threshold binaries for `-e 2` (for streaming through control register)

Reproduces mode-2's GEMM1 (`X·W1→A_hidden`, seeds 51/52 matching `main.cpp`),
computes `A_hidden` in FP16, and calibrates SAE weights + thresholds to its
distribution. Writes `<prefix>_weights.bin` and `<prefix>_thresholds.bin` (the
**streamed binary** path), then prints the matching `blackbox.sh` command.

```bash
python3 gen_mode2_files.py --tokens 16 --hidden 64 --features 32 --count-k 16
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-T, --tokens <n>` | 16 | num_tokens M |
| `-H, --hidden <n>` | 64 | hidden_size K (also GEMM1 output dim) |
| `-F, --features <n>` | 32 | SAE features (≤ `--max-features`) |
| `--max-features <n>` | 256 | weight-SRAM column count; must equal the RTL param |
| `-k, --count-k <n>` | 16 | count-k threshold |
| `-s, --seed <n>` | 0 | SAE weight RNG seed (X/W1 seeds fixed at 51/52) |
| `-o, --out-prefix <s>` | mode2 | output file prefix |

**Then run** (no `-A`: mode 2 generates activations on-device via GEMM1; `-N` is
the independent GEMM2 output width, any multiple of `-t`):

```bash
# With the checker active:
CONFIGS="-DCHECKER_ENABLE -DCACHE_PERSIST" ./ci/blackbox.sh --driver=rtlsim --cores=4 \
    --app=checker_test \
    "--args=-T16 -H64 -N128 -F32 -t4 \
     -W  $VORTEX_HOME/tests/regression/checker_test/mode2_weights.bin \
-C  $VORTEX_HOME/tests/regression/checker_test/mode2_thresholds.bin -e 2" | grep -E "latency|start GEMM|PERF|MISS_RATE"

# Baseline without the checker (rebuild with CONFIGS="" first):
CONFIGS="-DCACHE_PERSIST" ./ci/blackbox.sh     --driver=rtlsim     --cores=4     --app=checker_test     "--args=-T16 -H64 -N128 -F32 -t4 -e 2" | grep -E "latency|start GEMM|PERF|MISS_RATE"
```

### `gen_weights.py` — SAE weight hex (hard-coded path; no SAE weight streaming through control register)

Generates a `$readmemh` weight file (one SRAM row per line). This is the
**hard-coded hex** path — the file is *not* consumed by a `blackbox.sh` flag.
`run_tests.py` writes its own binaries and never calls this script.

```bash
python3 gen_weights.py --mode identity --hidden 64 --nfeat 16 --out sae_weights_test.hex
python3 gen_weights.py --mode saedec --weights W.npy --out sae_weights.hex
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode` | identity | `identity` (W[k][n]=k), `ones` (W=1), or `saedec` (from `.npy`) |
| `--hidden <n>` | 64 | hidden_size |
| `--nfeat <n>` | 256 | `MAX_FEATURES` (SRAM column count); must equal the RTL param |
| `--maxhidden <n>` | 2048 | `MAX_HIDDEN` (SRAM depth) |
| `--weights <f>` | none | `.npy` weight file, required for `--mode saedec` |
| `--out <f>` | sae_weights_test.hex | output hex path |

**Then run:** there is no runtime flag. Wire the hex into the build by setting
the `WEIGHT_FILE` parameter on the `VX_checker` instance in `VX_cluster.sv`, e.g.

```systemverilog
VX_checker #(
    ...
    .WEIGHT_FILE ("tests/regression/checker_test/sae_weights_test.hex")
) sem_checker ( ... );
```

then rebuild (`CONFIGS="-DCHECKER_ENABLE" make -s -j4`) and launch without `-W`:

```bash
CONFIGS="-DCHECKER_ENABLE" ./ci/blackbox.sh --driver=rtlsim --cores=2 \
  --app=checker_test --args="-T8 -H64 -N16 -F16 -t4 -e1 -A act_test.bin"
```

### `gen_thresholds.py` — threshold hex (hard-coded path; no threshold streaming through control register)

Generates a threshold file: line 0 is the count-k header (flag fires when
fired-count > k), lines 1..N are per-feature FP16 thresholds. Same **hard-coded
hex** path as `gen_weights.py` — consumed via `$readmemh`, not a runtime flag.

```bash
python3 gen_thresholds.py --mode zeros --num-features 16 --count-threshold 0 --out t.hex
python3 gen_thresholds.py --mode value --value 0.5 --num-features 64 --count-threshold 8 --out t.hex
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode` | zeros | `zeros`, `value` (constant), or `file` (from `.npy`) |
| `--value <x>` | 0.0 | per-feature threshold for `--mode value` |
| `--num-features <n>` | 16 | total SAE features (≤ `MAX_FEATURES`) |
| `--count-threshold <k>` | 0 | flag fires when fired-count > k (0 = any hit, N = require all) |
| `--weights <f>` | none | `.npy` file for `--mode file` |
| `--out <f>` | thresholds.hex | output hex path |

**Then run:** set the companion `THRESHOLD_FILE` parameter on the `VX_checker`
instance in `VX_cluster.sv` (alongside `WEIGHT_FILE`), rebuild, and launch
without `-C`. Typically you generate the weight and threshold hex together and
wire both params in the same build.




