# attest_test — Boot-Time Attestation / Verified Launch (Task C)

End-to-end test for the hardware boot verifier: a GEMM workload plus its kernel,
args, and SAE weights are described by a **signed manifest**; the boot verifier
hashes each region, checks the signature, and only releases the cores if
everything matches. A tampered image is blocked — the GEMM never runs.

## What it exercises

On launch the host (`main.cpp`):
1. Uploads matrix `A` (activations) and `B` (weights) to VRAM, pre-fills output
   `C` with a sentinel, uploads the GEMM kernel + args, and streams the SAE
   weights/thresholds into the checker's SRAM.
2. Computes SHA-256 of every region (`sha256.h`, matches `hw/rtl/libs/VX_sha256.sv`),
   builds the `manifest_t` (`manifest.h`), and signs it (Phase-0 keyed-hash stub).
3. Streams the manifest into the on-chip manifest SRAM and arms the verifier
   (`VX_DCR_ATTEST_*` DCRs), then calls `vx_start`.

Inside `vx_start`'s run the verifier hashes signature → kernel → args → SAE
weights → SAE thresholds → each per-layer weight region. All match ⇒ `boot_release`
⇒ cores boot ⇒ GEMM runs (and the checker taps its output). Any mismatch ⇒ cores
stay held ⇒ GEMM never runs.

**Pass/fail signal** is the GEMM output `C` (cache-independent): `C == reference`
⇒ attestation passed and the cores ran; `C == sentinel` ⇒ attestation blocked the
cores. Each run prints `PASSED!` when the observed behavior matches what the chosen
`-x` mode expects.

## Requirements

The RTL must be built with the checker **and** the attestation verifier enabled:

```
CONFIGS="-DCHECKER_ENABLE -DATTEST_ENABLE"
```

`ATTEST_ENABLE` requires `CHECKER_ENABLE` (the verifier reuses the checker's L2
port and SAE-SRAM read ports). `blackbox.sh` forwards `CONFIGS` to both the RTL
driver build and the app build, so the host sees the same `-DVX_CHECKER_MAX_FEATURES`
as the hardware.

## Build / environment setup

Vortex uses an **out-of-tree build** (everything lives under `build/`). From a
fresh clone, do the one-time setup (skip steps already done):

```sh
# 1. submodules (softfloat, ramulator, cvfpu, hardfloat) — required
git submodule update --init --recursive

# 2. system dependencies (Ubuntu; needs root/sudo)
./ci/install_dependencies.sh

# 3. create the build dir and configure
mkdir -p build && cd build
../configure --xlen=32 --tooldir=$HOME/tools

# 4. install the prebuilt toolchain: RISC-V GNU + LLVM + Verilator (large, one-time)
./ci/toolchain_install.sh --all
```

Then, in **every new shell**, source the toolchain environment:

```sh
cd build
source ./ci/toolchain_env.sh   # sets PATH/vars for RISC-V clang, Verilator, etc.
```

**Do not run the top-level `make`.** It builds every simulator, including the
OPAE/XRT FPGA shims this test never uses, which can exhaust a small or emulated
host (e.g. Docker on Apple Silicon). Instead build only the two components that
`blackbox.sh` does *not* build itself, then let blackbox build the rest (hw config,
runtime, the rtlsim driver — which Verilates the RTL — and the app):

```sh
# from build/, with toolchain_env sourced
make -C ../third_party    # softfloat + ramulator libs (the rtlsim driver links these)
make -C kernel            # kernel/libvortex.a (the GPU-side runtime the kernel links)
```

These two are one-time and persist. If the driver link later fails with
`cannot find .../softfloat.a` or `cannot find -lramulator`, those artifacts were
cleaned — just re-run `make -C ../third_party`.

The first test run below then triggers `blackbox.sh` to Verilate + compile the
rtlsim driver (slow — ~10-25 min under emulation; cached afterwards) and build the
app, then run the simulation.

## Running

From the Vortex build directory, with the toolchain sourced:

```sh
cd build
source ci/toolchain_env.sh

# valid manifest -> attestation PASS -> GEMM runs -> "PASSED!"
CONFIGS="-DCHECKER_ENABLE -DATTEST_ENABLE -DVX_CHECKER_MAX_FEATURES=32" \
  ./ci/blackbox.sh --driver=rtlsim --app=attest_test \
  --args="-T8 -H32 -N16 -F8 -t4 -x none"
```

Run every case (valid + all tampers) with the driver script:

```sh
python3 ../tests/regression/attest_test/run_tests.py
# or a subset:
python3 ../tests/regression/attest_test/run_tests.py --cases none,kernel,weight1
```

### Host args (`--args`)

| Flag | Meaning | Default |
|------|---------|---------|
| `-T` | tokens (M, GEMM rows / checker batch)     | 8 |
| `-H` | hidden size (K, GEMM inner / checker hidden) | 32 |
| `-N` | output width (N, GEMM cols)               | 16 |
| `-F` | SAE feature count                          | 8 |
| `-t` | GEMM tile size (M,H,N must be multiples)   | 4 |
| `-x` | tamper mode (see below)                    | none |

## Tamper modes (`-x`)

Every tamper is expected to **block** the GEMM; the test still prints `PASSED!`
because it verifies `C` stayed at the sentinel. There are two kinds:

**Manifest-field tampers** — flip a byte in the manifest. The Phase-0 signature
covers the whole manifest body, so these are caught by the signature check (magic
is caught even earlier at the header). They validate the outer seal.

| `-x` | caught at | status word |
|------|-----------|-------------|
| `none`   | — (valid) | `0x50415353` (PASS), `gemm_ran=1` |
| `magic`  | header    | `0x46410002` |
| `sig`, `kernel`, `args`, `sae`, `thresh`, `weight` | signature | `0x46410001` |

**Data tampers** — corrupt the actual device data *after* the manifest is signed,
so the signature stays valid and the per-region hash catches it (with per-layer
localization). These validate the region hashing.

| `-x` | corrupts | caught at | status word |
|------|----------|-----------|-------------|
| `kerneldata` | kernel code in VRAM   | kernel region  | `0x46410003` FAIL(kernel) |
| `argsdata`   | kernel args in VRAM   | args region    | `0x46410004` FAIL(args) |
| `weight0`    | matrix B, first half  | weight layer 0 | `0x46410007` FAIL(weight, layer 0) |
| `weight1`    | matrix B, second half | weight layer 1 | `0x46410107` FAIL(weight, layer 1) |

## Reading the output

- `status=0x50415353  gemm_ran=1  (expected PASS/run)` then `PASSED!` — valid case.
- `status=0x4641LLCC  gemm_ran=0  (expected FAIL/blocked)` then `PASSED!` — tamper
  correctly blocked the GEMM. `CC` = fail code, `LL` = offending layer index.
- On a FAIL the status word read back is often `0xbaadf00d`: the verifier *did*
  write the FAIL status, but with no GEMM to flush the write-back L2 it never
  reaches DRAM. This is expected — the test keys off `C`/`gemm_ran`, not the status
  word. On a PASS the GEMM's end-of-kernel flush makes the PASS status readable.
- With `--debug=3` the verifier prints a `[VERIFY]` trace of each region check
  (`check=<n> src=<n> base=0x… words=<n>`, `cmp_ok`, `STATUS`, `BOOT_RELEASE`).

## Notes / gotchas

- **Verification is slow** (~200k cycles, dominated by the ~7.4k-word kernel): the
  VRAM adapter reads one cache line per 32-bit word. Correct, just not bandwidth-
  optimal — see the future-work note in `hw/rtl/core/VX_boot_mem_adapter.sv`.
- **Constrained / emulated environments (e.g. Docker on Apple Silicon):** do *not*
  run the top-level `make` — it builds the OPAE/XRT FPGA sims you don't need and
  can OOM a small container. `blackbox.sh` builds only rtlsim. Cap Verilator jobs
  with `THREADS=2` (or `1`) and shrink the checker SRAM with
  `-DVX_CHECKER_MAX_FEATURES=32` to keep the build/sim light.

## Files

| File | Purpose |
|------|---------|
| `main.cpp`   | host: upload, build+sign manifest, stream it, launch, check `C` |
| `kernel.cpp` | tiled GEMM kernel (from `sgemm2`/`checker_test`) |
| `common.h`   | shared `kernel_arg_t` |
| `manifest.h` | shared signed-manifest layout (mirrors `hw/rtl/VX_attest_pkg.sv`) |
| `sha256.h`   | host SHA-256 reference (matches `hw/rtl/libs/VX_sha256.sv`) |
| `run_tests.py` | runs the valid case + all tamper cases via `blackbox.sh` |

RTL side: `hw/rtl/core/VX_boot_verifier.sv`, `VX_boot_mem_adapter.sv`,
`hw/rtl/VX_attest_pkg.sv`, `hw/rtl/libs/VX_sha256.sv`, and the `ATTEST_ENABLE`
block in `hw/rtl/VX_cluster.sv`. See CLAUDE.md → "Boot-Time Attestation" for the
full design.
