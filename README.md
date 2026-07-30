# Measured Models: Non-Bypassable Hardware Semantic Checking

A research fork of [Vortex](https://github.com/vortexgpgpu/vortex).
The base GPU, build system, toolchain, and runtime are all Vortex — credit goes to the Vortex authors (MICRO'21); their original README
is preserved verbatim at [README.upstream.md](README.upstream.md).

This page covers only the additions made in this fork: the non-bypassable
semantic checker (VX_checker), boot-time runtime image verification (VX_boot_verifier), 
the SHA-256 core, the manifest format, and the two regression tests. 

## Abstract
Runtime safety mechanisms based on model activations such as linear probes, sparse autoencoders, and activation steering can expose safety-relevant signals that are invisible to input/output filters. Their enforcement, however, is normally delegated to the same software stack that launches the model. That assumption is fragile for autonomous agents and malicious operators: the application can omit the checker, alter its inputs, ignore its verdict, or execute a different model entirely. We propose a GPU hardware root of trust for non-bypassable alignment monitoring. This repository demonstrates a Vortex-based prototype comprised of a upload-time verified hashing sequence and a sparse-autoencoder checker accelerator. 


This fork adds two main hardware mechanisms on top of the Vortex GPGPU:

1. **A sparse auto-encoder accelerator**: a systolic array that taps the
   model's hidden state directly off L2, runs a selected-feature SAE matmul + threshold in parallel with the main datapath, and raises a per-token flag. 
2. **Boot-time verified launch**: before the cores are
   allowed to execute, hardware independently hashes the model weights, kernel,
   kernel args, and the SAE weights/thresholds, and checks them
   against a signed manifest. Cores are held in reset until every hash matches, so
   a tampered image never runs.

---

## File Walkthrough

| File | Purpose |
|------|---------|
| hw/rtl/core/VX_checker.sv | Semantic checker: L2 snoop and output-stationary systolic sparse auto-encoder accelerator. |
| hw/rtl/core/VX_boot_verifier.sv | Boot verifier FSM: walks the signed manifest, hashes each region, compares, gates boot |
| hw/rtl/core/VX_boot_mem_adapter.sv | Adapter for the verifier's VRAM reads and status write |
| hw/rtl/libs/VX_sha256.sv | Streaming SHA-256 core, one 512-bit block / 64 cycles |
| hw/rtl/VX_attest_pkg.sv | Manifest layout package |
| hw/rtl/VX_cluster.sv | Checker control registers, L1 to L2 bus tap, Manifest SRAM, verifier/adapter instances, shared-L2-port mux, and boot gate |
| hw/rtl/VX_types.vh | VX_DCR_CHECKER_* and VX_DCR_ATTEST_* device-control registers |
| tests/regression/checker_test/ | Host + kernel test for the checker ([README](tests/regression/checker_test/README.md)) |
| tests/regression/attest_test/ | End-to-end execution image verification test: signed manifest verification, boot gate, and GEMM ([README](tests/regression/attest_test/README.md)) |

The RTL necessary for each test are behind compile-time flags (-DCHECKER_ENABLE,-DATTEST_ENABLE)
and are absent from the default build, so the upstream GPU is unchanged unless you
opt in.

---

## Architecture

### System overview

<img width="3183" height="2067" alt="Hashing Scheme Static drawio" src="https://github.com/user-attachments/assets/e37674c4-70b3-4ded-be6d-a57670b19222" />

The architecture of the verification system under both of the compile-time flags (-DCHECKER_ENABLE,-DATTEST_ENABLE) resembles the diagram above. On a kernel launch, the host loads the input data, model weights, checker weights, and a signed manifest all into their respective storage locations. Then, all of the data and metadata are read out by the verifier module and validated, ensuring that the subsequent execution is approved and recognized by the trusted signer. On global reset, core execution is blocked until verification is complete. Verification proceeds as described in the FSM below. 

### Boot verifier FSM 

On an arm edge the verifier walks the manifest header, then loops over every
region. First the manifest is hashed and verified against the signature. Next, the kernel code and 
arguments are hashed and verified against the manifest. Next, the SAE weights and thresholds are hashed and verified 
agaist the manifest. Finally, each per-layer weight region is hashed and verified. Each region is hashed through the single SHA-256 core and
compared against the golden hash stored in the manifest. The first mismatch aborts with a fail
code. The cores are allowed to start execution only on the
all-pass path, after all regions are correctly verified. Thus, it is structurally impossible to release the cores without every
check having passed, demonstrating the non-bypassability of our mechanism, in RTL.

<img width="735" height="1173" alt="static hash implementation fsm drawio" src="https://github.com/user-attachments/assets/8e97bb9a-5481-429e-b6f4-d5e873a72fbe" />


See [`hw/rtl/core/VX_boot_verifier.sv`](hw/rtl/core/VX_boot_verifier.sv)
for the full state machine.

### Semantic Sparse Autoencoder Accelerator

The checker is an output-stationary systolic array (default B_TILE × N_FEAT
= 4 × 16 PEs). Each PE holds one accumulator stationary across all K hidden
dimensions; activations flow rightward, SAE decoder weights flow downward, and the
indices always meet. After K + skew cycles the accumulators complete, go through
a per-feature threshold, and a token is flagged when its fired-feature count
exceeds a certain number. The SAE weights live in a private SRAM with no model read/write
path.

<img width="1892" height="1372" alt="SemanticChecker drawio (1)" src="https://github.com/user-attachments/assets/7800f1bd-0de5-454f-a9e8-60fc0f8ebadd" />


See [`hw/rtl/core/VX_checker.sv`](hw/rtl/core/VX_checker.sv) and the
[checker_test README](tests/regression/checker_test/README.md) for the dataflow
and skew details.

---

## Environment setup

From a fresh clone:

```sh
# 1. submodules (softfloat, ramulator, cvfpu, hardfloat)
git submodule update --init --recursive

# 2. system dependencies (Ubuntu; needs root/sudo)
./ci/install_dependencies.sh

# 3. create the build dir and configure
mkdir -p build && cd build
../configure --xlen=32 --tooldir=$HOME/tools

# 4. install the prebuilt toolchain: RISC-V GNU + LLVM + Verilator (large, one-time)
./ci/toolchain_install.sh --all
```

Then, in every new shell, source the toolchain environment:

```sh
cd build
source ./ci/toolchain_env.sh   # sets PATH/vars for RISC-V clang, Verilator, etc.
```

Build the two support libraries once (the rtlsim driver and the GPU kernels link
these):

```sh
# from build/, with toolchain_env sourced
make -C ../third_party    # softfloat + ramulator libs (rtlsim driver links these)
make -C kernel            # kernel/libvortex.a (GPU-side runtime the kernel links)
```

Python (for the test harnesses): Python == 3.8. checker_test's harness needs
NumPy; 

```sh
apt-get install -y python3-numpy   # or: pip3 install numpy
```

These dependencies are baked into Dockerfile.dev, so a freshly built dev image
already has them.

---

## Running the tests

Both tests run under the rtlsim (Verilator) driver, which simulates the actual
RTL. 

### checker_test 

Tests the correctness of the checker accelerator. 

```sh
# from build/, toolchain sourced
python3 ../tests/regression/checker_test/run_tests.py   # correctness sweep
```

Full reference:
[tests/regression/checker_test/README.md](tests/regression/checker_test/README.md).

### attest_test 

Tests the whole ingress verification chain on a GEMM kernel. 

```sh
# from build/, toolchain sourced
python3 ../tests/regression/attest_test/run_tests.py                 # valid + all tampers
python3 ../tests/regression/attest_test/run_tests.py --cases none    # valid case only (fastest first check)
```

Full reference:
[tests/regression/attest_test/README.md](tests/regression/attest_test/README.md).

