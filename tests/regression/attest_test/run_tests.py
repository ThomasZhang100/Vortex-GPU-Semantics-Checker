#!/usr/bin/env python3
"""
Boot-attestation end-to-end test driver (Task C).

Runs the attest_test GEMM through blackbox.sh once per case:

  * valid manifest              -> attestation PASS  -> GEMM runs  -> "PASSED!"
  * each tampered manifest field -> attestation FAIL -> GEMM blocked -> "PASSED!"
    (main.cpp expects a FAIL and verifies C stayed at its sentinel, so "PASSED!"
     means the attestation correctly blocked boot)

blackbox.sh is invoked from the BUILD directory (out-of-tree build) with the
CHECKER_ENABLE + ATTEST_ENABLE config, which it forwards to both the RTL driver
and the app build.  Source the toolchain first:

    cd build && source ci/toolchain_env.sh
    python3 ../tests/regression/attest_test/run_tests.py
    python3 ../tests/regression/attest_test/run_tests.py --cases none,kernel,weight1

Options:
    --build DIR    build directory (default: <repo>/build)
    --driver NAME  vortex driver (default: rtlsim)
    --configs STR  CONFIGS passed to blackbox (default enables checker+attest,
                   shrinks the checker SRAM so it fits constrained/emulated hosts)
    --threads N    Verilator build parallelism (default 2; use 1 on low-RAM hosts)
    --cases LIST   comma-separated subset of the cases below
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIGS = "-DCHECKER_ENABLE -DATTEST_ENABLE -DVX_CHECKER_MAX_FEATURES=32"
BASE_ARGS = "-T8 -H32 -N16 -F8 -t4"

# Each case must print "PASSED!": "none" runs the GEMM; every tamper is expected
# to block it (main.cpp verifies C stayed at its sentinel).
#  - manifest-field tampers (caught by the signature / magic outer seal)
#  - data tampers (manifest valid; caught by the per-region hash; weightN localizes layer)
ALL_CASES = ["none",
             "sig", "magic", "kernel", "args", "sae", "thresh", "weight",
             "weight0", "weight1", "argsdata", "kerneldata"]


def run_case(case, build_dir, driver, configs, threads):
    blackbox = build_dir / "ci" / "blackbox.sh"
    cmd = [
        str(blackbox),
        f"--driver={driver}",
        "--app=attest_test",
        f"--args={BASE_ARGS} -x {case}",   # one argument -> no shell word-splitting
    ]
    env = os.environ.copy()
    env["CONFIGS"] = configs
    env["THREADS"] = str(threads)
    print(f"\n=== case: {case} ===")
    print("CONFIGS=%s THREADS=%s %s" % (configs, threads, " ".join(cmd)))
    result = subprocess.run(
        cmd, cwd=str(build_dir), env=env,
        capture_output=True, text=True, timeout=3600,
    )
    out = result.stdout + result.stderr
    for line in out.splitlines():
        if any(t in line for t in ("status=", "gemm_ran", "PASSED!", "FAILED!",
                                   "Error", "Killed", "Segmentation")):
            print("   " + line)
    ok = ("PASSED!" in out) and (result.returncode == 0)
    print(f"   -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(REPO_ROOT / "build"))
    ap.add_argument("--driver", default="rtlsim")
    ap.add_argument("--configs", default=DEFAULT_CONFIGS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cases", default=",".join(ALL_CASES),
                    help="comma-separated subset of: " + ",".join(ALL_CASES))
    a = ap.parse_args()

    build_dir = Path(a.build).resolve()
    if not (build_dir / "ci" / "blackbox.sh").exists():
        sys.exit(f"error: {build_dir}/ci/blackbox.sh not found — run from the build "
                 f"directory or pass --build <dir>")

    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    results = {c: run_case(c, build_dir, a.driver, a.configs, a.threads) for c in cases}

    print("\n===== summary =====")
    for c, ok in results.items():
        print(f"  {c:10} {'OK' if ok else 'FAIL'}")
    n_fail = sum(1 for ok in results.values() if not ok)
    print(f"{len(results) - n_fail}/{len(results)} cases passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
