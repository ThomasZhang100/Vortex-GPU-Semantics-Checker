#!/bin/bash
# Sweep the checker's systolic-array geometry (B_TILE x N_FEAT) and record the
# estimated transistor count of the *logic* at each point (all RAM primitives
# black-boxed, so the compiled SRAM macro is excluded from the gate count).
#
# Two synth passes per geometry (the 2nd reuses project.v, so it's just seconds):
#   pass 1: black-box RAM only            -> total logic transistors
#   pass 2: black-box RAM + sa_array      -> control/glue "base" (the whole PE
#           array subtree is dropped; blackboxes are conservative sinks, so the
#           hpipe/FIFO plumbing that FEEDS the array is preserved, not DCE'd)
#   array = total - base                  -> the PE-array subtree alone
#
# One row per geometry is appended to checker_area_sweep.csv:
#     pes,geometry,total,array,base
#
# Run this from the *build tree* hw/syn/yosys directory (where `make` finds
# config.mk), with the toolchain env sourced:
#     $VORTEX_HOME/hw/syn/yosys/sweep_checker_area.sh
#
# Constraints on each (B N) pair: VX_CHECKER_MAX_FEATURES % N == 0  and
#                                 MAX_BATCH(=16)          % B == 0.
set -e

# (B_TILE N_FEAT) points to sweep.
GRID=(
  "4 4"
  "4 8"
  "4 16"
  "4 32"
  "8 16"
  "8 32"
)

RAMS="VX_dp_ram VX_sp_ram VX_async_ram_patch"
TOP=VX_checker_synth_top
BUILD=build_${TOP}
OUT=checker_area_sweep.csv

# `stat -tech cmos` prints:  "Estimated number of transistors:   <count>"
transistors() { grep -i "number of transistors" "$BUILD/yosys.log" | tail -1 | grep -oE '[0-9]+' | tail -1; }

echo "pes,geometry,total,array,base" > "$OUT"

for pair in "${GRID[@]}"; do
  read -r B N <<< "$pair"
  pes=$((B * N))
  cfg="-DCHECKER_ENABLE -DCHECKER_SYNTH -DVX_CHECKER_B_TILE=$B -DVX_CHECKER_N_FEAT=$N"
  echo "==================================================================="
  echo "=== B_TILE=$B  N_FEAT=$N   (${pes} PEs) ==="
  echo "==================================================================="

  # CONFIGS changed => project.v must be regenerated, so wipe the build dir.
  # NOTE: clean must get the same TOP_LEVEL_ENTITY, else BUILD_DIR defaults to
  # build_Vortex and the stale build_$TOP/project.v is silently reused.
  make clean TOP_LEVEL_ENTITY=$TOP >/dev/null 2>&1 || true

  # Pass 1: whole checker logic (RAM black-boxed) -> total
  # ALLOW_WARN=1: the flattened synth wrapper emits a few benign warnings
  # (unused request-payload ports, width tweaks); don't let them abort the sweep.
  make synthesis TOP_LEVEL_ENTITY=$TOP CONFIGS="$cfg" BLACKBOX="$RAMS" ALLOW_WARN=1
  total=$(transistors)

  # Pass 2: also black-box the PE array -> control/glue base (reuses project.v)
  make synthesis TOP_LEVEL_ENTITY=$TOP CONFIGS="$cfg" BLACKBOX="$RAMS sa_array" ALLOW_WARN=1
  base=$(transistors)

  array=$((total - base))
  echo "$pes,${B}x${N},$total,$array,$base" >> "$OUT"
  echo ">>> ${pes} PEs  ->  total=${total}  array=${array}  base=${base}"
done

echo
echo "Wrote $OUT:"
cat "$OUT"
