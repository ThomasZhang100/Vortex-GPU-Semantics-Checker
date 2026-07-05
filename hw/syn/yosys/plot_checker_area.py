#!/usr/bin/env python3
"""Stacked bar: systolic-array size (PEs) vs estimated logic transistor count,
split into the PE array subtree and the control/glue base.

Reads checker_area_sweep.csv (from sweep_checker_area.sh) and writes
checker_area_sweep.png.

Usage:
    python3 plot_checker_area.py [checker_area_sweep.csv]
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical pair (CVD-safe; see dataviz validator).
C_ARRAY = "#4269d0"   # PE array (sa_array subtree)
C_BASE  = "#e6772e"   # control / glue base
INK     = "#1b1b1f"
MUTED   = "#6b6b74"

csv_path = sys.argv[1] if len(sys.argv) > 1 else "checker_area_sweep.csv"

rows = []
with open(csv_path) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["pes"]), r["geometry"],
                     int(r["total"]), int(r["array"]), int(r["base"])))
rows.sort(key=lambda x: x[0])

labels = [f"{g}\n({p} PE)" for p, g, *_ in rows]
total  = [r[2] for r in rows]
array  = [r[3] for r in rows]
base   = [r[4] for r in rows]

x = range(len(rows))
fig, ax = plt.subplots(figsize=(8.5, 5.5))

# Base on the bottom, array stacked on top; 2px surface gap between segments.
b1 = ax.bar(x, base, width=0.65, color=C_BASE, zorder=3, label="Control / glue base")
b2 = ax.bar(x, array, width=0.65, bottom=base, color=C_ARRAY, zorder=3,
            label="PE array (sa_array)", linewidth=2, edgecolor="#fcfcfb")

ax.set_xticks(list(x))
ax.set_xticklabels(labels, color=INK)
ax.set_ylabel("Estimated transistors  (logic; SRAM black-boxed)", color=INK)
ax.set_xlabel("Systolic-array geometry  (B_TILE × N_FEAT)", color=INK)
ax.set_title("VX_checker logic size vs systolic-array size", color=INK, pad=12)

# Direct labels: segment values inside, total above the stack.
for i in x:
    if base[i] > 0.06 * max(total):
        ax.text(i, base[i] / 2, f"{base[i]:,}", ha="center", va="center",
                fontsize=8, color="#fcfcfb")
    if array[i] > 0.06 * max(total):
        ax.text(i, base[i] + array[i] / 2, f"{array[i]:,}", ha="center",
                va="center", fontsize=8, color="#fcfcfb")
    ax.text(i, total[i], f"{total[i]:,}", ha="center", va="bottom",
            fontsize=9, color=MUTED)

ax.yaxis.grid(True, color="#e5e5ea", zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.margins(y=0.14)
ax.legend(frameon=False, loc="upper left")

fig.tight_layout()
out = "checker_area_sweep.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
