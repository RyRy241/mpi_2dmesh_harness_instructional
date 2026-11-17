#!/usr/bin/env python3
"""
parse_timings.py -- turn the slurm-XXXX.out log(s) written by run_script.sh into a
timing table and three speedup charts (one per decomposition), using P=4 as baseline.

Usage (from the build directory, after the batch job finishes):
    python3 ../scripts/parse_timings.py slurm-XXXX.out [more slurm files...]

If several log files (repeated runs) are given, times for the same (P, decomp) are
averaged. Writes:
    timings.csv                 P, decomp, scatter/sobel/gather ms, and speedups
    speedup_row-slab.png        speedup of scatter, sobel, gather vs P (+ ideal line)
    speedup_column-slab.png
    speedup_tiled.png
"""
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DECOMPS = {1: "row-slab", 2: "column-slab", 3: "tiled"}
PHASES = ["scatter", "sobel", "gather"]
LABELS = {"scatter": "Scatter", "sobel": "Process (Sobel)", "gather": "Gather"}
BASE_P = 4

cmd_re = re.compile(r"srun\s+-n\s+(\d+)\s+.*-g\s+(\d)")
time_re = re.compile(r"(Scatter|Sobel|Gather) time:\s*([0-9.]+)")

samples = defaultdict(lambda: defaultdict(list))   # (P, decomp) -> phase -> [ms]
for path in sys.argv[1:]:
    key = None
    for line in open(path, errors="replace"):
        m = cmd_re.search(line)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            continue
        m = time_re.search(line)
        if m and key is not None:
            samples[key][m.group(1).lower()].append(float(m.group(2)))

if not samples:
    sys.exit("No timing results found; pass the slurm-XXXX.out file(s) from run_script.sh")

mean = {k: {ph: sum(v[ph]) / len(v[ph]) for ph in PHASES if v[ph]} for k, v in samples.items()}

with open("timings.csv", "w") as f:
    f.write("P,decomp,scatter_ms,sobel_ms,gather_ms,scatter_speedup,sobel_speedup,gather_speedup,runs\n")
    for (P, d) in sorted(mean):
        t = mean[(P, d)]
        base = mean.get((BASE_P, d), {})
        sp = [base[ph] / t[ph] if ph in base and ph in t and t[ph] > 0 else float("nan")
              for ph in PHASES]
        f.write(f"{P},{DECOMPS[d]},{t.get('scatter',''):.4f},{t.get('sobel',''):.4f},"
                f"{t.get('gather',''):.4f},{sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f},"
                f"{len(samples[(P,d)]['sobel'])}\n")

markers = {"scatter": "o", "sobel": "s", "gather": "^"}
for d, name in DECOMPS.items():
    Ps = sorted(P for (P, dd) in mean if dd == d)
    if (BASE_P, d) not in mean or not Ps:
        print(f"skipping {name}: no P={BASE_P} baseline in the logs")
        continue
    base = mean[(BASE_P, d)]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(Ps, [P / BASE_P for P in Ps], "k--", lw=1, label="Ideal (P/4)")
    for ph in PHASES:
        ax.plot(Ps, [base[ph] / mean[(P, d)][ph] for P in Ps],
                marker=markers[ph], label=LABELS[ph])
    ax.set_xticks(Ps)
    ax.set_xlabel("Concurrency (number of MPI ranks, 4 CPU nodes)")
    ax.set_ylabel(f"Speedup relative to P={BASE_P}")
    ax.set_title(f"MPI Sobel speedup by phase, {name} decomposition")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"speedup_{name}.png", dpi=200)
    plt.close(fig)

print("Mean times (ms):")
print(f"{'P':>3} {'decomp':>12} {'scatter':>10} {'sobel':>10} {'gather':>10}")
for (P, d) in sorted(mean, key=lambda k: (k[1], k[0])):
    t = mean[(P, d)]
    print(f"{P:>3} {DECOMPS[d]:>12} {t['scatter']:>10.2f} {t['sobel']:>10.2f} {t['gather']:>10.2f}")
print("\nWrote timings.csv and speedup_<decomp>.png")
