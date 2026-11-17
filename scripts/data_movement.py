#!/usr/bin/env python3
"""
data_movement.py -- compute the number of MPI messages and the amount of data moved
between ranks by mpi_2dmesh, for every concurrency level and decomposition strategy.

The counts follow directly from the code:
  * computeMeshDecomposition() builds exactly one tile per rank; rank 0 owns the tile
    at the origin (0,0), and the last tile along each axis absorbs any remainder.
  * scatterAllTiles(): rank 0 calls sendStridedBuffer() once for every tile it does NOT
    own; its own tile is a local memcpy (no message).
  * gatherAllTiles(): every other rank calls sendStridedBuffer() once to return its tile.
  * sendStridedBuffer() issues exactly ONE MPI_Send (an MPI_Type_vector describing the
    whole strided subregion), carrying width*height floats.
So per phase: messages = ntiles - 1, bytes = 4 * (W*H - area of rank 0's tile).

Usage:
    python3 data_movement.py [width height]      (default 7112 5146, the 4x zebra image)
Writes data_movement.csv and data_movement_table.tex into the current directory.
"""
import sys

W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (7112, 5146)
CONCURRENCY = [4, 9, 16, 25, 36, 49, 64, 81]
DECOMPS = {1: "row-slab", 2: "column-slab", 3: "tiled"}
BYTES_PER_VALUE = 4  # float


def bounds(n, extent):
    """Tile boundaries along one axis, same as computeMeshDecomposition()."""
    size = extent // n
    locs = [i * size for i in range(n)] + [extent]
    return [locs[i + 1] - locs[i] for i in range(n)]


def tiles(P, decomp):
    """Return a list of (width, height) per tile, in rank order."""
    if decomp == 1:
        return [(W, h) for h in bounds(P, H)]
    if decomp == 2:
        return [(w, H) for w in bounds(P, W)]
    r = int(P ** 0.5)
    return [(w, h) for h in bounds(r, H) for w in bounds(r, W)]


def movement(P, decomp):
    t = tiles(P, decomp)
    moved_values_per_phase = sum(w * h for w, h in t[1:])   # skip rank 0's tile
    msgs = 2 * (len(t) - 1)                                   # scatter + gather
    bytes_total = 2 * moved_values_per_phase * BYTES_PER_VALUE
    return msgs, bytes_total, t


rows = []
for P in CONCURRENCY:
    row = [P]
    for d in DECOMPS:
        msgs, b, _ = movement(P, d)
        row += [msgs, b]
    rows.append(row)

# CSV (exact byte counts)
with open("data_movement.csv", "w") as f:
    f.write("P," + ",".join(f"{n}_msgs,{n}_bytes" for n in DECOMPS.values()) + "\n")
    for r in rows:
        f.write(",".join(str(v) for v in r) + "\n")

# LaTeX table (data in MB = 10^6 bytes, to avoid large numbers / exponents)
with open("data_movement_table.tex", "w") as f:
    f.write("\\begin{tabular}{r|rr|rr|rr}\n\\hline\n")
    f.write(" & \\multicolumn{2}{c|}{Row-slab} & \\multicolumn{2}{c|}{Column-slab}"
            " & \\multicolumn{2}{c}{Tiled} \\\\\n")
    f.write("$P$ & \\#msgs & MB & \\#msgs & MB & \\#msgs & MB \\\\\n\\hline\n")
    for r in rows:
        cells = [str(r[0])]
        for i in range(3):
            cells += [str(r[1 + 2 * i]), f"{r[2 + 2 * i] / 1e6:.1f}"]
        f.write(" & ".join(cells) + " \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")

# Console summary
print(f"Image {W} x {H} floats = {W*H*BYTES_PER_VALUE/1e6:.1f} MB; "
      f"scatter+gather totals (MB = 1e6 bytes)\n")
print(f"{'P':>3} | {'row #msg':>8} {'row MB':>7} | {'col #msg':>8} {'col MB':>7} | "
      f"{'tile #msg':>9} {'tile MB':>7}")
for r in rows:
    print(f"{r[0]:>3} | {r[1]:>8} {r[2]/1e6:>7.1f} | {r[3]:>8} {r[4]/1e6:>7.1f} | "
          f"{r[5]:>9} {r[6]/1e6:>7.1f}")
print("\nWrote data_movement.csv and data_movement_table.tex")
