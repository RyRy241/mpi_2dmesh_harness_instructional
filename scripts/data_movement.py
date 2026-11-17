#!/usr/bin/env python3
"""
data_movement.py -- compute the number of MPI messages and the amount of data moved
between ranks by mpi_2dmesh, for every concurrency level and decomposition strategy.

The counts follow directly from the code:
  * computeMeshDecomposition() builds exactly one tile per rank; rank 0 owns the tile
    at the origin (0,0), and the last tile along each axis absorbs any remainder.
    Each tile gets HALO ghost cells on every edge it shares with another tile, and
    none on edges that lie on the boundary of the global mesh.
  * scatterAllTiles(): rank 0 calls sendStridedBuffer() once for every tile it does NOT
    own, sending the tile's base grid PLUS its halo cells; its own tile is a local
    memcpy (no message).
  * gatherAllTiles(): every other rank calls sendStridedBuffer() once to return its
    tile's base grid only (no halo cells).
  * sendStridedBuffer() issues exactly ONE MPI_Send (an MPI_Type_vector describing the
    whole strided subregion), carrying width*height floats.
So per phase there are ntiles - 1 messages.

Usage:
    python3 data_movement.py [--halo N] [width height]
        --halo N   halo width used by the code (default 1 = HALO_WIDTH in mpi_2dmesh.hpp;
                   use --halo 0 for the base version without halo cells)
        width height  image size (default 7112 5146, the 4x zebra image)
Writes data_movement.csv and data_movement_table.tex into the current directory.
"""
import sys

args = sys.argv[1:]
HALO = 1
if "--halo" in args:
    i = args.index("--halo")
    HALO = int(args[i + 1])
    del args[i:i + 2]
W, H = (int(args[0]), int(args[1])) if len(args) == 2 else (7112, 5146)

CONCURRENCY = [4, 9, 16, 25, 36, 49, 64, 81]
DECOMPS = {1: "row-slab", 2: "column-slab", 3: "tiled"}
BYTES_PER_VALUE = 4  # float


def bounds(n, extent):
    """(start, size) of each tile along one axis, same as computeMeshDecomposition()."""
    size = extent // n
    starts = [i * size for i in range(n)]
    sizes = [size] * (n - 1) + [extent - (n - 1) * size]
    return list(zip(starts, sizes))


def tiles(P, decomp):
    """List of (x, y, width, height) per tile, in rank order."""
    if decomp == 1:
        return [(0, y, W, h) for y, h in bounds(P, H)]
    if decomp == 2:
        return [(x, 0, w, H) for x, w in bounds(P, W)]
    r = int(P ** 0.5)
    return [(x, y, w, h) for y, h in bounds(r, H) for x, w in bounds(r, W)]


def padded_area(x, y, w, h):
    """Area of a tile's base grid plus its halo cells (halos only on interior edges)."""
    gxmin = HALO if x > 0 else 0
    gxmax = HALO if x + w < W else 0
    gymin = HALO if y > 0 else 0
    gymax = HALO if y + h < H else 0
    return (w + gxmin + gxmax) * (h + gymin + gymax)


def movement(P, decomp):
    """(messages, total bytes, halo bytes) for scatter + gather."""
    t = tiles(P, decomp)[1:]                         # rank 0's tile is a local copy
    scatter_values = sum(padded_area(*tt) for tt in t)
    gather_values = sum(w * h for _, _, w, h in t)
    msgs = 2 * len(t)
    total = (scatter_values + gather_values) * BYTES_PER_VALUE
    halo = (scatter_values - gather_values) * BYTES_PER_VALUE
    return msgs, total, halo


rows = []
for P in CONCURRENCY:
    row = [P]
    for d in DECOMPS:
        row += list(movement(P, d))
    rows.append(row)

# CSV (exact byte counts)
with open("data_movement.csv", "w") as f:
    f.write("P," + ",".join(f"{n}_msgs,{n}_bytes,{n}_halo_bytes" for n in DECOMPS.values()) + "\n")
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
            cells += [str(r[1 + 3 * i]), f"{r[2 + 3 * i] / 1e6:.1f}"]
        f.write(" & ".join(cells) + " \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")

# Console summary
print(f"Image {W} x {H} floats = {W*H*BYTES_PER_VALUE/1e6:.1f} MB; halo width = {HALO}; "
      f"scatter+gather totals (MB = 1e6 bytes)\n")
print(f"{'P':>3} | {'row #msg':>8} {'row MB':>7} | {'col #msg':>8} {'col MB':>7} | "
      f"{'tile #msg':>9} {'tile MB':>7}")
for r in rows:
    print(f"{r[0]:>3} | {r[1]:>8} {r[2]/1e6:>7.1f} | {r[4]:>8} {r[5]/1e6:>7.1f} | "
          f"{r[7]:>9} {r[8]/1e6:>7.1f}")
if HALO:
    print(f"\nHalo cells included in the above (scatter phase only), in MB:")
    print(f"{'P':>3} | {'row':>7} | {'col':>7} | {'tiled':>7}")
    for r in rows:
        print(f"{r[0]:>3} | {r[3]/1e6:>7.3f} | {r[6]/1e6:>7.3f} | {r[9]/1e6:>7.3f}")
print("\nWrote data_movement.csv and data_movement_table.tex")
