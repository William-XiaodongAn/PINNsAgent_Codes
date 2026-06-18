"""
Batch-convert cardiac-agent heat2d CSV instances -> pinnacle `.dat` files, one set per instance.

For each instance i in <src>, reads
    <src>/IC_i.csv         (t=0 frame)
    <src>/solution_i.csv   (t=T_END frame)
and writes (suffixed by the instance id):
    ref/heat2d_cardiac_init_u_<i>.dat   cols: x y u          (training IC)
    ref/heat2d_cardiac_<i>.dat          cols: x y t u        (reference solution, t=T_END)

CSV layout: single row of  2 + 128*128*4  values; first two are grid dims (128,128),
the rest is a flattened (128,128,4) RGBA array whose channel 0 is u and 1..3 are zero.

Usage (run from the pinnacle/ dir, or anywhere):
    python scripts/gen_heat2d_cardiac_batch.py --src /path/to/heat/generated_data --instances 0-49
    python scripts/gen_heat2d_cardiac_batch.py --src /path/to/heat/generated_data --instances 0,1,2
    python scripts/gen_heat2d_cardiac_batch.py --src /path/to/heat/generated_data   # all present
    python scripts/gen_heat2d_cardiac_batch.py                                       # default folder, all present
"""
import os
import re
import glob
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..",
                                           "cardiac-agent", "data", "heat", "generated_data"))
OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ref"))

N = 128              # grid points per axis
L = 1.0              # domain length
ALPHA = 0.01         # diffusivity
INTERNAL_STEPS = 30000
DX = L / N
DT = 0.25 * DX * DX / ALPHA
T_END = INTERNAL_STEPS * DT   # = 45.7763671875


def parse_instances(spec):
    ids = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            lo, hi = tok.split("-")
            ids += list(range(int(lo), int(hi) + 1))
        elif tok:
            ids.append(int(tok))
    return ids


def discover_instances(folder):
    """Return sorted instance ids that have BOTH an IC_*.csv and a solution_*.csv."""
    def ids_from(prefix):
        out = set()
        for path in glob.glob(os.path.join(folder, f"{prefix}_*.csv")):
            m = re.match(rf"{prefix}_(\d+)\.csv$", os.path.basename(path))
            if m:
                out.add(int(m.group(1)))
        return out

    ic_ids = ids_from("IC")
    sol_ids = ids_from("solution")
    return sorted(ic_ids & sol_ids)


def load_frame(csv_path):
    flat = np.loadtxt(csv_path, delimiter=",")
    gx, gy = int(flat[0]), int(flat[1])
    assert (gx, gy) == (N, N), f"unexpected grid dims {(gx, gy)} in {csv_path}"
    arr = flat[2:].reshape(N, N, 4)
    return arr[:, :, 0]


def grid_coords():
    x = np.arange(N) * DX
    X, Y = np.meshgrid(x, x, indexing="ij")
    return X, Y


def convert_one(src, i):
    ic_csv = os.path.join(src, f"IC_{i}.csv")
    sol_csv = os.path.join(src, f"solution_{i}.csv")
    X, Y = grid_coords()

    u0 = load_frame(ic_csv)
    out = np.column_stack([X.ravel(), Y.ravel(), u0.ravel()])
    path = os.path.join(OUT_DIR, f"heat2d_cardiac_init_u_{i}.dat")
    np.savetxt(path, out, fmt="%.8e")

    uT = load_frame(sol_csv)
    t_col = np.full(X.size, T_END)
    ref = np.column_stack([X.ravel(), Y.ravel(), t_col, uT.ravel()])
    path = os.path.join(OUT_DIR, f"heat2d_cardiac_{i}.dat")
    np.savetxt(path, ref, fmt="%.8e")
    print(f"instance {i}: wrote init_u_{i}.dat + heat2d_cardiac_{i}.dat", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="folder containing IC_*.csv / solution_*.csv (e.g. .../heat/generated_data)")
    ap.add_argument("--instances", default=None,
                    help="e.g. 0-49 or 0,1,2; omit to convert all present in the folder")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.instances:
        ids = parse_instances(args.instances)
    else:
        ids = discover_instances(args.src)
        if not ids:
            raise SystemExit(f"no IC_*/solution_* csv pairs found in {args.src}")
    print(f"src={args.src}\ninstances={ids}\nout={OUT_DIR}\n")
    for i in ids:
        convert_one(args.src, i)
    print(f"\ndone: {len(ids)} instances")


if __name__ == "__main__":
    main()
