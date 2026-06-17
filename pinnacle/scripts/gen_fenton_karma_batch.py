"""
Batch-convert Fenton-Karma CSV instances -> pinnacle `.dat` files, one set per instance.

For each instance i in <split> (train/test), reads
    <src>/<split>/IC_i.csv         (t=0 frame)
    <src>/<split>/solution_i.csv   (t=T_END frame)
and writes (suffixed by the instance id):
    ref/fenton_karma_init_u_<i>.dat   cols: x y u   (training IC)
    ref/fenton_karma_init_v_<i>.dat   cols: x y v
    ref/fenton_karma_init_w_<i>.dat   cols: x y w
    ref/fenton_karma_<i>.dat          cols: x y t u v w   (reference solution, t=T_END)

CSV layout: single row of  2 + 512*512*4  values; first two are grid dims (512,512),
the rest is a flattened (512,512,4) array whose channels are (u,v,w,fake). 4th dropped.

Usage (run from the pinnacle/ dir, or anywhere):
    python scripts/gen_fenton_karma_batch.py --split test --instances 10-59
    python scripts/gen_fenton_karma_batch.py --split test --instances 10,11,12
"""
import os
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..",
                                           "cardiac-agent", "data", "fenton_karma"))
OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ref"))

N = 512          # grid points per axis
L = 10.0         # domain length
T_END = 100.0    # solution_*.csv is the final-time snapshot


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


def load_frame(csv_path):
    flat = np.loadtxt(csv_path, delimiter=",")
    gx, gy = int(flat[0]), int(flat[1])
    assert (gx, gy) == (N, N), f"unexpected grid dims {(gx, gy)} in {csv_path}"
    arr = flat[2:].reshape(N, N, 4)
    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]


def grid_coords():
    dx = L / N
    x = np.arange(N) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    return X, Y


def convert_one(src, split, i):
    ic_csv = os.path.join(src, split, f"IC_{i}.csv")
    sol_csv = os.path.join(src, split, f"solution_{i}.csv")
    X, Y = grid_coords()

    u0, v0, w0 = load_frame(ic_csv)
    for name, field in (("u", u0), ("v", v0), ("w", w0)):
        out = np.column_stack([X.ravel(), Y.ravel(), field.ravel()])
        path = os.path.join(OUT_DIR, f"fenton_karma_init_{name}_{i}.dat")
        np.savetxt(path, out, fmt="%.8e")

    uT, vT, wT = load_frame(sol_csv)
    t_col = np.full(X.size, T_END)
    ref = np.column_stack([X.ravel(), Y.ravel(), t_col,
                           uT.ravel(), vT.ravel(), wT.ravel()])
    path = os.path.join(OUT_DIR, f"fenton_karma_{i}.dat")
    np.savetxt(path, ref, fmt="%.8e")
    print(f"instance {i}: wrote init_u/v/w_{i}.dat + fenton_karma_{i}.dat", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="cardiac-agent fenton_karma data dir")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--instances", default="10-59", help="e.g. 10-59 or 10,11,12")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ids = parse_instances(args.instances)
    print(f"src={args.src}\nsplit={args.split}\ninstances={ids}\nout={OUT_DIR}\n")
    for i in ids:
        convert_one(args.src, args.split, i)
    print(f"\ndone: {len(ids)} instances")


if __name__ == "__main__":
    main()
