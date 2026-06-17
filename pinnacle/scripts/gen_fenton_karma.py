"""
One-off converter: cardiac-agent Fenton-Karma CSVs  ->  pinnacle `.dat` reference files.

Each source CSV (IC_0.csv / solution_0.csv) is a SINGLE row of
    2 + 512*512*4  values
where the first two values are the grid dims (512, 512) and the rest is a
flattened (512, 512, 4) array whose channels are (u, v, w, fake_texture_value).
The 4th channel is a texture alpha and is dropped.

IC_0.csv       -> initial condition snapshot at t = 0
solution_0.csv -> reference solution snapshot at t = T_END (=100.0)

Outputs (written into pinnacle/ref/):
    fenton_karma_init_u.dat   columns: x y u      (t=0, for training IC)
    fenton_karma_init_v.dat   columns: x y v      (t=0, for training IC)
    fenton_karma_init_w.dat   columns: x y w      (t=0, for training IC)
    fenton_karma.dat          columns: x y t u v w (t=T_END, reference solution)

Spatial domain is [0, L]^2 with L=10, N=512 grid points per axis, dx = L/N.
Grid coords use x_i = i*dx  (i = 0..N-1)  -> spans [0, L-dx], inside the [0,L] bbox.
"""
import os
import numpy as np

# ----- config -----------------------------------------------------------------
SRC_DIR = r"C:/Users/xan37/Documents/GitHub/cardiac-agent/data/fenton_karma/train"
IC_CSV = os.path.join(SRC_DIR, "IC_0.csv")
SOL_CSV = os.path.join(SRC_DIR, "solution_0.csv")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "ref")
OUT_DIR = os.path.abspath(OUT_DIR)

N = 512          # grid points per axis
L = 10.0         # domain length
T_END = 100.0    # solution_*.csv is the final-time snapshot
STRIDE = 1       # keep full 512x512 resolution
# ------------------------------------------------------------------------------


def load_frame(csv_path):
    """Read one FK csv -> (u, v, w) each shaped (N, N)."""
    flat = np.loadtxt(csv_path, delimiter=",")
    gx, gy = int(flat[0]), int(flat[1])
    assert (gx, gy) == (N, N), f"unexpected grid dims {(gx, gy)} in {csv_path}"
    arr = flat[2:].reshape(N, N, 4)          # channels: u, v, w, fake
    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]


def grid_coords():
    dx = L / N
    x = np.arange(N) * dx                      # [0, L-dx]
    X, Y = np.meshgrid(x, x, indexing="ij")    # X[i,j]=x_i, Y[i,j]=y_j
    return X, Y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, Y = grid_coords()
    s = slice(None, None, STRIDE)
    Xc, Yc = X[s, s], Y[s, s]

    # ---- initial condition (t=0) -> one file per component -------------------
    u0, v0, w0 = load_frame(IC_CSV)
    for name, field in (("u", u0), ("v", v0), ("w", w0)):
        out = np.column_stack([Xc.ravel(), Yc.ravel(), field[s, s].ravel()])
        path = os.path.join(OUT_DIR, f"fenton_karma_init_{name}.dat")
        np.savetxt(path, out, fmt="%.8e")
        print(f"wrote {path}  shape={out.shape}")

    # ---- reference solution (t=T_END) -> single x y t u v w long table -------
    uT, vT, wT = load_frame(SOL_CSV)
    t_col = np.full(Xc.size, T_END)
    ref = np.column_stack([
        Xc.ravel(), Yc.ravel(), t_col,
        uT[s, s].ravel(), vT[s, s].ravel(), wT[s, s].ravel(),
    ])
    path = os.path.join(OUT_DIR, "fenton_karma.dat")
    np.savetxt(path, ref, fmt="%.8e")
    print(f"wrote {path}  shape={ref.shape}")


if __name__ == "__main__":
    main()
