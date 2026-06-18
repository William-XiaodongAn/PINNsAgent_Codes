"""
One-off converter: cardiac-agent heat2d CSVs  ->  pinnacle `.dat` reference files.

Each source CSV (IC_0.csv / solution_0.csv) is a SINGLE row of
    2 + 128*128*4  values
where the first two values are the grid dims (128, 128) and the rest is a
flattened (128, 128, 4) RGBA array whose channel 0 is the scalar field u and
channels 1..3 are zero padding (write_scalar_field_csv in generate_heat_data.py).
Only channel 0 is kept.

IC_0.csv       -> initial condition snapshot at t = 0
solution_0.csv -> reference solution snapshot at t = T_END

Outputs (written into pinnacle/ref/):
    heat2d_cardiac_init_u.dat   columns: x y u      (t=0, for training IC)
    heat2d_cardiac.dat          columns: x y t u    (t=T_END, reference solution)

Spatial domain is [0, L]^2 with L=1, N=128 grid points per axis, dx = L/N.
Grid coords use x_i = i*dx  (i = 0..N-1)  -> spans [0, L-dx], inside the [0,L] bbox
(matches np.linspace(0, L, N, endpoint=False) used by the generator).

T_END = internal_steps * dt, with dt = 0.25 * dx^2 / alpha  (heat-equation CFL).
"""
import os
import numpy as np

# ----- config -----------------------------------------------------------------
SRC_DIR = r"C:/Users/xan37/Documents/GitHub/cardiac-agent/data/heat/generated_data"
IC_CSV = os.path.join(SRC_DIR, "IC_0.csv")
SOL_CSV = os.path.join(SRC_DIR, "solution_0.csv")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "ref")
OUT_DIR = os.path.abspath(OUT_DIR)

N = 128              # grid points per axis
L = 1.0              # domain length
ALPHA = 0.01         # diffusivity
INTERNAL_STEPS = 30000
DX = L / N
DT = 0.25 * DX * DX / ALPHA
T_END = INTERNAL_STEPS * DT   # = 45.7763671875
STRIDE = 1           # keep full 128x128 resolution
# ------------------------------------------------------------------------------


def load_frame(csv_path):
    """Read one heat csv -> u shaped (N, N)."""
    flat = np.loadtxt(csv_path, delimiter=",")
    gx, gy = int(flat[0]), int(flat[1])
    assert (gx, gy) == (N, N), f"unexpected grid dims {(gx, gy)} in {csv_path}"
    arr = flat[2:].reshape(N, N, 4)          # channels: u, 0, 0, 0
    return arr[:, :, 0]


def grid_coords():
    x = np.arange(N) * DX                       # [0, L-dx]
    X, Y = np.meshgrid(x, x, indexing="ij")     # X[i,j]=x_i, Y[i,j]=y_j
    return X, Y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, Y = grid_coords()
    s = slice(None, None, STRIDE)
    Xc, Yc = X[s, s], Y[s, s]

    # ---- initial condition (t=0) --------------------------------------------
    u0 = load_frame(IC_CSV)
    out = np.column_stack([Xc.ravel(), Yc.ravel(), u0[s, s].ravel()])
    path = os.path.join(OUT_DIR, "heat2d_cardiac_init_u.dat")
    np.savetxt(path, out, fmt="%.8e")
    print(f"wrote {path}  shape={out.shape}")

    # ---- reference solution (t=T_END) -> x y t u long table -----------------
    uT = load_frame(SOL_CSV)
    t_col = np.full(Xc.size, T_END)
    ref = np.column_stack([Xc.ravel(), Yc.ravel(), t_col, uT[s, s].ravel()])
    path = os.path.join(OUT_DIR, "heat2d_cardiac.dat")
    np.savetxt(path, ref, fmt="%.8e")
    print(f"wrote {path}  shape={ref.shape}  T_END={T_END}")


if __name__ == "__main__":
    main()
