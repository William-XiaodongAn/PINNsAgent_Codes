import os
import numpy as np
import deepxde as dde
from scipy.interpolate import RegularGridInterpolator

from . import baseclass


class Heat2DCardiac(baseclass.BaseTimePDE):
    """2D linear heat equation (single field u) from the cardiac-agent benchmark.

        u_t = alpha * (u_xx + u_yy)

    Domain [0,1]x[0,1], periodic boundary conditions (the reference solver is an
    FFT spectral / periodic finite-difference update), alpha = 0.01.  The initial
    condition is a random smooth field supplied from data (single t=0 frame); the
    reference solution is a single t=T frame.  See scripts/gen_heat2d_cardiac.py.

    This is a NEW, distinct PDE -- it does not reuse PINNsAgent's existing Heat2D_*
    classes, whose coefficients, domains and BCs differ.
    """

    def __init__(
        self,
        datapath="ref/heat2d_cardiac.dat",
        icpath=("ref/heat2d_cardiac_init_u.dat",),
        bbox=[0, 1, 0, 1, 0, 45.7763671875],   # T_END = 30000 * dt, dt = 0.25*dx^2/alpha
        alpha=0.01,
        instance=None,
    ):
        super().__init__()

        # If an instance id is given, use the batch-generated indexed files
        # (ref/heat2d_cardiac_<i>.dat + ref/heat2d_cardiac_init_u_<i>.dat) produced
        # by scripts/gen_heat2d_cardiac_batch.py. Otherwise use the default pair above.
        if instance is not None:
            datapath = f"ref/heat2d_cardiac_{instance}.dat"
            icpath = (f"ref/heat2d_cardiac_init_u_{instance}.dat",)
            for p in (datapath,) + icpath:
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"Heat2DCardiac instance {instance}: missing file '{p}'. "
                        f"Run scripts/gen_heat2d_cardiac_batch.py to generate it first."
                    )

        # output dim: u
        self.output_dim = 1
        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        timedomain = dde.geometry.TimeDomain(bbox[4], bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # ---- PDE residual ---------------------------------------------------
        def pde(x, U):
            # time derivative (j=2 is t)
            u_t = dde.grad.jacobian(U, x, i=0, j=2)
            # diffusion (single output -> no `component`, unlike the 3-field FK model)
            u_xx = dde.grad.hessian(U, x, i=0, j=0)
            u_yy = dde.grad.hessian(U, x, i=1, j=1)
            return u_t - alpha * (u_xx + u_yy)

        self.pde = pde
        self.set_pdeloss(num=1)

        # reference solution (single t=T frame): columns x y t u
        self.load_ref_data(datapath, t_transpose=False)

        # ---- data-driven initial condition ----------------------------------
        self._ic_interp = [self._build_interp(p) for p in icpath]

        def ic_func(x, component):
            pts = x[:, 0:2]
            return self._ic_interp[component](pts).reshape(-1, 1)

        def boundary_ic(x, on_initial):
            return on_initial and np.isclose(x[2], bbox[4])

        # periodic BC on the x-walls and y-walls (single field u)
        def boundary_xb(x, on_boundary):
            return on_boundary and (np.isclose(x[0], bbox[0]) or np.isclose(x[0], bbox[1]))

        def boundary_yb(x, on_boundary):
            return on_boundary and (np.isclose(x[1], bbox[2]) or np.isclose(x[1], bbox[3]))

        self.add_bcs([
            {'component': 0, 'function': (lambda x: ic_func(x, 0)), 'bc': boundary_ic, 'type': 'ic'},
            {'component': 0, 'type': 'periodic', 'component_x': 0, 'bc': boundary_xb},
            {'component': 0, 'type': 'periodic', 'component_x': 1, 'bc': boundary_yb},
        ])

        # Training Config
        self.training_points(mul=4)

    @staticmethod
    def _build_interp(path):
        """Reconstruct a RegularGridInterpolator from an `x y val` grid file."""
        data = np.loadtxt(path).astype(np.float32)
        xs = np.unique(data[:, 0])
        ys = np.unique(data[:, 1])
        grid = data[:, 2].reshape(len(xs), len(ys))   # x-major ravel -> [i, j]
        return RegularGridInterpolator(
            (xs, ys), grid, method="linear", bounds_error=False, fill_value=None
        )
