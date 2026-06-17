import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

import deepxde as dde
from . import baseclass


class FentonKarma2D(baseclass.BaseTimePDE):
    """2D Fenton-Karma reaction-diffusion model (3 fields: u, v, w).

        u_t = D (u_xx + u_yy) - (I_fi + I_so + I_si) / C_m
        v_t = (1-v)/tau_mv(u)  if u <  V_c   else  -v/tau_pv
        w_t = (1-w)/tau_mw     if u <  V_c   else  -w/tau_pw

    Only u diffuses; v and w are pointwise ODEs. No-flux (Neumann) BC on u.
    Initial condition for u, v, w is supplied from data (single t=0 frame),
    the reference solution is a single t=T frame.  See scripts/gen_fenton_karma.py.
    """

    def __init__(
        self,
        datapath="ref/fenton_karma.dat",
        icpath=("ref/fenton_karma_init_u.dat",
                "ref/fenton_karma_init_v.dat",
                "ref/fenton_karma_init_w.dat"),
        bbox=[0, 10, 0, 10, 0, 100],
        tau_d=0.5714,
    ):
        super().__init__()
        # output dim: u, v, w
        self.output_dim = 3
        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        timedomain = dde.geometry.TimeDomain(bbox[4], bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # ---- model parameters (Fenton-Karma) --------------------------------
        D = 0.001
        C_m = 1.0
        tau_pv = 7.99
        tau_v1 = 9.8
        tau_v2 = 312.5
        tau_pw = 870.0
        tau_mw = 41.0
        tau_0 = 12.5
        tau_r = 33.83
        tau_si = 29.0
        k = 10.0
        V_csi = 0.861
        V_c = 0.13
        V_v = 0.04

        # ---- PDE residuals --------------------------------------------------
        def pde(x, U):
            u = U[:, 0:1]
            v = U[:, 1:2]
            w = U[:, 2:3]

            # time derivatives (j=2 is t)
            u_t = dde.grad.jacobian(U, x, i=0, j=2)
            v_t = dde.grad.jacobian(U, x, i=1, j=2)
            w_t = dde.grad.jacobian(U, x, i=2, j=2)

            # diffusion of u only
            u_xx = dde.grad.hessian(U, x, i=0, j=0, component=0)
            u_yy = dde.grad.hessian(U, x, i=1, j=1, component=0)

            # Heaviside H(u - V_c): hard switch, matches the FD reference solver
            H = (u >= V_c).float()

            I_fi = -v * H * (u - V_c) * (1 - u) / tau_d
            I_so = u * (1 - H) / tau_0 + H / tau_r
            I_si = -w * (1 + torch.tanh(k * (u - V_csi))) / (2 * tau_si)

            tau_mv = torch.where(u < V_v, torch.full_like(u, tau_v1), torch.full_like(u, tau_v2))

            res_u = u_t - (D * (u_xx + u_yy) - (I_fi + I_so + I_si) / C_m)
            res_v = v_t - torch.where(u < V_c, (1 - v) / tau_mv, -v / tau_pv)
            res_w = w_t - torch.where(u < V_c, (1 - w) / tau_mw, -w / tau_pw)

            return [res_u, res_v, res_w]

        self.pde = pde
        self.set_pdeloss(num=3)

        # reference solution (single t=T frame): columns x y t u v w
        self.load_ref_data(datapath, t_transpose=False)

        # ---- data-driven initial condition ----------------------------------
        # rebuild a regular grid interpolator per component from the init files
        self._ic_interp = [self._build_interp(p) for p in icpath]

        def ic_func(x, component):
            pts = x[:, 0:2]
            return self._ic_interp[component](pts).reshape(-1, 1)

        def boundary_ic(x, on_initial):
            return on_initial and np.isclose(x[2], bbox[4])

        self.add_bcs([
            {'component': 0, 'function': (lambda x: ic_func(x, 0)), 'bc': boundary_ic, 'type': 'ic'},
            {'component': 1, 'function': (lambda x: ic_func(x, 1)), 'bc': boundary_ic, 'type': 'ic'},
            {'component': 2, 'function': (lambda x: ic_func(x, 2)), 'bc': boundary_ic, 'type': 'ic'},
            # no-flux (Neumann) on u over all walls; v, w have no spatial BC
            {'component': 0, 'function': (lambda _: 0),
             'bc': (lambda _, on_boundary: on_boundary), 'type': 'neumann'},
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
