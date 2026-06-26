import os
import numpy as np
from scipy.interpolate import interp1d

import deepxde as dde
from . import baseclass


class AdvectionCardiac(baseclass.BaseTimePDE):
    """1D linear advection from the cardiac-agent (PDEBench) benchmark.

        u_t = -beta * u_x,   x in [0,1], t in [0,2], periodic BC.

    The initial condition is a data-driven smooth field (single t=0 frame); the
    reference solution is a single t=T frame. This is a NEW, distinct PDE -- it
    does not reuse PINNacle's existing Burgers1D (different domain, BC and physics).

    Data is produced by cardiac-agent/get_data.py -> transform_all_to_pinnsagent():
    the 1D profile was tiled into a 2D RGBA grid, then collapsed back to 1D here.

    Subclass with a concrete coefficient (PREFIX/BETA) so benchmark.py can map a
    pde_name straight to a class -- see AdvectionBeta01Cardiac / AdvectionBeta10Cardiac.
    """

    PREFIX = "advection"   # ref-file stem (e.g. ref/advection_beta0.1_<i>.dat)
    BETA = 0.1             # advection speed

    def __init__(self, datapath=None, icpath=None, bbox=[0, 1, 0, 2.0], instance=None):
        super().__init__()
        prefix, beta = type(self).PREFIX, type(self).BETA

        # Indexed test-set instance, or the default un-indexed pair.
        if instance is not None:
            datapath = f"ref/{prefix}_{instance}.dat"
            icpath = (f"ref/{prefix}_init_u_{instance}.dat",)
            for p in (datapath,) + icpath:
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"{type(self).__name__} instance {instance}: missing file '{p}'. "
                        f"Run cardiac-agent/get_data.py to generate it first.")
        else:
            datapath = datapath or f"ref/{prefix}.dat"
            icpath = icpath or (f"ref/{prefix}_init_u.dat",)

        # output dim: u
        self.output_dim = 1
        # geom: 1D space + time  -> coords are (x, t)
        self.bbox = bbox
        self.geom = dde.geometry.Interval(bbox[0], bbox[1])
        timedomain = dde.geometry.TimeDomain(bbox[2], bbox[3])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # ---- PDE residual: u_t + beta * u_x = 0 -----------------------------
        def pde(x, U):
            u_t = dde.grad.jacobian(U, x, i=0, j=1)   # j=1 is t
            u_x = dde.grad.jacobian(U, x, i=0, j=0)   # j=0 is x
            return u_t + beta * u_x

        self.pde = pde
        self.set_pdeloss(num=1)

        # reference solution (single t=T frame): columns x t u
        self.load_ref_data(datapath, t_transpose=False)

        # ---- data-driven initial condition ----------------------------------
        self._ic_interp = self._build_interp(icpath[0])

        def ic_func(x):
            return self._ic_interp(x[:, 0:1]).reshape(-1, 1)

        def boundary_ic(x, on_initial):
            return on_initial and np.isclose(x[1], bbox[2])

        # periodic BC on the two x-walls (single field u)
        def boundary_x(x, on_boundary):
            return on_boundary and (np.isclose(x[0], bbox[0]) or np.isclose(x[0], bbox[1]))

        self.add_bcs([
            {'component': 0, 'function': ic_func, 'bc': boundary_ic, 'type': 'ic'},
            {'component': 0, 'type': 'periodic', 'component_x': 0, 'bc': boundary_x},
        ])

        # Training Config
        self.training_points(mul=1)

    @staticmethod
    def _build_interp(path):
        """Linear interpolator u(x) from an `x u` profile file."""
        data = np.loadtxt(path).astype(np.float32)
        xs, us = data[:, 0], data[:, 1]
        return interp1d(xs, us, kind="linear", bounds_error=False,
                        fill_value=(us[0], us[-1]))


class AdvectionBeta01Cardiac(AdvectionCardiac):
    PREFIX = "advection_beta0.1"
    BETA = 0.1


class AdvectionBeta10Cardiac(AdvectionCardiac):
    PREFIX = "advection_beta1.0"
    BETA = 1.0
