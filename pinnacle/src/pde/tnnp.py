import os
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

import deepxde as dde
from . import baseclass


# State-variable order. MUST match cardiac-agent/get_data.py PINNSAGENT_SPECS["TNNP"]
# (texture-major: tex0 V,R_,Nai,Ki | tex1 Cai,CaSS,CaSR,ISumCa | tex2 m,h,j,Xs |
#  tex3 d,f,f2,fCaSS | tex4 r,s,Xr1,Xr2).
TNNP_FIELDS = ["V", "R_", "Nai", "Ki",
               "Cai", "CaSS", "CaSR", "ISumCa",
               "m", "h", "j", "Xs",
               "d", "f", "f2", "fCaSS",
               "r", "s", "Xr1", "Xr2"]


class TNNP2D(baseclass.BaseTimePDE):
    """2D ten Tusscher-Noble-Noble-Panfilov (TP06) human ventricular model.

    20 state variables; only V diffuses (no-flux/Neumann), the other 19 are
    pointwise ODEs (same structure as the 3-field Fenton-Karma class, just much
    larger). The reference description in cardiac-agent/pde_descriptions.py is a
    *discrete* time-stepping solver (x = x + dt*f, gating via exp(-dt/tau)); here
    we use the equivalent *continuous* ODEs:
        gating:  dy/dt = (y_inf - y) / tau_y
        conc.:   buffered rapid-equilibrium form (TP06)

    WARNING: this system is extremely stiff (tau from ~0.1 ms to hundreds of ms)
    and the 20 outputs span very different magnitudes (V~-80, Cai~1e-4, Ki~136).
    A vanilla PINN is not expected to reach useful accuracy without state
    normalisation and a stiff-aware training scheme -- this class is provided so
    TNNP is registered and *runnable* like the other PDEs, not because PINNs solve
    it well. Verify in your conda env before trusting any number.
    """

    def __init__(self, datapath=None, icpath=None,
                 bbox=[0, 12, 0, 12, 0, 100], diffCoef=0.001, instance=None):
        super().__init__()

        prefix = "tnnp"
        if instance is not None:
            datapath = f"ref/{prefix}_{instance}.dat"
            icpath = tuple(f"ref/{prefix}_init_{name}_{instance}.dat" for name in TNNP_FIELDS)
            for p in (datapath,) + icpath:
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"TNNP2D instance {instance}: missing file '{p}'. "
                        f"Run cardiac-agent/get_data.py to generate it first.")
        else:
            datapath = datapath or f"ref/{prefix}.dat"
            icpath = icpath or tuple(f"ref/{prefix}_init_{name}.dat" for name in TNNP_FIELDS)

        self.output_dim = len(TNNP_FIELDS)   # 20
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        timedomain = dde.geometry.TimeDomain(bbox[4], bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # ---- model parameters (TP06, from pde_descriptions.py) --------------
        Ko, Cao, Nao = 5.4, 2.0, 140.0
        Vc, Vsr, Vss = 0.016404, 0.001094, 0.00005468
        Bufc, Kbufc = 0.2, 0.001
        Bufsr, Kbufsr = 10.0, 0.3
        Bufss, Kbufss = 0.4, 0.00025
        Vmaxup, Kup = 0.006375, 0.00025
        Vrel, k3, k4 = 0.102, 0.060, 0.005
        k1prime, k2prime = 0.15, 0.045
        EC, maxsr, minsr = 1.5, 2.5, 1.0
        Vleak, Vxfer = 0.00036, 0.0038
        RR, FF, TT = 8314.3, 96486.7, 310.0
        CAP = 0.185
        Gks, Gto, Gkr, pKNa = 0.392, 0.294, 0.153, 0.03
        GK1, alphanaca, GNa, GbNa = 5.405, 2.5, 14.838, 0.00029
        KmK, KmNa, knak = 1.0, 40.0, 2.724
        GCaL, GbCa, knaca = 0.00003980, 0.000592, 1000.0
        KmNai, KmCa, ksat, nexp = 87.5, 1.38, 0.1, 0.35
        GpCa, KpCa, GpK = 0.1238, 0.0005, 0.0146

        inverseVcF2 = 1.0 / (2 * Vc * FF)
        inverseVcF = 1.0 / (Vc * FF)
        inversevssF2 = 1.0 / (2 * Vss * FF)
        rtof = RR * TT / FF
        fort = 1.0 / rtof
        KmNai3, Nao3 = KmNai ** 3, Nao ** 3
        EPS = 1e-8

        def sig(z):  # 1 / (1 + exp(z))
            return 1.0 / (1.0 + torch.exp(z))

        # ---- PDE / ODE residuals --------------------------------------------
        def pde(x, U):
            (V, R_, Nai, Ki, Cai, CaSS, CaSR, ISumCa, m, h, jg, Xs,
             d, f, f2, fCaSS, r, s, Xr1, Xr2) = [U[:, k:k + 1] for k in range(20)]

            # keep concentrations strictly positive for log / division safety
            Nai_p = torch.clamp(Nai, min=EPS)
            Ki_p = torch.clamp(Ki, min=EPS)
            Cai_p = torch.clamp(Cai, min=EPS)
            CaSS_p = torch.clamp(CaSS, min=EPS)
            CaSR_p = torch.clamp(CaSR, min=EPS)

            # time derivatives (coord 2 is t)
            d_dt = lambda i: dde.grad.jacobian(U, x, i=i, j=2)
            V_xx = dde.grad.hessian(U, x, i=0, j=0, component=0)
            V_yy = dde.grad.hessian(U, x, i=1, j=1, component=0)

            # ---- reversal potentials --------------------------------------
            Ek = rtof * torch.log(Ko / Ki_p)
            Ena = rtof * torch.log(Nao / Nai_p)
            Eks = rtof * torch.log((Ko + pKNa * Nao) / (Ki_p + pKNa * Nai_p))
            Eca = 0.5 * rtof * torch.log(Cao / Cai_p)

            # ---- membrane currents ----------------------------------------
            INa = GNa * m ** 3 * h * jg * (V - Ena)
            IKr = Gkr * np.sqrt(Ko / 5.4) * Xr1 * Xr2 * (V - Ek)
            IKs = Gks * Xs ** 2 * (V - Eks)
            Ito = Gto * r * s * (V - Ek)

            vmek = V - Ek
            Ak1 = 0.1 / (1.0 + torch.exp(0.06 * (vmek - 200.0)))
            Bk1 = (3.0 * torch.exp(0.0002 * (vmek + 100.0)) + torch.exp(0.1 * (vmek - 10.0))) \
                / (1.0 + torch.exp(-0.5 * vmek))
            IK1 = GK1 * Ak1 / (Ak1 + Bk1) * (V - Ek)

            IpK = GpK * sig((25.0 - V) / 5.98) * (V - Ek)
            IbNa = GbNa * (V - Ena)

            INaK = (1.0 / (1.0 + 0.1245 * torch.exp(-0.1 * V * fort)
                           + 0.0353 * torch.exp(-V * fort))) \
                * knak * (Ko / (Ko + KmK)) * (Nai_p / (Nai_p + KmNa))

            temp = torch.exp((nexp - 1.0) * V * fort)
            temp2 = knaca / ((KmNai3 + Nao3) * (KmCa + Cao) * (1.0 + ksat * temp))
            INaCa = temp2 * torch.exp(nexp * V * fort) * Cao * Nai_p ** 3 \
                - temp2 * temp * Nao3 * alphanaca * Cai_p

            # L-type Ca current (eps-guarded GHK denominators near V=15 mV)
            tical = torch.exp(2.0 * (V - 15.0) * fort)
            ical1t = GCaL * 4.0 * (V - 15.0) * (FF * fort) * (0.25 * tical) / (tical - 1.0 + EPS)
            ical2t = GCaL * 4.0 * (V - 15.0) * (FF * fort) * Cao / (tical - 1.0 + EPS)
            ICaL = d * f * f2 * fCaSS * (ical1t * CaSS - ical2t)

            IpCa = GpCa * Cai_p / (KpCa + Cai_p)
            IbCa = GbCa * (V - Eca)

            ISumNaK = INa + IbNa + INaK + IK1 + IKr + IKs + IpK + Ito
            ISumCa_expr = ICaL + IpCa + IbCa
            I_sum = ISumCa_expr + ISumNaK + INaCa

            # ---- calcium subsystem ----------------------------------------
            kCaSR = maxsr - (maxsr - minsr) / (1.0 + (EC / CaSR_p) ** 2)
            k1 = k1prime / kCaSR
            k2 = k2prime * kCaSR
            O = k1 * CaSS_p ** 2 * R_ / (k3 + k1 * CaSS_p ** 2)
            Irel = Vrel * O * (CaSR - CaSS)
            Ileak = Vleak * (CaSR - Cai)
            Iup = Vmaxup / (1.0 + (Kup / Cai_p) ** 2)
            Ixfer = Vxfer * (CaSS - Cai)

            bufc = 1.0 / (1.0 + Bufc * Kbufc / (Cai_p + Kbufc) ** 2)
            bufsr = 1.0 / (1.0 + Bufsr * Kbufsr / (CaSR_p + Kbufsr) ** 2)
            bufss = 1.0 / (1.0 + Bufss * Kbufss / (CaSS_p + Kbufss) ** 2)

            dCai = bufc * (-(IbCa + IpCa - 2.0 * INaCa) * inverseVcF2 * CAP
                           - (Iup - Ileak) * Vsr / Vc + Ixfer)
            dCaSR = bufsr * (Iup - Irel - Ileak)
            dCaSS = bufss * (-Ixfer * Vc / Vss + Irel * Vsr / Vss - ICaL * inversevssF2 * CAP)
            dR_ = k4 * (1.0 - R_) - k2 * CaSS * R_

            dNai = -(INa + IbNa + 3.0 * INaK + 3.0 * INaCa) * inverseVcF * CAP
            dKi = -(IK1 + Ito + IKr + IKs - 2.0 * INaK + IpK) * inverseVcF * CAP

            # ---- gating: steady states and time constants -----------------
            minf = sig((-56.86 - V) / 9.03) ** 2
            tau_m = sig((-60.0 - V) / 5.0) * (0.1 / (1.0 + torch.exp((V + 35.0) / 5.0))
                                              + 0.10 / (1.0 + torch.exp((V - 50.0) / 200.0)))
            hinf = sig((V + 71.55) / 7.43) ** 2
            AH = torch.where(V >= -40.0, torch.zeros_like(V), 0.057 * torch.exp(-(V + 80.0) / 6.8))
            BH = torch.where(V >= -40.0,
                             0.77 / (0.13 * (1.0 + torch.exp(-(V + 10.66) / 11.1))),
                             2.7 * torch.exp(0.079 * V) + 3.1e5 * torch.exp(0.3485 * V))
            tau_h = 1.0 / (AH + BH)
            jinf = hinf
            AJ = torch.where(V >= -40.0, torch.zeros_like(V),
                             ((-2.5428e4 * torch.exp(0.2444 * V) - 6.948e-6 * torch.exp(-0.04391 * V))
                              * (V + 37.78)) / (1.0 + torch.exp(0.311 * (V + 79.23))))
            BJ = torch.where(V >= -40.0,
                             0.6 * torch.exp(0.057 * V) / (1.0 + torch.exp(-0.1 * (V + 32.0))),
                             0.02424 * torch.exp(-0.01052 * V) / (1.0 + torch.exp(-0.1378 * (V + 40.14))))
            tau_j = 1.0 / (AJ + BJ)
            xsinf = sig((-5.0 - V) / 14.0)
            tau_xs = 1400.0 / torch.sqrt(1.0 + torch.exp((5.0 - V) / 6.0)) \
                * (1.0 / (1.0 + torch.exp((V - 35.0) / 15.0))) + 80.0
            rinf = sig((20.0 - V) / 6.0)
            tau_r = 9.5 * torch.exp(-(V + 40.0) ** 2 / 1800.0) + 0.8
            sinf = sig((V + 20.0) / 5.0)
            tau_s = 85.0 * torch.exp(-(V + 45.0) ** 2 / 320.0) \
                + 5.0 / (1.0 + torch.exp((V - 20.0) / 5.0)) + 3.0
            dinf = sig((-8.0 - V) / 7.5)
            tau_d = (1.4 / (1.0 + torch.exp((-35.0 - V) / 13.0)) + 0.25) \
                * (1.4 / (1.0 + torch.exp((V + 5.0) / 5.0))) + 1.0 / (1.0 + torch.exp((50.0 - V) / 20.0))
            finf = sig((V + 20.0) / 7.0)
            tau_f = 1102.5 * torch.exp(-(V + 27.0) ** 2 / 225.0) \
                + 200.0 / (1.0 + torch.exp((13.0 - V) / 10.0)) \
                + 180.0 / (1.0 + torch.exp((V + 30.0) / 10.0)) + 20.0
            f2inf = 0.67 / (1.0 + torch.exp((V + 35.0) / 7.0)) + 0.33
            tau_f2 = 600.0 * torch.exp(-(V + 25.0) ** 2 / 49.0) \
                + 31.0 / (1.0 + torch.exp((25.0 - V) / 10.0)) \
                + 16.0 / (1.0 + torch.exp((V + 30.0) / 10.0))
            fcassinf = 0.6 / (1.0 + (CaSS / 0.05) ** 2) + 0.4
            tau_fcass = 80.0 / (1.0 + (CaSS / 0.05) ** 2) + 2.0
            xr1inf = sig((-26.0 - V) / 7.0)
            tau_xr1 = (450.0 / (1.0 + torch.exp((-45.0 - V) / 10.0))) \
                * (6.0 / (1.0 + torch.exp((V + 30.0) / 11.5)))
            xr2inf = sig((V + 88.0) / 24.0)
            tau_xr2 = (3.0 / (1.0 + torch.exp((-60.0 - V) / 20.0))) \
                * (1.12 / (1.0 + torch.exp((V - 60.0) / 20.0)))

            # ---- residuals (continuous form), output order = TNNP_FIELDS ---
            return [
                d_dt(0) - (diffCoef * (V_xx + V_yy) - I_sum),     # V
                d_dt(1) - dR_,                                    # R_
                d_dt(2) - dNai,                                   # Nai
                d_dt(3) - dKi,                                    # Ki
                d_dt(4) - dCai,                                   # Cai
                d_dt(5) - dCaSS,                                  # CaSS
                d_dt(6) - dCaSR,                                  # CaSR
                ISumCa - ISumCa_expr,                             # ISumCa (algebraic)
                d_dt(8) - (minf - m) / tau_m,                     # m
                d_dt(9) - (hinf - h) / tau_h,                     # h
                d_dt(10) - (jinf - jg) / tau_j,                   # j
                d_dt(11) - (xsinf - Xs) / tau_xs,                 # Xs
                d_dt(12) - (dinf - d) / tau_d,                    # d
                d_dt(13) - (finf - f) / tau_f,                    # f
                d_dt(14) - (f2inf - f2) / tau_f2,                 # f2
                d_dt(15) - (fcassinf - fCaSS) / tau_fcass,        # fCaSS
                d_dt(16) - (rinf - r) / tau_r,                    # r
                d_dt(17) - (sinf - s) / tau_s,                    # s
                d_dt(18) - (xr1inf - Xr1) / tau_xr1,              # Xr1
                d_dt(19) - (xr2inf - Xr2) / tau_xr2,              # Xr2
            ]

        self.pde = pde
        self.set_pdeloss(num=20)

        # reference solution (single t=T frame): x y t + 20 fields
        self.load_ref_data(datapath, t_transpose=False)

        # ---- data-driven initial condition (all 20 fields) ------------------
        self._ic_interp = [self._build_interp(p) for p in icpath]

        def make_ic(component):
            return lambda x: self._ic_interp[component](x[:, 0:2]).reshape(-1, 1)

        def boundary_ic(x, on_initial):
            return on_initial and np.isclose(x[2], bbox[4])

        bcs = [{'component': c, 'function': make_ic(c), 'bc': boundary_ic, 'type': 'ic'}
               for c in range(self.output_dim)]
        # no-flux (Neumann) on V only; the other 19 are pointwise ODEs
        bcs.append({'component': 0, 'function': (lambda _: 0),
                    'bc': (lambda _, on_boundary: on_boundary), 'type': 'neumann'})
        self.add_bcs(bcs)

        self.training_points(mul=1)

    @staticmethod
    def _build_interp(path):
        """Reconstruct a RegularGridInterpolator from an `x y val` grid file."""
        data = np.loadtxt(path).astype(np.float32)
        xs = np.unique(data[:, 0])
        ys = np.unique(data[:, 1])
        grid = data[:, 2].reshape(len(xs), len(ys))
        return RegularGridInterpolator(
            (xs, ys), grid, method="linear", bounds_error=False, fill_value=None)
