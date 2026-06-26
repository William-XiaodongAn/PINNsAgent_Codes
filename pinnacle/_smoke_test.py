"""Smoke test for the newly registered cardiac PDE classes.

Converts a few instances, then instantiates each PDE, trains a handful of
iterations and predicts on the reference grid -- just checks the pipeline RUNS
(no NaN/exception), not that it is accurate.

Run:  conda run -n cardiac_agent python pinnacle/_smoke_test.py   (from repo root)
"""
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))                 # .../pinnacle
PINNSAGENT = os.path.abspath(os.path.join(HERE, ".."))            # .../PINNsAgent_Codes
CARDIAC = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # .../cardiac-agent

os.chdir(HERE)
# pinnacle MUST be first so `utils`/`src`/`deepxde` resolve to the vendored ones
# (PINNsAgent_Codes also has a `utils` package that would otherwise shadow them).
sys.path.insert(0, CARDIAC)   # for `import get_data`
sys.path.insert(0, HERE)      # pinnacle -> front of path

os.environ["DDE_BACKEND"] = "pytorch"
os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import get_data as gd   # parent converter

REF = os.path.join(HERE, "ref")
os.makedirs(REF, exist_ok=True)
DATA = os.path.join(CARDIAC, "data")


def convert_one(pde_name, inst):
    spec = gd.PINNSAGENT_SPECS[pde_name]
    srcs = gd._discover_instances(os.path.join(DATA, pde_name))
    if inst not in srcs:
        print(f"  ! {pde_name}: instance {inst} not found ({list(srcs)[:5]}...)")
        return False
    ic, sol = srcs[inst]
    gd._convert_instance(spec, ic, sol, inst, REF)
    print(f"  converted {pde_name} #{inst}")
    return True


print("=== converting test instances ===")
todo = [("advection_beta0.1", 10), ("burgers_nu0.001", 10),
        ("heat", 10), ("fenton_karma", 10), ("TNNP", 2)]
for name, i in todo:
    convert_one(name, i)

import deepxde as dde
import torch
dde.config.set_default_float("float32")

from src.pde.advection_cardiac import AdvectionBeta01Cardiac
from src.pde.burgers_cardiac import BurgersNu0001Cardiac
from src.pde.heat2d_cardiac import Heat2DCardiac
from src.pde.fenton_karma import FentonKarma2D
from src.pde.tnnp import TNNP2D


def smoke(label, pde, iters=20):
    pde.training_points(domain=400, boundary=100, initial=100, test=400, mul=1)
    net = dde.nn.FNN([pde.input_dim] + [32, 32, 32] + [pde.output_dim],
                     "tanh", "Glorot normal")
    model = pde.create_model(net)
    model.compile("adam", lr=1e-3)
    losshist, _ = model.train(iterations=iters, display_every=iters)
    last = float(np.sum(losshist.loss_train[-1]))
    d = pde.ref_data
    m = ~np.isnan(d).any(axis=1)
    y = model.predict(d[m, :pde.input_dim])
    finite = bool(np.isfinite(y).all()) and math.isfinite(last)
    print(f"[{'PASS' if finite else 'NaN '}] {label:24s} in_dim={pde.input_dim} "
          f"out_dim={pde.output_dim} loss={last:.3e} pred{y.shape} ref{d.shape}")
    return finite


print("\n=== smoke-training each PDE (20 iters) ===")
results = {}
for label, ctor in [
    ("AdvectionBeta01 #10", lambda: AdvectionBeta01Cardiac(instance=10)),
    ("BurgersNu0001 #10",   lambda: BurgersNu0001Cardiac(instance=10)),
    ("Heat2DCardiac #10",   lambda: Heat2DCardiac(instance=10)),
    ("FentonKarma2D #10",   lambda: FentonKarma2D(instance=10)),
    ("TNNP2D #2",           lambda: TNNP2D(instance=2)),
]:
    try:
        results[label] = smoke(label, ctor())
    except Exception as e:
        import traceback
        traceback.print_exc()
        results[label] = False
        print(f"[FAIL] {label}: {e}")

print("\n=== summary ===")
for k, v in results.items():
    print(f"  {'OK  ' if v else 'FAIL'}  {k}")
sys.exit(0 if all(results.values()) else 1)
