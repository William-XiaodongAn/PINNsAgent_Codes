"""
Evaluate the Fenton-Karma test set the faithful way:
for each test instance, TRAIN a fresh PINN with the agent's Best Config and that
instance's own IC, then score it against that instance's reference solution.

A PINN is per-IC: each instance is an independent training run. Hyperparameters
(the Best Config below) are shared; the IC and the reference change per instance.

Prereq: run scripts/gen_fenton_karma_batch.py first to produce, per instance i,
    ref/fenton_karma_init_{u,v,w}_<i>.dat   and   ref/fenton_karma_<i>.dat

Usage:
    python scripts/eval_test_set.py --instances 10-59 --iter 20000
    python scripts/eval_test_set.py --instances 10-11 --iter 2000   # quick smoke
"""
import os
import sys
import argparse
import csv
import numpy as np

ORIG_CWD = os.getcwd()
PINNACLE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PINNACLE)
os.chdir(PINNACLE)

os.environ["DDE_BACKEND"] = "pytorch"
import torch
import deepxde as dde
dde.config.set_default_float("float32")

from src.pde.fenton_karma import FentonKarma2D
from src.model.laaf import DNN_LAAF, DNN_GAAF
from src.utils.args import parse_width_depth

# ===== Best Config from the agent run (iter_1). Edit here if yours differs. =====
BEST = dict(
    net="laaf", activation="gaussian", optimizer="adam",
    width=148, depth=6, lr=1e-4, initializer="Glorot normal",
    num_domain=1100, num_boundary=1600, num_initial=4600,
)


def build_net(c, in_dim, out_dim):
    if c["net"] == "fnn":
        return dde.nn.FNN(
            layer_sizes=[in_dim] + parse_width_depth(c["width"], c["depth"]) + [out_dim],
            activation=c["activation"], kernel_initializer=c["initializer"])
    if c["net"] == "laaf":
        return DNN_LAAF(n_layers=c["depth"], n_hidden=c["width"], x_dim=in_dim, u_dim=out_dim,
                        activation=c["activation"], kernel_initializer=c["initializer"])
    if c["net"] == "gaaf":
        return DNN_GAAF(n_layers=c["depth"], n_hidden=c["width"], x_dim=in_dim, u_dim=out_dim,
                        activation=c["activation"], kernel_initializer=c["initializer"])
    raise ValueError(f"unknown net {c['net']}")


def eval_instance(i, iters):
    ref = f"ref/fenton_karma_{i}.dat"
    ic = (f"ref/fenton_karma_init_u_{i}.dat",
          f"ref/fenton_karma_init_v_{i}.dat",
          f"ref/fenton_karma_init_w_{i}.dat")
    for p in (ref,) + ic:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing — run gen_fenton_karma_batch.py first")

    pde = FentonKarma2D(datapath=ref, icpath=ic)
    pde.training_points(domain=BEST["num_domain"], boundary=BEST["num_boundary"],
                        initial=BEST["num_initial"], mul=1)

    net = build_net(BEST, pde.input_dim, pde.output_dim).float()
    model = pde.create_model(net)
    model.compile(BEST["optimizer"], lr=BEST["lr"])
    model.train(iterations=iters, display_every=max(1, iters // 4))

    d = pde.ref_data
    m = ~np.isnan(d).any(axis=1)
    test_x, test_y = d[m, :pde.input_dim], d[m, pde.input_dim:]
    with torch.no_grad():
        y = model.predict(test_x)
    mse = float(((y - test_y) ** 2).mean())
    l2re = float(np.sqrt(((y - test_y) ** 2).mean()) / np.sqrt((test_y ** 2).mean()))

    del model, net, pde
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mse, l2re


def parse_instances(spec):
    ids = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            lo, hi = tok.split("-"); ids += list(range(int(lo), int(hi) + 1))
        elif tok:
            ids.append(int(tok))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="10-59")
    ap.add_argument("--iter", type=int, default=20000)
    ap.add_argument("--out", default="test_eval_results.csv")
    args = ap.parse_args()

    ids = parse_instances(args.instances)
    print(f"Best Config: {BEST}")
    print(f"evaluating {len(ids)} instances x {args.iter} steps each\n", flush=True)

    results = []
    for i in ids:
        mse, l2re = eval_instance(i, args.iter)
        results.append((i, mse, l2re))
        print(f"instance {i:>3}: MSE={mse:.6e}  L2RE={l2re:.6e}", flush=True)

    mses = np.array([r[1] for r in results])
    l2s = np.array([r[2] for r in results])
    print(f"\n=== {len(results)} instances ===")
    print(f"mean MSE  = {mses.mean():.6e}  (std {mses.std():.2e})")
    print(f"mean L2RE = {l2s.mean():.6e}  (std {l2s.std():.2e})")

    out = args.out if os.path.isabs(args.out) else os.path.join(ORIG_CWD, args.out)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["instance", "mse", "l2re"]); w.writerows(results)
    print("written", out)


if __name__ == "__main__":
    main()
