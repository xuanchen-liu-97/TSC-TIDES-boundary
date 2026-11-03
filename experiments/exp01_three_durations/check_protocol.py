"""Check that the protocol of config.json is the TIDES structural benchmark with the same seed, and that the batch
integrator reproduces the TIDES integrator on one initial state.

    python check_protocol.py --tides /path/to/folder/containing/tides_benchmarks.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from ttboundary import make_protocol, simulate_batch      # noqa: E402
from ttboundary.storage import load_json                   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--tides", required=True, help="folder that contains tides_benchmarks.py")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.tides)
    import tides_benchmarks as tb
    cfg = load_json(a.config); pc = cfg["protocol"]
    P = make_protocol(**pc)
    tau, dt = float(cfg["durations"]["mid"]), float(cfg["dt"])
    B = tb.make_benchmark("structural", "kuramoto", N=pc["N"], mean_degree=pc["mean_degree"], n_stages=pc["n_stages"],
                          edge_changes=pc["edge_changes"], weight_range=tuple(pc["weight_range"]), seed=pc["seed"],
                          dt=dt, stage_duration=tau, noise_rel=0.0)
    same_w = all(P.edge_weight[e] == B.edge_weight[e] for e in B.candidate_edges)
    print(f"weights identical:   {same_w}")
    print(f"snapshots identical: {P.snapshots == B.snapshots}")
    sim = simulate_batch(P, B.X_clean[0][None, :], stage_duration=tau, dt=dt, save_dense=True)
    err = float(np.abs(sim["X_dense"][0] - B.X_clean).max())
    print(f"max |trajectory - TIDES trajectory| = {err:.1e}  ({'ok' if err < 1e-12 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
