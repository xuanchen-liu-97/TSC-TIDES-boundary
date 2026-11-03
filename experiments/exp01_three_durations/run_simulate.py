"""Experiment 01, step 1: one protocol, three stage durations, all trajectories integrated once.

Writes, under <data_dir>/:
    protocol.json, truth.json                 the hidden protocol and the quantities used to score reconstructions
    <duration>/tsc_sets.npz                   X0 (S, R, N), X_boundaries (S, R, K+1, N), boundary_spread (S, R, K+1)
    <duration>/tides_sets.npz                 X0 (S, T, N), X_dense (S, T, n, N), t_dense, X_boundaries,
                                              excitation / diff_spread / diff_maxabs (S, T, K)
    <duration>/test.npz                       held-out initial states and their boundary states (prediction tests)
    <duration>/summary.json                   interaction per stage, timings, diagnostics
States are clean: measurement noise is added at analysis time, so every noise level reuses these trajectories.
The same initial states are used at every duration.  Finished durations are skipped unless --force.

    python run_simulate.py                    # full experiment
    python run_simulate.py --quick            # small smoke test into data_quick/
    python run_simulate.py --durations long
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # repository root, so that `ttboundary` imports without install

from ttboundary import draw_initial_states, make_protocol, simulate_batch, stage_diagnostics   # noqa: E402
from ttboundary.simulate import boundary_spread                                                 # noqa: E402
from ttboundary.storage import exists, load_json, save_json, save_npz                           # noqa: E402


def quick(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["tsc"] = {**cfg["tsc"], "n_sets": 2, "trials_per_set": 300}   # above the largest library (180 columns)
    cfg["tides"] = {**cfg["tides"], "n_sets": 2}
    cfg["test"] = {**cfg["test"], "n_trials": 40}
    cfg["data_dir"] = cfg["data_dir"] + "_quick"
    return cfg


def initial_states(cfg: dict, N: int) -> dict:
    ic = cfg["initial_states"]
    tsc = np.stack([draw_initial_states(cfg["tsc"]["trials_per_set"], N, seed=cfg["tsc"]["seed"] + s, **ic)
                    for s in range(cfg["tsc"]["n_sets"])])
    tides = np.stack([draw_initial_states(cfg["tides"]["trajectories_per_set"], N, seed=cfg["tides"]["seed"] + s, **ic)
                      for s in range(cfg["tides"]["n_sets"])])
    test = draw_initial_states(cfg["test"]["n_trials"], N, seed=cfg["test"]["seed"], **ic)
    return dict(tsc=tsc, tides=tides, test=test)


def run_duration(label: str, tau: float, protocol, X0: dict, cfg: dict, out: Path) -> dict:
    dt, stride = cfg["dt"], cfg["tides"]["dense_stride"]
    S, R, N = X0["tsc"].shape
    T0 = time.perf_counter()
    # ---- TSC: boundary states of every trial
    sim = simulate_batch(protocol, X0["tsc"].reshape(S * R, N), stage_duration=tau, dt=dt)
    XB = sim["X_boundaries"].reshape(S, R, protocol.n_stages + 1, N)
    save_npz(out / "tsc_sets.npz", X0=X0["tsc"], X_boundaries=XB, t_boundaries=sim["t_boundaries"],
             boundary_spread=boundary_spread(XB.reshape(S * R, -1, N)).reshape(S, R, -1))
    t_tsc = time.perf_counter() - T0
    # ---- TIDES: dense trajectories (nested pooled sets: the first k trajectories of each set)
    T1 = time.perf_counter()
    St, Tt, _ = X0["tides"].shape
    simd = simulate_batch(protocol, X0["tides"].reshape(St * Tt, N), stage_duration=tau, dt=dt,
                          save_dense=True, dense_stride=stride)
    diag = stage_diagnostics(protocol, simd["X_dense"], steps_per_stage=simd["steps_per_stage"], dense_stride=stride)
    n_dense = simd["X_dense"].shape[1]
    save_npz(out / "tides_sets.npz", X0=X0["tides"], X_dense=simd["X_dense"].reshape(St, Tt, n_dense, N),
             t_dense=simd["t_dense"], X_boundaries=simd["X_boundaries"].reshape(St, Tt, -1, N),
             t_boundaries=simd["t_boundaries"], steps_per_stage=simd["steps_per_stage"], dense_stride=stride,
             **{k: v.reshape(St, Tt, -1) for k, v in diag.items()})
    t_tides = time.perf_counter() - T1
    # ---- held-out test states
    simt = simulate_batch(protocol, X0["test"], stage_duration=tau, dt=dt)
    save_npz(out / "test.npz", X0=X0["test"], X_boundaries=simt["X_boundaries"], t_boundaries=simt["t_boundaries"])
    spread = boundary_spread(XB.reshape(S * R, -1, N))
    summary = dict(duration=label, stage_duration=tau, dt=dt, steps_per_stage=sim["steps_per_stage"],
                   interaction=protocol.interaction_per_stage(tau),
                   boundary_spread_mean=spread.mean(axis=0).tolist(),
                   tides_excitation_mean=diag["excitation"].mean(axis=0).tolist(),
                   tides_diff_spread_mean=diag["diff_spread"].mean(axis=0).tolist(),
                   seconds=dict(tsc=t_tsc, tides=t_tides, total=time.perf_counter() - T0),
                   shapes=dict(tsc_X_boundaries=list(XB.shape), tides_X_dense=[St, Tt, n_dense, N]))
    save_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--durations", nargs="*", help="subset of duration labels (default: all)")
    ap.add_argument("--force", action="store_true", help="recompute durations that are already done")
    ap.add_argument("--quick", action="store_true", help="small smoke test into <data_dir>_quick/")
    a = ap.parse_args(argv)
    cfg = load_json(a.config)
    if a.quick:
        cfg = quick(cfg)
    data = HERE / cfg["data_dir"]
    protocol = make_protocol(**cfg["protocol"])
    data.mkdir(parents=True, exist_ok=True)
    protocol.save(data / "protocol.json")
    save_json(data / "truth.json", protocol.truth())
    save_json(data / "config_used.json", cfg)
    X0 = initial_states(cfg, protocol.N)
    for label, tau in cfg["durations"].items():
        if a.durations and label not in a.durations:
            continue
        out = data / label
        if exists(out / "summary.json") and not a.force:
            print(f"{label}: done, skipped"); continue
        s = run_duration(label, float(tau), protocol, X0, cfg, out)
        print(f"{label}: tau = {tau:g}, {s['steps_per_stage']} steps per stage, interaction per node per window "
              f"{s['interaction']['per_node_window']:.2f}, boundary spread {s['boundary_spread_mean'][0]:.2f} -> "
              f"{s['boundary_spread_mean'][-1]:.2f}, {s['seconds']['total']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
