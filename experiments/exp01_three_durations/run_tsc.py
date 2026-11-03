"""Experiment 01, step 2: TSC reconstructions from the stored boundary states.

One task = (duration, k, noise level, set).  Each task reconstructs the k sub-windows of the window from the set's
trials with both estimators (the perturbative 'flow' estimator of the TSC paper and the 'matched' flow-matching
estimator), and records everything -- coefficients, the BCH-2 reference, residuals, defect histories, design
diagnostics, triad analyses, composed predictions of the held-out test states -- without scoring it.

Measurement noise (relative to the RMS of the set's clean boundary states, as in the TIDES benchmarks) is added to
the observed boundary states at analysis time; test states and their true endpoints stay clean.

    python run_tsc.py --workers 16            # all tasks of config.json, heaviest first, finished ones skipped
    python run_tsc.py --quick --workers 2     # on the data of run_simulate.py --quick
    python run_tsc.py --durations long --k 1 2

Outputs: <data_dir>/tsc/<duration>/noise<level>/k<k>/set<ss>.npz and .json
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                    # one thread per worker process

import argparse
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from ttboundary import Protocol                                       # noqa: E402
from ttboundary import tsc                                            # noqa: E402
from ttboundary.storage import exists, load_json, load_npz, save_json, save_npz   # noqa: E402

TRIAD_KEYS = ("triads", "raw", "sym", "nat", "vec", "svec", "readout_keys", "c_true", "c_hat", "readout_wedges_in_triad")


def task_list(cfg: dict, durations=None, ks=None, sets=None) -> list:
    tasks = []
    for label in cfg["durations"]:
        if durations and label not in durations:
            continue
        for k in cfg["tsc"]["k_values"]:
            if ks and k not in ks:
                continue
            for noise in cfg["noise_levels"]:
                for s in range(cfg["tsc"]["n_sets"]):
                    if sets is None or s in sets:
                        tasks.append(dict(duration=label, tau=float(cfg["durations"][label]), k=int(k),
                                          noise=float(noise), set=s))
    # heaviest first: long windows, few sub-windows
    return sorted(tasks, key=lambda t: (-t["tau"] / t["k"], t["duration"], t["k"], t["noise"], t["set"]))


def out_path(data: Path, t: dict) -> Path:
    return data / "tsc" / t["duration"] / f"noise{t['noise']:g}" / f"k{t['k']}" / f"set{t['set']:02d}"


def add_noise(XB: np.ndarray, rel: float, seed: int) -> tuple:
    if rel <= 0:
        return XB, 0.0
    sigma = float(rel * np.sqrt(np.mean(XB ** 2)))
    return XB + sigma * np.random.default_rng(seed).standard_normal(XB.shape), sigma


def run_task(args) -> str:
    t, data, max_iter = args
    out = out_path(data, t)
    if exists(out.with_suffix(".json")):
        return f"{out}: done, skipped"
    T0 = time.perf_counter()
    protocol = Protocol.load(data / "protocol.json")
    XB = load_npz(data / t["duration"] / "tsc_sets.npz")["X_boundaries"][t["set"]]
    test = load_npz(data / t["duration"] / "test.npz")
    seed = zlib.crc32(f"{t['duration']}|{t['k']}|{t['noise']}|{t['set']}".encode())
    XB_obs, sigma = add_noise(XB, t["noise"], seed)
    r = tsc.run_subwindows(protocol, XB_obs, tau=t["tau"], k=t["k"], X_test=test["X0"],
                           matched_kwargs=dict(max_iter=max_iter) if max_iter else None)
    arrays, meta = {}, dict(task=t, sigma=sigma, n_trials=int(XB.shape[0]), T_sub=r["T_sub"], subwindows=[])
    for name in ("X_pred_test_flow", "X_pred_test_matched", "X_naive_test"):
        arrays[name] = r[name]
    arrays["X_true_test"] = test["X_boundaries"][:, -1]
    for j, sw in enumerate(r["subwindows"]):
        p = f"w{j}_"
        for key in ("coef_offsets", "coef_bch2", "coef_flow", "coef_matched", "endpoint_rms_flow",
                    "endpoint_rms_matched", "defect_history"):
            arrays[p + key] = np.asarray(sw[key])
        for est in ("flow", "matched", "bch2"):
            for key in TRIAD_KEYS:
                arrays[f"{p}triads_{est}_{key}"] = np.asarray(sw[f"triads_{est}"][key])
        meta["subwindows"].append(dict(stages=sw["stages"], lib=sw["lib"], design=sw["design"],
                                       matched_start=sw["matched_start"], start_defects=sw["start_defects"],
                                       iterations=int(len(sw["defect_history"])),
                                       final_defect=float(sw["defect_history"][-1]),
                                       endpoint_rms_flow=float(np.mean(sw["endpoint_rms_flow"])),
                                       endpoint_rms_matched=float(np.mean(sw["endpoint_rms_matched"]))))
    meta["seconds"] = time.perf_counter() - T0; meta["max_iter"] = max_iter
    save_npz(out.with_suffix(".npz"), **arrays)
    save_json(out.with_suffix(".json"), meta)                 # written last: its presence marks the task done
    return f"{out}: {meta['seconds']:.0f} s"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--durations", nargs="*"); ap.add_argument("--k", nargs="*", type=int)
    ap.add_argument("--sets", nargs="*", type=int)
    ap.add_argument("--quick", action="store_true", help="use the data of run_simulate.py --quick")
    ap.add_argument("--max-iter", type=int, default=None, help="cap on the flow-matching iterations (tests only)")
    a = ap.parse_args(argv)
    cfg = load_json(a.config)
    if a.quick:
        from run_simulate import quick
        cfg = quick(cfg)
    data = HERE / cfg["data_dir"]
    tasks = task_list(cfg, a.durations, a.k, a.sets)
    print(f"{len(tasks)} tasks, {a.workers} workers", flush=True)
    jobs = [(t, data, a.max_iter) for t in tasks]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            for msg in pool.imap_unordered(run_task, jobs):
                print(time.strftime("%H:%M:%S"), msg, flush=True)
    else:
        for j in jobs:
            print(time.strftime("%H:%M:%S"), run_task(j), flush=True)


if __name__ == "__main__":
    main()
