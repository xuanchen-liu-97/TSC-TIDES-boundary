"""Summary table of the TSC results (a first look; the scoring of the experiment is decided later).

Per duration and k, averaged over the sets found:
    pred      error of the composed prediction of the window on the held-out test states, relative to the
              displacement (naive: the time average of the stages)
    slope     slope of the chirality readout c_hat against the true net order c_ef of the aggregated wedges
    sign      fraction of unbalanced wedges (|c_ef| > 0.5) whose sign is read out correctly
    |coef-BCH2|  distance of the coefficients to the leading-order reference (meaningful at short durations)
    iter, defect  flow-matching iterations and final mean squared endpoint defect

    python inspect_tsc.py              # data/
    python inspect_tsc.py --quick      # data_quick/
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from ttboundary.storage import load_json, load_npz         # noqa: E402

ESTIMATORS = ("flow", "matched")


def summarise(data: Path, label: str, k: int, noise: float):
    files = sorted(glob.glob(str(data / "tsc" / label / f"noise{noise:g}" / f"k{k}" / "set*.npz")))
    if not files:
        return None
    X0 = load_npz(data / label / "test.npz")["X0"]
    acc = {e: dict(pred=[], slope=[], sign=[], coef=[]) for e in ESTIMATORS}; naive, iters, defect = [], [], []
    for f in files:
        z = load_npz(f); m = load_json(f.replace(".npz", ".json"))
        Xt = z["X_true_test"]; disp = np.linalg.norm(Xt - X0, axis=1).mean()
        naive.append(np.linalg.norm(z["X_naive_test"] - Xt, axis=1).mean() / disp)
        for e in ESTIMATORS:
            acc[e]["pred"].append(np.linalg.norm(z[f"X_pred_test_{e}"] - Xt, axis=1).mean() / disp)
        pairs = {e: ([], []) for e in ESTIMATORS}
        for j, sw in enumerate(m["subwindows"]):
            iters.append(sw["iterations"]); defect.append(sw["final_defect"])
            for e in ESTIMATORS:
                pairs[e][0].extend(z[f"w{j}_triads_{e}_c_true"]); pairs[e][1].extend(z[f"w{j}_triads_{e}_c_hat"])
                b = z[f"w{j}_coef_bch2"]; acc[e]["coef"].append(np.linalg.norm(z[f"w{j}_coef_{e}"] - b) / np.linalg.norm(b))
        for e in ESTIMATORS:
            c, ch = map(np.array, pairs[e]); unb = np.abs(c) > 0.5
            if unb.sum() > 2:
                acc[e]["slope"].append(np.polyfit(c, ch, 1)[0]); acc[e]["sign"].append(np.mean(np.sign(ch[unb]) == np.sign(c[unb])))
    return len(files), float(np.mean(naive)), acc, int(np.mean(iters)), float(np.mean(defect))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_json(a.config)
    data = HERE / (cfg["data_dir"] + ("_quick" if a.quick else ""))
    for noise in cfg["noise_levels"]:
        print(f"noise {noise:g}")
        print(f"{'':12s} {'naive':>7s} | {'flow: pred':>10s} {'slope':>6s} {'sign':>5s} {'|coef-BCH2|':>11s} | "
              f"{'matched: pred':>13s} {'slope':>6s} {'sign':>5s} {'|coef-BCH2|':>11s} | {'iter':>4s} {'defect':>7s}")
        for label in cfg["durations"]:
            for k in cfg["tsc"]["k_values"]:
                s = summarise(data, label, int(k), float(noise))
                if s is None:
                    continue
                n, naive, acc, it, de = s
                cells = []
                for e in ESTIMATORS:
                    q = acc[e]
                    sl = f"{np.mean(q['slope']):6.3f} {np.mean(q['sign']):5.2f}" if q["slope"] else f"{'-':>6s} {'-':>5s}"
                    cells.append(f"{np.mean(q['pred']):10.1e} {sl} {np.mean(q['coef']):11.1e}")
                print(f"{label:5s} k={k} ({n:2d}) {naive:7.1e} | {cells[0]} | {'':3s}{cells[1]} | {it:4d} {de:7.1e}")


if __name__ == "__main__":
    main()
