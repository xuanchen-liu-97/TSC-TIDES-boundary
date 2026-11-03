"""Batch integration of a protocol from many initial states.

The integrator is the one of the TIDES benchmarks (classical RK4 with step dt, the stage switching on the step grid),
vectorised over initial states: in stage r the field is  F(X) = (w * u(X[:, j] - X[:, i])) @ M_r,  where row m of the
edge-to-node matrix M_r has +1 at the first endpoint and -1 at the second endpoint of active edge m.

One integration serves both methods:
    boundary states   X(t_r) at the K+1 stage boundaries t_r = r * tau  (TSC takes any subset of them)
    dense trajectory  every `dense_stride`-th step                        (TIDES)
"""
from __future__ import annotations

import numpy as np

from .protocol import Protocol


def draw_initial_states(n: int, N: int, *, scale: float = 1.5, center: bool = True, seed: int = 0) -> np.ndarray:
    """n initial states, uniform in [-scale, scale]^N, optionally with their mean removed (the dynamics depend on
    state differences only, so the mean is a conserved, irrelevant offset)."""
    X0 = np.random.default_rng(seed).uniform(-scale, scale, (n, N))
    if center:
        X0 = X0 - X0.mean(axis=1, keepdims=True)
    return X0


def steps_per_stage(stage_duration: float, dt: float) -> int:
    ips = int(round(stage_duration / dt))
    if ips < 1 or abs(ips * dt - stage_duration) > 1e-9 * max(stage_duration, 1.0):
        raise ValueError(f"stage duration {stage_duration} is not a multiple of dt = {dt}")
    return ips


def _stage_operators(protocol: Protocol):
    ops = []
    for r in range(protocol.n_stages):
        i, j, w = protocol.stage_arrays(r)
        M = np.zeros((len(i), protocol.N))
        M[np.arange(len(i)), i] = 1.0
        M[np.arange(len(i)), j] = -1.0
        ops.append((i, j, w, M))
    return ops


def simulate_batch(protocol: Protocol, X0: np.ndarray, *, stage_duration: float, dt: float,
                   save_dense: bool = False, dense_stride: int = 1, chunk: int = 2000) -> dict:
    """Integrate every row of X0 through all stages.

    Returns
        t_boundaries   (K+1,)
        X_boundaries   (R, K+1, N)
        t_dense        (n_dense,)          if save_dense
        X_dense        (R, n_dense, N)     if save_dense; sample k is step k * dense_stride
        steps_per_stage
    """
    X0 = np.atleast_2d(np.asarray(X0, float))
    R, N = X0.shape
    K, u = protocol.n_stages, protocol.u()
    ips = steps_per_stage(stage_duration, dt)
    n_steps = K * ips
    if save_dense and (ips % dense_stride):
        raise ValueError("dense_stride must divide the number of steps per stage (stage boundaries are samples)")
    ops = _stage_operators(protocol)
    XB = np.empty((R, K + 1, N))
    n_dense = n_steps // dense_stride + 1 if save_dense else 0
    XD = np.empty((R, n_dense, N)) if save_dense else None
    for a in range(0, R, chunk):
        x = X0[a:a + chunk].copy()
        XB[a:a + chunk, 0] = x
        if save_dense:
            XD[a:a + chunk, 0] = x
        step = 0
        for r, (i, j, w, M) in enumerate(ops):
            F = lambda y: (w * u(y[:, j] - y[:, i])) @ M
            for _ in range(ips):
                k1 = F(x); k2 = F(x + 0.5 * dt * k1); k3 = F(x + 0.5 * dt * k2); k4 = F(x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                step += 1
                if save_dense and step % dense_stride == 0:
                    XD[a:a + chunk, step // dense_stride] = x
            XB[a:a + chunk, r + 1] = x
    out = dict(t_boundaries=np.arange(K + 1) * stage_duration, X_boundaries=XB, steps_per_stage=ips)
    if save_dense:
        out.update(t_dense=np.arange(n_dense) * dt * dense_stride, X_dense=XD)
    return out


def stage_diagnostics(protocol: Protocol, X_dense: np.ndarray, *, steps_per_stage: int, dense_stride: int = 1) -> dict:
    """Information diagnostics per trajectory and stage, from dense trajectories (R, n_dense, N):
        excitation     RMS norm of the field over the samples of the stage (as in the TIDES benchmarks)
        diff_spread    standard deviation of the active-edge state differences x_j - x_i over the stage
        diff_maxabs    largest |x_j - x_i| on an active edge during the stage
    Small spread means the stage explores little of the law u: a single trajectory then pins it down poorly."""
    R, K, u = X_dense.shape[0], protocol.n_stages, protocol.u()
    per = steps_per_stage // dense_stride
    ops = _stage_operators(protocol)
    exc, spread, maxabs = np.empty((R, K)), np.empty((R, K)), np.empty((R, K))
    for r, (i, j, w, M) in enumerate(ops):
        seg = X_dense[:, r * per:(r + 1) * per]                          # samples inside stage r (left-closed)
        d = seg[:, :, j] - seg[:, :, i]                                  # (R, per, E)
        F = (w * u(d)) @ M                                               # (R, per, N)
        exc[:, r] = np.sqrt(np.mean(np.sum(F * F, axis=2), axis=1))
        spread[:, r] = d.reshape(R, -1).std(axis=1)
        maxabs[:, r] = np.abs(d).reshape(R, -1).max(axis=1)
    return dict(excitation=exc, diff_spread=spread, diff_maxabs=maxabs)


def boundary_spread(X_boundaries: np.ndarray) -> np.ndarray:
    """(R, K+1): standard deviation of the node states at each stage boundary (how far from synchrony)."""
    return X_boundaries.std(axis=2)
