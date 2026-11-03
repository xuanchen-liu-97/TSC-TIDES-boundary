"""TSC reconstruction on a protocol: static effective generators from the endpoints of repeated trials.

Ported from the synthetic experiments of the TSC paper (tsc_synth60.py), specialised to the protocols of this
repository: Kuramoto coupling u(d) = sin d with the protocol's edge weights, no natural frequencies, no native
three-body terms.

Pieces
    stages_of(protocol, stages)       the stages of a (sub-)window in the TSC format (0-based edges -> weight)
    LocalLib                          local candidate library (pairwise partners within two hops of the window's
                                      aggregated network; connected candidate triads), as in tsc_synth60
    reconstruct                       least squares per node with the flow-consistent estimator
    bch2_field / project              leading-order (BCH degree 2) effective field, projected onto the library:
                                      the analytic reference for the coefficients at short stage durations
    analyse_triads                    raw strength, S3-symmetric part, matched native estimate, chirality readout
    run_subwindows                    k sub-windows of a window: reconstruct each, compose their flows

Sub-windows.  With k sub-windows of a K-stage window (k divides K), sub-window j covers stages
[j K/k, (j+1) K/k) and is observed at its two boundaries; k = 1 is "endpoints only", k = K resolves every stage.
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

FULL_DIRS = [(-2, 1, 1), (1, -2, 1), (1, 1, -2), (0, 1, -1)]   # zero-sum first harmonics on (own, partner1, partner2)
PERMS = list(itertools.permutations(range(3)))
NT = 8                                                           # columns per (node, triad): 4 directions x (sin, cos)


# ---------------------------------------------------------------------------------------------------------------
# model of the protocol
# ---------------------------------------------------------------------------------------------------------------

def stages_of(protocol, stage_indices) -> list:
    """TSC format for the given stages: one dict per stage, 0-based edge (i, j), i < j -> weight."""
    W = protocol.stage_weights()
    return [{(a - 1, b - 1): w for (a, b), w in W[s].items()} for s in stage_indices]


def stage_field(X: np.ndarray, E_s: dict, a: float = 1.0) -> np.ndarray:
    """Kuramoto field of one stage: dx_i/dt = a sum_j w_ij sin(x_j - x_i)."""
    F = np.zeros_like(X)
    if E_s:
        ij = np.array(list(E_s.keys())); w = np.array(list(E_s.values()), float)
        d = a * w * np.sin(X[:, ij[:, 1]] - X[:, ij[:, 0]])
        np.add.at(F.T, ij[:, 0], d.T); np.add.at(F.T, ij[:, 1], -d.T)
    return F


def time_average(E: list) -> dict:
    Eb = defaultdict(float)
    for E_s in E:
        for e, w in E_s.items():
            Eb[e] += w / len(E)
    return dict(Eb)


def net_order(E: list) -> dict:
    """c_ef in units of tau^2 (weights multiplied), keyed (e, f) with e < f; positive when e acts first."""
    c = defaultdict(float)
    for s in range(len(E)):
        for t in range(s + 1, len(E)):
            for e, we in E[s].items():
                for f, wf in E[t].items():
                    if len(set(e) & set(f)) != 1:
                        continue
                    if e < f: c[(e, f)] += we * wf
                    else: c[(f, e)] -= we * wf
    return dict(c)


def rk4_flow(field, X: np.ndarray, T: float, n: int = 40) -> np.ndarray:
    h = T / n
    for _ in range(n):
        k1 = field(X); k2 = field(X + h / 2 * k1); k3 = field(X + h / 2 * k2); k4 = field(X + h * k3)
        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


# ---------------------------------------------------------------------------------------------------------------
# local library (Kuramoto), least squares, flow-consistent estimator
# ---------------------------------------------------------------------------------------------------------------

class LocalLib:
    """Pairwise partners within two hops of the aggregated network; candidate triads (j, k) of node i with at least
    two of the pairs (i, j), (i, k), (j, k) aggregated.  Columns: per partner sin/cos of n (x_j - x_i), n = 1, 2;
    per triad sin/cos of the four zero-sum first-harmonic directions."""

    def __init__(self, N: int, agg_edges):
        self.N = N; agg = set(map(tuple, agg_edges)); self.agg = agg
        adj = {i: set() for i in range(N)}
        for u, v in agg:
            adj[u].add(v); adj[v].add(u)
        self.adj = adj
        self.partners, self.triads, self.tri_slot = [], [], [defaultdict(int) for _ in range(N)]
        for i in range(N):
            P = sorted((adj[i] | {k for j in adj[i] for k in adj[j]}) - {i})
            T = [(j, k) for j, k in itertools.combinations(P, 2)
                 if sum((min(p, q), max(p, q)) in agg for p, q in ((i, j), (i, k), (j, k))) >= 2]
            self.partners.append(P); self.triads.append(T)
            for n_, t in enumerate(T):
                self.tri_slot[i][t] = 4 * len(P) + NT * n_
        self.ncol = [(4 * len(P) + NT * len(T)) if P else 0 for P, T in zip(self.partners, self.triads)]
        self.all_triads = sorted({tuple(sorted((i, j, k))) for i in range(N) for j, k in self.triads[i]})
        self._prep = {}

    def _kprep(self, i):
        if i not in self._prep:
            P, T = self.partners[i], self.triads[i]; nc = self.ncol[i]
            D = np.zeros((self.N, nc)); S = np.zeros(nc, bool); c = 0
            for j in P:
                for n in (1, 2):
                    for sn in (True, False):
                        D[j, c] += n; D[i, c] -= n; S[c] = sn; c += 1
            for j, k in T:
                for d in FULL_DIRS:
                    for sn in (True, False):
                        D[i, c] += d[0]; D[j, c] += d[1]; D[k, c] += d[2]; S[c] = sn; c += 1
            self._prep[i] = (D, S)
        return self._prep[i]

    def design(self, X, i):
        if self.ncol[i] == 0:
            return np.zeros((X.shape[0], 0))
        D, S = self._kprep(i); A = X @ D; out = np.cos(A); out[:, S] = np.sin(A[:, S])
        return out

    def spec(self) -> dict:
        return dict(agg=[list(e) for e in sorted(self.agg)], partners=self.partners,
                    triads=[[list(t) for t in T] for T in self.triads], ncol=self.ncol)


class Fit:
    """Least-squares coefficients per node (no known part: the protocol has no natural frequencies)."""

    def __init__(self, lib: LocalLib, X0, Y, coef=None):
        self.lib = lib
        if coef is not None:
            self.coef = [np.asarray(c, float) for c in coef]; return
        self.coef = []
        for i in range(lib.N):
            if lib.ncol[i] == 0:
                self.coef.append(np.zeros(0)); continue
            self.coef.append(np.linalg.lstsq(lib.design(X0, i), Y[:, i], rcond=None)[0])

    def field(self, X):
        F = np.zeros_like(X, dtype=float)
        for i in range(self.lib.N):
            if self.lib.ncol[i]:
                F[:, i] += self.lib.design(X, i) @ self.coef[i]
        return F

    def flat(self):
        offs = np.cumsum([0] + [len(c) for c in self.coef])
        return np.concatenate(self.coef) if offs[-1] else np.zeros(0), offs


def reconstruct(lib: LocalLib, X0, X1, T, estimator: str = "flow", iters: int = 3, h: float = 1e-4):
    """Static field whose flow over T maps X0 to X1.  'flow': iterated flow-consistent correction (Y0 minus the
    finite-flow terms T/2 (DF F) + T^2/6 (D(DF F) F)); 'forward': the secant (X1 - X0)/T.
    Returns the fit and the residual RMS per node of the final regression."""
    Y0 = (X1 - X0) / T; mod = Fit(lib, X0, Y0); Yc = Y0
    if estimator == "flow":
        for _ in range(iters):
            F = mod.field(X0)
            a1 = (mod.field(X0 + h * F) - mod.field(X0 - h * F)) / (2 * h)

            def a1f(Z):
                Fz = mod.field(Z); return (mod.field(Z + h * Fz) - mod.field(Z - h * Fz)) / (2 * h)
            a2 = (a1f(X0 + h * F) - a1f(X0 - h * F)) / (2 * h)
            Yc = Y0 - T / 2 * a1 - T ** 2 / 6 * a2; mod = Fit(lib, X0, Yc)
    res = Yc - mod.field(X0)
    return mod, np.sqrt((res ** 2).mean(0))


class _FlowModel:
    """Vectorised static library field F(x; theta), its state Jacobian and its parameter derivative, for the
    flow-matching estimator.  theta is the concatenation of the per-node coefficient blocks.  All columns of all
    nodes are evaluated at once: A = X @ D_all, sin on the sine columns, cos on the cosine columns, and the columns
    are summed onto their nodes by a one-hot matrix."""

    def __init__(self, lib: LocalLib):
        self.lib = lib; N = lib.N
        offs = np.cumsum([0] + [lib.ncol[i] for i in range(N)]); self.offs = offs; self.P = int(offs[-1])
        self.D = np.zeros((N, self.P)); self.is_sin = np.zeros(self.P, bool); self.node_of = np.zeros(self.P, int)
        for i in range(N):
            if lib.ncol[i]:
                D, S = lib._kprep(i); sl = slice(offs[i], offs[i + 1])
                self.D[:, sl] = D; self.is_sin[sl] = S; self.node_of[sl] = i
        self.Is, self.Ic = np.where(self.is_sin)[0], np.where(~self.is_sin)[0]
        self.Ds, self.Dc = np.ascontiguousarray(self.D[:, self.Is]), np.ascontiguousarray(self.D[:, self.Ic])
        self.G = np.zeros((self.P, N)); self.G[np.arange(self.P), self.node_of] = 1.0
        self.blocks = [(i, slice(offs[i], offs[i + 1])) for i in range(N) if lib.ncol[i]]
        self.cols = np.arange(self.P)

    def _basis(self, X, derivative=False):
        As, Ac = X @ self.Ds, X @ self.Dc
        f = np.empty((X.shape[0], self.P)); f[:, self.Is] = np.sin(As); f[:, self.Ic] = np.cos(Ac)
        if not derivative:
            return f, None
        fp = np.empty_like(f); fp[:, self.Is] = np.cos(As); fp[:, self.Ic] = -np.sin(Ac)
        return f, fp

    def field(self, X, theta):
        """Fast path: two products, no column indexing."""
        Gs = self.G[self.Is] * theta[self.Is, None]; Gc = self.G[self.Ic] * theta[self.Ic, None]
        return np.sin(X @ self.Ds) @ Gs + np.cos(X @ self.Dc) @ Gc

    def field_and_derivatives(self, X, theta):
        """F (R, N), state Jacobian J (R, N, N) and the basis f (R, P): dF_i/dtheta_p = f_p for p in block i."""
        f, fp = self._basis(X, derivative=True); W = fp * theta
        F = (f * theta) @ self.G; J = np.zeros((X.shape[0], self.lib.N, self.lib.N))
        for i, sl in self.blocks:
            J[:, i, :] = W[:, sl] @ self.D[:, sl].T
        return F, J, f

    def flow_with_sensitivity(self, X0, theta, T, n):
        """Integrate x' = F(x; theta) and S' = J S + dF/dtheta (S = dx/dtheta, S(0) = 0) with RK4."""
        R, N = X0.shape; h = T / n; x = X0.copy(); S = np.zeros((R, N, self.P))

        def rhs(x, S):
            F, J, f = self.field_and_derivatives(x, theta)
            dS = np.matmul(J, S); dS[:, self.node_of, self.cols] += f
            return F, dS
        for _ in range(n):
            k1x, k1S = rhs(x, S); k2x, k2S = rhs(x + h / 2 * k1x, S + h / 2 * k1S)
            k3x, k3S = rhs(x + h / 2 * k2x, S + h / 2 * k2S); k4x, k4S = rhs(x + h * k3x, S + h * k3S)
            x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x); S = S + h / 6 * (k1S + 2 * k2S + 2 * k3S + k4S)
        return x, S

    def flow(self, X0, theta, T, n):
        return rk4_flow(lambda X: self.field(X, theta), X0, T, n)


def flow_steps(T: float, rate: float = 3.0, h_rate: float = 0.05, minimum: int = 8) -> int:
    """RK4 steps for a flow over T when fields change states at most ~rate per unit time: h * rate <= h_rate."""
    return max(minimum, int(np.ceil(T * rate / h_rate)))


def reconstruct_defect(lib: LocalLib, X0, X1, T, *, init: Fit | None = None, n_steps: int | None = None,
                       max_iter: int = 400, tol: float = 1e-9, anderson: int = 0):
    """Flow matching by defect correction, with Anderson acceleration.

    Fixed-point map on the coefficients:  G(theta) = theta + pinv(A) (X1 - flow_T(X0; theta)) / T,  A the library
    design at X0 (per node).  Its fixed point makes the endpoint defect orthogonal to the library at X0; when the
    window map is the flow of a field in the library (e.g. a single stage), that field is a fixed point, recovered at
    any window length.  Anderson mixing over the last `anderson` iterates (default 0, plain iteration: unguarded
    Anderson mixing oscillated on long windows).  The best iterate is kept.  One flow per
    iteration, no sensitivities.  Returns the fit, the endpoint residual RMS per node and the history of the mean
    squared defect."""
    model = _FlowModel(lib); n_steps = flow_steps(T) if n_steps is None else n_steps
    fit = init if init is not None else Fit(lib, X0, (X1 - X0) / T)
    pinv = [np.linalg.pinv(lib.design(X0, i)) if lib.ncol[i] else None for i in range(lib.N)]

    def step(theta):
        d = X1 - model.flow(X0, theta, T, n_steps)
        upd = np.concatenate([pinv[i] @ d[:, i] / T if pinv[i] is not None else np.zeros(0) for i in range(lib.N)])
        return upd, float(np.mean(d ** 2))
    theta = np.concatenate(fit.coef); hist, dTh, dF = [], [], []
    best_theta, best = theta, np.inf
    f_prev = th_prev = None
    for it in range(max_iter):
        f, msd = step(theta); hist.append(msd)
        if msd < best:
            best, best_theta = msd, theta
        if len(hist) > 3 and max(hist[-4:]) - min(hist[-4:]) <= tol * max(hist[0], 1e-300):
            break
        if anderson and f_prev is not None:
            dTh.append(theta - th_prev); dF.append(f - f_prev)
            dTh, dF = dTh[-anderson:], dF[-anderson:]
            Fm = np.column_stack(dF); g = np.linalg.lstsq(Fm, f, rcond=None)[0]
            new = theta + f - (np.column_stack(dTh) + Fm) @ g
        else:
            new = theta + f
        th_prev, f_prev, theta = theta, f, new
    theta = best_theta
    fit = Fit(lib, None, None, coef=[theta[model.offs[i]:model.offs[i + 1]] for i in range(lib.N)])
    res = model.flow(X0, theta, T, n_steps) - X1
    return fit, np.sqrt((res ** 2).mean(0)), np.array(hist)


def reconstruct_matched(lib: LocalLib, X0, X1, T, *, init: Fit | None = None, n_steps: int | None = None,
                        max_iter: int = 40, tol: float = 1e-12):
    """For checks only (slow: forward sensitivities).  Flow matching: the static library field whose integrated
    flow over T best reproduces the observed endpoints
    (least squares on X1), by Levenberg-Marquardt with forward sensitivities.  Exact at any window length when the
    window map is the flow of a field in the library; otherwise its residual measures how far the window is from any
    static description in the library.  Returns the fit, the endpoint residual RMS per node and a history."""
    model = _FlowModel(lib); n_steps = flow_steps(T) if n_steps is None else n_steps
    theta = np.concatenate([np.asarray(c, float) for c in (init.coef if init is not None else Fit(lib, X0, (X1 - X0) / T).coef)])

    def cost_of(th):
        return float(np.sum((model.flow(X0, th, T, n_steps) - X1) ** 2))
    cost = cost_of(theta); lam = 1e-3; hist = [cost]
    for _ in range(max_iter):
        xT, S = model.flow_with_sensitivity(X0, theta, T, n_steps)
        r = (xT - X1).reshape(-1); Jm = S.reshape(-1, model.P)
        g = Jm.T @ r; H = Jm.T @ Jm; d = np.diag(H).copy(); d[d == 0] = 1.0
        improved = False
        for _ in range(12):
            step = np.linalg.solve(H + lam * np.diag(d), -g)
            new = theta + step; c_new = cost_of(new)
            if np.isfinite(c_new) and c_new < cost:
                theta, rel = new, (cost - c_new) / max(cost, 1e-300); cost = c_new; lam = max(lam / 3, 1e-12)
                improved = True; break
            lam *= 4
        hist.append(cost)
        if not improved or rel < tol:
            break
    fit = Fit(lib, None, None, coef=[theta[model.offs[i]:model.offs[i + 1]] for i in range(lib.N)])
    res = model.flow(X0, theta, T, n_steps) - X1
    return fit, np.sqrt((res ** 2).mean(0)), np.array(hist)


def design_diagnostics(lib: LocalLib, X0) -> dict:
    """Per node: number of columns, rank and condition number of the design on the samples X0."""
    out = dict(ncol=[], rank=[], cond=[])
    for i in range(lib.N):
        if lib.ncol[i] == 0:
            out["ncol"].append(0); out["rank"].append(0); out["cond"].append(float("nan")); continue
        s = np.linalg.svd(lib.design(X0, i), compute_uv=False)
        out["ncol"].append(int(lib.ncol[i])); out["rank"].append(int(np.sum(s > s[0] * 1e-10)))
        out["cond"].append(float(s[0] / s[-1]) if s[-1] > 0 else float("inf"))
    return out


# ---------------------------------------------------------------------------------------------------------------
# leading-order (BCH degree 2) effective field and its projection onto the library
# ---------------------------------------------------------------------------------------------------------------

def bch2_field(E: list, tau: float, h: float = 1e-5):
    """G2(x) = Fbar(x) + tau^2 / (2 T) sum_{s<t} (DF_t F_s - DF_s F_t)(x),  T = len(E) tau, stage s acting first.
    Matches the flow of the composed stages to O(T^3) (checked against direct integration)."""
    T = len(E) * tau; Eb = time_average(E)

    def jvp(E_t, X, V):                                   # DF_t(X) V by central differences
        return (stage_field(X + h * V, E_t) - stage_field(X - h * V, E_t)) / (2 * h)

    def G(X):
        out = stage_field(X, Eb)
        Fs = [stage_field(X, E_s) for E_s in E]
        corr = np.zeros_like(X)
        for s in range(len(E)):
            for t in range(s + 1, len(E)):
                corr += jvp(E[t], X, Fs[s]) - jvp(E[s], X, Fs[t])
        return out + tau * tau / (2 * T) * corr
    return G


def project(lib: LocalLib, field, X) -> Fit:
    """Least-squares projection of a field onto the library on the states X (exact when the field is in the span)."""
    return Fit(lib, X, field(X))


# ---------------------------------------------------------------------------------------------------------------
# triad analysis on a canonical triad (0, 1, 2)
# ---------------------------------------------------------------------------------------------------------------

def _kcol(u, vec_dict):
    p1, p2 = [q for q in range(3) if q != u]
    v = (vec_dict.get(u, 0), vec_dict.get(p1, 0), vec_dict.get(p2, 0))
    for k, d in enumerate(FULL_DIRS):
        if v == d: return k, 1
        if v == tuple(-x for x in d): return k, -1
    raise ValueError(v)


def sym_operator() -> np.ndarray:
    dim = 3 * NT; S = np.zeros((dim, dim))
    for pi in PERMS:
        M = np.zeros((dim, dim))
        for u in range(3):
            p1, p2 = [q for q in range(3) if q != u]
            for c in range(NT):
                src = u * NT + c
                d, sn = FULL_DIRS[c // 2], c % 2 == 0
                newd = {pi[u]: d[0], pi[p1]: d[1], pi[p2]: d[2]}; k, sg = _kcol(pi[u], newd)
                M[pi[u] * NT + 2 * k + (0 if sn else 1), src] = sg if sn else 1
        S += M / 6
    return S


def native_pattern() -> np.ndarray:
    P = np.zeros(3 * NT)
    for u in range(3):
        P[u * NT + 0] = 1.0                                       # sin of (-2, 1, 1): sin(x_j + x_k - 2 x_i)
    return P


def wedge_pattern(A, Bc, C) -> np.ndarray:
    """Three-body content of [X_e, X_f], e = (A, B) acting first, f = (B, C), on the canonical triad."""
    P = np.zeros(3 * NT)
    for u, dct, coef in ((A, {A: 1, Bc: -2, C: 1}, -0.5), (Bc, {Bc: 0, A: -1, C: 1}, 1.0), (C, {C: 1, A: 1, Bc: -2}, 0.5)):
        k, sg = _kcol(u, dct); P[u * NT + 2 * k] += coef * sg
    return P


def triad_vector(lib: LocalLib, fit: Fit, h) -> np.ndarray:
    v = np.zeros(3 * NT)
    for pos, u in enumerate(h):
        others = tuple(q for q in h if q != u); s = lib.tri_slot[u].get(others)
        if s is None or lib.ncol[u] == 0:
            continue
        v[pos * NT:(pos + 1) * NT] = fit.coef[u][s:s + NT]
    return v


def analyse_triads(lib: LocalLib, fit: Fit, E: list, tau: float, a: float = 1.0) -> dict:
    """Per candidate triad: raw norm, symmetric norm, matched native estimate, and the readout of the net order of
    each aggregated wedge from the antisymmetric part (c_hat in units of tau^2, comparable with net_order(E))."""
    S, Pn = sym_operator(), native_pattern(); Pn2 = Pn @ Pn; T = len(E) * tau
    tri, vec, svec, raw, sym, nat = [], [], [], [], [], []
    for h in lib.all_triads:
        v = triad_vector(lib, fit, h); sv = S @ v
        tri.append(h); vec.append(v); svec.append(sv)
        raw.append(np.linalg.norm(v)); sym.append(np.linalg.norm(sv)); nat.append(sv @ Pn / Pn2)
    c_true_all = net_order(E); keys, c_true, c_hat, kinds = [], [], [], []
    for n, h in enumerate(tri):
        pos = {node: m for m, node in enumerate(h)}; pats, kk = [], []
        for v in h:
            u, w = [q for q in h if q != v]
            e, f = tuple(sorted((u, v))), tuple(sorted((v, w)))
            if e in lib.agg and f in lib.agg:
                if e > f: e, f = f, e
                A = [q for q in e if q != v][0]; C = [q for q in f if q != v][0]
                pats.append(wedge_pattern(pos[A], pos[v], pos[C])); kk.append((e, f))
        if not pats:
            continue
        P = np.column_stack(pats)
        if np.linalg.matrix_rank(P) < P.shape[1]:
            continue
        coef = np.linalg.lstsq(P, vec[n] - svec[n], rcond=None)[0]
        for (e, f), cf in zip(kk, coef):
            keys.append([*e, *f]); c_true.append(c_true_all.get((e, f), 0.0))
            c_hat.append(float(2 * T * cf / (tau * tau * a * a))); kinds.append(len(kk))
    return dict(triads=np.array(tri, int).reshape(-1, 3), vec=np.array(vec).reshape(-1, 3 * NT),
                svec=np.array(svec).reshape(-1, 3 * NT), raw=np.array(raw), sym=np.array(sym), nat=np.array(nat),
                readout_keys=np.array(keys, int).reshape(-1, 4), c_true=np.array(c_true), c_hat=np.array(c_hat),
                readout_wedges_in_triad=np.array(kinds, int))


# ---------------------------------------------------------------------------------------------------------------
# sub-windows of one window
# ---------------------------------------------------------------------------------------------------------------

def subwindow_stages(n_stages: int, k: int) -> list:
    if n_stages % k:
        raise ValueError(f"k = {k} does not divide the {n_stages} stages")
    L = n_stages // k
    return [list(range(j * L, (j + 1) * L)) for j in range(k)]


def run_subwindows(protocol, X_boundaries: np.ndarray, *, tau: float, k: int, X_test: np.ndarray | None = None,
                   n_project: int = 4000, project_seed: int = 0, matched_kwargs: dict | None = None) -> dict:
    """Reconstruct each of the k sub-windows from boundary states (R, K+1, N) with two estimators and compose their
    flows.

        flow      the perturbative flow-consistent estimator of the TSC paper (finite-flow correction to O(T^2))
        matched   flow matching by defect correction, started from 'flow': valid at any window length

    Returns per sub-window: the library spec, the coefficients of both estimators and the BCH-2 reference
    (flattened, with offsets), endpoint residuals, the defect history, design diagnostics and the triad analyses;
    and, if X_test is given, the composed prediction of the whole window from the test states for each estimator,
    together with the naive time-average prediction."""
    K, N = protocol.n_stages, protocol.N
    subs = subwindow_stages(K, k); L = K // k; T_sub = L * tau
    out = dict(k=k, tau=tau, T_sub=T_sub, subwindows=[])
    models = dict(flow=[], matched=[])
    rng = np.random.default_rng(project_seed)
    for j, st in enumerate(subs):
        E = stages_of(protocol, st)
        X0, X1 = X_boundaries[:, j * L], X_boundaries[:, (j + 1) * L]
        lib = LocalLib(N, sorted(set().union(*[set(E_s) for E_s in E])))
        fm = _FlowModel(lib); n_steps = flow_steps(T_sub)
        if not np.isfinite(X1).all():
            raise ValueError("non-finite boundary states")
        f_flow, _ = reconstruct(lib, X0, X1, T_sub, estimator="flow")
        # start the flow matching from the candidate whose flow lands closest to the observed endpoints
        Xp = rng.uniform(-np.pi, np.pi, (n_project, N))
        starts = dict(flow=f_flow, secant=Fit(lib, X0, (X1 - X0) / T_sub),
                      time_average=project(lib, lambda X: stage_field(X, time_average(E)), Xp))
        start_defect = {name: float(np.mean((fm.flow(X0, np.concatenate(f.coef), T_sub, n_steps) - X1) ** 2))
                        for name, f in starts.items()}
        start = min(start_defect, key=start_defect.get)
        f_m, rms_m, hist = reconstruct_defect(lib, X0, X1, T_sub, init=starts[start], **(matched_kwargs or {}))
        rms_flow = np.sqrt(((fm.flow(X0, np.concatenate(f_flow.coef), T_sub, n_steps) - X1) ** 2).mean(0))
        ref = project(lib, bch2_field(E, tau), rng.uniform(-np.pi, np.pi, (n_project, N)))
        sw = dict(stages=st, lib=lib.spec(), design=design_diagnostics(lib, X0), coef_offsets=f_flow.flat()[1],
                  coef_bch2=ref.flat()[0], triads_bch2=analyse_triads(lib, ref, E, tau), defect_history=hist,
                  matched_start=start, start_defects=start_defect)
        for name, f, rms in (("flow", f_flow, rms_flow), ("matched", f_m, rms_m)):
            sw[f"coef_{name}"] = f.flat()[0]; sw[f"endpoint_rms_{name}"] = rms
            sw[f"triads_{name}"] = analyse_triads(lib, f, E, tau)
            models[name].append((fm, np.concatenate(f.coef)))
        out["subwindows"].append(sw)
    if X_test is not None:
        for name, mods in models.items():
            Z = np.array(X_test, float)
            for fm, theta in mods:
                Z = fm.flow(Z, theta, T_sub, flow_steps(T_sub))
            out[f"X_pred_test_{name}"] = Z
        Eb = time_average(stages_of(protocol, range(K)))
        out["X_naive_test"] = rk4_flow(lambda X: stage_field(X, Eb), np.array(X_test, float), K * tau, n=flow_steps(K * tau))
    return out
