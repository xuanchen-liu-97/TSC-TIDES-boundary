"""The hidden protocol of a switching network, shared by TSC and TIDES.

A protocol is a sequence of interaction configurations (stages) on N nodes.  Every candidate edge carries one fixed
weight, used whenever the edge is active; stages differ by which edges are active.  The dynamics in stage r is

    dx_i/dt = sum_j w_ij^(r) u(x_j - x_i),      u odd (here the Kuramoto law u(d) = sin d),

the model class of the TIDES benchmarks.  `make_protocol` reproduces the random draws of
`tides_benchmarks.make_benchmark(mode="structural")` exactly (weights first, then the first snapshot, then the
rewirings), so a protocol built here with seed s is the protocol of the TIDES structural benchmark with seed s.

Conventions
    nodes            1-based, as in TIDES (`candidate_edges` in combinations order)
    stage_weights()  TSC format: one dict per stage, active edge -> weight
    net_order()      c_ef in units of tau^2 (weights multiplied), keyed (e, f) with e < f, positive when e acts
                     first -- the convention of the TSC synthetic experiments (`tsc_synth60.imbalance`)
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np


def kuramoto_law(shape: float = 0.0):
    """u(d) = (sin d + a sin 2d) / (1 + 2a); a = 0 is the Kuramoto law (unit slope at d = 0)."""
    a = float(shape)
    if a == 0.0:
        return np.sin
    return lambda d: (np.sin(d) + a * np.sin(2.0 * d)) / (1.0 + 2.0 * a)


# ---------------------------------------------------------------------------------------------------------------
# random networks with controlled change (identical to tides_benchmarks)
# ---------------------------------------------------------------------------------------------------------------

def _connected(edges, N) -> bool:
    adj = {v: set() for v in range(1, N + 1)}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    seen, stack = {1}, [1]
    while stack:
        u = stack.pop()
        for v in adj[u] - seen:
            seen.add(v); stack.append(v)
    return len(seen) == N


def _random_snapshot(N, mean_degree, rng, max_tries=1000):
    E = int(round(mean_degree * N / 2))
    pairs = list(combinations(range(1, N + 1), 2))
    for _ in range(max_tries):
        pick = rng.choice(len(pairs), size=E, replace=False)
        edges = tuple(sorted(pairs[k] for k in pick))
        if _connected(edges, N):
            return edges
    raise RuntimeError("could not draw a connected snapshot; raise mean_degree")


def _rewire(snapshot, n_changes, N, rng, max_tries=1000):
    """Remove n_changes/2 active edges and add as many inactive ones, keeping the graph connected."""
    k = n_changes // 2
    pairs = list(combinations(range(1, N + 1), 2))
    active = list(snapshot); inactive = [p for p in pairs if p not in set(snapshot)]
    for _ in range(max_tries):
        out = set(rng.choice(len(active), size=k, replace=False))
        new = set(rng.choice(len(inactive), size=k, replace=False))
        edges = tuple(sorted([e for m, e in enumerate(active) if m not in out] + [inactive[m] for m in new]))
        if _connected(edges, N):
            return edges
    raise RuntimeError("could not rewire while keeping connectivity")


# ---------------------------------------------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Protocol:
    N: int
    candidate_edges: tuple            # 1-based node pairs, combinations order
    edge_weight: dict                 # edge -> weight (fixed per edge, used whenever the edge is active)
    snapshots: tuple                  # per stage: sorted tuple of active edges
    law: str = "kuramoto"
    shape: float = 0.0
    meta: dict = field(default_factory=dict)

    # ---- basic views
    @property
    def n_stages(self) -> int:
        return len(self.snapshots)

    def u(self):
        if self.law != "kuramoto":
            raise ValueError("only the Kuramoto family is implemented")
        return kuramoto_law(self.shape)

    def stage_weights(self) -> list:
        """TSC format: per stage, {active edge: weight}."""
        return [{e: self.edge_weight[e] for e in snap} for snap in self.snapshots]

    def stage_matrices(self) -> np.ndarray:
        """(K, N, N) symmetric weight matrices, 0-based indices."""
        W = np.zeros((self.n_stages, self.N, self.N))
        for r, snap in enumerate(self.snapshots):
            for (a, b) in snap:
                W[r, a - 1, b - 1] = W[r, b - 1, a - 1] = self.edge_weight[(a, b)]
        return W

    def stage_arrays(self, r: int):
        """Endpoint indices (0-based) and weights of the active edges of stage r."""
        snap = self.snapshots[r]
        i = np.array([e[0] - 1 for e in snap], int)
        j = np.array([e[1] - 1 for e in snap], int)
        w = np.array([self.edge_weight[e] for e in snap], float)
        return i, j, w

    # ---- truth used to score TSC
    def net_order(self) -> dict:
        """c_ef in units of tau^2 for every pair of edges sharing exactly one node, keyed (e, f) with e < f,
        positive when e acts first (stages of equal duration)."""
        E = self.stage_weights(); c = defaultdict(float)
        for s in range(len(E)):
            for t in range(s + 1, len(E)):
                for e, we in E[s].items():
                    for f, wf in E[t].items():
                        if len(set(e) & set(f)) != 1:
                            continue
                        if e < f: c[(e, f)] += we * wf
                        else: c[(f, e)] -= we * wf
        return {k: v for k, v in c.items() if abs(v) > 1e-14}

    def unbalanced_wedges(self) -> dict:
        """Triads carrying a nonzero net order: {triad (sorted 1-based nodes): [(e, f, c_ef), ...]}."""
        out = defaultdict(list)
        for (e, f), c in self.net_order().items():
            out[tuple(sorted(set(e) | set(f)))].append((e, f, c))
        return dict(out)

    def time_average(self) -> dict:
        """Naive aggregate (stages of equal duration): edge -> mean weight over the stages."""
        Eb = defaultdict(float)
        for snap in self.snapshots:
            for e in snap:
                Eb[e] += self.edge_weight[e] / self.n_stages
        return dict(Eb)

    def interaction_per_stage(self, stage_duration: float) -> dict:
        """How much interaction a stage carries: per edge (w tau) and per node (weighted degree times tau)."""
        W = self.stage_matrices()
        per_edge = [stage_duration * np.mean([self.edge_weight[e] for e in s]) for s in self.snapshots]
        per_node = [stage_duration * float(np.mean(W[r].sum(1))) for r in range(self.n_stages)]
        return dict(per_edge=float(np.mean(per_edge)), per_node=float(np.mean(per_node)),
                    per_node_window=float(np.sum(per_node)))

    # ---- serialisation
    def to_dict(self) -> dict:
        return dict(N=self.N, law=self.law, shape=self.shape, meta=self.meta,
                    candidate_edges=[list(e) for e in self.candidate_edges],
                    edge_weight=[[a, b, w] for (a, b), w in self.edge_weight.items()],
                    snapshots=[[list(e) for e in s] for s in self.snapshots])

    @classmethod
    def from_dict(cls, d: dict) -> "Protocol":
        return cls(N=int(d["N"]), law=d["law"], shape=float(d["shape"]), meta=dict(d.get("meta", {})),
                   candidate_edges=tuple(tuple(e) for e in d["candidate_edges"]),
                   edge_weight={(int(a), int(b)): float(w) for a, b, w in d["edge_weight"]},
                   snapshots=tuple(tuple(tuple(e) for e in s) for s in d["snapshots"]))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path) -> "Protocol":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def truth(self) -> dict:
        """Everything used later to score reconstructions, in JSON-friendly form."""
        return dict(net_order=[[list(e), list(f), c] for (e, f), c in sorted(self.net_order().items())],
                    unbalanced_triads=[list(t) for t in sorted(self.unbalanced_wedges())],
                    time_average=[[list(e), w] for e, w in sorted(self.time_average().items())],
                    native_triads=[])                       # purely pairwise protocol: no native hyperedges


def make_protocol(*, N: int = 8, mean_degree: float = 2.5, n_stages: int = 6, edge_changes: int = 4,
                  weight_range=(0.8, 1.2), law: str = "kuramoto", shape: float = 0.0, seed: int = 1000) -> Protocol:
    """Structural protocol: weights and topology drawn exactly as tides_benchmarks.make_benchmark(mode='structural')."""
    rng = np.random.default_rng(seed)
    cand = tuple(combinations(range(1, N + 1), 2))
    edge_weight = {e: float(rng.uniform(*weight_range)) for e in cand}
    snaps = [_random_snapshot(N, mean_degree, rng)]
    for _ in range(n_stages - 1):
        snaps.append(_rewire(snaps[-1], edge_changes, N, rng))
    return Protocol(N=N, candidate_edges=cand, edge_weight=edge_weight, snapshots=tuple(snaps), law=law,
                    shape=float(shape), meta=dict(seed=seed, mean_degree=mean_degree, n_stages=n_stages,
                                                  edge_changes=edge_changes, weight_range=list(weight_range)))
