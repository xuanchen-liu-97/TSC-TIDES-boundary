"""ttboundary: experiments on the regime boundaries of TSC (coarse, repeated observations) and TIDES (fine, single
trajectories) -- where each method holds, where it fails, and what data it needs to cross."""
from .protocol import Protocol, make_protocol, kuramoto_law
from .simulate import draw_initial_states, simulate_batch, stage_diagnostics

__all__ = ["Protocol", "make_protocol", "kuramoto_law", "draw_initial_states", "simulate_batch", "stage_diagnostics"]
__version__ = "0.1.0"
