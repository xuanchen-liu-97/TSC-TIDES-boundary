# TSC-TIDES-boundary

Experiments on the regime boundaries of TSC (coarse, repeated observations) and TIDES (fine, single trajectories):
where each method holds, where it fails, and what data it needs to cross.

Package: `ttboundary`

    ttboundary/protocol.py    hidden protocol (TIDES-identical draws), stage views, truth for scoring (net order c_ef)
    ttboundary/simulate.py    vectorised RK4 over many initial states; boundary states (TSC) and dense trajectories (TIDES)
    ttboundary/storage.py     atomic npz/json saving, so interrupted jobs can be resumed
    experiments/exp01_three_durations/
        config.json           all parameters of experiment 01
        run_simulate.py       integrates every duration once (clean states; noise is added at analysis time)

Quick check:

    cd experiments/exp01_three_durations
    python run_simulate.py --quick
