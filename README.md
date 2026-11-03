# TSC-TIDES-boundary

Experiments on the regime boundaries of TSC (coarse, repeated observations) and TIDES (fine, single trajectories):
where each method holds, where it fails, and what data it needs to cross.

Package: `ttboundary`

    ttboundary/protocol.py    hidden protocol (TIDES-identical draws), stage views, truth for scoring (net order c_ef)
    ttboundary/simulate.py    vectorised RK4 over many initial states; boundary states (TSC) and dense trajectories (TIDES)
    ttboundary/storage.py     atomic npz/json saving, so interrupted jobs can be resumed
    ttboundary/tsc.py         TSC reconstruction (ported from tsc_synth60.py) on k sub-windows, two estimators:
                                flow     the perturbative flow-consistent estimator of the TSC paper
                                matched  flow matching by defect correction (valid at any window length), started
                                         from the best of {flow, secant, projected time average}
                              plus the leading-order (BCH-2) reference projected onto the library, triad analysis
                              (raw / symmetric / native estimate) and the chirality readout of the net order
    experiments/exp01_three_durations/
        config.json           all parameters of experiment 01
        run_simulate.py       step 1: integrates every duration once (clean states; noise is added at analysis time)
        run_tsc.py            step 2: one task per (duration, k, noise, set); records everything, scores nothing
        check_protocol.py     the protocol equals the TIDES structural benchmark; same trajectories as TIDES
        inspect_tsc.py        first-look summary table of the TSC results

Run (16 cores):

    cd experiments/exp01_three_durations
    python run_simulate.py                      # < 2 min
    python run_tsc.py --workers 16              # resumable: finished tasks are skipped on resubmission

Smoke test:

    python run_simulate.py --quick
    python run_tsc.py --quick --workers 2 --max-iter 60

Outputs of run_tsc.py: data/tsc/<duration>/noise<level>/k<k>/set<ss>.npz (arrays) and .json (metadata; written
last, so its presence marks the task as done).  Prefix w<j>_ = sub-window j; triads_<estimator>_<field> for
estimator in {flow, matched, bch2}.
