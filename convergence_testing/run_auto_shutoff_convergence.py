"""
Auto-shutoff convergence testing — Pi-Shift Bragg Grating FDTD.

Sweeps the FDTD `auto shutoff min` (field-energy threshold for early termination)
and reports Q convergence. Same convergence rule as run_mesh_convergence.py:
pick the coarsest value where the metric stabilizes within THRESHOLD for two
consecutive (coarse → fine) values.

Hard requirement: every task must terminate via auto-shutoff (status==2),
NOT by hitting the simulation-time limit. Tasks that hit the time limit are
marked invalid and excluded from the convergence rule. If that happens, bump
SIM_TIME_S and re-run only the affected values — interpreting a timed-out
row as converged is wrong (the field never actually decayed to that level).

Single-phase sweep (one parameter), so no aggregator dependency job —
local `--aggregate SHUTOFF` after Athena array completion is sufficient.

Usage:
    # Local sequential sweep
    python convergence_testing/run_auto_shutoff_convergence.py

    # Aggregate Athena array per-task results into the checkpoint
    python convergence_testing/run_auto_shutoff_convergence.py --aggregate SHUTOFF
"""

# PHASES / KIND_PREFIX: read by athena/deploy_dgx.sh's Convergence picker.
# Single phase ⇒ one menu entry, submitted as a SLURM array with
# SWEEP_KIND=shutoff_conv_shutoff.
PHASES      = ["SHUTOFF"]
KIND_PREFIX = "shutoff_conv"

# IS_HELPER: hide from the Single-runner picker (defensive — Single only scans
# runners/single/, but this stays consistent with run_mesh_convergence.py).
IS_HELPER = True

import argparse
import glob
import json
import os
import sys
import time

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# config is stdlib-safe (no scipy/lumapi) — needed for CONV_DIR at module level.
import config
# Simulation imports (bragg_device → analysis → scipy) deferred to inside
# _run_one / _make_cfg so --aggregate runs without triggering them.

# ═════════════════════════════════════════════════════════════════════════════
# User-editable settings
# ═════════════════════════════════════════════════════════════════════════════

# Convergence metric: "Q" (default — most sensitive to truncated field decay),
# "lambda", or "T". Q is the natural metric for a shutoff sweep because the
# auto-shutoff truncates the slow ringdown that determines FWHM (and hence Q).
CONVERGENCE_METRIC   = "Q"
N_PERIODS_EACH_SIDE  = 80      # device length; encoded in checkpoint filename

THRESHOLD_LAMBDA = 0.010
THRESHOLD_Q      = 0.020       # 2% — same as mesh convention
THRESHOLD_T      = 0.015

# Coarse → fine. The convergence rule walks left-to-right and picks the
# coarsest value whose two successors are within THRESHOLD. Add 1e-9 if
# 1e-8 hasn't converged (and bump SIM_TIME_S accordingly).
SHUTOFF_VALUES = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

# 2× current production (1000 ps in bragg_device.py). Long enough that the
# field reaches 1e-8 before the limit on the user's high-Q device. If any row
# reports status=TIMEOUT, raise this and re-run those values.
SIM_TIME_S = 2000e-12

# Pinned to current production mesh — fixing the mesh isolates the shutoff
# parameter as the only variable in the sweep.
DEFAULT_CELLS  = 5
DEFAULT_DYZ_NM = 50.0

# ═════════════════════════════════════════════════════════════════════════════
# Output paths
# ═════════════════════════════════════════════════════════════════════════════

CONV_DIR     = os.path.join(config.BASE_SAVE_DIR, "auto_shutoff_convergence")
LAYOUTS_DIR  = os.path.join(CONV_DIR, "layouts")
RESULTS_DIR  = os.path.join(CONV_DIR, "results")
CHECKPOINT   = os.path.join(CONV_DIR,
    f"checkpoint_p{N_PERIODS_EACH_SIDE}_{CONVERGENCE_METRIC}.json")

os.makedirs(LAYOUTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Simulation config
# ═════════════════════════════════════════════════════════════════════════════

def _make_cfg():
    """SimulationConfig tuned for convergence testing: no 2D/3D/far-field monitors."""
    from simulation_config import SimulationConfig
    cfg = SimulationConfig()
    cfg.apodization.enabled        = False
    cfg.monitors.record_2d_fields  = False
    cfg.monitors.record_3d_fields  = False
    cfg.farfield.enabled           = False
    cfg.run.cleanup_lumerical_data = True
    cfg.grating.n_periods_each_side = N_PERIODS_EACH_SIDE
    # Narrow scan: 20 nm captures the cavity peak without wasting DFT points.
    cfg.spectral.scan_width_nm = 20.0
    cfg.spectral.n_wl_points   = 1001
    return cfg


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}


def _save_checkpoint(data: dict) -> None:
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f, indent=2)


def _threshold_for(metric: str) -> float:
    return {"Q": THRESHOLD_Q, "lambda": THRESHOLD_LAMBDA, "T": THRESHOLD_T}[metric]


def _read_termination_status(fdtd):
    """Return (early_shutoff_triggered, status_code).

    Lumerical FDTD's runtime status code:
      1 = simulation time limit reached (auto-shutoff did NOT trigger — bad)
      2 = auto-shutoff triggered (good)
    Returns (None, None) if the status can't be read on this Lumerical version.
    """
    try:
        raw = fdtd.getresult("FDTD", "status")
        code = int(round(float(raw)))
        return (code == 2), code
    except Exception:
        return None, None


# ═════════════════════════════════════════════════════════════════════════════
# Single simulation runner
# ═════════════════════════════════════════════════════════════════════════════

def _run_one(cfg, shutoff_min: float, tag: str) -> dict:
    """
    Build and run one FDTD simulation with a swept `auto shutoff min`,
    extract Q and check whether auto-shutoff actually triggered.

    Returns a record with keys: auto_shutoff_min, sim_time_s_configured,
    cells, dyz_nm, dx_nm, Q, resonance_nm, fwhm_pm, transmission, sim_time_s,
    early_shutoff_triggered (bool|None), status_code (int|None), invalid (bool).
    """
    import matplotlib.pyplot as plt
    from bragg_device import PiShiftBraggFDTD
    from post_processing import extract_s_parameters, find_resonance
    from sim_helpers import apply_monitor_overrides

    class _ShutoffBraggFDTD(PiShiftBraggFDTD):
        # Override the hard-coded `auto shutoff min` and `simulation time` from
        # bragg_device.py:299-300 with the swept value. Lumerical `set` on the
        # FDTD object overwrites the prior call, so this cleanly replaces them.
        def __init__(self, *args, _shutoff_min=1e-6, _sim_time_s=SIM_TIME_S, **kwargs):
            self._shutoff_min = float(_shutoff_min)
            self._sim_time_s  = float(_sim_time_s)
            super().__init__(*args, **kwargs)

        def _add_fdtd_region(self):
            super()._add_fdtd_region()
            self.fdtd.set("auto shutoff min", self._shutoff_min)
            self.fdtd.set("simulation time", self._sim_time_s)

    half_pitch = cfg.grating.pitch_m / 2.0
    cells = int(DEFAULT_CELLS)
    dx_nm = half_pitch / cells * 1e9

    print(f"\n{'─' * 62}")
    print(f"  [{tag}]")
    print(f"  auto_shutoff_min = {shutoff_min:.0e}  sim_time = {SIM_TIME_S * 1e12:.0f} ps")
    print(f"  cells={cells} (dx={dx_nm:.1f} nm)  dyz={DEFAULT_DYZ_NM:.0f} nm")
    print(f"{'─' * 62}")

    device_kwargs = cfg.to_device_kwargs()
    device_kwargs["cells_per_half_period"] = cells
    sim = _ShutoffBraggFDTD(
        **device_kwargs,
        _shutoff_min=float(shutoff_min),
        _sim_time_s=float(SIM_TIME_S),
    )

    layout_path = os.path.join(LAYOUTS_DIR, f"layout_{tag}.fsp")

    sim.build()
    sim.update_scan(
        center_lambda_m=cfg.spectral.center_wavelength_m,
        width_nm=cfg.spectral.scan_width_nm,
        n_points=cfg.spectral.n_wl_points,
    )
    apply_monitor_overrides(sim, cfg)
    sim.fdtd.save(layout_path)

    t0 = time.perf_counter()
    sim.fdtd.run()
    elapsed = time.perf_counter() - t0
    print(f"  Simulation done in {elapsed:.1f} s wall")

    early_shutoff, status_code = _read_termination_status(sim.fdtd)

    try:
        s_params  = extract_s_parameters(sim, cfg)
        resonance = find_resonance(s_params)
        # Lumerical wavelengths come back in descending order (uniform in f),
        # so dw < 0 and find_resonance's FWHM is negative — abs() corrects.
        fwhm_m = abs(resonance.spectral_fwhm_m)
        Q = resonance.wavelength_m / fwhm_m
        print(f"  lambda_res = {resonance.wavelength_m * 1e9:.4f} nm  "
              f"FWHM = {fwhm_m * 1e12:.2f} pm  "
              f"Q = {Q:.2f}  "
              f"T_res = {resonance.transmission:.4f}")
        if early_shutoff is True:
            print(f"  ✓ auto-shutoff triggered (status={status_code})")
        elif early_shutoff is False:
            print(f"  ✗ INVALID — hit simulation time limit (status={status_code}). "
                  f"Bump SIM_TIME_S above {SIM_TIME_S * 1e12:.0f} ps and re-run.")
        else:
            print(f"  ? termination status unknown (could not read).")
    finally:
        plt.close("all")
        sim.close()

    return {
        "auto_shutoff_min":         float(shutoff_min),
        "sim_time_s_configured":    float(SIM_TIME_S),
        "cells":                    cells,
        "dyz_nm":                   float(DEFAULT_DYZ_NM),
        "dx_nm":                    round(dx_nm, 2),
        "Q":                        float(Q),
        "resonance_nm":             float(resonance.wavelength_m * 1e9),
        "fwhm_pm":                  float(fwhm_m * 1e12),
        "transmission":             float(resonance.transmission),
        "sim_time_s":               round(elapsed, 1),
        "early_shutoff_triggered":  early_shutoff,
        "status_code":              status_code,
        "invalid":                  (early_shutoff is False),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Sweep phase
# ═════════════════════════════════════════════════════════════════════════════

def _run_phase(values, cfg, checkpoint, metric, threshold) -> tuple:
    """
    Sweep auto_shutoff_min coarse → fine. Skip rows that hit the time limit
    (`invalid=True`) when applying the convergence rule.

    Returns (results_list, best_value).
    """
    print(f"\n{'=' * 62}")
    print(f"  Phase SHUTOFF: sweeping auto_shutoff_min  "
          f"[metric={metric}, threshold={threshold * 100:.1f}%]")
    print(f"  Periods={N_PERIODS_EACH_SIDE}  cells={DEFAULT_CELLS}  "
          f"dyz={DEFAULT_DYZ_NM:.0f} nm  sim_time={SIM_TIME_S * 1e12:.0f} ps")
    print(f"{'=' * 62}")

    results         = []
    prev_wl         = None
    prev_Q          = None
    prev_T          = None
    converged_count = 0
    best_value      = values[-1]   # fallback: finest

    for i, val in enumerate(values):
        ck_key = f"phSHUTOFF_{val:.0e}"

        if ck_key in checkpoint:
            print(f"\n  [SKIP] auto_shutoff_min={val:.0e} — loaded from checkpoint")
            rec = checkpoint[ck_key]
        else:
            tag = f"phSHUTOFF_{val:.0e}"
            rec = _run_one(cfg, val, tag)
            checkpoint[ck_key] = rec
            _save_checkpoint(checkpoint)

        results.append(rec)

        # A timed-out row is not a valid measurement of that shutoff value —
        # exclude it and reset the consecutive-stable counter.
        if rec.get("invalid"):
            print(f"  [convergence rule] skipping invalid (timed-out) row at {val:.0e}")
            converged_count = 0
            prev_wl = prev_Q = prev_T = None
            continue

        wl = rec["resonance_nm"]
        Q  = rec["Q"]
        T  = rec["transmission"]

        if prev_wl is not None:
            if metric == "Q":
                delta = abs(Q - prev_Q) / max(abs(prev_Q), 1e-12)
                label = f"|ΔQ/Q| = {delta * 100:.3f}%"
            elif metric == "T":
                delta = abs(T - prev_T) / max(abs(prev_T), 1e-12)
                label = f"|ΔT/T| = {delta * 100:.3f}%"
            else:
                delta = abs(wl - prev_wl) / max(abs(prev_wl), 1e-12)
                label = f"|Δλ/λ| = {delta * 100:.3f}%"

            if delta < threshold:
                converged_count += 1
                if converged_count >= 2:
                    best_value = values[i - 1]
                    print(f"\n  Converged at auto_shutoff_min={val:.0e}  ({label})")
                    print(f"  Using auto_shutoff_min = {best_value:.0e}  "
                          f"(coarser, already stable)")
                    break
            else:
                converged_count = 0

        prev_wl = wl
        prev_Q  = Q
        prev_T  = T
    else:
        print(f"\n  No convergence within sweep range. "
              f"Using finest value: auto_shutoff_min = {best_value:.0e}")

    return results, best_value


# ═════════════════════════════════════════════════════════════════════════════
# Results table
# ═════════════════════════════════════════════════════════════════════════════

def _print_table(results: list) -> None:
    sep = "─" * 116
    header = (f"  {'shutoff':<10} {'Q':>10} {'lambda(nm)':>12} {'FWHM(pm)':>10} "
              f"{'T_res':>7} {'ΔQ%':>10} {'cfg(ps)':>8} {'wall(s)':>8} {'status':>14}")
    print(f"\n{sep}")
    print(f"  Auto-shutoff convergence — periods={N_PERIODS_EACH_SIDE}, "
          f"cells={DEFAULT_CELLS}, dyz={DEFAULT_DYZ_NM:.0f} nm, "
          f"sim_time={SIM_TIME_S * 1e12:.0f} ps")
    print(sep)
    print(header)
    print(sep)
    prev_Q = None
    for r in results:
        es = r.get("early_shutoff_triggered")
        if es is True:
            status = "ok (shutoff)"
        elif es is False:
            status = "TIMEOUT"
        else:
            status = "unknown"
        dQ_str = "         —"
        if prev_Q is not None and not r.get("invalid"):
            dQ = (r["Q"] - prev_Q) / abs(prev_Q) * 100
            dQ_str = f"{dQ:+.4f}%"
        print(f"  {r['auto_shutoff_min']:<10.0e} {r['Q']:>10.2f} "
              f"{r['resonance_nm']:>12.4f} {r['fwhm_pm']:>10.2f} "
              f"{r['transmission']:>7.4f} {dQ_str:>10} "
              f"{r['sim_time_s_configured'] * 1e12:>8.0f} "
              f"{r['sim_time_s']:>8.1f} {status:>14}")
        if not r.get("invalid"):
            prev_Q = r["Q"]
    print(sep)
    if any(r.get("invalid") for r in results):
        print("\n  WARNING: one or more rows hit the simulation time limit.")
        print(f"           Bump SIM_TIME_S above {SIM_TIME_S * 1e12:.0f} ps "
              f"and re-run those values.")


# ═════════════════════════════════════════════════════════════════════════════
# Aggregator: fold Athena array per-task JSONs into the checkpoint
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_array_parts(phase: str = "SHUTOFF") -> dict:
    """
    Fold per-task Athena results from
    auto_shutoff_convergence/array_part_phSHUTOFF_*.json into the checkpoint.
    Run this locally after pulling Athena results.
    """
    phase = phase.upper()
    if phase != "SHUTOFF":
        raise ValueError(f"phase must be 'SHUTOFF', got {phase!r}")

    pattern = os.path.join(CONV_DIR, f"array_part_ph{phase}_*.json")
    parts = sorted(glob.glob(pattern))
    if not parts:
        print(f"No array_part files matched: {pattern}")
        return _load_checkpoint()

    checkpoint = _load_checkpoint()
    threshold  = _threshold_for(CONVERGENCE_METRIC)

    print(f"\nAggregating {len(parts)} Phase {phase} part files into checkpoint")
    print(f"  checkpoint: {CHECKPOINT}")

    records = []
    for p in parts:
        with open(p) as f:
            rec = json.load(f)
        val = float(rec["auto_shutoff_min"])
        ck_key = f"ph{phase}_{val:.0e}"
        checkpoint[ck_key] = rec
        records.append((val, rec))
        es = rec.get("early_shutoff_triggered")
        flag = "ok" if es is True else ("TIMEOUT" if es is False else "?")
        print(f"  [{phase}] {ck_key:<22s} Q={rec['Q']:>8.2f}  "
              f"λ={rec['resonance_nm']:.4f} nm  [{flag}]  ({os.path.basename(p)})")

    # Coarse → fine for the convergence rule (largest shutoff value first).
    records.sort(key=lambda kv: -float(kv[0]))

    converged_count = 0
    best_value      = records[-1][0]
    prev_wl = prev_Q = prev_T = None
    for i, (val, rec) in enumerate(records):
        if rec.get("invalid"):
            converged_count = 0
            prev_wl = prev_Q = prev_T = None
            continue
        wl = rec["resonance_nm"]
        Q  = rec["Q"]
        T  = rec["transmission"]
        if prev_wl is not None:
            if CONVERGENCE_METRIC == "Q":
                delta = abs(Q - prev_Q) / max(abs(prev_Q), 1e-12)
            elif CONVERGENCE_METRIC == "T":
                delta = abs(T - prev_T) / max(abs(prev_T), 1e-12)
            else:
                delta = abs(wl - prev_wl) / max(abs(prev_wl), 1e-12)
            if delta < threshold:
                converged_count += 1
                if converged_count >= 2:
                    best_value = records[i - 1][0]
                    break
            else:
                converged_count = 0
        prev_wl = wl
        prev_Q  = Q
        prev_T  = T

    checkpoint["best_auto_shutoff_min"] = float(best_value)
    print(f"\n  best_auto_shutoff_min = {best_value:.0e}")

    _save_checkpoint(checkpoint)
    print(f"  Wrote {CHECKPOINT}")
    return checkpoint


# ═════════════════════════════════════════════════════════════════════════════
# Main (local sequential sweep)
# ═════════════════════════════════════════════════════════════════════════════

def main():
    cfg        = _make_cfg()
    checkpoint = _load_checkpoint()
    threshold  = _threshold_for(CONVERGENCE_METRIC)

    print("\n" + "=" * 62)
    print("  AUTO-SHUTOFF CONVERGENCE")
    print(f"  Convergence metric    : {CONVERGENCE_METRIC}  "
          f"(threshold {threshold * 100:.1f}%)")
    print(f"  Periods each side     : {cfg.grating.n_periods_each_side}")
    print(f"  Sim time limit        : {SIM_TIME_S * 1e12:.0f} ps")
    print(f"  Sweep values          : {[f'{v:.0e}' for v in SHUTOFF_VALUES]}")
    print(f"  Checkpoint            : {CHECKPOINT}")
    print("=" * 62)

    results, best = _run_phase(
        values=SHUTOFF_VALUES, cfg=cfg, checkpoint=checkpoint,
        metric=CONVERGENCE_METRIC, threshold=threshold,
    )
    _print_table(results)
    checkpoint["best_auto_shutoff_min"] = float(best)
    _save_checkpoint(checkpoint)

    print(f"\n{'=' * 62}")
    print("  RECOMMENDED SHUTOFF SETTING")
    print(f"{'=' * 62}")
    print(f"  auto_shutoff_min : {best:.0e}")
    print(f"\n  (Production default: 1e-6 in bragg_device.py:300)")
    print(f"  Results saved to: {CONV_DIR}")
    print(f"{'=' * 62}\n")


# Auto-discovery contract: athena_run.py looks for `run` on each runner module.
# Ignores cfg argument — this script builds its own SimulationConfig via _make_cfg().
def run(cfg=None):
    main()


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", choices=["SHUTOFF"], default=None,
                        help="Fold Athena array per-task JSONs "
                             "(array_part_phSHUTOFF_*.json) into the checkpoint. "
                             "Skips simulation entirely.")
    args = parser.parse_args()

    if args.aggregate is not None:
        aggregate_array_parts(args.aggregate)
    else:
        main()
