"""
Two-stage 3D-field comparison: prelim resonance finder → locked-wavelength
3D + far-field comparison of two devices that differ only in
`grating.innermost_tooth_shift_m`.

  Stage 0 (prelim) — quick S-parameter scan of the default device with
                     2D/3D monitors and far-field disabled, narrow domain
                     (1.8 λ). Returns the resonance wavelength λ_res.
                     Runs in a few minutes.

  Stages A & B   — full 3D + far-field, but with the source CENTERED on
                   λ_res and the 3D monitor recording at exactly ONE
                   frequency point (= λ_res). This keeps the 3D field
                   array around ~1 GB instead of the ~87 GB that the
                   51-frequency-point version produces with the 5 λ
                   far-field domain (which OOM-kills the FDTD engine
                   during getresult).

The two final .mat files are loaded by matlab_plotting/plot_field_3d.m
for side-by-side inspection.

Run from the repo root (single Python process — runs all three sims sequentially):
    python runners/single/compare_3d_field_default_vs_shift.py

Server entry points:
    RUN_SCRIPT=compare_3d_field_default_vs_shift
        → run_compare_3d_field_default_vs_shift(cfg)
        Sequential: prelim + both big runs in the same job. Use on Zeus
        (no job arrays) or whenever you don't want to coordinate two jobs.

    RUN_SCRIPT=compare_3d_field_prelim
        → run_compare_3d_field_prelim(cfg)
        PRELIM ONLY — runs Stage 0, writes λ_res to LOCKED_LAMBDA_FILE
        (env var; defaults to <RESULTS_DIR>/compare_3d_lambda_res.json).
        Used by the Athena chained workflow:
          Job 1 (this prelim, single sim) → writes JSON
          Job 2 (sweep array, --dependency=afterok:Job1) → reads JSON,
                runs the two big sims in parallel.
        See runners/sweeps/compare_3d_field_shift.py for the sweep half.
"""

import copy
import json
import os
import sys

# Make the project root importable when invoked as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.single.run_simulation import run_single_sim
from simulation_config import SimulationConfig


SHIFT_M              = 100e-9   # second device's innermost-tooth shift
LOCKED_SCAN_WIDTH_NM = 6.0      # narrowed scan around the locked resonance


def _resolve_locked_lambda_path() -> str:
    """Path the prelim writes to / array tasks read from. Configurable via env."""
    p = os.environ.get("LOCKED_LAMBDA_FILE", "").strip()
    if p:
        return p
    import config as _config
    return os.path.join(_config.RESULTS_DIR, "compare_3d_lambda_res.json")


def _make_prelim_config(cfg: SimulationConfig | None) -> SimulationConfig:
    """Quick prelim sim: no 3D, no 2D, no far-field — just locate the resonance."""
    cfg = copy.deepcopy(cfg) if cfg is not None else SimulationConfig()
    cfg.monitors.record_3d_fields  = False
    cfg.monitors.record_2d_fields  = False
    cfg.farfield.enabled           = False        # narrow 1.8 λ domain → fast
    cfg.grating.cavity_width_option = "narrow"
    cfg.grating.innermost_tooth_shift_m = 0.0      # default device
    return cfg


def _make_locked_compare_config(base: SimulationConfig, lambda_res_m: float) -> SimulationConfig:
    """Locked-wavelength 3D + far-field config used for both comparison runs."""
    cfg = copy.deepcopy(base)
    cfg.monitors.record_3d_fields    = True
    cfg.monitors.record_2d_fields    = False     # everything sliced from 3D in MATLAB
    cfg.monitors.n_3d_freq_points    = 1          # record at λ_res only — keeps 3D array small
    cfg.farfield.enabled             = True       # widens domain to 5 λ for far-field capture
    cfg.grating.cavity_width_option  = "narrow"
    cfg.spectral.center_wavelength_m = lambda_res_m
    cfg.spectral.scan_width_nm       = LOCKED_SCAN_WIDTH_NM  # tight band around λ_res
    return cfg


def run_compare_3d_field_prelim(cfg: SimulationConfig | None = None) -> dict:
    """Stage 0 only: run the prelim sim and write λ_res to LOCKED_LAMBDA_FILE.

    Used by the Athena chained workflow as the prerequisite of the
    `compare_3d_field_shift` sweep array. Returns the prelim result dict
    and writes a small JSON sidecar so each downstream array task can
    read the locked wavelength and override its center_wavelength_m.
    """
    base = cfg if cfg is not None else SimulationConfig()
    cfg_prelim = _make_prelim_config(base)

    print("\n" + "=" * 60)
    print("PRELIM ONLY — resonance scan (no 3D, no far-field, 1.8 λ domain)")
    print("=" * 60)
    res_prelim = run_single_sim(cfg_prelim)
    lam_res_nm = float(res_prelim["resonance_wavelength_nm"])
    lam_res_m  = lam_res_nm * 1e-9

    out_path = _resolve_locked_lambda_path()
    payload = {
        "lambda_res_nm": lam_res_nm,
        "lambda_res_m":  lam_res_m,
        "scan_width_nm": cfg_prelim.spectral.scan_width_nm,
        "source":        "compare_3d_field_prelim",
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[prelim] λ_res = {lam_res_nm:.4f} nm")
    print(f"[prelim] wrote locked-wavelength sidecar to: {out_path}")
    return res_prelim


def run_compare_3d_field_default_vs_shift(cfg: SimulationConfig | None = None) -> list[dict]:
    """Run prelim + the two comparison sims back-to-back. Returns the two final result dicts."""
    base = cfg if cfg is not None else SimulationConfig()

    # ── Stage 0: prelim — locate the resonance wavelength ────────────────────
    cfg_prelim = _make_prelim_config(base)
    print("\n" + "=" * 60)
    print("STAGE 0 — prelim resonance scan (no 3D, no far-field, 1.8 λ domain)")
    print("=" * 60)
    res_prelim = run_single_sim(cfg_prelim)
    lam_res_nm = float(res_prelim["resonance_wavelength_nm"])
    lam_res_m  = lam_res_nm * 1e-9
    print(f"\n[compare] prelim resonance λ_res = {lam_res_nm:.4f} nm")
    print(f"[compare] locking source center + 3D-monitor frequency to λ_res for runs A and B.")

    # ── Build the locked base used for both comparison runs ──────────────────
    base_locked = _make_locked_compare_config(base, lam_res_m)

    # ── Stage A: default device (no tooth shift) ─────────────────────────────
    cfg_a = copy.deepcopy(base_locked)
    cfg_a.grating.innermost_tooth_shift_m = 0.0
    print("\n" + "=" * 60)
    print(f"RUN A — innermost_tooth_shift = 0 nm   (λ_lock = {lam_res_nm:.4f} nm)")
    print("=" * 60)
    res_a = run_single_sim(cfg_a)

    # ── Stage B: 100 nm innermost-tooth shift ────────────────────────────────
    cfg_b = copy.deepcopy(base_locked)
    cfg_b.grating.innermost_tooth_shift_m = SHIFT_M
    print("\n" + "=" * 60)
    print(f"RUN B — innermost_tooth_shift = {SHIFT_M*1e9:.0f} nm   (λ_lock = {lam_res_nm:.4f} nm)")
    print("=" * 60)
    res_b = run_single_sim(cfg_b)

    print("\n" + "=" * 60)
    print("All runs complete. Open MATLAB → run plot_field_3d.m → select")
    print("the two 3D-field result_*.mat files (skip the prelim one — it has no field_3d).")
    print("=" * 60)
    return [res_a, res_b]


if __name__ == "__main__":
    run_compare_3d_field_default_vs_shift()
