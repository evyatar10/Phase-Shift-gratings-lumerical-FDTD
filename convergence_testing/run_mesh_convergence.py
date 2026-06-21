"""
Mesh convergence testing — Pi-Shift Bragg Grating FDTD.

Tests two mesh parameters sequentially (coordinate descent).  The convergence
metric and threshold are configurable (see user-editable settings below):
  "Q"      — Q factor (lambda/FWHM); threshold 2.0% (typical published FDTD standard)
  "lambda" — resonance wavelength;   threshold 1.0% (Ansys/Lumerical standard)

Sweep order:
  Phase X  — cells_per_half_period  [4, 5, 6, 7, 8, 10]   controls dx via dx = pitch/(2N)
  Phase YZ — dyz_max_step_nm        [60, 50, 40, 30, 20]  controls global + box dy = dz

Each phase fixes the best-converged value from the previous phase.
Early stopping: if |Δmetric/metric| < threshold for two consecutive mesh values,
the phase stops and recommends the coarser of those two.

Mesh model — mirrors production (bragg_device.py, single mesh-override box):
  • Global FDTD region uses "custom non-uniform" with max-mesh-step in x (=dx_override),
    y (=dyz), z (=dyz). Grading enabled in y/z.
  • One mesh-override box centered at x=0 with dx=pitch/(2N), dy=dyz, dz=dyz.
  • Source/monitor positions snap to dx_override.

Results are checkpointed to a JSON file named checkpoint_p{N_PERIODS}_{METRIC}.json
after every run.  Different (periods, metric) combinations write separate files and
never overwrite each other.

Usage:
    # Local sequential sweep
    python convergence_testing/run_mesh_convergence.py

    # Aggregate Athena array per-task results into the checkpoint.
    # Normally invoked automatically by the server-side aggregator job
    # submitted with --dependency=afterok after each mesh_conv array.
    python convergence_testing/run_mesh_convergence.py --aggregate X
    python convergence_testing/run_mesh_convergence.py --aggregate YZ
"""

# PHASES / KIND_PREFIX: read by athena/deploy_dgx.sh's Convergence picker.
# Expands this file into one menu entry per phase ("run_mesh_convergence X" /
# "...YZ") that submit as parallel SLURM arrays with SWEEP_KIND=mesh_conv_x /
# mesh_conv_yz. Files in convergence_testing/ without these constants are
# offered as a single sequential entry instead.
PHASES      = ["X", "YZ"]
KIND_PREFIX = "mesh_conv"

# IS_HELPER: hide from the Single-runner picker (legacy guard — Single now
# only scans runners/single/, so this is defensive only).
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
# Simulation imports (bragg_device → analysis → scipy) are deferred to inside
# _run_one / _make_cfg so that --aggregate runs without triggering them.

# ═════════════════════════════════════════════════════════════════════════════
# User-editable settings
# ═════════════════════════════════════════════════════════════════════════════

# ── Convergence metric ───────────────────────────────────────────────────────
# Set CONVERGENCE_METRIC to "Q", "lambda", or "T".
#   lambda — 1.0%: Ansys/Lumerical stated standard for resonance wavelength
#   Q      — 2.0%: typical published FDTD standard for mesh convergence of Q
#   T      — 1.5%: T_res (transmission at resonance); useful when peak height
#                  matters more than Q or λ (e.g. for filter design).
CONVERGENCE_METRIC   = "Q"      # "Q", "lambda", or "T"
N_PERIODS_EACH_SIDE  = 80       # device length; also encoded in checkpoint filename

THRESHOLD_LAMBDA = 0.010        # 1.0% — for lambda convergence metric
THRESHOLD_Q      = 0.020        # 2.0% — for Q convergence metric
THRESHOLD_T      = 0.015        # 1.5% — for T_res convergence metric

# Sweep values for each phase (coarse → fine).
PHASE_X_VALUES  = [4, 5, 6, 7, 8, 9]        # cells_per_half_period (controls dx)
PHASE_YZ_VALUES = [50, 25, 10]               # dyz_max_step_nm: only divisors of GCD(350,700,800,900)=50, so every value tiles cleanly into core_height and all three widths (no Lumerical mesh snapping artifacts).

# Default (current production) values — used as fixed values in other phases
DEFAULT_CELLS    = 5
DEFAULT_DYZ_NM   = 50.0   # production pins global dy = dz = 50 nm (max-mesh-step)

# ── Polarization ─────────────────────────────────────────────────────────────
# CONV_POL selects the polarization for the convergence study. The default "TE"
# reproduces the original behavior (and the existing TE checkpoint) byte-for-byte
# — nothing below the `if POLARIZATION == "TM"` guard runs unless CONV_POL=TM.
#
# CONV_POL=TM runs the TM convergence: SourceConfig.polarization="TM" flips the
# port mode (fundamental TM) and the symmetry-plane parity automatically, and we
# switch to the TM-study device (calibrated pitch + constant indices) and
# re-center the scan window on the TM resonance. All TM artifacts use a separate
# CONV_DIR + checkpoint (suffix "_TM"), so the TE results are never touched.
POLARIZATION = (os.environ.get("CONV_POL") or "TE").upper()   # unset OR empty → TE

# TM-study device parameters (used ONLY when POLARIZATION == "TM"):
#   • pitch 518.3 nm — the calibrated pitch that re-centers the TM resonance onto
#     λ_TE ≈ 1570.7 nm (runners/tm/PITCH_ALIGNMENT.md). Measured TM resonance at
#     this pitch: 1568.6 nm (accurate mesh) / 1570.6 nm (optimization mesh).
#   • n_core/n_clad — match the TM study (Si3N4 "Luke" / SiO2 "Palik" @ 1.55 µm)
#     so the recommended mesh applies to the actual TM device being compared.
#   • 40 nm window centered at 1567 nm brackets the TM stopband (~1554–1568 nm)
#     and the cavity peak across every mesh point (~2 nm mesh-induced shift), with
#     margin. find_resonance isolates the cavity peak inside the band (verified on
#     the existing 150 nm scan).
TM_PITCH_NM      = 518.3
TM_N_CORE        = 1.9963
TM_N_CLAD        = 1.444
TM_CENTER_M      = 1.567e-6
TM_SCAN_WIDTH_NM = 40.0
TM_N_POINTS      = 2001

# CONV_POL=TE_CMP runs TE on the SAME comparison device (identical indices) at
# pitch 500 nm → TE resonance ≈ 1570.7 nm. This is the apples-to-apples partner
# of the TM run: same materials/geometry, only the polarization differs, so a
# TE-vs-TM mesh comparison isolates the polarization effect (esp. dz). It is
# DISTINCT from the default CONV_POL=TE, which keeps the original 1.977/1.44
# device + 1560 nm window byte-for-byte. TE_CMP uses the comparison indices /
# window width / point count above (TM_N_CORE/TM_N_CLAD/TM_SCAN_WIDTH_NM/TM_N_POINTS).
TE_CMP_PITCH_NM = 500.0
TE_CMP_CENTER_M = 1.571e-6

# ═════════════════════════════════════════════════════════════════════════════
# Output paths
# ═════════════════════════════════════════════════════════════════════════════

# TM artifacts live in a separate dir + checkpoint so they never collide with the
# TE results. For POLARIZATION == "TE" the tag is empty → original paths verbatim.
_POL_TAG     = "" if POLARIZATION == "TE" else f"_{POLARIZATION}"
CONV_DIR     = os.path.join(config.BASE_SAVE_DIR, "mesh_convergence" + _POL_TAG.lower())
LAYOUTS_DIR  = os.path.join(CONV_DIR, "layouts")
RESULTS_DIR  = os.path.join(CONV_DIR, "results")
CHECKPOINT   = os.path.join(CONV_DIR,
    f"checkpoint_p{N_PERIODS_EACH_SIDE}_{CONVERGENCE_METRIC}{_POL_TAG}.json")

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
    # Narrow scan: 20 nm is enough to capture the cavity resonance peak
    # and avoids wasting DFT points outside the stopband.
    cfg.spectral.scan_width_nm = 20.0
    cfg.spectral.n_wl_points = 1001
    if POLARIZATION in ("TM", "TE_CMP"):
        # Comparison device (TM-study constant indices, same for both
        # polarizations). SourceConfig.polarization flips the port mode +
        # symmetry parity automatically (see bragg_device). TM uses the
        # calibrated pitch (resonance ~1568.4 nm); TE_CMP uses pitch 500 nm
        # (resonance ~1570.7 nm). Each window is re-centered on its own peak.
        is_tm = POLARIZATION == "TM"
        cfg.source.polarization = "TM" if is_tm else "TE"
        cfg.grating.pitch_m = (TM_PITCH_NM if is_tm else TE_CMP_PITCH_NM) * 1e-9
        cfg.material.use_constant_materials = True
        cfg.material.n_core_const = TM_N_CORE
        cfg.material.n_clad_const = TM_N_CLAD
        cfg.spectral.center_wavelength_m = TM_CENTER_M if is_tm else TE_CMP_CENTER_M
        cfg.spectral.scan_width_nm = TM_SCAN_WIDTH_NM
        cfg.spectral.n_wl_points = TM_N_POINTS
        # Seed the mode solver near the effective index (λ = 2·n_eff·Λ).
        cfg.material.n_eff_guess = (
            cfg.spectral.center_wavelength_m / (2.0 * cfg.grating.pitch_m))
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


# ═════════════════════════════════════════════════════════════════════════════
# Single simulation runner
# ═════════════════════════════════════════════════════════════════════════════

def _run_one(cfg, cells: int, dyz_nm: float, tag: str) -> dict:
    """
    Build and run one FDTD simulation, extract Q factor from S-parameters.

    Parameters
    ----------
    cfg     : SimulationConfig (monitors already disabled)
    cells   : cells_per_half_period (controls dx = pitch/(2*cells))
    dyz_nm  : max-mesh-step in nm applied to both global and override-box dy/dz
    tag     : short string used in layout / results filenames

    Returns
    -------
    dict with keys: cells, dyz_nm, dx_nm, dy_nm, dz_nm,
                    Q, resonance_nm, fwhm_pm, transmission, sim_time_s
    """
    import matplotlib.pyplot as plt
    from bragg_device import PiShiftBraggFDTD
    from post_processing import extract_s_parameters, find_resonance
    from sim_helpers import apply_monitor_overrides

    class _ConvergenceBraggFDTD(PiShiftBraggFDTD):
        def __init__(self, *args, dyz_max_step_m=DEFAULT_DYZ_NM * 1e-9, **kwargs):
            self._dyz_m = float(dyz_max_step_m)
            super().__init__(*args, **kwargs)

        def _setup_fdtd_region(self):
            super()._setup_fdtd_region()
            self.fdtd.set("dy", self._dyz_m)
            self.fdtd.set("dz", self._dyz_m)

        def _add_aligned_mesh_override(self):
            super()._add_aligned_mesh_override()
            self.fdtd.setnamed("mesh_override", "dy", self._dyz_m)
            self.fdtd.setnamed("mesh_override", "dz", self._dyz_m)

    half_pitch = cfg.grating.pitch_m / 2.0
    dx_nm = half_pitch / cells * 1e9
    dy_nm = float(dyz_nm)
    dz_nm = float(dyz_nm)

    print(f"\n{'─' * 62}")
    print(f"  [{tag}]")
    print(f"  cells={cells}  dyz={dyz_nm:.1f} nm")
    print(f"  dx={dx_nm:.1f} nm   dy={dy_nm:.1f} nm   dz={dz_nm:.1f} nm")
    print(f"{'─' * 62}")

    device_kwargs = cfg.to_device_kwargs()
    # Force the swept cells through to the parent (override whatever the cfg
    # supplied via simulation_mode).
    device_kwargs["cells_per_half_period"] = int(cells)
    sim = _ConvergenceBraggFDTD(
        **device_kwargs,
        dyz_max_step_m=float(dyz_nm) * 1e-9,
    )

    layout_path  = os.path.join(LAYOUTS_DIR, f"layout_{tag}.fsp")

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
    print(f"  Simulation done in {elapsed:.1f} s")

    try:
        s_params  = extract_s_parameters(sim, cfg)
        resonance = find_resonance(s_params)
        # Lumerical returns wavelengths in descending order (uniform in frequency),
        # so dw < 0 and find_resonance's FWHM comes out negative. abs() corrects.
        fwhm_m = abs(resonance.spectral_fwhm_m)
        Q = resonance.wavelength_m / fwhm_m
        print(f"  lambda_res = {resonance.wavelength_m * 1e9:.4f} nm  "
              f"FWHM = {fwhm_m * 1e12:.2f} pm  "
              f"Q = {Q:.2f}  "
              f"T_res = {resonance.transmission:.4f}")
    finally:
        plt.close("all")
        sim.close()

    return {
        "cells":        int(cells),
        "dyz_nm":       float(dyz_nm),
        "dx_nm":        round(dx_nm,  2),
        "dy_nm":        round(dy_nm,  2),
        "dz_nm":        round(dz_nm,  2),
        "Q":            float(Q),
        "resonance_nm": float(resonance.wavelength_m * 1e9),
        "fwhm_pm":      float(fwhm_m * 1e12),
        "transmission": float(resonance.transmission),
        "sim_time_s":   round(elapsed, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# One sweep phase
# ═════════════════════════════════════════════════════════════════════════════

def _run_phase(phase: str,
               param: str,
               values: list,
               fixed_cells: int,
               fixed_dyz_nm: float,
               cfg,
               checkpoint: dict,
               metric: str,
               threshold: float) -> tuple:
    """
    Sweep one parameter while fixing the other.

    param     : "cells_per_half_period" or "dyz_max_step_nm"
    metric    : "Q" or "lambda" — drives convergence checking.
    threshold : fractional threshold (e.g. 0.02 = 2%).

    Returns (results_list, best_value).
    """
    print(f"\n{'=' * 62}")
    print(f"  Phase {phase}: sweeping {param}  [metric={metric}, threshold={threshold*100:.1f}%]")
    print(f"  Fixed: cells={fixed_cells}, dyz={fixed_dyz_nm} nm")
    print(f"{'=' * 62}")

    results         = []
    prev_wl         = None
    prev_Q          = None
    prev_T          = None
    converged_count = 0
    best_value      = values[-1]   # fallback: finest mesh

    for i, val in enumerate(values):
        if param == "cells_per_half_period":
            cells  = int(val)
            dyz_nm = float(fixed_dyz_nm)
        else:
            cells  = int(fixed_cells)
            dyz_nm = float(val)

        ck_key = f"ph{phase}_{val}"

        if ck_key in checkpoint:
            print(f"\n  [SKIP] {param}={val} — loaded from checkpoint")
            rec = checkpoint[ck_key]
        else:
            tag = f"ph{phase}_{param.replace('_', '')}_{val}"
            rec = _run_one(cfg, cells, dyz_nm, tag)
            checkpoint[ck_key] = rec
            _save_checkpoint(checkpoint)

        results.append(rec)
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
                    print(f"\n  Converged at {param}={val}  ({label})")
                    print(f"  Using {param} = {best_value}  (coarser, already stable)")
                    break
            else:
                converged_count = 0

        prev_wl = wl
        prev_Q  = Q
        prev_T  = T
    else:
        print(f"\n  No convergence within sweep range. "
              f"Using finest value: {param} = {best_value}")

    return results, best_value


# ═════════════════════════════════════════════════════════════════════════════
# Results table
# ═════════════════════════════════════════════════════════════════════════════

def _print_table(phase: str, param_label: str, results: list) -> None:
    col = 14
    sep = "─" * 118
    header = (f"  {'Param val':<{col}} {'dx(nm)':<8} {'dy(nm)':<8} {'dz(nm)':<8}"
              f" {'lambda(nm)':>12}  {'FWHM(pm)':>10}  {'Q':>10}"
              f"  {'T_res':>7}  {'Δλ%':>10}  {'ΔQ%':>10}  {'Time(s)':>8}")
    print(f"\n{sep}")
    print(f"  Phase {phase} — {param_label}")
    print(sep)
    print(header)
    print(sep)

    prev_wl = None
    prev_Q  = None
    for r in results:
        val = r.get("cells") if param_label == "cells_per_half_period" \
              else r.get("dyz_nm")
        dlam_str = "         —"
        dQ_str   = "         —"
        if prev_wl is not None:
            dlam = (r["resonance_nm"] - prev_wl) / abs(prev_wl) * 100
            dlam_str = f"{dlam:+.4f}%"
        if prev_Q is not None:
            dQ = (r["Q"] - prev_Q) / abs(prev_Q) * 100
            dQ_str = f"{dQ:+.4f}%"
        t_res = r.get("transmission", float("nan"))
        print(f"  {str(val):<{col}} {r['dx_nm']:<8.1f} {r['dy_nm']:<8.1f} "
              f"{r['dz_nm']:<8.1f} {r['resonance_nm']:>12.4f}  {r['fwhm_pm']:>10.2f}"
              f"  {r['Q']:>10.2f}  {t_res:>7.4f}  {dlam_str:>10}  {dQ_str:>10}"
              f"  {r['sim_time_s']:>8.1f}")
        prev_wl = r["resonance_nm"]
        prev_Q  = r["Q"]
    print(sep)


# ═════════════════════════════════════════════════════════════════════════════
# Aggregator: fold Athena array per-task JSONs into the checkpoint
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_array_parts(phase: str) -> dict:
    """
    Fold per-task results from Athena array runs into the checkpoint.

    Athena writes each task's record to mesh_convergence/array_part_ph{X|YZ}_NNNN.json
    (avoids concurrent writes to the shared checkpoint). This function reads
    those files in numerical order, reconstructs ck_key = ph{X|YZ}_{value} entries,
    and — for the X-phase — computes best_cells_phase_x using the same convergence
    rule as _run_phase so build_sweep_list.py can pin SWEEP_FIXED_CELLS for the
    follow-on YZ array. Normally invoked by athena/jobs/run_mesh_aggregate.sh
    via --dependency=afterok after the mesh_conv_{x,yz} array completes.
    """
    phase = phase.upper()
    if phase not in {"X", "YZ"}:
        raise ValueError(f"phase must be 'X' or 'YZ', got {phase!r}")

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
        if phase == "X":
            val = rec["cells"]
        else:
            val = rec.get("dyz_nm", rec.get("dz_nm"))
        ck_key = f"ph{phase}_{val}"
        checkpoint[ck_key] = rec
        records.append((val, rec))
        print(f"  [{phase}] {ck_key:<20s}  Q={rec['Q']:>8.2f}  "
              f"λ={rec['resonance_nm']:.4f} nm  ({os.path.basename(p)})")

    # Sort by the swept value so the convergence rule sees coarse → fine order.
    if phase == "X":
        # cells: ascending → finer
        records.sort(key=lambda kv: int(kv[0]))
    else:
        # dyz_nm: descending → finer
        records.sort(key=lambda kv: -float(kv[0]))

    # Recompute the converged "best" value with the same rule used by _run_phase.
    converged_count = 0
    best_value = records[-1][0]   # fallback: finest
    prev_wl = None
    prev_Q  = None
    prev_T  = None
    for i, (val, rec) in enumerate(records):
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

    if phase == "X":
        checkpoint["best_cells_phase_x"] = int(best_value)
        print(f"\n  best_cells_phase_x = {int(best_value)}  "
              f"(picked up by build_sweep_list.py for mesh_conv_yz SWEEP_FIXED_CELLS)")
    else:
        checkpoint["best_dyz_nm_phase_yz"] = float(best_value)
        print(f"\n  best_dyz_nm_phase_yz = {float(best_value):.1f} nm")

    _save_checkpoint(checkpoint)
    print(f"  Wrote {CHECKPOINT}")
    return checkpoint


# ═════════════════════════════════════════════════════════════════════════════
# Main (local sequential sweep)
# ═════════════════════════════════════════════════════════════════════════════

def main():
    cfg        = _make_cfg()
    checkpoint = _load_checkpoint()

    half_pitch = cfg.grating.pitch_m / 2.0
    threshold  = _threshold_for(CONVERGENCE_METRIC)

    print("\n" + "=" * 62)
    print("  MESH CONVERGENCE")
    print(f"  Convergence metric    : {CONVERGENCE_METRIC}  "
          f"(threshold {threshold * 100:.1f}%)")
    print(f"  Periods each side     : {cfg.grating.n_periods_each_side}")
    print(f"  Current defaults      : cells={DEFAULT_CELLS} "
          f"(dx={half_pitch / DEFAULT_CELLS * 1e9:.0f} nm), "
          f"dyz={DEFAULT_DYZ_NM:.0f} nm (global + box, max-mesh-step)")
    print(f"  Checkpoint            : {CHECKPOINT}")
    print("=" * 62)

    # ── Phase X: cells_per_half_period (controls dx) ──────────────────────
    res_X, best_cells = _run_phase(
        phase="X", param="cells_per_half_period", values=PHASE_X_VALUES,
        fixed_cells=DEFAULT_CELLS, fixed_dyz_nm=DEFAULT_DYZ_NM,
        cfg=cfg, checkpoint=checkpoint,
        metric=CONVERGENCE_METRIC, threshold=threshold,
    )
    _print_table("X", "cells_per_half_period", res_X)
    checkpoint["best_cells_phase_x"] = int(best_cells)
    _save_checkpoint(checkpoint)

    # ── Phase YZ: dyz_max_step_nm (global + box dy=dz) ────────────────────
    res_YZ, best_dyz = _run_phase(
        phase="YZ", param="dyz_max_step_nm", values=PHASE_YZ_VALUES,
        fixed_cells=best_cells, fixed_dyz_nm=DEFAULT_DYZ_NM,
        cfg=cfg, checkpoint=checkpoint,
        metric=CONVERGENCE_METRIC, threshold=threshold,
    )
    _print_table("YZ", "dyz_max_step_nm", res_YZ)
    checkpoint["best_dyz_nm_phase_yz"] = float(best_dyz)
    _save_checkpoint(checkpoint)

    # ── Final recommendation ──────────────────────────────────────────────
    dx_rec = half_pitch / best_cells * 1e9

    print(f"\n{'=' * 62}")
    print("  RECOMMENDED MESH SETTINGS")
    print(f"{'=' * 62}")
    print(f"  cells_per_half_period : {best_cells}      ->  dx = {dx_rec:.1f} nm")
    print(f"  dyz_max_step_nm       : {best_dyz:.1f}   ->  dy = dz = {best_dyz:.1f} nm")
    print(f"\n  (Production defaults: cells=5 ⇒ dx=50 nm, dy=dz=50 nm)")
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
    parser.add_argument("--aggregate", choices=["X", "YZ"], default=None,
                        help="Fold Athena array per-task JSONs (array_part_ph{X|YZ}_*.json) "
                             "into the checkpoint. Skips simulation entirely.")
    args = parser.parse_args()

    if args.aggregate is not None:
        aggregate_array_parts(args.aggregate)
    else:
        main()
