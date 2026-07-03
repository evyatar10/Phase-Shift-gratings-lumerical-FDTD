"""
TM lateral-scatterer recycling scan — axial position line, mirrored SiN cylinder pair.

Physics (paper_8 §5-§7)
-----------------------
The TM defect mode radiates IN the horizontal (chip) plane, hugging the guide axis
(half-power half-angle ~7-12 deg); TE radiates mostly vertically. A small SiN cylinder
pair placed in the oxide beside the guide is driven by the shedding leaky field and,
by reciprocity, re-couples part of it back into the guided mode with round-trip phase
2*k_clad*rho. Constructive placements recur every lambda_res/(2*n_clad) ~ 0.540 um in
the radial distance rho from the defect. The readout of this scan is the predicted
OSCILLATION of Q / peak T / resonant loss / lambda_res vs position (including the
destructive positions where Q must DROP); the recoverable budget is bounded by the
small TM radiation loss (1-R-T ~ 0.04), so expect sub-% to few-% modulation, not a
big Q boost. A flat null within the control noise band is a valid (negative) result.

Device (anchored TM baseline — confirmed with user, CLAUDE.md §4)
-----------------------------------------------------------------
  height 350 nm, pitch 516.83 nm, corrugation 400 nm, N = 80 periods/side,
  constant indices n_core 1.97 / n_clad 1.444, mesh "optimization" (dx = 50 nm).
  Target resonance 1558.46 nm (T = 0.827, |FWHM| ~ 1.31 nm); scan window
  [1543.5, 1573.5] nm (30 nm, 3001 pts) — the proven narrow window from
  side_by_side_tm_400nm that EXCLUDES the ~1577 nm band-edge ripple which lures
  find_bragg_resonance onto the wrong peak.

Scatterer
---------
  Vertical SiN cylinder (n = 1.97, full 350 nm core height, z-centered) at
  (x_s, +/-1.0 um): a y-MIRRORED PAIR, which keeps the y=0 symmetry plane valid
  (same per-task cost as the plain device, ~2x the interference signal). A single
  off-axis scatterer with the TM 'Symmetric' y BC left on would silently simulate
  the pair anyway — bragg_device raises on that combination.
  Domain: y_span_override_m = 4.8 um so the outermost cylinder edge (1.2 um)
  keeps >= 1.2 um > lambda/n_clad clearance from the y PML (z span untouched).

Grid (mode='zipped' — lockstep lists, one SLURM array task per row)
-------------------------------------------------------------------
  idx 0        : r = 0 control (no scatterer, IDENTICAL numerics/domain)
  idx 1-91     : r = 150 nm, x_s = 0 .. 12.15 um, step 135 nm (4 samples per
                 0.540 um recoupling period)                       [91 tasks]
  idx 92-137   : r = 100 nm, x_s = 0 .. 12.15 um, step 270 nm     [46 tasks]
  idx 138-183  : r = 200 nm, x_s = 0 .. 12.15 um, step 270 nm     [46 tasks]
  idx 184-186  : mesh-jitter controls: r = 150 nm at x_s = {2.725, 5.425, 8.125} um
                 (+25 nm = half a mesh cell off the main-line points)  [3 tasks]
  => 187 tasks total. Forward side only (the cavity field is x-symmetric).

File names encode _scR{r}_X{x}_Y{y}_pair (integer nm) -> unique .fsp/.h5/.mat per
task, no clobbering. Positions/radii are module literals (deterministic on both the
local and cluster side of the deploy — NEVER env-derived: sbatch --export truncates
comma lists and --option3 forwards no TM_* envs).

Run on Athena as a parallel SLURM array (default partition — independent tasks;
serialize after any in-flight --option3 sweep, shared data/sweep_list.txt):
    smoke:  bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_scatterer_scan --array-tasks=0
    full:   bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_scatterer_scan

Output -> results/tm_scatterer_scan/results/ (RUN_NAME = module short name) ->
results_from_athena/tm_scatterer_scan/. Per-task .mat: T/R/loss spectra,
resonance_transmission / resonance_wavelength_nm / spectral_fwhm_nm (NEGATIVE for
TM -> use abs for Q), fwhm_m, plus scatterer_r_m / scatterer_x_m / scatterer_y_m.
Plot with matlab_plotting/plot_scatterer_scan.m.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


AXIAL_Y_NM = 1000.0        # fixed lateral offset of the pair (+/- y), nm from the axis
X_MAX_NM = 12150.0         # line end: ~22.5 recoupling periods, well inside the arms
STEP_FULL_NM = 135.0       # 4 samples per lambda_res/(2*n_clad) ~ 540 nm period
STEP_HALF_NM = 270.0       # half-resolution line for the radius bracket


def build_base() -> SimulationConfig:
    """Anchored TM device + proven narrow window — pinned explicitly (no env reads)
    so the Athena dispatcher's global tweaks are NOT inherited and the local and
    cluster expansions are identical. Mirrors side_by_side_tm_400nm.build_base,
    single-device, y-symmetry ON (the mirrored pair preserves it)."""
    cfg = SimulationConfig()
    cfg.grating.pitch_m = 516.83e-9                   # TM pitch (co-resonant with TE)
    cfg.grating.n_periods_each_side = 80
    cfg.grating.cavity_neg_detuning_nm = 0.0
    cfg.apodization.enabled = False

    cfg.geometry.corrugation_depth_m = 400e-9         # TE-mode-width-matched value
    cfg.geometry.core_height_m = 350e-9

    cfg.material.use_constant_materials = True        # const_material_mode default "object"
    cfg.material.n_core_const = 1.97
    cfg.material.n_clad_const = 1.444

    cfg.mesh.simulation_mode = "optimization"         # dx = 50 nm
    cfg.source.polarization = "TM"

    # NARROW window centered on the anchored TM defect resonance (1558.46 nm,
    # T=0.827, |FWHM|~1.31 nm). Excludes the ~1577 nm band-edge ripple. 3001
    # points (~10 pm) resolve the ~1.3 nm peak (>100 samples across the FWHM).
    cfg.spectral.center_wavelength_m = 1.5585e-6
    cfg.spectral.scan_width_nm = 30.0
    cfg.spectral.n_wl_points = 3001

    # Ports-only (cheap, ~3.5 GB/task). Fields/far-field OFF (MonitorConfig's
    # record_2d_fields defaults True — must be pinned off for the array).
    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled = False

    # Scatterer pair: enabled study-wide; per-task radius 0 = the control row.
    cfg.scatterer.enabled = True
    cfg.scatterer.mirrored_y = True                   # keeps the y=0 symmetry plane
    # Y-only domain widening: outermost cylinder edge 1.0+0.2 um -> >=1.2 um
    # clearance to the y PML (> lambda/n_clad = 1.08 um). z span stays default.
    cfg.y_span_override_m = 4.8e-6
    return cfg


BASE = build_base()


# ── Position/radius lists (module literals — deterministic everywhere) ────────
def _line(step_nm):
    n = int(round(X_MAX_NM / step_nm))
    return [round(i * step_nm, 1) for i in range(n + 1)]


_x_full = _line(STEP_FULL_NM)      # 91 points, 0 .. 12150 nm
_x_half = _line(STEP_HALF_NM)      # 46 points, 0 .. 12150 nm
_x_jitter = [2725.0, 5425.0, 8125.0]   # main-line points +25 nm (half a mesh cell)

_rs = [0.0] + [150.0] * len(_x_full) + [100.0] * len(_x_half) \
      + [200.0] * len(_x_half) + [150.0] * len(_x_jitter)
_xs = [0.0] + _x_full + _x_half + _x_half + _x_jitter
_ys = [AXIAL_Y_NM] * len(_xs)

assert len(_rs) == len(_xs) == len(_ys) == 187, f"task count changed: {len(_rs)}"
assert len(set(zip(_rs, _xs, _ys))) == len(_rs), \
    "duplicate (r, x, y) row -> file-tag collision (same .fsp/.h5/.mat)"


SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_nm      = _xs,
    scatterer_y_nm      = _ys,
    mode  = "zipped",
    label = "tm_scatterer_scan",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
