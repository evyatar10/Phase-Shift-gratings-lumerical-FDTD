"""
TM lateral-scatterer at the KNOWN-BEST axial X (r=100, x=810 nm), three DISTANCES.

Follow-up to tm_scatterer_scan. That study swept the whole axial x-line at y = 1.0 um
and found the best r = 100 nm mirrored SiN pair at x = 810 nm (ΔT = +0.0020..+0.0026,
ACC-confirmed at accurate mesh; tm_scatterer_scan/FINDINGS.md + tm_scatterer_radius).
Here we DO NOT re-scan x: the cylinder is pinned at that optimum (r = 100 nm,
x = 810 nm) and only the lateral offset varies, y = 800 / 1000 / 1200 nm, to read how
the recoupling benefit changes as the pair moves in/out from the guide (the y-line the
original study never took). The r = 0 control comes along so every reported delta is vs
an in-study, identical-numerics baseline (CLAUDE.md 2).

Physics (paper_8 5-7)
-----------------------
The TM defect mode radiates IN the horizontal (chip) plane, hugging the guide axis.
A mirrored SiN cylinder pair placed in the oxide beside the guide is driven by the
shedding leaky field and, by reciprocity, re-couples part of it back into the guided
mode with round-trip phase 2*k_clad*rho. Constructive placements recur every
lambda_res/(2*n_clad) ~ 0.540 um in the radial distance rho. Moving the pair OUT in y
(a) lengthens rho at fixed x, phase-shifting the constructive x-positions, and
(b) drops the driving field amplitude (larger rho), so the modulation should shrink.
The recoverable budget is bounded by the small TM radiation loss (1-R-T ~ 0.04): expect
sub-% to few-% modulation, not a big Q boost. A flat null within the control noise band
is a valid (negative) result.

Device (anchored TM baseline — CLAUDE.md 4; identical to tm_scatterer_scan)
-----------------------------------------------------------------------------
  height 350 nm, pitch 516.83 nm, corrugation 400 nm, N = 80 periods/side,
  constant indices n_core 1.97 / n_clad 1.444, mesh "optimization" (dx = 50 nm).
  Target resonance 1558.46 nm (T = 0.827, |FWHM| ~ 1.31 nm); narrow scan window
  [1543.5, 1573.5] nm (30 nm, 3001 pts) — excludes the ~1577 nm band-edge ripple.

Scatterer
---------
  Vertical SiN cylinder (n = 1.97, full 350 nm core height, z-centered), r = 100 nm,
  a y-MIRRORED PAIR at (x_s, +/- y) so the y=0 symmetry plane stays valid.
  Domain: y_span_override_m = 4.8 um (unchanged from tm_scatterer_scan) so numerics
  are directly comparable. Outermost case y=1200 nm -> edge 1.30 um -> clearance
  2.40 - 1.30 = 1.10 um > lambda/n_clad (1.079 um): passes with no PML warning.

Grid (mode='zipped' — lockstep lists, one SLURM array task per row)
-------------------------------------------------------------------
  idx 0          : r = 0 control (no scatterer, IDENTICAL numerics/domain)
  idx 1-3        : r = 100 nm, x = 810 nm, y = {800, 1000, 1200} nm
  => 4 tasks total. Forward side only (the cavity field is x-symmetric). Row 2
     (y = 1000 nm) reproduces the known ΔT = +0.0020 optimum as an internal check.

File names encode _scR{r}_X{x}_Y{y}_pair (integer nm) -> unique .fsp/.h5/.mat per
task, no clobbering (the _Y tag separates the three distances). Positions/radii are
module literals — deterministic on both local and cluster (NEVER env-derived:
sbatch --export truncates comma lists and --option3 forwards no TM_* envs).

Run on Athena as a parallel SLURM array (default partition — independent tasks;
serialize after any in-flight --option3 sweep, shared data/sweep_list.txt). Only
4 tasks, well under the QOS 24h_1g 100-submitted / 4-running cap — one shot:
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_scatterer_ydist_scan

Output -> results/tm_scatterer_ydist_scan/results/ ->
results_from_athena/tm_scatterer_ydist_scan/. Per-task .mat: T/R/loss spectra,
resonance_transmission / resonance_wavelength_nm / spectral_fwhm_nm (NEGATIVE for
TM -> use abs for Q), fwhm_m, plus scatterer_r_m / scatterer_x_m / scatterer_y_m.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


R_NM = 100.0               # FIXED cylinder radius (nm) — the "size 100" pair
X_OPT_NM = 810.0           # known-best axial position for r=100 (tm_scatterer_scan)
Y_LIST_NM = [700.0, 800.0, 1000.0, 1200.0]   # lateral distances of the pair (+/- y), nm
# y=700 nm = closest-in probe: r=100 inner edge at 600 nm, 100 nm oxide gap to the
# 500 nm wide-tooth edge (y=600 would graze it). Added after the first 3 distances.
REF_Y_NM = 1000.0          # reference distance (where x=810 was found + the control sits)


def build_base() -> SimulationConfig:
    """Anchored TM device + proven narrow window — pinned explicitly (no env reads)
    so the Athena dispatcher's global tweaks are NOT inherited and the local and
    cluster expansions are identical. Identical to tm_scatterer_scan.build_base."""
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

    # Ports-only (cheap, ~3.5 GB/task). Fields/far-field OFF.
    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled = False

    # Scatterer pair: enabled study-wide; per-task radius 0 = the control row.
    cfg.scatterer.enabled = True
    cfg.scatterer.mirrored_y = True                   # keeps the y=0 symmetry plane
    # Domain unchanged from tm_scatterer_scan so the y=1000 line is directly
    # comparable; y=1200 outermost edge 1.30 um still clears the y PML by 1.10 um.
    cfg.y_span_override_m = 4.8e-6
    return cfg


BASE = build_base()


# ── Position lists (module literals — deterministic everywhere) ───────────────
# idx 0: r=0 control (numerics-identical baseline, at the reference distance).
# idx 1..3: the known-best pillar (r=100, x=810) at each lateral distance.
_rs = [0.0] + [R_NM] * len(Y_LIST_NM)
_xs = [0.0] + [X_OPT_NM] * len(Y_LIST_NM)
_ys = [REF_Y_NM] + list(Y_LIST_NM)


assert len(_rs) == len(_xs) == len(_ys) == 5, f"task count changed: {len(_rs)}"
assert len(set(zip(_rs, _xs, _ys))) == len(_rs), \
    "duplicate (r, x, y) row -> file-tag collision (same .fsp/.h5/.mat)"


SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_nm      = _xs,
    scatterer_y_nm      = _ys,
    mode  = "zipped",
    label = "tm_scatterer_ydist_scan",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
