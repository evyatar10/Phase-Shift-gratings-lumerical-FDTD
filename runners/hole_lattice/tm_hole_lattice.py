"""TM in-core SiO2 hole LATTICE — photonic-crystal circles inside the pi-shift grating.

Study dir: tm_hole_lattice   |   Created 2026-07-18   |   Job(s): 123303
Purpose: 6-task discriminator for the idea "SiO2 circles (n = 1.444, NOT air) inside
the SiN core, one per period, sharing the teeth's pi-shift defect — can they reduce
radiation loss at fixed mode width?". Successor of the single-hole scan
(runners/archive/sweeps/scatterers/tm_hole_scan.py, jobs 116152/116272: a single
in-core hole was parasitic-to-neutral, never beneficial).

Each task answers one mechanism question (predictions = 2026-07-16 theory round):

  idx 0  control, no holes                     in-study baseline, identical numerics
  idx 1  corr 400 + matched lattice            kappa-split, untrimmed
  idx 2  corr 300 + matched lattice            kappa-split, corr-trimmed; idx 1+2
                                               bracket the fwhm_m (mode width) match
  idx 3  corr 400 + lattice at period 545 nm   PERIOD detune — predict loss UP
                                               (545 nm phase-matches into the light cone)
  idx 4  corr 400 + lattice shifted +pitch/4   DEFECT/phase detune — predict kappa
                                               renorm only, no loss benefit
  idx 5  corr 400 + idx-1 lattice +25 nm       jitter-floor twin (half mesh cell, §2)

Read out per task: peak T, loss, resonance lambda, Q, fwhm_m — all deltas vs idx 0.

Device: anchored TM W800 baseline (pitch 516.83 nm, corr 400 nm, height 350 nm,
N = 80/side, n 1.97/1.444, optimization mesh dx = 50 nm). Holes: r = 100 nm cylinders
at y = 0, full core height, one per NARROW segment center — removing SiN where the
core is already narrow deepens the index modulation, i.e. ADDS grating strength.
Tooth layout (bragg_device rect branch): left-arm narrow centers at -d*pitch
(d = 1..N), right-arm at (d+1/2)*pitch (d = 0..N-1), cavity spans [-pitch/4, +pitch/4]
— so the hole lattice inherits the pi-shift defect from the teeth. No hole in the
cavity itself.

WIDE scan window for ALL tasks (1545 +/- 75 nm, 6001 pts): a hole in every period is
a strong perturbation (~20% of each narrow-segment area) and blue-shifts lambda_res
by up to ~35 nm. The bare-device resonance (1558.5 nm) is well inside the window.

Dispatch (SERIALIZE — check `bash athena/deploy_athena.sh --status` first; 6 tasks):
    bash athena/deploy_athena.sh --option3 --spec=runners.hole_lattice.tm_hole_lattice
Output -> results/tm_hole_lattice/results/ -> results_from_athena/tm_hole_lattice/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base

# ── User knobs ────────────────────────────────────────────────────────────────
PITCH_NM     = 516.83     # device pitch (anchored TM value — do not change alone)
N_SIDE       = 80         # grating periods per side
R_NM         = 100.0      # hole radius (2 mesh cells at dx=50; < narrow half-width)
TRIM_CORR_NM = 300.0      # idx-2 corrugation (idx 1 keeps the full 400)
DETUNED_P_NM = 545.0      # idx-3 hole period; anything > ~528 nm radiates directly
JITTER_NM    = 25.0       # idx-5 lattice offset = half a dx=50 mesh cell

X_GE_NM = N_SIDE * PITCH_NM + PITCH_NM / 4.0    # grating end (41475.6 nm)


def narrow_centers_nm(p_nm):
    """Hole sites at the narrow-segment centers of a pi-shift lattice with period
    p_nm, keeping every circle fully inside the grating extent."""
    x_max = X_GE_NM - R_NM
    left  = [-d * p_nm for d in range(1, N_SIDE + 1) if d * p_nm <= x_max]
    right = [(d + 0.5) * p_nm for d in range(N_SIDE) if (d + 0.5) * p_nm <= x_max]
    return sorted(left + right)


# ── Task table: (corrugation nm, hole radius nm, hole x-centers nm) ───────────
_L0 = narrow_centers_nm(PITCH_NM)                          # matched lattice, 160 sites
assert len(_L0) == 2 * N_SIDE, f"matched lattice truncated: {len(_L0)}"

_rows = [
    (400.0,        0.0,  []),                                  # 0 control
    (400.0,        R_NM, _L0),                                 # 1 matched, untrimmed
    (TRIM_CORR_NM, R_NM, _L0),                                 # 2 matched, corr-trimmed
    (400.0,        R_NM, narrow_centers_nm(DETUNED_P_NM)),     # 3 period detune
    (400.0,        R_NM, [x + PITCH_NM / 4.0 for x in _L0]),   # 4 defect/phase detune
    (400.0,        R_NM, [x + JITTER_NM for x in _L0]),        # 5 jitter twin
]

_corrs = [r[0] for r in _rows]
_rs    = [r[1] for r in _rows]
_xs    = [[round(x, 1) for x in r[2]] for r in _rows]

# File-tag collision guard: the tag carries (r, site count, first/last x, corr) —
# see sim_helpers.generate_file_tag array form. Shared names clobber .h5 mid-run.
_keys = {(c, r, len(xs), xs[0] if xs else 0.0, xs[-1] if xs else 0.0)
         for c, r, xs in zip(_corrs, _rs, _xs)}
assert len(_keys) == len(_rows), "rows must be unique in (corr, r, N, x0, x-1)"
# Holes must sit fully inside the narrow segment (r < narrow half-width).
assert R_NM < (800.0 - max(_corrs)) / 2.0, "hole radius exceeds narrow half-width"


def _build_base():
    cfg = build_base()                    # anchored TM W800; scatterer arrives ENABLED
    cfg.scatterer.y_m = 0.0               # on-axis: keeps both symmetry planes
    cfg.scatterer.index = 1.444           # SiO2 hole (mesh-order override wins overlaps)
    cfg.y_span_override_m = None          # default transverse box (no off-axis pillars)
    cfg.spectral.center_wavelength_m = 1.545e-6   # WIDE window (lattice blue-shift)
    cfg.spectral.scan_width_nm = 150.0
    cfg.spectral.n_wl_points = 6001
    return cfg


BASE = _build_base()

SPEC = SweepSpec(
    corrugation_depth_nm = _corrs,
    scatterer_radius_nm  = _rs,
    scatterer_x_list_nm  = _xs,
    mode  = "zipped",
    label = "tm_hole_lattice",
)

if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
