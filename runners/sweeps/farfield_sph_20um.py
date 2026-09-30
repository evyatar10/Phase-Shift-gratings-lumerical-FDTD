"""Far field of three 20-um-mode pi-shift gratings at N=98 -> spherical-harmonic content.

Study dir: runners/sweeps/   |   Created 2026-09-29   |   Job(s): Athena 164883 (smoke, FAIL), 164891 (smoke, PASS), 164893 (round A, 6 tasks)
Purpose (user): three devices at the SAME mode width (~20 um) and the SAME length
(N=98/side) -- (A) plain TE corr 250, (B) Itai's re-optimized Nt60 "overshoot"
apodization (TE, the job-63722 geometry, untouched), (C) plain TM corr 325 at the
co-resonant pitch 516.83 -- record the complex far field on the top (+z) and
side (+y) planes at the resonance, then decompose OFFLINE into vector spherical
harmonics (python_tools/farfield_multipole.py): power fraction per (E/M, l, m).

Nothing to reuse: the 641 stored *_ff.mat rows are N80/N100 scatterer rows at
other corrugations, and every one was projected at the band CENTRE, not the
resonance (stored TE example: lam 1558.94 vs res 1559.39 = 41% of a linewidth
off). First study on the 2026-09-29 engine change: farfield_freq_points > 1 makes
the monitors record the band and the projection is taken at the recorded point
nearest resonance_wavelength_nm (default 1 = old behaviour, snapshot-gated).

ROUND A (this SPEC, 6 tasks):
  rows 0-3  TE far-field BOX LADDER on device A. The far-field box was never
            converged for TE (te_span_z_check: z only, ports only, y 3.76; the
            6.8/8.8 standard came from TM corr 400). Boxes y/z (um): 6.8/6.81
            (every stored 20-um row), 8.0/8.8 (TM c325 family), 10.0/10.8,
            12.0/12.8. Verdict = the multipole spectrum itself + T, T+R and the
            E^2-weighted mean |ux| per monitor; converged when the last two rungs
            agree to the rung-to-rung jitter.
  row 4     device B at 6.8/6.81 = IDENTICAL numerics to its stored row (63722:
            lam 1559.8597, T 0.97498, Q_L 7680, fwhm_m 19.633 um) -> that row is
            the control. Re-run bigger only if the ladder says 6.8 is not enough.
  row 5     device C at 8.0/8.8 = the stored TM c325 numerics
            (result_N100_TM_avg_C325_Ybox8p0_Zbox8p8: lam 1559.006, |FWHM| 0.886,
            fwhm_m 19.24 um). TM box convergence is settled (tm_span_conv_c325).
Mode widths EXPECTED at N=98: A ~20.0 (19.99 at N166 / box 6.8), B 19.63
(measured), C ~19.2 (19.24 at N100). Half-lengths 49.0 / 48.1 / 50.6 um.

Numerics: mesh "optimization" (dx 50 nm), z-symmetry ON, ports 2001 pts, 2D field
planes OFF (350 MB/row at N=10 already; the 1D field_profile gives fwhm_m). Windows
10 nm (A, C; linewidths ~0.5 / 0.9 nm) and 4 nm (B; 0.2 nm), each centred on the
stored resonance at the row's own box. A: 1559.986 nm MEASURED at box 6.8 (IGUM
itai_hh_apod row result_N166_avg_C250_Ybox6p8_Zbox6p8.mat) -- +0.20 nm vs the
default-box 1559.79, which is why the projection keys off the FOUND resonance.
Far-field monitors: x-span 80 um (devices are 96-101 um long; captures grazing
rays to |ux| ~ 0.995 from the mode's +-15 um), 401 x 401 direction-cosine grid,
complex E, 81 frequency points (spacing 0.125 / 0.05 nm -> the projected point
is <= 12% of a linewidth from the resonance), near-field surfaces OFF.
RAM: the ports record the full cross-section at every point -> ~130 GB at
12 x 12.8 um (memory: 58 GB at ~70 um^2 / 2001 pts) -> SBATCH_MEM=200G.

SMOKE first (CLAUDE.md 5: hardware-touching engine change gets a minutes-scale
end-to-end pass). SMOKE = True swaps the rows for ONE N=10 / 201-pt / 11-ff-point
row that drives the new path end to end: monitors with >1 point, the
resonance-indexed projection, the .mat write. Expected log markers: "Override: Far-field monitors set to 11
frequency point(s)" and "lam = <x> nm  (point k/11, target <res> nm)" with
|x - res| <= half the point spacing. Smoke 164883 (2026-09-29) FAILED this:
apply_monitor_overrides reset the monitors to 1 point (fixed in sim_helpers).
Numbers are never physics. Then SMOKE = False.

Dispatch (cluster: user choice):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 \
        --spec=runners.sweeps.farfield_sph_20um --max-concurrent=3
Output -> results/farfield_sph_20um/results/  (download to results_from_<cluster>/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.sweeps import itai_hh_nt60w20 as hh
from simulation_config import SimulationConfig

SMOKE = False                     # True = one tiny end-to-end row (see docstring)

N_SIDE        = 98
FF_X_SPAN_UM  = 80.0
FF_RES        = 401
FF_FREQ_PTS   = 81
N_WL_POINTS   = 2001

LAM_A_NM, LAM_B_NM, LAM_C_NM = 1559.986, 1559.8597, 1559.006   # stored, see docstring
WIN_A_NM, WIN_B_NM, WIN_C_NM = 10.0, 4.0, 10.0


def zmult(z_um, lam_nm):
    """span_multiplier giving z_span = core 0.35 um + mult * lambda_centre."""
    return round((z_um - 0.35) / (lam_nm * 1e-3), 2)


# (pol, pitch_nm, corr_nm, avg_w_nm, cavity_w_nm, teeth, centre_nm, window_nm, y_um, z_um)
HH_TEETH = hh.teeth(N_SIDE)
ROWS = [
    ("TE", 500.0,   250.0, 800.0,  None,           None,     LAM_A_NM, WIN_A_NM,  6.8,  6.81),
    ("TE", 500.0,   250.0, 800.0,  None,           None,     LAM_A_NM, WIN_A_NM,  8.0,  8.80),
    ("TE", 500.0,   250.0, 800.0,  None,           None,     LAM_A_NM, WIN_A_NM, 10.0, 10.80),
    ("TE", 500.0,   250.0, 800.0,  None,           None,     LAM_A_NM, WIN_A_NM, 12.0, 12.80),
    ("TE", hh.PITCH_NM, round(hh.BULK_WIDE_NM - hh.BULK_NARROW_NM, 1), hh.AVG_WIDTH_NM,
                                   hh.CAVITY_W_NM, HH_TEETH, LAM_B_NM, WIN_B_NM,  6.8,  6.81),
    ("TM", 516.83,  325.0, 800.0,  None,           None,     LAM_C_NM, WIN_C_NM,  8.0,  8.80),
]
if SMOKE:
    ROWS = [("TE", 500.0, 250.0, 800.0, None, None, 1560.0, 20.0, 6.8, 6.81)]

for _r in ROWS:
    assert (N_SIDE if not SMOKE else 10) * _r[1] * 1e-3 >= (2.0 * 20.0 if not SMOKE else 0), "containment"

BASE = SimulationConfig()
BASE.spectral.n_wl_points = 201 if SMOKE else N_WL_POINTS
BASE.monitors.record_2d_fields = False   # 2D planes were 350 MB at N=10 (smoke 164883); GBs at N=98 / 12 um; fwhm_m comes from the 1D field_profile
BASE.farfield.enabled = True
BASE.farfield.save_complex = True
BASE.farfield.save_nearfield = False
BASE.farfield.farfield_x_span_m = (8.0 if SMOKE else FF_X_SPAN_UM) * 1e-6
BASE.farfield.ff_resolution = 101 if SMOKE else FF_RES
BASE.farfield.farfield_freq_points = 11 if SMOKE else FF_FREQ_PTS

SPEC = SweepSpec(
    polarization              = [r[0] for r in ROWS],
    pitch_nm                  = [r[1] for r in ROWS],
    corrugation_depth_nm      = [r[2] for r in ROWS],
    avg_width_nm              = [r[3] for r in ROWS],
    cavity_width_nm           = [r[4] for r in ROWS],
    width_narrow_per_tooth_nm = [None if r[5] is None else r[5][0] for r in ROWS],
    width_wide_per_tooth_nm   = [None if r[5] is None else r[5][1] for r in ROWS],
    n_periods_each_side       = [10 if SMOKE else N_SIDE] * len(ROWS),
    center_wavelength_nm      = [r[6] for r in ROWS],
    scan_width_nm             = [r[7] for r in ROWS],
    y_span_um                 = [r[8] for r in ROWS],
    span_mult                 = [zmult(r[9], r[6]) for r in ROWS],
    mode  = "zipped",
    label = "farfield_sph_20um",
)

if __name__ == "__main__":
    print(SPEC.describe().split("width_narrow")[0])
    for i, cfg in enumerate(SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        print(f"  task {i}: {cfg.source.polarization} pitch {cfg.grating.pitch_m*1e9:.2f} "
              f"corr {cfg.geometry.corrugation_depth_m*1e9:.1f} N {cfg.grating.n_periods_each_side} "
              f"| window {cfg.spectral.center_wavelength_m*1e9:.4f} +- {cfg.spectral.scan_width_nm/2:.1f} nm "
              f"x {cfg.spectral.n_wl_points} | box y {cfg.y_span*1e6:.2f} z {cfg.z_span*1e6:.2f} um "
              f"| ff monitors at y {kw['farfield_y_dist_m']*1e6:.2f} z {kw['farfield_z_dist_m']*1e6:.2f} um, "
              f"x-span {kw['farfield_x_span_m']*1e6:.0f} um, {kw['farfield_freq_points']} pts, "
              f"teeth {'per-tooth' if kw.get('width_narrow_per_tooth_m') else 'uniform'}")
