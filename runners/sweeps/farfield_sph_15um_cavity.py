"""BEYOND-INVERSE-DESIGN screen on the Nt80x +20 nm parent: features INSIDE the cavity segment, and sub-period notches.

Study dir: runners/sweeps/ | Created 2026-10-21 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Why (user 2026-10-21: what inverse design can reach is not interesting): the adjoint lane's parameters are per-tooth corr/avg/shift,
the comb and the WHOLE cavity-segment width; its first step was the inner contrast we already measured. Sidewall tabs on the teeth
(farfield_sph_15um_phase) are closed: they radiate 2-15 % of the leak with <= 8 % pattern overlap. Two families remain outside the
optimizer and sit where the leak is born:
  rows 0-3  100-nm SiN tabs on the walls of the 257-nm cavity segment (centre / both edges / right edge only), protrusion d.
  rows 4-5  NOTCHES (cladding-index rects cut d into the wall; the scatterer mesh order wins overlaps) at the left-arm sites of the
            half-period tab set: the measured left-arm response there (rh_L = rh both-arm row - rh right-only row) adds to the leak
            with Re<c,b>/|c|^2 = +0.066, |b|^2/|c|^2 = 0.049 at 5 nm, so a notch (-b at first order) predicts P/P_parent
            1 - 0.132 + 0.049 = 0.92 (5 nm) / 0.95 (2.5 nm). First order only: the tab basis failed its linearity gate (0.34).
Reading: scratch solve_v4.py --short vs the +20 parent (diff_g0001_p20nm), parity_readout.py; pass = P/P_parent <= 0.95 above the
per-row floor (|b|^2/|c|^2 ~ 0.012), lambda/W held.
Dispatch (Athena): SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_cavity \
                     --max-concurrent=4 --qos=4d_1g --gpu=a100
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_PHASE_DEV"] = "NT80X"
from runners.sweeps import farfield_sph_15um_phase as ph  # noqa: E402  (same +20 parent, numerics, rect size, wall rule)
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

HALF_CAV_NM = ph.PITCH_NM / 4                         # cavity segment |x| < 128.5 nm
EDGE_NM = HALF_CAV_NM - ph.SITE_LEN_NM / 2 - 5.0      # edge tab centre: footprint ends 5 nm inside the segment
RH_LEFT = [x for x in ph.sites(ph.PITCH_NM / 2, 5.0)[0] if x < 0]     # left-arm sites of the half-period set (rows rh_d5 / rh_d5_R)
ROWS = [  # (name, x list nm, depth nm, index): index 1.97 = added SiN tab, 1.444 = notch
    ("cav_c_d5",       [0.0],                 5.0, 1.97),
    ("cav_c_d2p5",     [0.0],                 2.5, 1.97),
    ("cav_e_d5",       [-EDGE_NM, EDGE_NM],   5.0, 1.97),
    ("cav_eR_d5",      [EDGE_NM],             5.0, 1.97),
    ("notch_rhL_d5",   RH_LEFT,               5.0, 1.444),
    ("notch_rhL_d2p5", RH_LEFT,               2.5, 1.444),
    # rows 6-7 APPENDED 2026-10-21 after 266486 (MEASURED vs the +20 parent: notch 5 nm P 0.896, Q_i +9 %; 2.5 nm 0.934): depth scan.
    ("notch_rhL_d7p5", RH_LEFT,               7.5, 1.444),
    ("notch_rhL_d10",  RH_LEFT,              10.0, 1.444),
]


def y_list(xs, d, n):
    """Tab: outer edge at wall + d (rect sits 60 - d inside the core). Notch: inner edge at wall - d (rect 60 - d outside)."""
    s = 1 if n > 1.5 else -1
    return [ph.half_width_nm(x) + s * (d - ph.SITE_Y_SPAN_NM / 2) for x in xs]


n = len(ROWS)
SPEC = SweepSpec(
    polarization              = [ph.POL] * n,
    pitch_nm                  = [ph.PITCH_NM] * n,
    corrugation_depth_nm      = [ph.CORR_NM] * n,
    avg_width_nm              = [ph.AVG_NM] * n,
    cavity_width_nm           = [ph.CAV_W_NM] * n,
    width_narrow_per_tooth_nm = [ph.TEETH[0]] * n,
    width_wide_per_tooth_nm   = [ph.TEETH[1]] * n,
    n_periods_each_side       = [ph.N_SIDE] * n,
    center_wavelength_nm      = [ph.LAM_NM] * n,
    scan_width_nm             = [ph.WIN_NM] * n,
    y_span_um                 = [ph.BOX_Y] * n,
    span_mult                 = [ph.zmult(ph.BOX_Z, ph.LAM_NM)] * n,
    scatterer_shape           = ["rect"] * n,
    scatterer_x_span_um       = [ph.SITE_LEN_NM * 1e-3] * n,
    scatterer_y_span_nm       = [ph.SITE_Y_SPAN_NM] * n,
    scatterer_index           = [r[3] for r in ROWS],
    scatterer_mirrored_y      = [True] * n,
    scatterer_x_list_nm       = [r[1] for r in ROWS],
    scatterer_y_list_nm       = [y_list(r[1], r[2], r[3]) for r in ROWS],
    mode  = "zipped",
    label = "farfield_sph_15um_cavity",
)
BASE = ph.BASE

if __name__ == "__main__":
    assert abs(ph.NAR[0] - 940.3) < 1e-6 and abs(ph.WID[0] - 960.3) < 1e-6, "parent is not the +20 nm contrast device"
    for (name, xs, d, idx), cfg in zip(ROWS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        xb = [v * 1e9 for v in kw["scatterer_x_list_m"]]; yb = [v * 1e9 for v in kw["scatterer_y_list_m"]]
        for x, y in zip(xb, yb):                       # property asserts (CLAUDE.md 5): footprint on one flat wall, sign of d right
            for e in (-ph.SITE_LEN_NM / 2, ph.SITE_LEN_NM / 2):
                assert abs(ph.half_width_nm(x + e) - ph.half_width_nm(x)) < 1e-6, ("footprint straddles a step", name, x)
            edge = y + ph.SITE_Y_SPAN_NM / 2 if idx > 1.5 else y - ph.SITE_Y_SPAN_NM / 2
            assert abs(edge - (ph.half_width_nm(x) + (d if idx > 1.5 else -d))) < 1e-6, ("wrong protrusion/notch depth", name)
        if name.startswith("cav"):
            assert all(abs(x) + ph.SITE_LEN_NM / 2 < HALF_CAV_NM for x in xb), ("cavity tab leaves the segment", name)
        else:
            assert all(x < 0 for x in xb) and abs(kw["scatterer_index"] - 1.444) < 1e-9, ("notch row wrong", name)
        print(f"{name:15s}: n {idx} sites {[round(x) for x in xb]} y {[round(y, 1) for y in yb]} | d {d} | short {cfg.farfield.short_run} "
              f"sim ps {os.environ.get('TM_SIM_TIME_PS')} box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f}")
