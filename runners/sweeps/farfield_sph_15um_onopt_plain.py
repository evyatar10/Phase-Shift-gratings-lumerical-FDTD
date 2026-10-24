"""PLAIN TE: do added features still help ON TOP OF the wide (better) cavity? (owner question 2026-10-21)

Study dir: runners/sweeps/ | Created 2026-10-21 | Job(s): Athena (see docs/HANDOFF_radiation_cancellation_2026-10-21.md)
Why: on the 800-nm cavity the right-step shoulder gave R 0.958, but the cavity-width ladder reaches 0.899 at 1050 nm by itself
(farfield_sph_15um_cavsize_plain). A feature is interesting only if it still cuts radiation on the better cavity. Rows:
  0 cav1050_J_R_d10    1050-nm cavity + 10-nm shoulder on the narrow wall right of the +125 nm step   (vs cavsize card cavW_1050nm)
  1 cav1050_notch_d5   1050-nm cavity + the left-arm notch set (7 sites, 5 nm, half-period registration) (vs cavW_1050nm)
  2 cav800_notch_d5    800-nm cavity + the same notch set                                              (vs the stored round-A control)
The notch set is the plain-device analogue of the Nt80x notches that cut 0.896 (left-arm sites of the half-period tab set,
farfield_sph_15um_phase PLAIN geometry). Read with python_tools/radcancel/solve_v4.py (no --short), each vs its own control.
Dispatch (Athena): bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_onopt_plain --max-concurrent=3 --gpu=a100
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_PHASE_DEV"] = "PLAIN"
from runners.sweeps import farfield_sph_15um_phase as ph  # noqa: E402  (plain round-A device, numerics, rect size)
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

XR = ph.PITCH_NM / 4 + ph.SITE_LEN_NM / 2 + 5.0                  # shoulder centre, 5 nm clear of the +125 nm step (narrow wall)
NOTCH_X = [x for x in ph.sites(ph.PITCH_NM / 2, 5.0)[0] if x < 0]  # left-arm sites of the plain half-period set
ROWS = [  # (name, cavity width nm, x list, depth nm, index)
    ("cav1050_J_R_d10",  1050.0, [XR],     10.0, 1.97),
    ("cav1050_notch_d5", 1050.0, NOTCH_X,   5.0, 1.444),
    ("cav800_notch_d5",   800.0, NOTCH_X,   5.0, 1.444),
]


def y_list(xs, d, n):
    s = 1 if n > 1.5 else -1                                         # tab: outer edge wall + d; notch: inner edge wall - d
    return [ph.half_width_nm(x) + s * (d - ph.SITE_Y_SPAN_NM / 2) for x in xs]


n = len(ROWS)
SPEC = SweepSpec(
    polarization              = [ph.POL] * n,
    pitch_nm                  = [ph.PITCH_NM] * n,
    corrugation_depth_nm      = [ph.CORR_NM] * n,
    avg_width_nm              = [ph.AVG_NM] * n,
    cavity_width_nm           = [r[1] for r in ROWS],
    width_narrow_per_tooth_nm = [None] * n,
    width_wide_per_tooth_nm   = [None] * n,
    n_periods_each_side       = [ph.N_SIDE] * n,
    center_wavelength_nm      = [ph.LAM_NM] * n,
    scan_width_nm             = [ph.WIN_NM] * n,
    y_span_um                 = [ph.BOX_Y] * n,
    span_mult                 = [ph.zmult(ph.BOX_Z, ph.LAM_NM)] * n,
    scatterer_shape           = ["rect"] * n,
    scatterer_x_span_um       = [ph.SITE_LEN_NM * 1e-3] * n,
    scatterer_y_span_nm       = [ph.SITE_Y_SPAN_NM] * n,
    scatterer_index           = [r[4] for r in ROWS],
    scatterer_mirrored_y      = [True] * n,
    scatterer_x_list_nm       = [r[2] for r in ROWS],
    scatterer_y_list_nm       = [y_list(r[2], r[3], r[4]) for r in ROWS],
    mode  = "zipped",
    label = "farfield_sph_15um_onopt_plain",
)
BASE = ph.BASE

if __name__ == "__main__":
    assert ph.TEETH is None and abs(ph.PITCH_NM - 500.0) < 1e-9, "not the plain TE device"
    for (name, cw, xs, d, idx), cfg in zip(ROWS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        assert abs(kw["cavity_width_m"] - cw * 1e-9) < 1e-12, ("cavity width did not reach the builder", name)
        xb = [v * 1e9 for v in kw["scatterer_x_list_m"]]; yb = [v * 1e9 for v in kw["scatterer_y_list_m"]]
        for x, y in zip(xb, yb):                 # property asserts: one flat wall (arm walls; the cavity width does not enter these sites)
            assert abs(x) > ph.PITCH_NM / 4 + ph.SITE_LEN_NM / 2, ("site inside the cavity segment", name, x)
            for e in (-ph.SITE_LEN_NM / 2, ph.SITE_LEN_NM / 2):
                assert abs(ph.half_width_nm(x + e) - ph.half_width_nm(x)) < 1e-6, ("footprint straddles a step", name, x)
            edge = y + ph.SITE_Y_SPAN_NM / 2 if idx > 1.5 else y - ph.SITE_Y_SPAN_NM / 2
            assert abs(edge - (ph.half_width_nm(x) + (d if idx > 1.5 else -d))) < 1e-6, ("wrong depth", name)
        if idx < 1.5:
            assert all(x < 0 for x in xb) and abs(kw["scatterer_index"] - 1.444) < 1e-9, ("notch row wrong", name)
        print(f"{name:17s}: cavity {kw['cavity_width_m']*1e9:.0f} nm | n {idx} sites {[round(v) for v in xb]} d {d} | sim ps "
              f"{os.environ.get('TM_SIM_TIME_PS')} window {cfg.spectral.center_wavelength_m*1e9:.1f}+-{cfg.spectral.scan_width_nm/2}")
