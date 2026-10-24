"""PLAIN TE pi-shift device: local SHOULDERS at the two cavity-junction wall steps (the compact leak source).

Study dir: runners/sweeps/ | Created 2026-10-21 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Why (user 2026-10-21: understand cancellation on the plain device first, semi-analytically, then transfer to the envelope):
the plain TE leak is NOT the envelope Fourier tail (that fits TM, rms 0.35; TE rms 1.56) - it is broad in u_x and its band-limited
source image puts 90 % within |x| <= 2 um, strongest within +-1 um. The cavity segment (pitch/2 at the mean width 800 nm) meets the
right arm's NARROW half (642) at x = +125 nm and the left arm's WIDE half (958) at x = -125 nm: each sidewall steps 79 nm there.
GPT (01a15a66) proposes those two junctions as the localized radiators; a short SiN shoulder on the SMALLER-width side of a step
softens it locally. Rows (100-nm footprint, both sidewalls, full height, 5 nm clear of the step):
  0 J_R_d10  shoulder on the right-arm narrow wall just right of +125 nm
  1 J_L_d10  shoulder on the cavity wall just right of -125 nm (the left step's smaller side)
  2 J_LR_d10 both
  3 J_R_d5   linearity of row 0
Control = the stored round-A plain TE far field (farfield_sph_15um/results/result_N96_avg_C316_*_ff.mat), Lorentzian normalisation
(scratch solve_v4.py without --short). 10 nm (not GPT's 5) so one shoulder pair resolves above the plain-row floor (the 14-tab
10-nm array was |b|^2/|c|^2 0.012). Pass: a resolved >= 10 % total reduction with lambda/W held -> half-depth + model fit next.
Dispatch (Athena): bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_junction_plain --max-concurrent=1 --gpu=a100
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_PHASE_DEV"] = "PLAIN"
from runners.sweeps import farfield_sph_15um_phase as ph  # noqa: E402  (plain round-A device, numerics, rect size, wall rule)
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

A_NM = ph.PITCH_NM / 4                                   # junction position (cavity segment ends)
OFF_NM = ph.SITE_LEN_NM / 2 + 5.0                        # shoulder centre 5 nm clear of the step
XR, XL = A_NM + OFF_NM, -A_NM + OFF_NM                   # right: narrow-arm wall; left: cavity wall (right of the -125 step)
ROWS = [("J_R_d10", [XR], 10.0), ("J_L_d10", [XL], 10.0), ("J_LR_d10", [XL, XR], 10.0), ("J_R_d5", [XR], 5.0)]


def shoulder_x(xs_nm):
    """Centre of a shoulder on the SMALLER-width side of the wall step at xs_nm, 5 nm clear of the step."""
    left, right = ph.half_width_nm(xs_nm - 60), ph.half_width_nm(xs_nm + 60)
    return xs_nm + OFF_NM if right < left else xs_nm - OFF_NM


# rows 4-8 APPENDED 2026-10-21 (GPT 01a15a66 s22): complete a one-shoulder-per-step basis over the six steps that carry 99 % of the
# plain TE leak (source fit): left-junction half depth, then the second and third steps on each side (+-3a, +-5a). Depth 10 nm,
# revised after rows 0-3 if their responses are unresolved or nonlinear. Indices 0-3 (266507) must not move.
ROWS += [("J_L_d5", [XL], 5.0)] + [(f"S_{'p' if j > 0 else 'm'}{abs(j)}a_d10", [shoulder_x(j * A_NM)], 10.0) for j in (-5, -3, 3, 5)]

n = len(ROWS)
SPEC = SweepSpec(
    polarization              = [ph.POL] * n,
    pitch_nm                  = [ph.PITCH_NM] * n,
    corrugation_depth_nm      = [ph.CORR_NM] * n,
    avg_width_nm              = [ph.AVG_NM] * n,
    cavity_width_nm           = [ph.CAV_W_NM] * n,
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
    scatterer_index           = [1.97] * n,
    scatterer_mirrored_y      = [True] * n,
    scatterer_x_list_nm       = [r[1] for r in ROWS],
    scatterer_y_list_nm       = [[ph.half_width_nm(x) + d - ph.SITE_Y_SPAN_NM / 2 for x in xs] for _, xs, d in ROWS],
    mode  = "zipped",
    label = "farfield_sph_15um_junction_plain",
)
BASE = ph.BASE

if __name__ == "__main__":
    assert ph.TEETH is None and abs(ph.PITCH_NM - 500.0) < 1e-9, "not the plain TE device"
    for (name, xs, d), cfg in zip(ROWS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        xb = [v * 1e9 for v in kw["scatterer_x_list_m"]]; yb = [v * 1e9 for v in kw["scatterer_y_list_m"]]
        for x, y in zip(xb, yb):                       # property asserts: one flat wall, protrusion d, on the SMALLER side of a step
            for e in (-ph.SITE_LEN_NM / 2, ph.SITE_LEN_NM / 2):
                assert abs(ph.half_width_nm(x + e) - ph.half_width_nm(x)) < 1e-6, ("footprint straddles a step", name, x)
            assert abs(y + ph.SITE_Y_SPAN_NM / 2 - (ph.half_width_nm(x) + d)) < 1e-6, ("wrong protrusion", name)
            step_x = A_NM if x > 0 else -A_NM
            assert ph.half_width_nm(x) < ph.half_width_nm(step_x - 60 if x > 0 else step_x - 60), ("not the smaller side", name, x)
        print(f"{name:9s}: x {[round(v) for v in xb]} wall {[ph.half_width_nm(v) for v in xb]} y {[round(v, 1) for v in yb]} d {d} | "
              f"sim ps {os.environ.get('TM_SIM_TIME_PS')} short {cfg.farfield.short_run} box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} lam {cfg.spectral.center_wavelength_m*1e9:.1f}")
