"""PLAIN TE second tooth with a POSITION RAMP along the device: an apodization written by the second tooth's phase.

Study dir: runners/sweeps/ | Created 2026-10-23 | Job(s): Athena (see dispatch note in the session; user: "work independently tonight")
Why: at a FIXED position the second tooth is only a coupling knob (plain_te_second_tooth 267575: rows sit on the plain
Q_i-vs-width curve; Nt80x 267814 + width-matched control 268447: tooth -8 %). What the fixed sweep never did is vary the
position ALONG the device. Tooth on the narrow half (DELTA 0) weakens the local Bragg coupling (kappa x~0.6), on the wide half
(DELTA 250) strengthens it (x~1.4): ramping DELTA from 0 at the cavity to 250 over X_RAMP makes kappa grow outward -- the
overshoot's kind of envelope, from one tooth size and one lattice. Same tooth as plain_te_second_tooth (3 x 50-nm units,
+79 nm above the local wall, both arms on their own lattice, edges on the mesh lines: DELTA quantised to 50 nm).
Read against the MEASURED plain Q_i(W) curve of 267575 (rows that do not touch the cavity), and against the plain device.
Numerics = plain_te_second_tooth (100 ps, 20-nm window centred 1567, far field + short_run); no control row.
Dispatch (Athena): SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.plain_te_tooth_ramp --max-concurrent=4 --gpu=a100
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runners.sweeps import plain_te_second_tooth as st  # noqa: E402  (device, tooth, numerics, half_width_nm)
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

X_RAMPS_UM = [5.0, 10.0, 20.0, 40.0]       # distance over which DELTA goes 0 -> 250 nm (then stays on the wide half)


def delta_at(s_nm, x_ramp_nm):
    """Tooth offset from the narrow-half centre at distance s from the first narrow centre, on the 50-nm grid."""
    return st.UNIT_NM * round(250.0 * min(1.0, max(0.0, s_nm) / x_ramp_nm) / st.UNIT_NM)


def sites(deltas):
    """Units of the second tooth at offset deltas[j] in period j (both arms, each on its own lattice)."""
    xs = []
    for j in range(st.N_SIDE):
        d = deltas[j]
        for c in (st.PITCH_NM / 4 + (j + 0.25) * st.PITCH_NM + d, -(st.PITCH_NM / 4 + (j + 0.75) * st.PITCH_NM) + d):
            for k in range(3):
                x = c + (k - 1) * st.UNIT_NM
                if st.PITCH_NM / 4 < abs(x) < st.END_NM and (x > 0) == (c > 0):
                    xs.append(x)
    return xs, [st.half_width_nm(x) for x in xs]


# ROUND 2 (2026-10-23 04:50, after 269066 + control 269113: ramp 5 um 2.68x the plain curve, its contrast-only control 2.24x,
# longer ramps worse): APPENDED rows (indices 0-3 = 269066, never move). Shorter ramp; mirrored ramp (0 -> -250: the tooth
# crosses the OTHER quadrature, opposite chirp sign); abrupt steps (0 for k periods, then 250: same magnitude change, no chirp).
RAMP = lambda r_um: [delta_at(j * st.PITCH_NM, r_um * 1e3) for j in range(st.N_SIDE)]
STEP = lambda k: [0.0 if j < k else 250.0 for j in range(st.N_SIDE)]
ROWS = [(f"ramp{r:g}um", RAMP(r)) for r in X_RAMPS_UM] + [
    ("ramp2p5um", RAMP(2.5)),
    ("ramp5um_mirror", [-d for d in RAMP(5.0)]),
    ("step3", STEP(3)), ("step5", STEP(5)), ("step10", STEP(10))]
XYD = [sites(ds) + (ds,) for _, ds in ROWS]
# the result/.fsp/.h5 tag reads only the count and the FIRST/LAST site (sim_helpers); every ramp has the same 576 units and
# the same first/last site, so the 4 tasks wrote ONE file and clobbered each other (268992, 2026-10-23). Rotating each row's
# list by its row index makes the tags unique; the drawn geometry is identical.
XYD = [(xs[k + 1:] + xs[:k + 1], ys[k + 1:] + ys[:k + 1], ds) for k, (xs, ys, ds) in enumerate(XYD)]   # k+1: never reuse the clobbered 268992 name
BASE = copy.deepcopy(st.BASE)
n = len(ROWS)
SPEC = SweepSpec(
    polarization              = [st.POL] * n,
    pitch_nm                  = [st.PITCH_NM] * n,
    corrugation_depth_nm      = [st.CORR_NM] * n,
    avg_width_nm              = [st.AVG_NM] * n,
    cavity_width_nm           = [None] * n,
    width_narrow_per_tooth_nm = [None] * n,
    width_wide_per_tooth_nm   = [None] * n,
    n_periods_each_side       = [st.N_SIDE] * n,
    center_wavelength_nm      = [st.LAM_NM] * n,
    scan_width_nm             = [st.WIN_NM] * n,
    y_span_um                 = [st.BOX_Y] * n,
    span_mult                 = [st.zmult(st.BOX_Z, st.LAM_NM)] * n,
    scatterer_shape           = ["rect"] * n,
    scatterer_x_span_um       = [st.UNIT_NM * 1e-3] * n,
    scatterer_y_span_nm       = [2 * st.PROT_NM] * n,
    scatterer_index           = [1.97] * n,
    scatterer_mirrored_y      = [True] * n,
    scatterer_x_list_nm       = [xyd[0] for xyd in XYD],
    scatterer_y_list_nm       = [xyd[1] for xyd in XYD],
    mode  = "zipped",
    label = "plain_te_tooth_ramp",
)

if __name__ == "__main__":
    tags = [(len(xs), round(xs[0]), round(xs[-1]), round(ys[0]), round(ys[-1])) for xs, ys, _ in XYD]
    assert len(set(tags)) == len(tags), ("file tags collide", tags)
    for (r, _), (xs, ys, ds), cfg in zip(ROWS, XYD, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        assert cfg.farfield.enabled and cfg.farfield.short_run and kw.get("short_run"), "no time traces"
        assert ds[0] == 0.0 and abs(ds[-1]) == 250.0 and all(abs(b) >= abs(a) for a, b in zip(ds, ds[1:])), ("ramp not monotonic 0 -> +-250", r)
        assert min(abs(x) for x in xs) >= 150.0, ("a unit touches the cavity", r)
        span = kw["scatterer_y_span_m"] * 1e9
        for x, y in zip(xs, ys):                 # one flat wall, edges on mesh lines, outer edge = wall + 79
            hw = st.half_width_nm(x)
            assert abs(st.half_width_nm(x - 24.999) - hw) < 1e-6 and abs(st.half_width_nm(x + 24.999) - hw) < 1e-6, ("straddles", r, x)
            assert abs(((x - 50.0) / 50.0) - round((x - 50.0) / 50.0)) < 1e-9, ("unit edge off the mesh lines", r, x)
            assert abs(y + span / 2 - (hw + st.PROT_NM)) < 1e-6, ("protrusion", r, x)
        k_wide = [abs(d) for d in ds].index(250.0)
        print(f"{r:15s}: {len(xs)} units | tag X{round(xs[0])}to{round(xs[-1])} | DELTA per period (first 12) {[int(d) for d in ds[:12]]} ... | wide centre from period "
              f"{k_wide} | window {cfg.spectral.center_wavelength_m*1e9:.0f}+-{cfg.spectral.scan_width_nm/2} | ps {os.environ.get('TM_SIM_TIME_PS')}")
