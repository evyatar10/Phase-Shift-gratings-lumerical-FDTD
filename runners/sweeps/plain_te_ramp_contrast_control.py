"""Control for plain_te_tooth_ramp: the SAME per-period Bragg strength written by the tooth widths, no second tooth.

Study dir: runners/sweeps/ | Created 2026-10-23 | Job(s): Athena (see dispatch note in the session; overnight autonomous block)
Why: the tooth-position ramp (269066, X_ramp 5 um) gave Q_i 4.07e4 at W 14.26 um = 2.68x the plain Q_i(W) curve of 267575
(plain device 1.12x). Is that the second tooth, or just an apodization (kappa low at the cavity, high outside)?
Control: per period j the first Fourier (Bragg) component of tooth + second tooth is (1 + 0.4045 e^{i phi_j}) times the plain
one (0.4045 = 0.5 * sin(0.3 pi): a c/2 bump over 0.3 pitch; phi_j = 2 pi (DELTA_j - 250) / 500 from the wide-half centre).
This row draws only the MAGNITUDE S_j = |1 + 0.4045 e^{i phi_j}| as a contrast: S >= 1 -> wide = narrow + S c (narrow 642
kept), S < 1 -> narrow = wide - S c (wide 958 kept; the tooth sat on the narrow wall). The phase part (a local chirp) and the
tooth's own scattering are what this control leaves out. Same plain numerics as the ramp (100 ps, 20-nm window, short_run).
Dispatch (Athena): SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.plain_te_ramp_contrast_control --gpu=a100
"""
import cmath
import copy
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runners.sweeps import plain_te_second_tooth as st  # noqa: E402
from runners.sweeps import plain_te_tooth_ramp as tr  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

X_RAMP_UM = 5.0
A = 0.5 * math.sin(0.3 * math.pi)
NAR0, WID0 = st.AVG_NM - st.CORR_NM / 2, st.AVG_NM + st.CORR_NM / 2
DELTAS = tr.XYD[tr.X_RAMPS_UM.index(X_RAMP_UM)][2]          # the ramp row's per-period DELTA (nm)
S = [abs(1 + A * cmath.exp(1j * 2 * math.pi * (d - 250.0) / st.PITCH_NM)) for d in DELTAS]
NAR = [NAR0 if s >= 1 else WID0 - s * st.CORR_NM for s in S]
WID = [NAR0 + s * st.CORR_NM if s >= 1 else WID0 for s in S]

# ROW 1 (appended 2026-10-23 05:50, GPT review 01a166a6): DC-AND-FUNDAMENTAL-matched control. The tooth ramp adds a CONSTANT
# 158 * 150/500 = 47.4 nm to the period-averaged width everywhere (mean 847.4 nm), while row 0's mean width varies 807..864 nm
# (a local detuning profile). Row 1 keeps every period's mean at 847.4 nm and its contrast at S_j * 316 nm; cavity unchanged.
MEAN_TOOTH_NM = st.AVG_NM + (st.CORR_NM / 2) * (st.UNIT_NM * 3 / st.PITCH_NM)
NAR_DC = [MEAN_TOOTH_NM - s * st.CORR_NM / 2 for s in S]
WID_DC = [MEAN_TOOTH_NM + s * st.CORR_NM / 2 for s in S]
ROWS = [("contrast_only", NAR, WID), ("dc_matched", NAR_DC, WID_DC)]
BASE = copy.deepcopy(st.BASE)
BASE.scatterer.enabled = False
n = len(ROWS)
SPEC = SweepSpec(
    polarization              = [st.POL] * n,
    pitch_nm                  = [st.PITCH_NM] * n,
    corrugation_depth_nm      = [st.CORR_NM] * n,
    avg_width_nm              = [st.AVG_NM] * n,
    cavity_width_nm           = [st.AVG_NM] * n,
    width_narrow_per_tooth_nm = [r[1] for r in ROWS],
    width_wide_per_tooth_nm   = [r[2] for r in ROWS],
    n_periods_each_side       = [st.N_SIDE] * n,
    center_wavelength_nm      = [st.LAM_NM] * n,
    scan_width_nm             = [st.WIN_NM] * n,
    y_span_um                 = [st.BOX_Y] * n,
    span_mult                 = [st.zmult(st.BOX_Z, st.LAM_NM)] * n,
    mode  = "zipped",
    label = "plain_te_ramp_contrast_control",
)

if __name__ == "__main__":
    tags = [(round(r[1][0], 1), round(r[1][-1], 1), round(r[2][0], 1), round(r[2][-1], 1)) for r in ROWS]
    assert len(set(tags)) == len(tags), ("file tags collide", tags)
    assert abs(S[0] - (1 - A)) < 1e-9 and abs(S[-1] - (1 + A)) < 1e-9, "ramp ends are not the narrow/wide-centre tooth"
    for (name, nar, wid), cfg in zip(ROWS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        n_b = [v * 1e9 for v in kw["width_narrow_per_tooth_m"]]; w_b = [v * 1e9 for v in kw["width_wide_per_tooth_m"]]
        assert len(n_b) == st.N_SIDE and all(abs(a - b) < 1e-6 for a, b in zip(n_b, nar)) and all(abs(a - b) < 1e-6 for a, b in zip(w_b, wid)), name
        assert all(abs((w - q) - s * st.CORR_NM) < 1e-6 for q, w, s in zip(n_b, w_b, S)), ("contrast != S * corr", name)
        assert not kw.get("scatterer_enabled", False) and abs(kw["cavity_width_m"] * 1e9 - 800.0) < 1e-6, ("scatterers on / cavity changed", name)
        if name == "dc_matched":
            assert all(abs((q + w) / 2 - 847.4) < 1e-6 for q, w in zip(n_b, w_b)), "mean width not 847.4"
        means = [(q + w) / 2 for q, w in zip(n_b, w_b)]
        print(f"{name:14s}: narrow {[round(v) for v in n_b[:8]]}.. {n_b[-1]:.1f} | wide {[round(v) for v in w_b[:8]]}.. {w_b[-1]:.1f} | mean {min(means):.1f}-{max(means):.1f}"
              f" | cavity {kw['cavity_width_m']*1e9:.0f} | ps {os.environ.get('TM_SIM_TIME_PS')}")
