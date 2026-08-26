"""Itai Lev-Ran's 60-period "HH" apodization at OUR Q3dB operating point (IGUM).

Study dir: runners/sweeps/   |   Created 2026-08-25   |   Job(s): TBD
Purpose: put his custom apodization at the Q3dB point (spatial mode fwhm_m = 20 um
AND peak T = 0.5) in BOTH polarizations, so the loaded Q can be compared against
our own Q3dB devices:
    TE  result_N166_avg_C250.mat                 lam 1559.79 nm, Q 12903, mode 20.46 um
    TM  result_N165_TM_avg_C325_Ybox8p0_Zbox8p8  lam 1559.00 nm, Q 13930, mode 19.97 um

GEOMETRY is taken AS DRAWN from share_with_Evyatar/ -- his 61-value
Nt60_highbulk_HighCorr_dw.npy pushed through his own bragg_lib chain
(advanced_dw_correction 3D LUT, then advanced_index_correction which holds the
period-averaged neff at its bulk value). His profile is NOT a taper: dw runs 0 at
the cavity -> OVERSHOOT 1200 nm (2.4x the 500 nm bulk) at d~24-26 -> dip 603 ->
second lobe 1188 at d~50-54 -> bulk 500 from d=61. avg_wg 1.0 um, cavity =
pitch/2 at 950.3 nm. Widths below are the drawn (narrow, wide) pairs.

Polarization of his design = TE: his MEASURED neff LUT (analyse_neff_report.pdf,
fabricated chip IB1) matches our simulated TE0 at n=1.97/h=350nm to 0.2-0.5% over
W=0.6-1.3 um, while TM is off by 7%.

TWO KNOBS, established by a pre-dispatch transfer-matrix pass (no GPU):
  * mode width  <- the apodization AMPLITUDE only. fwhm is invariant at 14.24 um
    over N=55..130 AND over bulk corrugation 500..200 nm (the mode lives entirely
    inside the 60-period apodized core), but moves 14.2 -> 25.7 um as the profile
    is scaled 1.0 -> 0.7. Scale ~0.78 is the 20 um candidate -> the ladder below.
  * peak T      <- N.  (round 2)
Per user decision: 20 um is the spec, so the profile SHAPE is kept and only its
depth is scaled; no unscaled rows. TM keeps the drawn (TE-index-corrected) widths
and changes pitch only -- the residual TM chirp is accepted and reported.

ROUND 1 (this SPEC, 6 tasks): scale ladder at N=98, both polarizations, each arm
pitch-retuned to its own Q3dB resonance. Delivers fwhm_m(scale) -> lock 20 um,
AND lam_res -> the exact pitch correction for round 2 (lam_res is proportional to
pitch for a Bragg device, so one round locks both). Q_i comes out of each row via
Q_i = Q_L/(1-sqrt(T)), which predicts round 2's N.
ROUND 2 (later edit): N ladder at the locked scale + corrected pitch -> ln(T) vs
N -> the -3 dB crossing -> the deliverable Q.

Containment: half-device = N*pitch must be >= 2x the 20 um target (asserted).
N=98 -> 49.6 um (TE) / 51.2 um (TM). This also sets the floor N >= 78.

Windows: TE centred 1559.79 nm (its pitch comes from the TMM, which is anchored
on our own stack -> good to a few nm) 30 nm wide; TM centred 1559.00 nm and 60 nm
wide because its starting pitch is a cruder TE/TM ratio estimate. 4001 pts.

Dispatch (IGUM -- Athena is holding the lumopt2 campaign 137075 under a HANDOFF
deploy freeze and its home is at 289/300 GB; re-probe IGUM's queue + seats first
and leave headroom for what is already running there):
    SBATCH_MEM=160G bash igum/deploy_igum.sh         --option3 --spec=runners.sweeps.itai_hh_apod --max-concurrent=3
Output -> results/itai_hh_apod/results/ (download to results_from_igum/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from runners.sweeps.sweep_spec import SweepSpec

AVG_WIDTH_NM = 1000.0
CAVITY_W_NM  = 950.3           # his index-corrected mid-section width
BULK_NARROW_NM, BULK_WIDE_NM = 746.9, 1257.1     # d >= 61 (drawn dw = 510.2 nm)
N_APOD       = 60
N_SIDE       = 98
SCALES       = [0.72, 0.78, 0.84]

# Each arm is retuned onto ITS OWN measured Q3dB resonance. lambda_target and
# the bias factor K = lambda_measured/lambda_predicted come from our two stored
# anchors, which the FDE curve above reproduces to -0.46% (TE) / -0.23% (TM):
#   TE  result_N166_avg_C250.mat                lam 1559.79  (pitch 500.00, W800 corr250)
#   TM  result_N165_TM_avg_C325_Ybox8p0_Zbox8p8 lam 1559.00  (pitch 516.83, W800 corr325)
LAMBDA = {"TE": 1559.79, "TM": 1559.00}
K_BIAS = {"TE": 1559.79 / 1552.65, "TM": 1559.00 / 1555.38}
WINDOW = {"TE": 30.0, "TM": 60.0}   # TM wider: its arm carries TE-corrected widths

N_WL_POINTS = 4001

# Our own FDE n_eff(W) for TE0 at 1559.4 nm (n=1.97/1.444, h=350 nm; PML, 500x400).
# Used ONLY to re-solve the per-period mid width when the profile is scaled --
# see teeth(). Gated against our two measured Q3dB resonances (TE -0.46%,
# TM -0.23%); those residuals are folded into PITCH_*_NM below.
NEFF_W_UM = [0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95]
NEFF_TE   = [1.49682, 1.52198, 1.54473, 1.56456, 1.58130, 1.59533, 1.60695, 1.61666, 1.62488, 1.63182, 1.63776, 1.64280, 1.64717, 1.65097, 1.65430]
NEFF_TM   = [1.47793, 1.49018, 1.50132, 1.51119, 1.51976, 1.52720, 1.53363, 1.53921, 1.54406, 1.54827, 1.55197, 1.55523, 1.55811, 1.56065, 1.56293]

APOD_NARROW_NM = [   # d=1 (cavity centre) .. d=61, from Nt60_highbulk_HighCorr_dw.npy
    950.3, 912.5, 894.4, 867.9, 842.6, 819.2, 797.2, 776.3,
    756.4, 737.5, 719.8, 703.0, 687.3, 672.7, 659.0, 646.1,
    634.0, 622.6, 612.0, 601.9, 592.7, 584.9, 579.3, 577.0,
    577.0, 577.0, 579.4, 583.8, 596.4, 610.9, 625.8, 641.4,
    657.8, 675.2, 690.5, 702.7, 711.2, 715.5, 715.5, 711.0,
    702.2, 689.7, 674.5, 658.0, 642.3, 627.2, 612.6, 597.9,
    583.0, 580.4, 579.9, 579.7, 579.9, 580.4, 594.9, 620.8,
    647.4, 676.5, 707.9, 732.3, 746.9,
]
APOD_WIDE_NM = [
    950.3, 990.5, 1011.7, 1045.4, 1080.9, 1117.1, 1154.7, 1194.2,
    1235.8, 1279.3, 1325.0, 1372.8, 1422.4, 1473.5, 1526.1, 1579.8,
    1634.0, 1687.8, 1740.4, 1790.8, 1837.1, 1876.2, 1903.7, 1915.1,
    1915.1, 1915.1, 1903.1, 1881.7, 1818.9, 1745.8, 1672.5, 1600.6,
    1531.0, 1464.4, 1411.8, 1373.6, 1348.8, 1336.6, 1336.8, 1349.5,
    1375.3, 1414.7, 1466.7, 1529.8, 1596.4, 1665.8, 1737.5, 1811.0,
    1885.7, 1898.5, 1900.9, 1901.6, 1900.8, 1898.5, 1826.4, 1696.9,
    1574.1, 1459.6, 1358.4, 1292.2, 1257.1,
]

assert len(APOD_NARROW_NM) == len(APOD_WIDE_NM) == N_APOD + 1


def _neff(w_nm):
    return float(np.interp(w_nm * 1e-3, NEFF_W_UM, NEFF_TE))


def _solve_mid(dw_nm, n_target):
    """Mid width whose period-averaged n_eff equals n_target, at corrugation dw.

    This is Itai's own advanced_index_correction criterion
    (bragg_lib.apodized_bragg_resonator), re-solved with OUR n_eff curve. It has
    to be re-solved whenever the profile is scaled: scaling dw about a FIXED mid
    width breaks the correction and chirps the cavity (measured: 0.18% spread at
    scale 1.0 -> 0.98% at 0.72, i.e. ~15 nm of Bragg lambda across the mode).
    """
    lo, hi = 0.30 * 1e3, 2.60 * 1e3
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if (_neff(m + dw_nm / 2) + _neff(m - dw_nm / 2)) / 2 < n_target:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def teeth(n_side, scale):
    """Per-tooth (narrow, wide) widths in nm, innermost tooth first.

    His SHAPE is preserved exactly; only its depth is scaled (user decision:
    20 um is the spec). The mid width of every period is then re-solved so the
    period-averaged n_eff stays at the scaled BULK value, which is what his
    index correction does -- at scale 1.0 this reproduces his drawn widths.
    """
    nar = list(APOD_NARROW_NM) + [BULK_NARROW_NM] * (n_side - N_APOD - 1)
    wid = list(APOD_WIDE_NM) + [BULK_WIDE_NM] * (n_side - N_APOD - 1)
    dw_bulk = (BULK_WIDE_NM - BULK_NARROW_NM) * scale
    n_target = (_neff(AVG_WIDTH_NM + dw_bulk / 2) + _neff(AVG_WIDTH_NM - dw_bulk / 2)) / 2
    out_n, out_w = [], []
    for a, b in zip(nar, wid):
        dw = (b - a) * scale
        mid = _solve_mid(dw, n_target)
        out_n.append(round(mid - dw / 2.0, 1))
        out_w.append(round(mid + dw / 2.0, 1))
    return out_n, out_w


def pitch_for(pol, scale):
    """Pitch that lands this row on its target resonance: lambda = K*2*<n>*pitch.

    <n> is the per-period (narrow+wide)/2 averaged over the mode envelope (a
    20 um FWHM Gaussian on the cavity -- the target width), so the periods the
    mode actually samples set the pitch. Iterated because the envelope is
    measured in periods.
    """
    curve = NEFF_TE if pol == "TE" else NEFF_TM
    tn, tw = teeth(N_SIDE, scale)
    nd = np.array([(np.interp(x * 1e-3, NEFF_W_UM, curve)
                    + np.interp(y * 1e-3, NEFF_W_UM, curve)) / 2 for x, y in zip(tn, tw)])
    x = (np.arange(N_SIDE) + 0.5)
    pitch = 500.0
    for _ in range(30):
        w = np.exp(-4 * np.log(2) * (x * pitch * 1e-3 / 20.0) ** 2)
        pitch = LAMBDA[pol] / (K_BIAS[pol] * 2 * float(np.sum(w * nd) / np.sum(w)))
    return round(pitch, 2)


# ── ROUND 2 (2026-08-26): the 20 um comparison, at boxes we have now MEASURED.
# Round 1a settled the box: TE converged between y6.0/z5.0 and y7.5/z6.9
# (T 0.9536 -> 0.9566, T+R 0.9576 -> 0.9582, both < 1). The TM box was already
# known from the project's own N100 TM series (converged by ~5.8-6.8 um) and the
# inverse-design standard 8.0 x 8.8 -- so TM uses 8.0/z-mult 5.4, not a ladder.
#
# Why these scales: fwhm_m is box-INDEPENDENT (17.61 um at every box, to 0.01),
# so the scale->width line measured on the bad-box rows is still valid:
#     0.72 -> 17.61 um, 0.78 -> 16.88 um  =>  -12.17 um per unit scale
#     20 um  =>  scale 0.524 (TE).  TM already gives 20.08 um at scale 0.72.
# TE rows bracket it (0.52, 0.58) because 0.72 -> 0.52 is a long extrapolation.
#
# Row 4 re-measures OUR TE Q3dB baseline at the same box: its stored value used
# z = 3.16 um, the same undersized vertical box that made his device read T > 1.
# A named box change is the one justification CLAUDE.md 6 allows for that.
BOX_Y_UM, BOX_Z_MULT = 6.8, 4.14      # IDENTICAL to the inverse-design programme
#   (campaign_c325_seed*.py: box_y_um=6.8, box_z_mult=4.14 -> z 6.81 um). Our own
#   round-1a ladder converged EARLIER than this (y6.0/z5.03 -> y7.5/z6.9 moved T+R
#   0.9576 -> 0.9582), so 6.8x6.8 is comfortably converged AND makes every number
#   here directly comparable to the inverse-design results (CLAUDE.md 2).

# (polarization, scale, pitch_nm, centre_nm, window_nm, y_span_um, span_mult, baseline)
ROWS = [("TE", 0.52, pitch_for("TE", 0.52), LAMBDA["TE"], WINDOW["TE"], BOX_Y_UM, BOX_Z_MULT, False),
        ("TE", 0.58, pitch_for("TE", 0.58), LAMBDA["TE"], WINDOW["TE"], BOX_Y_UM, BOX_Z_MULT, False),
        ("TM", 0.72, pitch_for("TM", 0.72), LAMBDA["TM"], WINDOW["TM"], BOX_Y_UM, BOX_Z_MULT, False),
        ("TE", None, 500.0, 1559.79, 30.0, BOX_Y_UM, BOX_Z_MULT, True)]

_T = [(None, None) if b else teeth(N_SIDE, sc) for (_p, sc, _pi, _c, _w, _y, _m, b) in ROWS]

SPEC = SweepSpec(
    n_periods_each_side       = [166 if b else N_SIDE for *_r, b in ROWS],
    pitch_nm                  = [p for _pol, _s, p, _c, _w, _y, _m, _b in ROWS],
    avg_width_nm              = [800.0 if b else AVG_WIDTH_NM for *_r, b in ROWS],
    cavity_width_nm           = [None if b else CAVITY_W_NM for *_r, b in ROWS],
    corrugation_depth_nm      = [250.0 if b else round((BULK_WIDE_NM - BULK_NARROW_NM) * sc, 1)
                                 for _pol, sc, _p, _c, _w, _y, _m, b in ROWS],
    width_narrow_per_tooth_nm = [t[0] for t in _T],
    width_wide_per_tooth_nm   = [t[1] for t in _T],
    polarization              = [pol for pol, *_r in ROWS],
    center_wavelength_nm      = [c for _pol, _s, _p, c, _w, _y, _m, _b in ROWS],
    scan_width_nm             = [w for _pol, _s, _p, _c, w, _y, _m, _b in ROWS],
    y_span_um                 = [y for *_r, y, m, _b in [(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]) for r in ROWS]],
    span_mult                 = [r[6] for r in ROWS],
    mode  = "zipped",
    label = "itai_hh_apod",
)

if __name__ == "__main__":
    print(SPEC.describe().split("width_narrow")[0])
    for i, (pol, s, pitch, c, w) in enumerate(ROWS):
        tn, tw = _T[i]
        print(f"  task {i}: {pol} scale={s:.2f} pitch={pitch:.1f} N={N_SIDE} "
              f"| window {c:.2f}+-{w/2:.0f} nm | teeth {len(tn)} "
              f"| narrow {min(tn):.1f}-{max(tn):.1f} | wide {min(tw):.1f}-{max(tw):.1f} "
              f"| dw_max {max(b - a for a, b in zip(tn, tw)):.1f}")
