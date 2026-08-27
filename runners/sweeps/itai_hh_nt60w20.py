"""Itai's RE-OPTIMIZED Nt60 design, natively FWHM 20 um -- N ladder to -3 dB.

Study dir: runners/sweeps/   |   Created 2026-08-26   |   Job(s): 63722 (N=98)

Purpose: his previous Nt60 ("HH", target FWHM 15.1 um) only reached our 20 um
spec because WE scaled its whole dw profile by ~0.562. He has now re-optimized
the profile to give 20 um directly (Nt60_FWHM_20_dw.npy, his optimizer reports
FWHM 19.9889 um, Er 0.0050). This runner measures that device, untouched.

ROUND 1 -- N=98, HIS design length -- IS MEASURED, job 63722, and is NOT in the
ladder below (CLAUDE.md 6: never re-measure a stored result). Reuse it from
    results/itai_hh_nt60w20/results/result_N98_..._Ybox6p8_Zbox6p8.mat
    lam_res 1559.8597 nm (+0.07 nm from target -> the pitch is RIGHT, left alone)
    peak T 0.97498 | R 0.00014 | T+R 0.97511 < 1 | Q_L 7680 | fwhm_m 19.633 um
    -> Q_i ~ 6.1e5, but POORLY CONDITIONED at T=0.975 (1% in T = 39% in Q_i).
User decision 2026-08-26: 19.633 um is on spec, run his design UNTOUCHED (no
rescale) -- so the ladder varies N only.

ROUND 2 (job 63752) -- ONE rung, N=130, and that is the whole answer. Method and
sizing come from memory/project_q3db_measurement_method.md, distilled by reading
te_q3db_20um (9 sims) and trench_q3db_20um (29 sims) BEFORE dispatching more:
  * Q(-3 dB) = 0.29289*Q_i is exact two-port algebra, and it reproduces all FOUR
    of our own directly-measured -3 dB anchors to ~2% (TE N166 -1.9%, TM N165
    -2.1%, TM N170 +0.5%, TM N169 +3.2%). So a 17 h crossing run buys ~2%.
  * The only assumption is that Q_i has stopped drifting with N. MEASURED
    containment benchmark from the TM study: Q_i still drifted at an end-field of
    1.8e-2 of peak and had saturated by 2.4e-3. This device's measured envelope
    (exponential bulk tail, 1/e = 15.3 um) gives 5.5e-3 at N=98 -- ambiguous --
    and 2.0e-3 at N=130, PAST the saturation benchmark.
  * Conditioning A = sqrt(T)/(2(1-sqrt(T))): 39.2 at the N=98 row, 7.9 at N=130.
    At the measured dT = 0.0018 mesh floor that is 7.2% vs 1.6% in Q_i.
  So N=130 gives Q_i both saturated and well conditioned -> quote 0.29289*Q_i
  from it, and use N=98 only to show how much drift was left behind it.
  N=150 was dispatched in error and CANCELLED before starting (63752_1): with Q_c
  extrapolated instead of ln T (Q_c is linear in N by construction) two rows 32
  periods apart pin the growth rate to ~1%, which moves N* by 0.3 periods -- so
  the third rung bought nothing. TE-only by user decision; TM is a measured 3.4x
  weaker in Q_i on his old design (light-cone headroom 10.0% TE vs 5.5% TM) and
  his index correction is TE-derived.
IT IS NOT OUR SCALED DEVICE (checked before dispatch, no GPU): the second lobe
is gone and the bulk/apodization balance is inverted --
    raw dw            his old   our 20um (old x0.562)   his new
    bulk                500          281                 484.9 nm
    peak               1200          674                 615   nm
    peak/bulk          2.40         2.40                 1.27
    widest drawn tooth 1915.1       1429.5               1347.5 nm
so the previous 20 um rows do not answer for it.

GEOMETRY = his .npy through HIS OWN chain, at scale 1.0: advanced_dw_correction
(3D LUT dw_correction_LUT.mat) then advanced_index_correction (per-period mid
width holding the period-averaged neff at its bulk value), target_lambda 1600,
avg_wg 1.0 um, round_gds 0.1 nm. No re-solve with our FDE curve -- that only
exists in itai_hh_apod.teeth() to repair a SCALED profile, and this one is not
scaled. Gate: the same chain on his OLD .npy reproduces the stored
itai_hh_apod.APOD_*_NM tables to 0.10 nm in dw (mid width carries a constant
-2.0 nm common-mode offset, ~0.3 nm of lambda, absorbed by the pitch retune).

PITCH 491.06 nm, not his 514: his design targets 1600 nm, we measure at our own
TE Q3dB anchor 1559.79 nm (result_N166_avg_C250.mat). Same envelope-weighted
retune that landed -0.16 nm on the previous TE round -- <n> = 1.58090 over a
20 um Gaussian on the cavity, K-biased on that anchor. N-independent to 0.01 nm.

BOX y 6.8 um / span_mult 4.14 (z 6.81 um) = the previous 20 um rows AND the
inverse-design programme, so every number here is directly comparable. The
derived box is NOT used: sizing y from the scalar width_wide is what made his
device read T+R > 1 in round 1. Widest tooth here is 1347.5 nm -> ratio 5.05.
Fine-mesh override box = 1617 nm (verified in the local build smoke; the engine
deployed on IGUM already carries the per-tooth mesh-box fix, so remote == local).
It is not load-bearing here either way -- half the widest tooth is 674 nm, inside
both the fixed box (808 nm) and the old scalar one (748 nm). Stored 20 um rows
measured before that fix was deployed used the scalar box; the difference cannot
move a sidewall in or out of the fine mesh, but it is a numerics difference and is
recorded here rather than buried. No engine code is pushed by this study.

WINDOWS are per-rung and centred on the MEASURED 1559.8597 nm, not the design
target. lambda barely moves with N here -- the mode occupies ~40 periods, so
periods 99+ sit outside it -- but each window still keeps >=1 nm of margin.
n_wl_points is FIXED at 3001 on the sweep path (not sweepable), so the window
width is the only sampling knob (memory: high-Q measurement adequacy):
    N=130  EXPECTED Q_L ~3.5e4  linewidth ~45 pm  window 4.0 nm -> 1.33 pm, 34 pts
Ring-down 16.1*tau = 461 ps, well inside the 2000 ps default, so no
TM_SIM_TIME_PS override is needed. (It would be, past ~N=160: the sweep array at
igum/deploy_igum.sh:1232 does NOT forward that env var -- set os.environ at the
top of this module instead, it is imported on the node before the scene builds.)

Containment: min N * pitch = 130 * 0.49106 = 63.8 um >= 2 x 20 um (asserted).

Dispatch (IGUM, user choice; probe seats + queue first). --max-concurrent=1 for
TWO reasons: the IGUM ansyscl startup race, and seats -- 24/50 were in use at
round 1, and two concurrent tasks (~7 seats each) would put the pool at ~38/50,
past the >=35 HIGH band where the rule is to hold fan-outs. Serial costs wall
time, not GPU-hours.
    SBATCH_MEM=160G bash igum/deploy_igum.sh \
        --option3 --spec=runners.sweeps.itai_hh_nt60w20 --max-concurrent=1
Output -> results/itai_hh_nt60w20/results/ (download to results_from_igum/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from runners.sweeps.sweep_spec import SweepSpec
from runners.sweeps.itai_hh_apod import AVG_WIDTH_NM, K_BIAS, LAMBDA, NEFF_TE, NEFF_W_UM

N_APOD         = 60
N_MEASURED     = 98                       # job 63722 -- reused, NOT re-run
N_LADDER       = [130]                    # one rung; see round-2 note above
WINDOW_NM      = [4.0]
SCAN_CENTER_NM = 1559.8597                # MEASURED at N=98, not the design target
MODE_FWHM_UM   = 20.0                     # the spec the pitch envelope is weighted on
BOX_Y_UM, BOX_Z_MULT = 6.8, 4.14
N_WL_POINTS    = 3001                     # the sweep path's default; not sweepable

# His drawn widths, cavity-first (d=1 .. d=61). Produced by his own chain (see
# docstring) from Nt60_FWHM_20_dw.npy; the generator is scratch, these are data.
APOD_NARROW_NM = [
    951.4, 923.9, 914.6, 899.5, 882.8, 866.7, 851.7, 837.5,
    824.0, 811.2, 799.0, 787.6, 776.8, 766.9, 757.7, 749.4,
    742.0, 735.2, 729.1, 723.9, 719.7, 716.5, 714.4, 713.3,
    713.2, 714.2, 716.3, 719.5, 723.3, 727.7, 732.7, 738.2,
    744.0, 750.0, 755.9, 761.6, 767.2, 772.5, 777.5, 781.5,
    784.4, 786.1, 786.6, 786.1, 784.8, 782.8, 780.6, 778.5,
    776.2, 773.9, 771.4, 769.0, 766.6, 764.3, 762.2, 760.1,
    758.0, 756.0, 754.4, 753.3, 752.9,
]
APOD_WIDE_NM = [
    951.4, 980.1, 990.5, 1008.0, 1028.6, 1049.6, 1070.5, 1091.3,
    1112.3, 1133.4, 1154.5, 1175.6, 1196.4, 1216.7, 1236.4, 1255.0,
    1272.5, 1288.9, 1304.5, 1318.1, 1329.4, 1338.1, 1344.1, 1347.3,
    1347.5, 1344.7, 1338.8, 1329.9, 1319.8, 1308.2, 1295.4, 1281.7,
    1267.7, 1253.8, 1240.4, 1227.8, 1216.1, 1205.2, 1195.2, 1187.3,
    1181.6, 1178.4, 1177.4, 1178.3, 1180.9, 1184.8, 1188.9, 1193.2,
    1197.6, 1202.4, 1207.3, 1212.3, 1217.3, 1222.1, 1226.7, 1231.2,
    1235.7, 1240.1, 1243.8, 1246.3, 1247.1,
]
BULK_NARROW_NM, BULK_WIDE_NM = 752.9, 1247.1
CAVITY_W_NM = APOD_NARROW_NM[0]

assert len(APOD_NARROW_NM) == len(APOD_WIDE_NM) == N_APOD + 1


def teeth(n_side):
    """Per-tooth (narrow, wide) in nm, innermost first: his apodized core + bulk."""
    return (list(APOD_NARROW_NM) + [BULK_NARROW_NM] * (n_side - N_APOD - 1),
            list(APOD_WIDE_NM) + [BULK_WIDE_NM] * (n_side - N_APOD - 1))


def pitch_for(n_side):
    """Pitch that lands this device on our TE anchor: lambda = K*2*<n>*pitch.

    <n> is the per-period (narrow+wide)/2 averaged over the mode envelope (a
    20 um FWHM Gaussian on the cavity), so the periods the mode actually samples
    set the pitch. Iterated because the envelope is measured in periods. Same
    method as itai_hh_apod.pitch_for, re-derived here because that one is
    hard-wired to its own scaled-profile teeth().
    """
    nar, wid = teeth(n_side)
    nd = np.array([(np.interp(a * 1e-3, NEFF_W_UM, NEFF_TE)
                    + np.interp(b * 1e-3, NEFF_W_UM, NEFF_TE)) / 2 for a, b in zip(nar, wid)])
    x = np.arange(n_side) + 0.5
    pitch = 500.0
    for _ in range(30):
        w = np.exp(-4 * np.log(2) * (x * pitch * 1e-3 / MODE_FWHM_UM) ** 2)
        pitch = LAMBDA["TE"] / (K_BIAS["TE"] * 2 * float(np.sum(w * nd) / np.sum(w)))
    return round(pitch, 2)


# pitch is N-independent to 0.01 nm (the mode samples only the first ~40 periods),
# and the N=98 row MEASURED it right to +0.07 nm -- so every rung keeps it.
PITCH_NM = pitch_for(N_MEASURED)
_T = [teeth(n) for n in N_LADDER]
assert min(N_LADDER) * PITCH_NM >= 2.0 * MODE_FWHM_UM * 1e3        # containment

SPEC = SweepSpec(
    n_periods_each_side       = list(N_LADDER),
    pitch_nm                  = [PITCH_NM] * len(N_LADDER),
    avg_width_nm              = [AVG_WIDTH_NM] * len(N_LADDER),
    cavity_width_nm           = [CAVITY_W_NM] * len(N_LADDER),
    corrugation_depth_nm      = [round(BULK_WIDE_NM - BULK_NARROW_NM, 1)] * len(N_LADDER),
    width_narrow_per_tooth_nm = [t[0] for t in _T],
    width_wide_per_tooth_nm   = [t[1] for t in _T],
    polarization              = ["TE"] * len(N_LADDER),
    center_wavelength_nm      = [SCAN_CENTER_NM] * len(N_LADDER),
    scan_width_nm             = list(WINDOW_NM),
    y_span_um                 = [BOX_Y_UM] * len(N_LADDER),
    span_mult                 = [BOX_Z_MULT] * len(N_LADDER),
    mode  = "zipped",
    label = "itai_hh_nt60w20",
)

if __name__ == "__main__":
    # Growth calibrated on the ONE measured point (N=98, Q_L 7680) at 4.7 %/period
    # -- deliberately used only to size windows and predict cost, never to place
    # the crossing. Q_i 6.1e5 from the same row.
    print(SPEC.describe().split("width_narrow")[0])
    for i, n in enumerate(N_LADDER):
        tn, tw = _T[i]
        win = WINDOW_NM[i]
        dlam = win / (N_WL_POINTS - 1) * 1e3                       # pm per sample
        Q = 7680 * np.exp(0.047 * (n - N_MEASURED))
        lw = SCAN_CENTER_NM * 1e3 / Q
        tau = Q * SCAN_CENTER_NM * 1e-9 / (2 * np.pi * 2.99792458e8) * 1e12
        Tex = (1 - Q / 6.1e5) ** 2
        print(f"  task {i}: TE N={n:3d} pitch {PITCH_NM:.2f} | half-device "
              f"{n*PITCH_NM/1000:.1f} um | teeth {len(tn)} | widest {max(tw):.1f} nm "
              f"| window {SCAN_CENTER_NM}+-{win/2:.2f} nm = {dlam:.2f} pm")
        print(f"           EXPECTED Q_L {Q:7.0f} -> linewidth {lw:5.1f} pm = "
              f"{lw/dlam:4.1f} samples | 16.1*tau {16.1*tau:5.0f} ps vs 2000 ps "
              f"| T ~ {Tex:.2f}")
