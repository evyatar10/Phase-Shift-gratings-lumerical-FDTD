"""Itai's RE-OPTIMIZED Nt60 design, natively FWHM 20 um -- short run at his length.

Study dir: runners/sweeps/   |   Created 2026-08-26   |   Job(s): TBD
Purpose: his previous Nt60 ("HH", target FWHM 15.1 um) only reached our 20 um
spec because WE scaled its whole dw profile by ~0.562. He has now re-optimized
the profile to give 20 um directly (Nt60_FWHM_20_dw.npy, his optimizer reports
FWHM 19.9889 um, Er 0.0050). This runner measures that device, untouched.

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

WINDOW 1559.79 +- 4 nm at the sweep-path default 3001 points = 2.67 pm.
Q_L extrapolates to ~6e3 (bracket 3e3-1.5e4) from the two measured TE anchors at
N=98, i.e. a 100-500 pm linewidth = 40-190 samples across it, and a ring-down
16.1*tau of 80-250 ps against the 2000 ps default -- both comfortable. If the
measured Q comes back above ~5e4 the window must be narrowed for round 2
(see memory: high-Q measurement adequacy).

Containment: 98 * 0.49106 = 48.1 um >= 2 x 20 um (asserted below).

Dispatch (IGUM, user choice; probe seats + queue first, --max-concurrent=1 for
the ansyscl startup race):
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

N_APOD, N_SIDE = 60, 98
MODE_FWHM_UM   = 20.0                     # the spec the pitch envelope is weighted on
BOX_Y_UM, BOX_Z_MULT = 6.8, 4.14
SCAN_WIDTH_NM  = 8.0
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


PITCH_NM = pitch_for(N_SIDE)
_NAR, _WID = teeth(N_SIDE)
assert N_SIDE * PITCH_NM >= 2.0 * MODE_FWHM_UM * 1e3        # containment

SPEC = SweepSpec(
    n_periods_each_side       = [N_SIDE],
    pitch_nm                  = [PITCH_NM],
    avg_width_nm              = [AVG_WIDTH_NM],
    cavity_width_nm           = [CAVITY_W_NM],
    corrugation_depth_nm      = [round(BULK_WIDE_NM - BULK_NARROW_NM, 1)],
    width_narrow_per_tooth_nm = [_NAR],
    width_wide_per_tooth_nm   = [_WID],
    polarization              = ["TE"],
    center_wavelength_nm      = [LAMBDA["TE"]],
    scan_width_nm             = [SCAN_WIDTH_NM],
    y_span_um                 = [BOX_Y_UM],
    span_mult                 = [BOX_Z_MULT],
    mode  = "zipped",
    label = "itai_hh_nt60w20",
)

if __name__ == "__main__":
    dlam = SCAN_WIDTH_NM / (N_WL_POINTS - 1) * 1e3          # pm per sample
    print(SPEC.describe().split("width_narrow")[0])
    print(f"  task 0: TE N={N_SIDE} pitch {PITCH_NM:.2f} nm | half-device "
          f"{N_SIDE*PITCH_NM/1000:.1f} um | teeth {len(_NAR)} "
          f"| narrow {min(_NAR):.1f}-{max(_NAR):.1f} | wide {min(_WID):.1f}-{max(_WID):.1f} "
          f"| bulk dw {BULK_WIDE_NM-BULK_NARROW_NM:.1f} | dw_max "
          f"{max(b-a for a, b in zip(_NAR, _WID)):.1f}")
    print(f"  window {LAMBDA['TE']} +- {SCAN_WIDTH_NM/2:.0f} nm at {N_WL_POINTS} pts "
          f"= {dlam:.2f} pm | box y {BOX_Y_UM} um (ratio {BOX_Y_UM*1e3/max(_WID):.2f}) "
          f"/ z 6.81 um")
    for Q in (3e3, 6e3, 1.5e4, 5e4):
        lw = LAMBDA["TE"] * 1e3 / Q
        tau = Q * LAMBDA["TE"] * 1e-9 / (2 * np.pi * 2.99792458e8) * 1e12
        print(f"  Q_L {Q:8.0f}: linewidth {lw:6.1f} pm = {lw/dlam:5.1f} samples "
              f"| 16.1*tau {16.1*tau:6.0f} ps vs 2000 ps sim time")
