"""Compact tooth-group BUMP knobs on the 15.5-um Itai Nt80x overshoot (lab device): radiation response basis.

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Same recipe as farfield_sph_20um_bumps on the device whose loss is 90 % radiation (265046: Q_i 5.25e5, Q_rad 5.9e5).
Its leak source (band-limited inverse FT of the 265046 far field) is broader: 42/63/83 % within |x| <= 5/10/20 um ->
8 groups {0,1} {2,3} {4,5} {6,7} {8,9} {10,11} {12,14} {15,17} x two families: MEAN width +5 nm (cards 0-7, route B)
+ {2,3} -5 nm (pair) and +2.5 nm (half-step) (cards 8-9), and DIFF = corrugation contrast +5 nm (wide +2.5 / narrow -2.5,
cards 10-17, route A's family): 18 cards. Per stored energy this device radiates 3.7x faster than the 20-um one (1/Q_rad), so
5-nm knobs give responses ~1/3.7 of the 20-um ones (cross terms up to ~+-25 % of the leak: measurable). Numerics = farfield_sph_15um ROUND C_NT80 PROD exactly
(short run 340 ps, tube 100 um + caps, own +-0.25 nm far-field window at 1619.455, box 6.8/6.81, pitch 514, N 96);
control = the 265046 row. Analysis: bump_solve.py (ringdown gamma^2 per row for the amplitude), then one validation card.
Dispatch (Athena; the 20-um validation 265103/265105 showed real reductions):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --cards=runners.sweeps.farfield_sph_15um_bumps --max-concurrent=4
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "C_NT80"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um import SPEC as SPEC_C, BASE as BASE_C, ROWS, N_SIDE  # noqa: E402
from experiment_card import ExperimentCard  # noqa: E402

BUMP_NM = 5.0
GROUPS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13, 14), (15, 16, 17)]
# two knob FAMILIES (both validated at 20 um): "mean" = +dw on wide AND narrow tooth (route B);
# "diff" = +dw/2 on the wide, -dw/2 on the narrow tooth = corrugation contrast (route A's family).
KNOBS = ([(g, +BUMP_NM, "mean") for g in GROUPS] + [((2, 3), -BUMP_NM, "mean"), ((2, 3), +BUMP_NM / 2, "mean")]
         + [(g, +BUMP_NM, "diff") for g in GROUPS])
NAR0, WID0 = ROWS[0][5]


def bump_teeth(teeth_idx, dw_nm, fam):
    nar, wid = list(NAR0), list(WID0)
    for d in teeth_idx:
        if fam == "mean":
            nar[d] += dw_nm
            wid[d] += dw_nm
        else:
            nar[d] -= dw_nm / 2
            wid[d] += dw_nm / 2
    return nar, wid


BASE = next(iter(SPEC_C.expand(base=BASE_C)))
CARDS = [ExperimentCard(width_narrow_per_tooth_nm=bump_teeth(g, dw, fam)[0], width_wide_per_tooth_nm=bump_teeth(g, dw, fam)[1],
                        label=f"{fam}_g{g[0]:02d}{g[-1]:02d}_{'p' if dw > 0 else 'm'}{abs(dw):g}nm")
         for g, dw, fam in KNOBS]

if __name__ == "__main__":
    for i, ((g, dw, fam), card) in enumerate(zip(KNOBS, CARDS)):
        cfg = card.to_config(base=BASE)
        nar = [v * 1e9 for v in cfg.grating.width_narrow_per_tooth_m]; wid = [v * 1e9 for v in cfg.grating.width_wide_per_tooth_m]
        diff = [d for d in range(N_SIDE) if abs(nar[d] - NAR0[d]) > 1e-6 or abs(wid[d] - WID0[d]) > 1e-6]
        assert diff == list(g), diff
        dn, dwd = nar[g[0]] - NAR0[g[0]], wid[g[0]] - WID0[g[0]]
        assert (fam == "mean" and abs(dn - dwd) < 1e-6) or (fam == "diff" and abs(dn + dwd) < 1e-6)
        print(f"card {i} {card.label:18s}: {g} {dw:+.1f} dn {dn:+.2f} dw {dwd:+.2f} | pitch {cfg.grating.pitch_m*1e9:.1f} N {cfg.grating.n_periods_each_side} "
              f"| win {cfg.spectral.center_wavelength_m*1e9:.3f}+-{cfg.spectral.scan_width_nm/2:.0f} | box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} "
              f"| short {cfg.farfield.short_run} ffx {cfg.farfield.farfield_x_span_m*1e6:.0f} 2dwin {cfg.farfield.use_2d_window} {cfg.monitors.monitor_2d_center_nm} | sim ps {os.environ.get('TM_SIM_TIME_PS')}")
