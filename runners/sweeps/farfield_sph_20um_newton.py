"""NEWTON ROUND on the 20-um overshoot: response basis re-measured at the best point (combo_sum = 0.49 A + 1.0 B, leak 0.711).

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Why: the first linear solve removed the low orders (l<=5: 30.5 % -> 9.3 % of the seed leak) but left l > 10 (half the leak)
untouched, and extrapolating the same two directions turned the leak back up (265108). A second linear round needs the
responses AT the new point and knobs further out (l > 10 lives at |x| > 2 um). New seed = combo_sum teeth (farfield_sph_20um_combo,
MEASURED 265105: T 0.98523, Q_i 1.044e6). Knobs: 8 period groups {0,1} {2,3} {4,5} {6,7} {8,9} {10,11} {12,14} {15,17} x
{mean +5 nm, diff +5 nm (wide +2.5 / narrow -2.5)} = 16 one-sided cards. Control = the measured combo_sum row.
Numerics = round C exactly. Analysis: bump_solve-style two-family solve with lambda/W held, then ONE validation card.
Dispatch (Athena, queued behind the 15.5-um Jacobian which has priority):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --cards=runners.sweeps.farfield_sph_20um_newton --max-concurrent=4 --after=265111
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF_SPH_ROUND"] = "C"
from runners.sweeps.farfield_sph_20um_bumps import BASE, N_SIDE  # noqa: E402
from runners.sweeps.farfield_sph_20um_combo import teeth as combo_teeth  # noqa: E402
from experiment_card import ExperimentCard  # noqa: E402

NAR0, WID0 = combo_teeth(0.49, 1.0)                 # the new seed = best measured point
BUMP_NM = 5.0
GROUPS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13, 14), (15, 16, 17)]
KNOBS = [(g, "mean") for g in GROUPS] + [(g, "diff") for g in GROUPS]


def knob_teeth(teeth_idx, fam):
    nar, wid = list(NAR0), list(WID0)
    for d in teeth_idx:
        if fam == "mean":
            nar[d] += BUMP_NM
            wid[d] += BUMP_NM
        else:
            nar[d] -= BUMP_NM / 2
            wid[d] += BUMP_NM / 2
    return nar, wid


CARDS = [ExperimentCard(width_narrow_per_tooth_nm=knob_teeth(g, fam)[0], width_wide_per_tooth_nm=knob_teeth(g, fam)[1],
                        label=f"{fam}_g{g[0]:02d}{g[-1]:02d}_p5nm") for g, fam in KNOBS]

if __name__ == "__main__":
    for (g, fam), card in zip(KNOBS, CARDS):
        cfg = card.to_config(base=BASE)
        nar = [v * 1e9 for v in cfg.grating.width_narrow_per_tooth_m]; wid = [v * 1e9 for v in cfg.grating.width_wide_per_tooth_m]
        diff = [d for d in range(N_SIDE) if abs(nar[d] - NAR0[d]) > 1e-6 or abs(wid[d] - WID0[d]) > 1e-6]
        assert diff == list(g), diff
        print(f"{card.label:18s} dn {nar[g[0]]-NAR0[g[0]]:+.2f} dw {wid[g[0]]-WID0[g[0]]:+.2f} | box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} shutoff {cfg.mesh.auto_shutoff_min} sim ps {os.environ.get('TM_SIM_TIME_PS')}")
