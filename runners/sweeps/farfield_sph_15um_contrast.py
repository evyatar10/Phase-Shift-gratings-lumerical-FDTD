"""Nt80x (15.5-um lab device): CONTRAST line search on the two periods beside the pi shift.

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Why (MEASURED, Athena 265111 card diff_g0001_p5nm vs 265046): +5 nm contrast (wide +2.5 / narrow -2.5) on period group {0,1}
alone gives far power per stored energy 0.862 of the control, T 0.664 -> 0.696, Q_L 99.1k -> 101.4k, Q_i 5.35e5 -> 6.13e5
(+15 %), lambda -0.002 nm, W -0.004 um; real overlap eta^2 0.32 (the best knob measured on any device). The bounded solve wants it
beyond +5 nm. Cards: {0,1} at +10 / +15 / +20 nm, and {0,1} +10 with {2,3} +10. Same C_NT80 PROD numerics (340 ps short run).
Verdict: far power per stored energy and Q_i vs contrast (where does it turn), lambda/W held, and the harmonic table of the best.
Dispatch (Athena, outside the 24h_1g cap):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --cards=runners.sweeps.farfield_sph_15um_contrast --max-concurrent=4 --qos=4d_1g --gpu=a100
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "C_NT80"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um_bumps import BASE, NAR0, WID0, N_SIDE  # noqa: E402
from experiment_card import ExperimentCard  # noqa: E402

STEPS = [((0, 1), 10.0), ((0, 1), 15.0), ((0, 1), 20.0), ("both", 10.0), ((0, 1), 25.0), ((0, 1), 30.0)]   # 25/30 added 2026-10-20 (MEASURED +5/+10/+15: P/U 0.862/0.767/0.720, Q_i +15/+27/+35 %)


def contrast_teeth(spec, dc):
    nar, wid = list(NAR0), list(WID0)
    groups = [(0, 1), (2, 3)] if spec == "both" else [spec]
    for g in groups:
        for d in g:
            nar[d] -= dc / 2
            wid[d] += dc / 2
    return nar, wid


CARDS = [ExperimentCard(width_narrow_per_tooth_nm=contrast_teeth(s, dc)[0], width_wide_per_tooth_nm=contrast_teeth(s, dc)[1],
                        label=(f"diff_g0001_p{dc:g}nm" if s != "both" else f"diff_g0003_p{dc:g}nm")) for s, dc in STEPS]

if __name__ == "__main__":
    for (s, dc), card in zip(STEPS, CARDS):
        cfg = card.to_config(base=BASE)
        nar = [v * 1e9 for v in cfg.grating.width_narrow_per_tooth_m]; wid = [v * 1e9 for v in cfg.grating.width_wide_per_tooth_m]
        diff = [d for d in range(N_SIDE) if abs(nar[d] - NAR0[d]) > 1e-6]
        print(f"{card.label:16s}: changed {diff} dn {nar[0]-NAR0[0]:+.1f} dw {wid[0]-WID0[0]:+.1f} | sim ps {os.environ.get('TM_SIM_TIME_PS')} short {cfg.farfield.short_run} box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f}")
