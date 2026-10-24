"""VALIDATION of the linear cancellation solve on the 20-um overshoot: the solved mean-width bump profile at full and half amplitude.

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Basis = farfield_sph_20um_bumps (Athena 265089, +5 nm bumps on period groups {0,1}..{10,11}); solve = scratch bump_solve.py
(G alpha = -h with lambda and W held). DW_NM below = the solved per-period mean-width change (nm, index 0 = first period
beside the cavity), pasted from the solve; cards = full profile and half profile (slope check). Same round-C numerics
and control as the basis. Verdict = measured far power per stored energy vs the control AND vs the solve's prediction,
plus T, Q_L, Q_i, W, lambda.
Job 265100 (labels solved_x*) ran a ZERO profile by mistake (the solve's npy was overwritten by a single-row run) = cards-path
control rows only. Real validation = labels solved6_x*.
Dispatch (Athena):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --cards=runners.sweeps.farfield_sph_20um_bumps_val --max-concurrent=2
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF_SPH_ROUND"] = "C"
from runners.sweeps.farfield_sph_20um_bumps import BASE, N_SIDE, hh  # noqa: E402  (expanded round-C seed config)
from experiment_card import ExperimentCard  # noqa: E402

DW_NM = {0: 1.756, 1: 1.756, 2: 0.355, 3: 0.355, 4: 1.423, 5: 1.423, 6: -1.045, 7: -1.045, 8: -8.368, 9: -8.368, 10: 2.416, 11: 2.416}   # constrained 6-knob solve 2026-10-20 (Athena 265089): eta^2 0.204, predicted leak 0.796 (full) / 0.847 (half) of control
SCALES = [1.0, 0.5]


def profile_teeth(scale):
    nar, wid = (list(v) for v in hh.teeth(N_SIDE))
    for d, dw in DW_NM.items():
        nar[d] += scale * dw
        wid[d] += scale * dw
    return nar, wid


CARDS = [ExperimentCard(width_narrow_per_tooth_nm=profile_teeth(s)[0], width_wide_per_tooth_nm=profile_teeth(s)[1],
                        label=f"solved6_x{s:g}") for s in SCALES]

if __name__ == "__main__":
    assert DW_NM and max(abs(v) for v in DW_NM.values()) > 0.5, "DW_NM must hold the real solved profile (job 265100 ran zeros: the npy had been overwritten)"
    nar0, wid0 = hh.teeth(N_SIDE)
    for card, s in zip(CARDS, SCALES):
        cfg = card.to_config(base=BASE)
        nar = [v * 1e9 for v in cfg.grating.width_narrow_per_tooth_m]
        print(f"{card.label}: " + " ".join(f"{d}:{nar[d]-nar0[d]:+.2f}" for d in sorted(DW_NM)) +
              f" | box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} shutoff {cfg.mesh.auto_shutoff_min} sim ps {os.environ.get('TM_SIM_TIME_PS')}")
