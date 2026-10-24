"""Headroom check beyond the best measured point (combo_sum = 0.49 A + 1.0 B, leak 0.711, Q_i 1.044e6): two extrapolations.

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Cards: ext_a49_b15 = 0.49 A + 1.5 B (push the bump profile), ext_a74_b15 = 0.74 A + 1.5 B (push both). Same numerics and
control as farfield_sph_20um_combo. Verdict: does the leak keep falling (headroom -> next Newton round from this point) or turn.
Dispatch (Athena):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --cards=runners.sweeps.farfield_sph_20um_combo2 --max-concurrent=2
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF_SPH_ROUND"] = "C"
from runners.sweeps.farfield_sph_20um_combo import BASE, teeth  # noqa: E402
from experiment_card import ExperimentCard  # noqa: E402

MIXES = {"ext_a49_b15": (0.49, 1.5), "ext_a74_b15": (0.74, 1.5)}
CARDS = [ExperimentCard(width_narrow_per_tooth_nm=teeth(a, s)[0], width_wide_per_tooth_nm=teeth(a, s)[1], label=lab)
         for lab, (a, s) in MIXES.items()]

if __name__ == "__main__":
    for card, (lab, (a, s)) in zip(CARDS, MIXES.items()):
        cfg = card.to_config(base=BASE); nar = [v * 1e9 for v in cfg.grating.width_narrow_per_tooth_m]
        print(f"{lab}: A {a} B {s} | nar d8 {nar[8]:.2f} d10 {nar[10]:.2f} | box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} shutoff {cfg.mesh.auto_shutoff_min} sim ps {os.environ.get('TM_SIM_TIME_PS')}")
