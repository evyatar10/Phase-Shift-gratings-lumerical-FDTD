"""COMBINATION of the two validated radiation-cancellation routes on the 20-um overshoot (final overnight run 2026-10-20).

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Routes (both MEASURED at round-C numerics): (A) the seed -> T9739 design direction at alpha 0.49 (265098: leak 0.862 of control,
Q_i 9.76e5); (B) the solved 6-knob mean-width bump profile x1 (265103: leak 0.842, Q_i 9.81e5). Their far-field responses are
correlated (2-knob solve: eta^2 0.253 at mix 0.42 A + 0.40 B vs 0.21 / 0.19 alone). Cards: combo_solve = 0.21 A + 0.40 B
(predicted leak ~0.75); combo_sum = 0.49 A + 1.0 B (naive sum, nonlinearity probe). Control = 182131 seed.
Dispatch (Athena):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --cards=runners.sweeps.farfield_sph_20um_combo --max-concurrent=2
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF_SPH_ROUND"] = "C"
from runners.sweeps.farfield_sph_20um_bumps import BASE, N_SIDE, hh  # noqa: E402
from runners.sweeps.farfield_sph_20um_dirval import NAR0, WID0, NAR1, WID1  # noqa: E402
from runners.sweeps.farfield_sph_20um_bumps_val import DW_NM  # noqa: E402
from experiment_card import ExperimentCard  # noqa: E402

MIXES = {"combo_solve": (0.21, 0.40), "combo_sum": (0.49, 1.0)}     # (alpha along A, scale of B)


def teeth(a, s):
    nar = [n0 + a * (n1 - n0) + s * DW_NM.get(d, 0.0) for d, (n0, n1) in enumerate(zip(NAR0, NAR1))]
    wid = [w0 + a * (w1 - w0) + s * DW_NM.get(d, 0.0) for d, (w0, w1) in enumerate(zip(WID0, WID1))]
    return nar, wid


CARDS = [ExperimentCard(width_narrow_per_tooth_nm=teeth(a, s)[0], width_wide_per_tooth_nm=teeth(a, s)[1], label=lab)
         for lab, (a, s) in MIXES.items()]

if __name__ == "__main__":
    for card, (lab, (a, s)) in zip(CARDS, MIXES.items()):
        cfg = card.to_config(base=BASE)
        nar = [v * 1e9 for v in cfg.grating.width_narrow_per_tooth_m]
        dn = [n - n0 for n, n0 in zip(nar, NAR0)]
        print(f"{lab}: A {a} B {s} | max |dnar| {max(map(abs, dn)):.2f} nm | d0..d11 {[round(v, 2) for v in dn[:12]]} | box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} shutoff {cfg.mesh.auto_shutoff_min} sim ps {os.environ.get('TM_SIM_TIME_PS')}")
