"""Linear-response VALIDATION on the 20-um overshoot: the seed -> T9739 design direction at alpha = 0.49 and 0.25.

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Why: under the per-stored-energy, Lorentzian-at-lam_ff normalisation, the measured far-field CHANGE from the round-C seed
to the inverse-design best TE_S2_N98_T9739 (both MEASURED, Athena 182131) carries 68 % of the leak power with a real
overlap eta^2 = 0.166 and alpha* = 0.49 (dlam +0.02 nm, dW +0.002 um); every mean-width bump/ripple knob overlaps < 1 %.
Prediction (linear): at alpha = 0.49 the leak per stored energy falls to 0.834 of the seed's; alpha = 0.25 = half-step
(slope check). Cards: per-tooth widths = seed + alpha * (T9739 - seed) (T9739's tooth shifts <= 0.017 nm are not drawn,
as in round C). Numerics = round C exactly (expanded seed config from farfield_sph_20um_bumps.BASE); control = 182131 seed.
Dispatch (Athena):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --cards=runners.sweeps.farfield_sph_20um_dirval --max-concurrent=2
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF_SPH_ROUND"] = "C"
from runners.sweeps.farfield_sph_20um_bumps import BASE, N_SIDE, hh  # noqa: E402
from runners.sweeps.farfield_sph_20um import design_teeth  # noqa: E402
from runners.lumopt2_design.best_designs import TE_S2_N98_T9739  # noqa: E402
from experiment_card import ExperimentCard  # noqa: E402

ALPHAS = [0.49, 0.25]
NAR0, WID0 = (list(v) for v in hh.teeth(N_SIDE))
NAR1, WID1 = design_teeth(TE_S2_N98_T9739)


def mix(alpha):
    return ([a + alpha * (b - a) for a, b in zip(NAR0, NAR1)], [a + alpha * (b - a) for a, b in zip(WID0, WID1)])


CARDS = [ExperimentCard(width_narrow_per_tooth_nm=mix(a)[0], width_wide_per_tooth_nm=mix(a)[1], label=f"dir_a{a:g}") for a in ALPHAS]

if __name__ == "__main__":
    for a, card in zip(ALPHAS, CARDS):
        cfg = card.to_config(base=BASE)
        nar = [v * 1e9 for v in cfg.grating.width_narrow_per_tooth_m]; wid = [v * 1e9 for v in cfg.grating.width_wide_per_tooth_m]
        dn = [n - n0 for n, n0 in zip(nar, NAR0)]; dw = [w - w0 for w, w0 in zip(wid, WID0)]
        print(f"{card.label}: max |dnar| {max(map(abs, dn)):.2f} nm, max |dwid| {max(map(abs, dw)):.2f} nm, teeth changed {sum(1 for v in dn if abs(v) > 1e-6)}, "
              f"first 4 dnar {[round(v, 2) for v in dn[:4]]} | box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} shutoff {cfg.mesh.auto_shutoff_min} sim ps {os.environ.get('TM_SIM_TIME_PS')} pol {cfg.source.polarization}")
