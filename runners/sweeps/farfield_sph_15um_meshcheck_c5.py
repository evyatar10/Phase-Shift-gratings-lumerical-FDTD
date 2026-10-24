"""Nt80x at dy ~37.5 nm WITH the +5-nm inner contrast knob (matched pair to farfield_sph_15um_meshcheck = the dy-37 control).

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Why (GPT review 01a1586c): the +5-nm contrast on period pair {0,1} (265111 diff_g0001_p5nm: P/U 0.862, Q_i +15 %) is the best knob
measured but is not certified above the ~0.1|c| per-row noise and the known high-order mesh sensitivity; the certifying comparison
is control vs the SAME card on the SAME finer mesh: R = (P/U)_contrast / (P/U)_control at dy 37 vs 0.862 at dy 50, plus decay Q_L,
T, W at each geometry's own resonance. One card, chained after the dy-37 control (265195) so the seat budget stays <= ~30.
Dispatch (Athena): SBATCH_MEM=200G bash athena/deploy_athena.sh --cards=runners.sweeps.farfield_sph_15um_meshcheck_c5 --max-concurrent=1 --qos=4d_1g --gpu=a100 --after=265195
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "C_NT80"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um_meshcheck import BASE, Y_CELLS  # noqa: E402  (dy 37.5 base)
from runners.sweeps.farfield_sph_15um_bumps import bump_teeth  # noqa: E402
from experiment_card import ExperimentCard  # noqa: E402

# card 1 added 2026-10-21: +20 nm (the ladder minimum, 265648/265649 + contrast diff_g0001_p20nm: R 0.684 at dy 50); run ONLY
# task 1 (--array-tasks=1): card 0 is stored (meshcheck_c5 265205-chain). Pass: |R_37(+20) - 0.684| <= 0.05 vs the stored dy-37 control.
CARDS = [ExperimentCard(width_narrow_per_tooth_nm=bump_teeth((0, 1), dc, "diff")[0], width_wide_per_tooth_nm=bump_teeth((0, 1), dc, "diff")[1],
                        label=f"nt80x_ycells{Y_CELLS}_diff0001_p{dc:g}nm") for dc in (5.0, 20.0)]

if __name__ == "__main__":
    for card in CARDS:
        cfg = card.to_config(base=BASE); kw = cfg.to_device_kwargs()
        n = [v * 1e9 for v in cfg.grating.width_narrow_per_tooth_m]; w = [v * 1e9 for v in cfg.grating.width_wide_per_tooth_m]
        print(f"{card.label}: y_cells {kw['y_mesh_cells']} | teeth 0,1 narrow {n[0]:.1f},{n[1]:.1f} wide {w[0]:.1f},{w[1]:.1f} | short {cfg.farfield.short_run} sim ps {os.environ.get('TM_SIM_TIME_PS')}")
