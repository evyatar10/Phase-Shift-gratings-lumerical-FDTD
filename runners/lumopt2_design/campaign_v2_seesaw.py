"""V2 see-saw-seeded campaign — start at the apodization optimum, chase shifts.

Study dir: runners/lumopt2_design/  |  Created 2026-08-24  |  Job(s): TBD
Strategy (settled this session): apodization SATURATES at ~0.938 (the see-saw
amplitude sweep peaked at d090 then declined) and tooth shifts are what carry
past that ceiling (+0.025 measured on the 136465 lineage). So instead of asking
a uniform-seeded optimizer to re-discover the profile, seed AT the measured
see-saw optimum and let the campaign spend its evals on the one direction the
ladders could not close: the shift block (plus fine profile trims).

Seed = seesaw_d090 (inner-8 corr 235, outer-17 corr 393, shifts 0, wcav 800,
winner comb): MEASURED T 0.93836 / fwhm_env 18.33113 um / Q_load 2074 /
lam 1564.558 (IGUM seesaw_d090_evals.jsonl — in band, nothing extrapolated).

Wall config = the fully corrected stack, first campaign to carry all three:
  fw_tooth_w   MEASURED 3-block per-tooth corr price (the Fable-audit fix —
               the see-saw direction is priced correctly, so the optimizer can
               refine the profile without being told it costs 0.82 um)
  fw_curve     MEASURED 6-point elongation threshold curve (shifts free below
               e=65, so the shift direction is explorable from this seed)
  fw_pen_cap   2.0 rational saturation (the 136640 lnsrch fix)
trust_nm on corr/avg keeps the warm seed from being slammed by unit-norm
probes; shifts sit ON the 0-bound so the clamp deliberately skips them (they
must travel). wcav left free: unpriced by the wall (known gap), owned by the
measured fwhm_env guard + fail-closed _best_from_log, like everything else.

Dispatch:  SBATCH_MEM=160G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \\
  bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_v2_seesaw
Resume: re-dispatch the same module (cold-start resume; REQUEUE-safe).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.seesaw_ladder import rung_params

SEED_D, SEED_B = 90.0, 68.0          # the measured amplitude-sweep optimum
SEED_P = rung_params(SEED_D, SEED_B)

SPEC = eng.CampaignSpec(
    label="lumopt2_v2_seesaw",
    scan_width_nm=10.0, n_wl_points=501,          # v2 window
    region_dx_nm=eng.DX_PITCHLOCK_NM,             # pitch-locked (plan §24)
    scan_center_nm=1564.558,                      # MEASURED d090 resonance
    free_comb=False,
    rho_band=False,
    fwhm0_um=18.3460,        # band reference = the ORIGIN width (program-wide
                             # spec band, NOT this seed's own width)
    adj_phase_fix=True, adj_fix_re=1.0561, adj_fix_im=0.1239,
    trust_nm={"corr": 40.0, "avg": 15.0},
    fwhm_wall=True,
    fw_curve=True,
    fw_pen_cap=2.0,
    fw_tooth_w=eng.FW_TOOTH_W,
    fw_anchor={"fwhm": 18.33113,                  # MEASURED d090 fwhm_env
               "mcorr": float(np.mean(SEED_P[eng.SL_CORR])),
               "elong": 0.0,
               "corr_vec": tuple(float(v) for v in SEED_P[eng.SL_CORR])},
    seed_override=tuple(SEED_P),
    max_iter=60, max_feval=100,
)
N_TASKS = 1


def main(task_idx=0):
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    best = eng.run_campaign(SPEC, out_dir)
    print(f"[v2seesaw] done: best_fom {best['fom']:.5f} (delivered design = "
          f"width-filtered log, never this number)")


if __name__ == "__main__":
    print("see-saw-seeded campaign: seed d090 (T 0.93836), dispatch via deploy")
