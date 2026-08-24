"""V2 PROJECTION-FIRST campaign — T-only adjoint, measured-FWHM guard, seeded
from the RE-TRIMMED BEST (the +0.0665-at-equal-width verdict design).

Study dir: runners/lumopt2_design/  |  Created 2026-08-22  |  Job(s): TBD
Architecture settled by measurement (V2_FWHM_PLAN §8): the CPU width-adjoint
costs 8.7 h/solve — priced out. This campaign optimizes softmax-T with the
VALIDATED port adjoint only (C = 1.0561+0.1239i, calibrated at these exact
numerics: 50 nm region + PVA — deliberately kept so every measured anchor and
the seed's own row stay bit-comparable; the pitch-locked/conformal migration
happens at the NEXT anchor reset with MX-GRAD's verdict). Width control:
  - measured fwhm_env guard: fwhm0 17.713551 (W2), band +2%/−5%, WidthTrip on
    accepted-best + restart/final filter (engine, shipped);
  - rho band RETIRED (rho_band=False — item-27 hole; would also crush this
    legitimately-deep-corr seed); elongation cheat-wall kept;
  - stage-boundary re-trim = the proven uniform corr-add bisection (manual,
    ~5 forwards) before any cross-width comparison.
Seed: BEST_T9635 + 52.5 nm uniform corr — MEASURED in-band at these numerics
(17.755 µm = ratio 1.0023, T 0.95968; job 136051 row d+52.5).

Dispatch:  SBATCH_MEM=160G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \\
  bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_v2_projection
Resume: re-dispatch the same module (cold-start resume; REQUEUE-safe).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9635

RETRIM_DELTA_NM = 42.0           # ★DERIVED for the SYMMETRIC band: lands the
                                 # seed at the band CENTRE 18.346 um (from the
                                 # two pitch-locked points, jobs 136296 and
                                 # mx_retrim: dW/ddelta -0.04704 um/nm). The old
                                 # +52.5 seed sat at ratio 0.9731 = OUTSIDE the
                                 # new +/-2% band, and cost T for width we have
                                 # no use for (predicted T 0.9609 vs 0.9594).

# ★★CLOSED 2026-08-24 — CONVERGED, then cancelled (Athena 136465, 24 h).
# Evals 10/11/12 are identical to 5 decimals (FOM 0.71405, W 18.35309) =
# L-BFGS-B's zero-step termination. Its winner is preserved as BEST_T9636 in
# best_designs.py (T 0.96361 / lam 1566.444 / W 18.35309 / Q 2021.6 / Q_i
# 110079) and its .fsp is exported locally. THE SPEC BELOW IS LEFT EXACTLY AS
# IT RAN — it is the record of that run, not a template.
# ★DO NOT RE-DISPATCH AS-IS: this spec predates the per-tooth width price, so
# its fwhm_wall is the RANK-1 version (mean-corr only, no fw_tooth_w, anchor
# without corr_vec) that made profile shaping invisible to the optimizer
# (skill item 34). Any new stage must add fw_tooth_w=eng.FW_TOOTH_W,
# "corr_vec" in fw_anchor, fw_curve and fw_pen_cap — and take a NEW label,
# since FOMs are not comparable across a penalty change.
SPEC = eng.CampaignSpec(
    label="lumopt2_v2proj_s2",
    scan_width_nm=10.0, n_wl_points=501,      # v2 window
    # ★PITCH-LOCKED MESH (2026-08-23) — dx = PITCH/10 = 51.683 nm, NOT 50.0.
    # Measured this night on real profiles: the standing wave has exactly the
    # pitch period (516.6 nm measured), so dx=50 gives 10.34 samples/period —
    # a NON-integer ratio whose sampling phase drifts across the mode, and
    # fwhm_env (built through standing-wave PEAKS) then mis-reads by up to
    # 700 nm (3.9%) purely from where the wave sits on the grid. Against a
    # +2%/-5% band that is most of the tolerance, and it is DESIGN-dependent
    # (tooth shifts translate the wave), so it does NOT cancel in a ratio —
    # exactly the +3.6%/+0.6%/+2.6% spread §11 measured across MX rows.
    # At dx = pitch/10 the ratio is EXACTLY 10.00: on a known-truth synthetic
    # the error falls to +0.1 nm and is phase-INDEPENDENT. Anchors below are
    # the MEASURED corrected-mesh rows (job 136077 t15/t16), so nothing here
    # is extrapolated. softW is immune either way (3 nm spread) — it is
    # fwhm_env, the AUTHORITY, that needed this.
    region_dx_nm=eng.DX_PITCHLOCK_NM,
    scan_center_nm=1566.398,                  # DERIVED for the +42.0 nm seed
    free_comb=False,                          # user: comb fixed at winner
    rho_band=False,                           # retired; fwhm guard owns width
    fwhm0_um=18.3460,                         # MEASURED corrected ORIGIN width
                                              # (mx_origin) = the band reference
    adj_phase_fix=True, adj_fix_re=1.0561, adj_fix_im=0.1239,
    trust_nm={"corr": 12.0, "avg": 10.0, "shift": 12.0, "wcav": 12.0},
    fwhm_wall=True,
    fw_anchor={"fwhm": 18.3460,          # DERIVED seed width; re-anchors to the
               "mcorr": 368.5,           # set exactly in main() from the seed
               "elong": 130.6},
    max_iter=40, max_feval=70,
)
N_TASKS = 1


def main(task_idx=0):
    p = np.asarray(BEST_T9635, dtype=float).copy()
    p[eng.SL_CORR] = np.minimum(p[eng.SL_CORR] + RETRIM_DELTA_NM,
                                SPEC.corr_max_nm)
    # Order matters: trust_nm bounds center on the module seed until
    # seed_override is set — set it FIRST so replay's bounds check (and the
    # optimizer's trust box) center on THIS seed, not the uniform one.
    SPEC.seed_override = tuple(p)
    SPEC.seed_override = tuple(eng.replay_params(SPEC, p))
    SPEC.fw_anchor["mcorr"] = float(np.mean(p[eng.SL_CORR]))
    SPEC.fw_anchor["elong"] = float(2.0 * p[eng.SL_SHIFT].sum())
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    best = eng.run_campaign(SPEC, out_dir)
    print(f"[v2proj] finished: best_fom {best['fom']:.5f} — delivered design "
          f"comes from the width-filtered log; re-trim before ANY comparison")


if __name__ == "__main__":
    print("projection-first campaign; seed = BEST_T9635 + 52.5 nm corr")
