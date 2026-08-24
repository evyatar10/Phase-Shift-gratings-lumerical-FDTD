"""V2 second-basin campaign — STRICTLY UNIFORM initial seed, comb fixed.

Study dir: runners/lumopt2_design/  |  Created 2026-08-23  |  Job(s): TBD
User decision 2026-08-23: basin 2 starts from the pure initial design
(corr 325 / avg 800 / shifts 0 / wcav 800 + the winner comb) — NOT the
see-saw variant. Basin 1 = campaign_v2_projection (the retrimmed best,
job 136141). Two independent basins = the program's convergence evidence.

This is NOT a re-run of the old seedA campaign: same seed, but the v2
regime — ±5 nm/501 window, measured-FWHM wall + guard (fwhm_wall), rho
band retired, comb FROZEN (was free), width authority = fwhm_env. The old
result's width numbers are void (profile_line bug) and its best grew +14.9%
wide through the rho hole; this campaign is the corrected experiment.

Knobs follow the PROVEN uniform-seed precedent (campaign_c325_seedA):
NO trust_nm — a uniform seed must be able to travel (the best design sits
160 nm away in wcav; a ±15 nm trust box would imprison it in the seed's
local basin), and seedA proved the C-fixed gradient survives that travel.
Width safety comes from the wall + guard, not from clamps. The wall's
slopes are extrapolations at mcorr 325 (measured near 368); the anchor is
EXACT for this seed (W2, job 135971 t10) and re-anchors at every accepted
best, and the measured-FWHM guard stays the authority.

Dispatch (projection mode, EXACT_WIDTH_GRAD=False — no gate needed):
  SBATCH_MEM=160G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \\
    bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_v2_uniform
Resume: re-dispatch the same module (cold-start resume; REQUEUE-safe).

★MEMORY if EXACT_WIDTH_GRAD is ever turned on (job 136122 OOM, plan §18 /
skill item 30): a MixedFom loads a FULL-λ-grid region array per FOM entry
(fdtd_session.get_fields_at_wavelengths fetches all recorded λ, then
slices) — dispatch exact mode at SBATCH_MEM=300G, only after the C_field
fit from the W3-GPU gate (validate_c325 task 20) sets ADJ_FIX_FIELD.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from runners.lumopt2_design import lumopt2_design as eng

EXACT_WIDTH_GRAD = False  # False = projection (v2proj architecture); True
                          # needs ADJ_FIX_FIELD from fit_c_field + 300G.
ADJ_FIX_FIELD = None      # (re, im) from fit_c_field.py; None = not fitted
ADJ_FIX_PORT = (1.0561, 0.1239)   # validated port C (skill item 6)

# ★MEASURED at the PITCH-LOCKED mesh (job 136077 t15, mx_origin) — the
# corrected anchors. The old 50.0-nm pair (17.713551 / 18.160689) is VOID
# for this campaign: at 10.34 samples per standing-wave period fwhm_env
# mis-reads by up to 3.9% depending on where the wave sits on the grid
# (plan §24). This seed IS the origin, so its row is the band reference.
FWHM0_UM = 18.3460        # MEASURED, mx_origin, dx = pitch/10
SOFTW0_UM = 18.476441     # MEASURED, same eval

SPEC = eng.CampaignSpec(
    # ★s3 (2026-08-23): NEW label, not a resume of s2. s2 ran 4 evals under the
    # FW_A_ELONG secant, which over-taxes small shifts ~10x (measured: it charges
    # 0.813 um of predicted widening at e=60 where jobs 61742/61782 measure ZERO).
    # s2 was therefore forbidden from the very direction this campaign exists to
    # test: it spent 2 of 3 post-seed evals on rejected out-of-band probes and
    # gained +0.0005 T in ~7 h, while shift-FROZEN 136468 gained +0.0076. FOMs
    # are not comparable across a wall change, so s2's log must NOT be resumed —
    # its rows stay as the record of the old regime.
    # ★s3 RAN AND TERMINATED EARLY — 2026-08-24, Athena 136640, 8h16m, exit 0,
    # ~6 of 100 evals, "Optimization did not converge. Message:
    # ABNORMAL_TERMINATION_IN_LNSRCH". Best row: T 0.9041 / W 18.315 / e 59,
    # IN BAND (best FOM 0.66924 vs the seed's 0.66722).
    # CAUSE: fw_curve is physically right but far STEEPER than the old linear
    # wall (e=287 scores -793 vs -51; e=163 scores -68). The shift block's wide
    # bounds (0-200 nm/tooth) mean L-BFGS-B's unit-norm scaled probe lands
    # straight in that cliff, the line search finds no acceptable decrease, and
    # scipy aborts.
    # ★WHY trust_nm IS NOT THE FIX HERE (tried, verified non-functional): the
    # clamp is CENTRED on the seed and param_bounds deliberately skips it when
    # the seed sits ON a physical edge — the uniform seed's shifts are exactly
    # 0, the lower bound. Setting trust_nm={"shift": ...} leaves the bounds at
    # [0, 200] unchanged. Tightening shift_bounds is forbidden by standing rule
    # (project_grating_geometry_facts: "do NOT tighten"), and giving the seed
    # nonzero shifts would break the strictly-uniform-initial design the user
    # chose. So the remaining principled fix is a BOUNDED penalty (cap the
    # hinge, e.g. at a few FOM units — large enough to reject, small enough to
    # keep the landscape navigable). That is a COST-FUNCTION change and is
    # PARKED for the user, not taken autonomously.
    # ★s4 (2026-08-24, user: "fix it and continue"): same physics as s3, with
    # the width hinge SATURATED (fw_pen_cap) so the line search cannot be handed
    # an unnavigable cliff. New label because FOM values are not comparable
    # across a penalty change — s3's log stays as the record of that run.
    # ★s5 (2026-08-24, Fable audit + user "fix all bugs"): s4 cancelled at
    # ~1.5 h / 0 rows (136695) — superseded before its first eval by the
    # per-tooth width price. The rank-1 mean-corr wall told the optimizer all
    # 25 teeth cost the same width per nm; the see-saw direction (inner down /
    # outer up, MEASURED +0.037 T at constant width) sat in its null space,
    # which is why uniform-seeded campaigns pinned at the band ceiling near
    # T 0.917. s5 = same seed, same saturated hinge, corr term now the
    # MEASURED 3-block FW_TOOTH_W. THE EXPERIMENT: if the optimizer now finds
    # the see-saw basin from uniform on its own, the stalls were wrong prices,
    # not local minima.
    label="lumopt2_v2_uniform_s5",
    fw_tooth_w=eng.FW_TOOTH_W,
    fw_pen_cap=2.0,
    scan_width_nm=10.0, n_wl_points=501,          # v2 window (plan §5a)
    region_dx_nm=eng.DX_PITCHLOCK_NM,             # ★pitch-locked (plan §24)
    scan_center_nm=1564.614,                      # MEASURED mx_origin resonance
    free_comb=False,                              # user: comb fixed at winner
    rho_band=False,                               # retired (skill item 27)
    fwhm0_um=FWHM0_UM,
    adj_phase_fix=True, adj_fix_re=ADJ_FIX_PORT[0], adj_fix_im=ADJ_FIX_PORT[1],
    fwhm_wall=True,
    fw_curve=True,        # ★MEASURED 6-point elongation curve (61742/61782),
                          # not the secant. Opens the explorable shift range
                          # from e=27.5 to e=81.0 nm before the band edge.
    fw_anchor={"fwhm": FWHM0_UM, "mcorr": eng.CORR_NM, "elong": 0.0,
               "corr_vec": (eng.CORR_NM,) * eng.N_FREE},
    # ★shift trust box added 2026-08-24. The "no trust_nm" precedent above was
    # written so a uniform seed could TRAVEL, and that still holds for the
    # geometry blocks — corr/avg/wcav stay unbounded here. But ev2 of the first
    # dispatch showed the shift block does not need freedom, it needs SCALE:
    # L-BFGS-B's uniform ~0.0575 probe in bounds-scaled space became 5.75
    # nm/tooth over the 200 nm range, i.e. e=287.4 and W 32.27 um (FOM -1.554),
    # skipping clean over the 1-65 nm window the ladder measures to be FREE.
    # 30 nm lands the same probe at e~29. The box re-centres on the best design
    # at every restart, so the seed can still travel anywhere over the run.
    trust_nm={"shift": 30.0},
    max_iter=60, max_feval=100,                   # seedA budgets
)
N_TASKS = 1


def main(task_idx=0):
    if EXACT_WIDTH_GRAD:
        assert ADJ_FIX_FIELD is not None, \
            "C_field not fitted — run fit_c_field on the W3 vectors first"
        SPEC.width_grad = True
        SPEC.wg_anchor = {"softw": SOFTW0_UM, "fwhm": FWHM0_UM}
        SPEC.wg_source, SPEC.wg_adj_resource = "import", "GPU"
        SPEC.adj_fix_field_re, SPEC.adj_fix_field_im = ADJ_FIX_FIELD
    # no seed_override: the module-default seed IS the uniform initial design
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    best = eng.run_campaign(SPEC, out_dir)
    print(f"[v2uniform] done: best_fom {best['fom']:.5f} (delivered design = "
          f"width-filtered log, never this number)")


if __name__ == "__main__":
    print("v2 basin-2 campaign: strictly uniform seed; dispatch via deploy")
