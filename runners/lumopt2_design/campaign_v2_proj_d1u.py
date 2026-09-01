"""d1u — two-constraint (W AND λ_pk) null-space campaign, UNIFORM lane
continuation (inherits c1's 15 iterates — never re-derive, user rule
2026-08-30).

Study dir: runners/lumopt2_design/  |  Created 2026-08-30  |  Job(s): TBD
Purpose: the from-uniform arm of the d1 generation — same ns2 + adaptive-cap
engine as campaign_v2_proj_d1, continuing the cancelled c1 lane (Athena
138535) from its best logged row (t_pk 0.9273 / W 18.658 / λ_pk 1565.594,
eval 5 of the resumed log) instead of re-deriving ~15 iterates at ~2.5 GPU-h
each. Science question it keeps alive: can the machine reach a BEST-class
design from uniform on its own, now that the resonance-drift channel is
constrained?

★STATE INHERITANCE IS THE DISPATCH PREREQUISITE (CLAUDE.md §6, 2026-08-30):
before dispatch, server-side copy c1's eval log into this label's out_dir:
    ssh athena 'mkdir -p <d1u_out_dir> && \
      cp .../lumopt2_v2_proj_c1/lumopt2_v2_proj_c1_evals.jsonl \
         <d1u_out_dir>/lumopt2_v2_proj_d1u_evals.jsonl'
(c1 predates the optstate sidecar — cap/λ-target start at spec defaults; the
λ target latches at the resumed row's own resonance on eval 0.)
_best_from_log then warm-starts from c1's best in-band row; trust boxes
re-centre on it per the resume path. The dispatch note names the inherited
rows.

Differences vs campaign_v2_proj_d1: no seed_override (the copied log IS the
seed), wgp_target_um stays C1's ceiling-ride level 18.613 (c1's best row sits
0.045 above it — inside the ±0.05 deadband, so the null-space law engages
immediately; the hard ±2% band still has 0.055 µm of headroom above), and
scan_center stays C1's (resume re-centres onto the row's λ anyway).

Dispatch (both d-lanes on Athena, user-approved 2026-08-30):
    SBATCH_MEM=300G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \
        bash athena/deploy_athena.sh \
        --lumopt2-design=runners.lumopt2_design.campaign_v2_proj_d1u
"""
import dataclasses
import os

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_v2_proj import ADJ_FIX_FIELD, SPEC as C1

SPEC = dataclasses.replace(
    C1,
    label="lumopt2_v2_proj_d1u",         # label bumped; c1's log is COPIED in,
                                         # not resumed under the old label
    wgp_ns2=True,
    wgp_lam_target_nm=None,              # latch the resumed row's resonance
    # 2026-09-01 restart knobs (user-approved, same rationale as d1):
    # λ-deadband 0.2 (λ = tool not spec; d1u's 0.30 nm residual stops
    # fighting), start cap 20 (measured-safe), k=5 reuse (0.685°/step probe
    # + smoke PASS), filter slack at the noise floor.
    wgp_lam_margin_nm=0.2,
    wgp_cap_adapt=True,
    wgp_step_max_nm=20.0,
    wgp_cap_max_nm=60.0,
    wgp_cap_grow=1.5,
    wgp_reuse_k=5,
    wgp_fom_slack=1.5e-3,
    wg_dwdlam_fit=True,
    max_iter=60, max_feval=120,
)

N_TASKS = 1


def main(task_idx=0):
    assert ADJ_FIX_FIELD is not None, "C_field not fitted — see campaign_v2_proj"
    SPEC.adj_fix_field_re, SPEC.adj_fix_field_im = ADJ_FIX_FIELD
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    log = os.path.join(out_dir, f"{SPEC.label}_evals.jsonl")
    # ★fail loud if the inheritance step was skipped: a cold uniform start
    # here would silently re-derive c1's 15 iterates (~20 GPU-h).
    assert os.path.exists(log), (
        f"{log} missing — copy c1's evals jsonl into this label FIRST "
        "(see module docstring); dispatching without it re-derives c1.")
    best = eng.run_campaign(SPEC, out_dir)
    print(f"[v2proj-d1u] done: best_fom {best['fom']:.5f} — uniform-lane ns2 "
          f"continuation; read {SPEC.label}_proj.jsonl for rho_T/cap_nm")
