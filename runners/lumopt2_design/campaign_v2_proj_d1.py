"""d1 — two-constraint (W AND λ_pk) null-space campaign from BEST_T9636.

Study dir: runners/lumopt2_design/  |  Created 2026-08-30  |  Job(s): TBD
Purpose: raise t_pk from BEST_T9636 (0.96361 @ fwhm_env 18.354) while holding
BOTH the envelope width and the resonance wavelength, with a step engine that
can actually move.

★WHY THIS IS NOT b2 (job 138595, REJECTED 2026-08-30 — understand before
re-proposing anything similar): b2 changed wgp_step 0.25→0.5, which is a
mathematical NO-OP — the delivered step was a constant wgp_step_max_nm inf-norm
move because _cap(a)=cap0·min(1,a/a0) scales the cap in lockstep with alpha
(raw gradient step ~73 nm ⇒ cap always binds; alpha and wgp_step cancel).
d1 changes the two things that were actually binding:
  1. wgp_ns2: the step is projected into the null space of BOTH the raw
     fixed-λ ∇W and gλ = dλ_pk/dp (already computed exactly per iterate via
     the IFT selector passes, zero extra solves). Measured basis: ΔW ≈
     0.3655·Δλ_pk on every gradient path (c1 live ride evals grew W
     +0.031/+0.050 µm — λ drift, not envelope reshaping), and task 49
     (job 138575) proved T can rise at fixed λ while pitch-rescale moves λ
     without moving W. Holding λ kills the width-creep channel at the root,
     and the fitted 0.3655 coefficient cancels out of the feasible directions.
     Feppon range-space restoration replaces the all-or-nothing restore.
  2. wgp_cap_adapt: the cap becomes a real trust radius — grows ×1.5 on
     accepts where |Δλ|<0.10 nm AND |ΔW|<0.020 µm both measurably held
     (ceiling 60 nm), halves on rejects (floor 2 nm), persisted in the
     <label>_optstate.json sidecar across REQUEUEs/restarts.

Seed lessons inherited (campaign_v2_proj_best.py, MEASURED 2026-08-29):
wgp_target_um = the seed's OWN fwhm_env (18.3545) so the null-space law is
active from iterate 0 — at BEST the shadow price lam=0.062 (∇T aligned with
∇W), and a width-blind climb bought +0.00097 T for +0.272 µm (0.0036 T/µm).
param_bounds derives the frozen-comb slivers from the seed, so BEST's evolved
winner comb rides along frozen at its own values (gen-1 ruling 2026-08-30:
comb stays frozen; free it in a d2 arm only if d1 is healthy but saturates).

FALSIFICATION READOUT (task 51 toy first): rho_T in <label>_proj.jsonl = the
fraction of D^½∇T surviving the two-constraint projection. rho_T < 0.15 on
all toy iterates ⇒ BEST is locked under (W, λ) — a real physics verdict; the
pivot is the AL trade-curve (AL_COMBINED_DESIGN.md), not a bigger cap.

★INHERIT THE TOY'S STATE BEFORE DISPATCH (user rule 2026-08-30 — never
re-derive iterates a prior lane already computed at ~2.5 GPU-h each). The
toy (task 51, label lumopt2_v2_ns2toy) runs the SAME spec knobs and seed, so
the campaign must warm-start from its last accepted point, not re-derive it:
    ssh athena 'mkdir -p <d1_out_dir> && \
      cp <toy_out_dir>/lumopt2_v2_ns2toy_evals.jsonl \
         <d1_out_dir>/lumopt2_v2_proj_d1_evals.jsonl && \
      cp <toy_out_dir>/lumopt2_v2_ns2toy_optstate.json \
         <d1_out_dir>/lumopt2_v2_proj_d1_optstate.json'
(_best_from_log then resumes from the toy's best in-band row and the grown
trust cap + λ target carry over via the optstate sidecar. The dispatch note
must name the inherited rows.)

Dispatch (after gates + task 50 smoke + task 51 toy + explicit user approval):
    SBATCH_MEM=300G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \
        bash athena/deploy_athena.sh \
        --lumopt2-design=runners.lumopt2_design.campaign_v2_proj_d1
"""
import dataclasses
import os

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9636
from runners.lumopt2_design.campaign_v2_proj import ADJ_FIX_FIELD, SPEC as C1

SPEC = dataclasses.replace(
    C1,
    label="lumopt2_v2_proj_d1",          # the label IS the resume key — bumped
    seed_override=BEST_T9636,
    scan_center_nm=1566.44,              # BEST's own resonance (PVA v2 numerics)
    wgp_target_um=18.3545,               # seed fwhm_env — null-space from it-0
    # ── the d1 step engine ──────────────────────────────────────────────────
    wgp_ns2=True,
    wgp_lam_target_nm=None,              # latch the first measured λ_pk
    # 0.05→0.2 (2026-09-01, user λ-policy): λ-hold is an algorithmic tool,
    # not a spec — never fight T for the last 0.1 nm; final trim is free.
    wgp_lam_margin_nm=0.2,
    wgp_cap_adapt=True,
    # 10→20 (2026-09-01): 30+ nm measured safe on this landscape; earn the
    # rest. Cap state resumes from the optstate sidecar where present.
    wgp_step_max_nm=20.0,
    wgp_cap_max_nm=60.0,
    wgp_cap_grow=1.5,
    # ★k=5 reuse (2026-09-01, user-approved): gW rotates 0.685°/10 nm step
    # (probe 139256) ⇒ 4 stale steps ≈ 2.8° ≈ 5% leak, inside the guards;
    # smoke 139345 PASS (2/4 reused, constraints held, cap grew).
    wgp_reuse_k=5,
    # filter slack at the T noise floor (Sun–Nocedal): spurious noise
    # rejects must not fake cap-collapse convergence.
    wgp_fom_slack=1.5e-3,
    wg_dwdlam_fit=True,                  # diagnostics-only under ns2 (the
                                         # coefficient is out of the step math)
    # 60 iterates × ~1.1-1.5 h measured on H200 fits the 96 h lane with margin;
    # resume-by-label continues from the best logged row if walltime ends it.
    max_iter=60, max_feval=120,
)

N_TASKS = 1


def main(task_idx=0):
    assert ADJ_FIX_FIELD is not None, "C_field not fitted — see campaign_v2_proj"
    SPEC.adj_fix_field_re, SPEC.adj_fix_field_im = ADJ_FIX_FIELD
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    best = eng.run_campaign(SPEC, out_dir)
    print(f"[v2proj-d1] done: best_fom {best['fom']:.5f} — two-constraint "
          f"lane; read {SPEC.label}_proj.jsonl for rho_T/cap_nm/rLam_nm")
