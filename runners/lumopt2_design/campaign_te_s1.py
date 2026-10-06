"""TE lane, seed S1 — the PLAIN uniform TE pi-shift grating (corr 250 / pitch
500 / W800), run through the d1 two-constraint (W AND λ_pk) null-space engine.

Study dir: runners/lumopt2_design/  |  Created 2026-10-04  |  Job(s): TBD
Purpose: raise t_pk of the TE device while holding its own envelope FWHM
(±2 %) and resonance — the TE lane is the only route to contesting Itai's
absolute Q_i (TE/TM factor 3.4× measured same-geometry, HANDOFF_2026-09-01 §6).

DEVICE (user, 2026-10-04): TE, h350, n 1.97/1.444, pitch 500 nm, uniform
corr 250 nm, W800, NO comb and no scatterers of any kind (bare=True — circles
were measured unhelpful for TE). Box 6.8 × 6.81 µm = the TE far-field box
ladder's converged value (job 164893: 6.8 vs 12/12.8 agree, T 0.912 both) and
the inverse-design numerics family. 60 FREE periods per side (Itai's
apodization footprint; HANDOFF next-step #2) inside N = 98/side = Itai's own
device length (user decision 2026-10-04, same N as the stored far-field rows of
both families). 2κL = 3.36 with κ_TE(corr 250) = 0.0343 /µm (DERIVED from the
stored te_q3db_20um ladder N=166..215, d ln Q_c/dN = 0.03426/period): below the
3.5 surrogate rule but above the 3.2 hard floor, so two_kl_floor is set to 3.3
here BY USER ORDER (N is a spec choice, not a surrogate), stated explicitly.

★EVERYTHING FITTED ON THE TM DEVICE IS VOID HERE until re-measured (skill
items 6, 28, 35): C_port, C_field, the noise floor behind wgp_fom_slack, the
PVA resonance (scan centre) and the seed's own fwhm_env/softW anchors. The
constants below marked None are filled from validate_te tasks IN THIS ORDER
(each is a hard stop): λ-finder (task 0) → SCAN_CENTER; production-window
canary (task 1) → FWHM0/SOFTW0; noise floor (tasks 2,3) → slack; C_port
(tasks 4+5 → `validate_te fit`); C_field (tasks 6+7 → fit_c_field.py) — all at
the SAME scan centre (validate_c325 task 45 lesson: a C fitted at a different
centre is a different functional); pipeline smoke (task 8); toy (task 9);
then this campaign (warm-started from the toy's evals+optstate, §6).
main() refuses to run before.

★TE-SPECIFIC RISK (research 2026-10-04, Johnson PRE 65 066611 / Kottke PRE
77 036611 / MEEP adjoint docs): in TE the field E_y is NORMAL to the walls
that corr and avg move — the hard case for FDTD shape gradients (TM had E
parallel to every moving wall). The C_port FD gate therefore also reports
the per-CLASS residual (corr/avg vs shift): if corr/avg miss by >10 % after
the global C while shifts pass, the engine's `bc_patch` (Johnson E∥/D⊥
normal reweight, FD-gated by validate_te) is the fallback, not a bigger C.

Dispatch (after ALL gates + explicit user approval; cluster per user):
    SBATCH_MEM=256G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \
        bash athena/deploy_athena.sh \
        --lumopt2-design=runners.lumopt2_design.campaign_te_s1
"""
import dataclasses
import os

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_v2_proj_d1 import SPEC as D1

# ── MEASURED by validate_te, pasted here (None = not yet measured) ──────────
SCAN_CENTER_NM = 1560.936    # MEASURED task 0 (job 168375_0, PVA): λ_pk 1560.9357; conformal stored 1559.990 (+0.95 nm) | T 0.9060 Q 1538 W 19.12 — old note: lam_pk at PVA numerics (conformal stored
                             # row: 1559.990, job 164893 — PVA shifts it; TM
                             # moved +5.2 nm, TE EXPECTED larger, unmeasured)
FWHM0_UM       = 19.121      # MEASURED task 1 (168530_1, 10 nm/501): fwhm_env_um 19.1209; T 0.9053 Q 1539 λ 1560.900 — task 1: fwhm_env_um of the seed = the width SPEC
SOFTW0_UM      = 18.738      # MEASURED task 1: softw_adj_um (the twin's OWN sample — wg_anchor must be measured through it; raw-line softw 18.756) — task 1: softw_um of the seed (anchor)
ADJ_FIX_PORT   = (0.945335, +0.117012)   # MEASURED, EXACT LSQ 2026-10-05 (FD+Re 168581_4, Im 168582_5): resid corr +3.6/-3.7 %,
                                     # avg +3.4 %, shift_1 -0.6 %, shift_30 -0.1 %, wcav -5.0 % -> PASS (all <=5 %). Held-out shift_30 -16.5 %
                                     # (1300x cancellation: known only to ~15 % out of sample). The earlier grid fit's -12.3 % was a fit artifact.
ADJ_FIX_FIELD  = (0.966720, +0.036560)   # MEASURED fit 2026-10-05 (FD+Re 168641_6, Im 168642_7): vector resid 0.012 %, per-param <=0.2 %, signs 3/3 PASS
FOM_SLACK      = 5e-4        # MEASURED tasks 2,3 (168579): T spread 1e-5 for +0.5 nm sub-cell tooth moves (PVA is smooth); 50× margin, 3× tighter than TM's 1.5e-3

N_SIDE   = 98                # Itai's length (user 2026-10-04); 2κL = 3.36 (DERIVED)
KAPPA_TE = 0.0343            # /µm at corr 250 (DERIVED, te_q3db_20um ladder)

SPEC = dataclasses.replace(
    D1,
    label="lumopt2_te_s1",
    # ── device profile ─────────────────────────────────────────────────────
    polarization="TE",
    pitch_nm=500.0,
    corr0_nm=250.0,
    avg_w_nm=800.0,
    kappa_per_um=KAPPA_TE,
    n_free=60,
    n_periods_side=N_SIDE,
    bare=True, free_comb=False,          # no circles anywhere (user 2026-10-04)
    seed_override=None,                  # uniform seed from seed_params
    corr_seed_nm=None, avg_seed_nm=None,
    corr_min_nm=100.0, corr_max_nm=450.0,   # 250 ∓ 150/+200 (TM: 325 in 150..500)
    avg_bounds_nm=(775.0, 825.0),
    wcav_bounds_nm=(750.0, 1150.0),
    region_dx_nm=50.0,                   # = dx_pitchlock at pitch 500 (device mesh pitch/10)
    # widest tooth at bounds (825+450/2)/2 = 525 nm + 500 margin; region DFT ≈ 40 MB/λ
    # → host estimate ≈ 31 GiB at 501 λ, GPU ≈ 7 GiB (audit calibration; 168240's 88.8 GiB was HOST)
    region_y_half_nm=1050.0,
    # ── recording window: ±5 nm @ 20 pm (501 pts). Q_L MEASURED 1538 (λ-finder) →
    # spectral FWHM ~0.9 nm → ~46 pts/FWHM (resolution target for the IFT
    # λ selector; the engine has no hard assert on it — task 1 prints it).
    # Narrower than d1's ±5/501 to keep the 2×-longer region's field arrays
    # inside the 256G lane (item 30); recenter stays 2.0 (< half-window).
    scan_center_nm=SCAN_CENTER_NM or 1563.0,   # placeholder overwritten in main()
    # 2026-10-04 audit: FOM window ±2.5·FWHM = ±2.5 nm (MEASURED fwhm 1.015 nm) would be
    # clipped at the band edge inside a 6 nm window before recenter trips (clipping INFLATES
    # the softmax FOM). TM d1's ±5 nm / 20 pm window keeps the 2.0 nm recenter valid.
    scan_width_nm=10.0, n_wl_points=501, recenter_nm=2.0,
    # ── width spec / anchors (filled from the canary) ─────────────────────
    fwhm0_um=FWHM0_UM, wgp_target_um=FWHM0_UM,
    wg_anchor=None,
    fw_tooth_w=None, fwhm_wall=False,    # 25-entry TM table must stay OFF at n_free 60
    # ── d1 step engine, TE-scaled (dn_eff/dW ≈ 1.8× TM — research 2026-10-04):
    # nm caps ×~0.55 vs the TM-settled 20/40; the adaptive cap earns the rest.
    wgp_ns2=True, wgp_cap_adapt=True,
    wgp_step_max_nm=10.0, wgp_cap_max_nm=30.0, wgp_cap_grow=1.5,
    trust_nm={"shift": 15.0},
    wgp_lam_margin_nm=0.2, wgp_lam_target_nm=None,
    wgp_reuse_k=5, wgp_reuse_travel_nm=40.0,
    wgp_fom_slack=FOM_SLACK,
    wg_dwdlam=0.3655,                    # TM path fit — DIAGNOSTIC ONLY under ns2
    wg_dwdlam_fit=True,                  # (cancels from the step); refit online
    wg_src_tiles=4,                      # 2·(98·0.5+0.125)+2 = 100 µm / 50 nm = 2005 cells → 501/tile ≤ 1000
    # ── 2026-10-04 optimizer upgrades (user-approved; default-inert elsewhere,
    # gate_projection_local §10). TE lane is their first hardware user — the
    # k=8 pipeline smoke must show each one's log marker before the campaign.
    wgp_noise_freeze=True, wgp_noise_stop=3,   # cap frozen on noise-level rejects
    wgp_filter_band=True,                      # filter on band VIOLATION, not distance to centre (review A5)
    wgp_total_cap=True, wgp_cond_norm=True,    # one cap on the summed step; scale-free degeneracy test (review A2/A6)
    wgp_reuse_broyden=True,                    # rank-1 update of the reused width row
    wgp_mode_mac=0.9,                          # mode-hop reject below this overlap
    wgp_range_alpha=0.5,                       # restore half the violation per step
    wgp_range_cap_frac=None,                   # range cap = trust cap (Feppon α_C)
    # ── C factors: VOID until fitted (asserted in main) ────────────────────
    adj_phase_fix=True, adj_fix_re=1.0, adj_fix_im=0.0,
    two_kl_floor=3.3,                    # user-ordered N=98: 2κL 3.36 > hard floor 3.2
    max_iter=60, max_feval=120,
)

# ── v3 step engine variant (2026-10-05, user "go v3") — SPEC above stays the
# BASELINE (d1 ns2 engine + review fixes) so the two can be compared on the same
# seed, anchors and C factors (GPT F6 paired study). v3 = bounded QP step, total
# moving-resonance width row with measured dW/dλ, λ as a re-centred per-step
# trust bound (¼ of the 1.0 nm linewidth), 3-point-parabola peak objective.
SPEC_V3 = dataclasses.replace(
    SPEC, label="lumopt2_te_s1_v3",
    # width-row reuse OFF (toy 169105, MEASURED): its gate needs |ΔW| ≤ 0.025 um
    # since the last fresh solve, but every v3 step moved W 0.08-0.12 um, so it
    # never opened; near the band edge a stale row would also cost rejects.
    wgp_reuse_k=0,
    wgp_v3=True, wgp_v3_peak=True, wgp_v3_dlam_nm=0.25,
)

N_TASKS = 1


def main(task_idx=0):
    for name, val in (("SCAN_CENTER_NM", SCAN_CENTER_NM), ("FWHM0_UM", FWHM0_UM),
                      ("SOFTW0_UM", SOFTW0_UM), ("ADJ_FIX_PORT", ADJ_FIX_PORT),
                      ("ADJ_FIX_FIELD", ADJ_FIX_FIELD)):
        assert val is not None, f"{name} not measured — run validate_te first"
    # v3 engine (user 2026-10-06 "yes"): warm-starts from the v3 toy 169105_29 —
    # its evals.jsonl + optstate.json are copied into out_dir before dispatch.
    spec = dataclasses.replace(
        SPEC_V3, scan_center_nm=SCAN_CENTER_NM, fwhm0_um=FWHM0_UM,
        wgp_target_um=FWHM0_UM, wg_anchor={"softw": SOFTW0_UM, "fwhm": FWHM0_UM},
        adj_fix_re=ADJ_FIX_PORT[0], adj_fix_im=ADJ_FIX_PORT[1])
    spec.adj_fix_field_re, spec.adj_fix_field_im = ADJ_FIX_FIELD
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    best = eng.run_campaign(spec, out_dir)
    print(f"[te-s1] done: best_fom {best['fom']:.5f} — read "
          f"{spec.label}_proj.jsonl for rho_T/cap_nm/rLam_nm")
