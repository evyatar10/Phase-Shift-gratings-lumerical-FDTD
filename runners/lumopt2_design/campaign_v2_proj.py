"""Ceiling-riding PROJECTED-gradient campaign — the width-constant optimizer.

Study dir: runners/lumopt2_design/  |  Created 2026-08-25  |  Job(s): TBD
Purpose: maximise T at CONSTANT mode width using the EXACT width gradient,
by riding the band ceiling and stepping in the null space of grad-W — instead
of a penalty that is exactly zero while the design is in spec.
Formulation + gates: HANDOFF.md "THE FORMULATION, DECIDED 2026-08-24 23:00".

★A SEPARATE MODULE, not a new label inside campaign_v2_uniform.py, for one
reason: that module is what job 136753 is RUNNING, and every Athena partition
is PreemptMode=REQUEUE — editing it would let a requeued 136753 restart under
a different FOM. (§6. The FOM here genuinely differs: pure T, no width
penalty, width steered by the projection.)

PREREQUISITES — all three, or this must not be dispatched:
  1. the tiled width gradient RUNS and its per-tile max|src| are all non-zero
     (task 37 / job 136967, the `[wg-tiles]` log line);
  2. its signs match the keep-forever FD [-0.00365, +0.01825, +0.02026];
  3. C_field FITTED AT THIS TILING (fit_c_field.py) — magnitudes are needed
     for step-clipping and restoration, and a cropped-region C is void.
ADJ_FIX_FIELD stays None until 3 passes; main() refuses to run without it.

★MEMORY — MEASURED, NOT GUESSED (job 137012, P2, OOM-killed exit 137 at
"Computing gradient fields"): `wg_project` runs `calculate_gradient_fields`
TWICE (once for grad-T, once for grad-W), so TWO full
(nx,ny,nz,3,n_wl) field arrays are live at once — and lumopt2 fetches the
FULL lambda grid per entry before slicing (fdtd_session.py:1355). At
n_wl_points=501 that OOMs a 160G job. The single-pass tiled gradient tasks
(37/40) fit in 160G only because they used 151 points.
⇒ **Dispatch this campaign at SBATCH_MEM=300G**, and gates that inherit this
SPEC at 300G too.

Dispatch (after the gates):
    SBATCH_MEM=300G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \
        bash athena/deploy_athena.sh \
        --lumopt2-design=runners.lumopt2_design.campaign_v2_proj
"""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_v2_uniform import (
    ADJ_FIX_PORT, FWHM0_UM, SOFTW0_UM)

# ── the width spec, and where on it we ride ──────────────────────────────────
# Band is SYMMETRIC ±2% about the benchmark (user ruling 2026-08-23).
W_HI_UM = 1.02 * FWHM0_UM            # ceiling, 18.713 for the 18.346 family
MARGIN_UM = 0.10                     # >> the 5.5 nm width noise floor,
                                     # << the 367 nm half-band
# ★Ride the CEILING minus the margin, NOT the benchmark: the band's upper half
# is free real estate and BEST_T9636 leaves 0.36 um of it unspent (worth an
# EXPECTED +0.005..+0.015 T at measured marginal rates). HANDOFF item ②-3.
W_TARGET_UM = W_HI_UM - MARGIN_UM

# ★★★C_field for the TILED width adjoint — FITTED AND PASSED 2026-08-25.
# Source: fit_c_field.py on FD (job 136189, CPU full region) + Re (137003
# task 37) + Im (137003 task 40), all at THIS tiling (wg_src_tiles=4, GPU,
# full region, `[wg-tiles] live 4/4` on both).
#   fit prints C = 0.4554 - 0.1336i (s 0.4746, phi -16.35 deg), but the
#   ENGINE APPLIES a*RE + b*IM, which needs the CONJUGATE of that.
#   vector residual 0.1% | per-param resid -0.4% / +0.1% / -0.1% | signs 3/3
# The W3 gate is <=10% per param — this passes by ~25x. Note the RAW ratio
# Re/FD was 1.990 / 2.035 / 2.036, i.e. constant to +-1.2% across three
# different parameter classes: the signature of a correct adjoint awaiting
# one global constant, which is exactly what C_field is.
# ★STANDING ORDER (fit_c_field docstring): re-verify at a SECOND operating
# point before trusting it broadly, and re-fit on every Lumerical version
# bump. The P2/P3 gates below are at different points and serve as that
# second check for SIGNS; a full magnitude re-fit elsewhere is still owed.
# ★★SIGN VERIFIED 2026-08-25 (Fable audit B1, then re-derived here):
#   stored (0.4554, -0.1336) -> resid -18.4/-14.8/-14.4%  FAILS W3
#   conj   (0.4554, +0.1336) -> resid  -0.4/ +0.1/ -0.1%  PASSES
# Signs are unaffected either way, which is exactly why this was
# invisible: it would have run with ~15% wrong step MAGNITUDES.
ADJ_FIX_FIELD = (0.4554, +0.1336)

SPEC = dataclasses.replace(
    eng.CampaignSpec(),
    label="lumopt2_v2_proj",
    # numerics identical to the s5 campaign, so the two are comparable
    scan_width_nm=10.0, n_wl_points=501,
    region_dx_nm=eng.DX_PITCHLOCK_NM,
    scan_center_nm=1564.614,
    free_comb=False,
    rho_band=False,
    fwhm0_um=FWHM0_UM,
    adj_phase_fix=True, adj_fix_re=ADJ_FIX_PORT[0], adj_fix_im=ADJ_FIX_PORT[1],
    # ── the exact width gradient, tiled around the CUDA 1024-thread ceiling ──
    width_grad=True,
    wg_source="fieldregion",
    wg_adj_resource="GPU",
    wg_src_tiles=4,                  # ~511 cells/tile at the pitch-locked
                                     # mesh (528 at dx=50); cap is ~1024
    wg_anchor={"softw": SOFTW0_UM, "fwhm": FWHM0_UM},
    # ★wg_track_resonance (audit B2, 2026-08-25): DEFAULT IS FALSE, and the
    # twin (plus the tiles, whose lambda follows sim_result.wavelengths)
    # would stay pinned at the scan centre while the resonance walks up to
    # RECENTER_NM = 2.0 nm ~ 2.7 linewidths across a campaign. Standing user
    # ruling (2026-08-23, plan §23): softW is measured ON RESONANCE, always.
    # Off-resonance the width gradient's sign structure is demonstrably
    # different, and both restoration and the null space are sign-sensitive.
    wg_track_resonance=True,
    # ★wg_lam_chain (DEFECT #19, 2026-08-25): WITHOUT this the projection nulls
    # the WRONG gradient. gW from the adjoint is dW/dp at FIXED λ, but W is
    # specced at the device's own MOVING resonance, and W is near-linear in λ
    # (MEASURED dW/dlam = 0.3655 µm/nm, r = 0.984, n = 9). On job 137075_41 the
    # ENTIRE +0.0110 µm width change of iterate 0→1 was explained by the
    # +0.04 nm λ drift, and the T-per-µm exchange rate did not beat the
    # unprojected baseline (0.097 vs 0.091). Unfixed, restore pushes along a
    # fixed-λ gW that cannot undo λ-mediated growth while climb re-drifts λ.
    # Costs ZERO extra adjoint solves (two selector passes off the same fields).
    # ★REQUIRES ≥~40 spectrum points per spectral FWHM — satisfied here
    # (10 nm / 501 pts = 20 pm vs FWHM ~810 pm). Do NOT widen scan_width_nm
    # without raising n_wl_points in step.
    wg_lam_chain=True,
    # ── the projection driver ───────────────────────────────────────────────
    wg_project=True,
    wgp_target_um=W_TARGET_UM,
    wgp_margin_um=MARGIN_UM,
    wgp_step=0.25,                   # calibrated by gate P2
    wgp_step_max_nm=5.0,             # C_field magnitude guard
    # ★wgp_autogain: the projection direction is EXACTLY scale-invariant in
    # |C_field| (verified: scaling |C| moves the null space 0.00 deg), so the
    # magnitude only sets step-length prediction and restoration gain — and
    # those are MEASURABLE. This compares predicted vs measured dW each
    # iterate and self-calibrates, which removes the entire class of "C was
    # fitted at the wrong mesh" error that cost a dispatch cycle on
    # 2026-08-25. Only the PHASE of C still has to be right.
    wgp_autogain=True,
    # ★NO fwhm_wall: the penalty is replaced, not stacked. Width is steered by
    # the projection; WidthTrip still fail-closes the delivered design.
    fwhm_wall=False,
    trust_nm={"shift": 30.0},        # the bounds box the step is clipped to
    # ★max_iter 60 -> 30 (DERIVED 2026-08-25): a projected iterate costs THREE
    # solves (forward + port adjoint + tiled width adjoint) ~= 2.7 h at the
    # measured congested-node speeds, so 60 iters = ~162 h and would be KILLED
    # by the 96 h 4d_1g walltime two-thirds through. 30 x 2.7 = ~81 h fits with
    # margin. Resume is proven (136753 warm-started correctly), so a second
    # dispatch continues from the best logged row if 30 is not enough.
    max_iter=30, max_feval=60,
)
N_TASKS = 1


def main(task_idx=0):
    assert ADJ_FIX_FIELD is not None, (
        "C_field not fitted at the TILED config — run fit_c_field.py on the "
        "task-37 vectors first. Step-clipping and restoration need MAGNITUDES; "
        "signs alone cannot set a step length.")
    SPEC.adj_fix_field_re, SPEC.adj_fix_field_im = ADJ_FIX_FIELD
    # seed = the strictly uniform initial design (module default), so this is
    # the honest from-uniform result the programme wants.
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    best = eng.run_campaign(SPEC, out_dir)
    print(f"[v2proj] done: best_fom {best['fom']:.5f} | rode W_tgt "
          f"{W_TARGET_UM:.3f} um (ceiling {W_HI_UM:.3f}) | read "
          f"{SPEC.label}_proj.jsonl for phase/lambda/dw_pred per iterate")


if __name__ == "__main__":
    print(f"projected campaign: ride {W_TARGET_UM:.3f} um "
          f"(ceiling {W_HI_UM:.3f}); dispatch via deploy after the gates")
