# One-adjoint combined objective (AL family) — implementation design

2026-08-28. Status: DESIGN ONLY — implementation starts after c1/b1 shadow-price
traces justify it (decision criteria at the bottom). User-approved as the
zero-GPU exploration; formulation may still change, so everything is
spec-switched and nothing replaces the projection path.

## Formulation

Objective per iterate: J = T_soft − λ·softW, with λ FIXED within the iterate.
Two λ policies (spec-selectable):
- `al`: equality-form AL — λ ← λ + μ·(W_meas − W_tgt) between iterates
  (signed, never dormant; μ = spec.wg_mu; the engine's wg_lam_hi/lo machinery
  generalizes — reuse, do not duplicate).
- `lagged`: λ ← the previous iterate's exact projection shadow price
  (`lam` in the proj jsonl — already logged; one-iterate-stale, exact machinery).

## Why ONE adjoint suffices

The adjoint problem is linear ⇒ sources superpose (same fact as the 4-tile
source). Build ONE adjoint source
    S = (∂J/∂T-slice)·S_port + (∂J/∂softW)·S_width
      =  jac_T·S_port − λ·S_width_weighted
and the single run returns ∇J directly. 3 solves/iterate → 2 (−33%).

## Code-touch points (all in lumopt2_design.py)

1. `CampaignSpec.wg_formulation: str = "projection"` — values
   "projection" (default, bit-identical behavior) | "al" | "lagged".
2. `make_fct_v2`: for al/lagged, fct = softmax_T(x[:n_wl]) − λ_live·x[-1]
   (λ_live read via spec._wg_lam_live, a plain float — autograd sees it as
   constant within the iterate; the softW jacobian entry −λ then WEIGHTS the
   width entry's adjoint source through the existing MixedFom machinery — this
   may already suffice WITHOUT new source code, because lumopt2 scales each
   entry's adjoint source by its jacobian entry. ★VERIFY FIRST: if
   base_fom scales per-entry sources by jac_of_fom, the combined run is free —
   just run BOTH entries' adjoints in ONE adjoint phase. Check
   `_compute_adjoint_fields_phased`: entries are separate .fsp files ⇒ they are
   separate SOLVES. The saving requires merging the two sources into ONE fsp:
   extend `build_base_fsp`'s width twin to also carry the port adjoint source,
   OR inject the port-mode source as an import source into the width-adjoint
   fsp (superposition happens in the source dataset, amplitudes = jacobian
   entries × C-calibrations).★ This is the real engineering: ONE adjoint .fsp
   carrying S_port·jac_T·C_port + S_width·(−λ)·C_field. C's differ per source
   type — apply per-component BEFORE summing, i.e. bake C_port into the port
   part and C_field into the width part of the dataset.
3. `run_campaign`: dispatch on wg_formulation — al/lagged use plain L-BFGS-B
   (ScipyOptimizer, the pre-projection path) with the combined fct; no
   run_projected. λ update in the per-iterate callback; log λ, dw_pred,
   dlam_pred exactly as run_projected does (same jsonl schema).
4. Guards carried over verbatim: λ-chain on gW? — NOT needed for the gradient
   (∂T/∂λ=0 at peak kills the T chain term) but the WIDTH part still needs it:
   the combined source's width component must use the CORRECTED width gradient
   ⇒ the two selector passes still run (they are assembly-only, zero solves).
   wgp_lam_step_nm, curvature floor, WidthTrip, resume: unchanged.

## Gates (before any dispatch)

- Math gate: combined-fct jacobian == jac_T − λ·jac_W (autograd, flat-x layout).
- Plumbing gate: source-dataset superposition — assert the merged source ==
  α·S_port + β·S_width on a synthetic pair, and that per-component C's landed
  on the right parts (the known failure mode).
- FD gate on hardware (smoke tier, task TBD): combined ∇J vs finite
  differences at the N=60 surrogate — the C-recipe's standard.
- Known-answer: at λ=0 the combined gradient must equal the pure-T gradient
  bit-for-bit; at jac_T=0 (wg_pure), equal −λ·∇W.

## Decision criteria (read from c1/b1 proj jsonl, ~10+ iterates)

- `lam` trace smooth & slowly varying (|Δλ/λ| < ~30%/iterate) ⇒ lagged/AL is
  near-exact ⇒ implement + smoke + a 3-iterate toy, then adopt for campaign 2.
- `lam` oscillating/sign-flipping ⇒ stay with the exact two-adjoint
  projection; bank this design.
- Either way the adaptive step-cap (separate small feature) is worth having:
  cap doubles after 5 consecutive verified dlam_pred iterates, halves on any
  filter reject. Implement together with this or standalone.
