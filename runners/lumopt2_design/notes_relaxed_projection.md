# Relaxed Gradient Projection (RGP) — Antonau/Hojjat/Bletzinger, SMO 63:1633–1651 (2021)
`doi:10.1007/s00158-020-02821-y`. Pages = **journal pages** (PDF p.1 = 1633). Their regime = ours: minimize `f(x)` s.t. `g_j ≤ 0`, ~1e5–1e6 vars, adjoint gradients, **constant step** (line search declared too expensive / inaccurate for highly non-linear responses, p.1634).
## 1. Buffer (critical) zone — eqs (12)(13)(14), p.1636–37
Where a constraint counts as active. **Linear ramp only** — "non-linear distribution is non-applicable … reduces the stability of the method" (p.1636).
```
LBV_j = CBV_j - BS_j                              # lower buffer edge       (12)
w_j   = (g_j(x) - LBV_j)/BS_j                     # inequality, w in [0,2]  (12)
w_j   = 1 + |h_j(x) - LV_j|/BS_j                  # equality, always >= 1   (13)
BS_j  = BSF * max_k(|g_j(x_k) - g_j(x_{k-1})|)    # size from history       (14)
```
`CBV` = buffer centre (init = limit `LV`) ⇒ zone spans `[LV-BS, LV+BS]`: **w=0 at entry, w=1 exactly at the limit, w=2 at the far edge, w>1 ⇒ violated.** `BS` init = 1e-12 or 1% of the limit; `BSF_init = 2.0` so "the algorithm has at least one optimization iteration inside the buffer zone before the constraint value reaches its limit" (p.1637).
## 2. Relaxation / correction split — eqs (15)(16), p.1637
```
w_r = w if w <= 1 else 1            # relaxation in [0,1]; equality: w_r == 1 always   (15)
w_c = 0                if w <= 1    # correction                                       (16)
    = BSF_init*(w-1)   if 1 < w < w_max
    = BSF_init*w_max   if w >= w_max
```
`w_max = 2` default; raise to 10–100 only when starting deep in the infeasible domain (p.1637). ⚠ The clamp as printed jumps (`BSF_init*(w_max-1)=2` → `BSF_init*w_max=4`) — apparent typo; use `BSF_init*(w_max-1)`.
## 3. Search direction — eq (17), p.1637 (the whole method)
```
p   = -[I - N w_r (N^T N)^{-1} N^T] grad_f   # relaxed projection
s_h =  p - N w_c                             # correction ROTATES the direction
s   =  s_h/||s_h||_max                       # bound the norm
```
Gradients pre-scaled by max norm; `w_r` diagonal r×r, `w_c` length-r. Geometry (Fig. 3, p.1638): **w=0 ⇒ plain steepest descent; w=1 ⇒ exact Rosen projection; w>1 ⇒ rotated toward the feasible side.** ★Contrast with classical GP (eq (11), p.1635) where the correction is a *separate design update* `-N(N^T N)^{-1} g_a` needing re-linearization and **extra primal/adjoint solves**: here it is additive in the same direction — **zero extra solves, and the objective keeps improving while restoring.**
## 4. Active-set / adaptation — §3.3, p.1638
No hard on/off switch anywhere; `w` switches continuously. Two adaptation rules:
- **Zigzag** (eq 19, 4 iterates): `Dg_i*Dg_{i-1} < 0` AND `Dg_{i-1}*Dg_{i-2} < 0` ⇒ widen the buffer,
  `BSF_{i+1} = BSF_i + |w_i - w_{i-1}|*factor`, `factor = 1` in every run (eq 18). Wider `BS` ⇒ smaller `w_c`,
  larger `w_r` ⇒ the constraint moves less per iterate. ("Alternatively … BSF can be doubled.")
- **Infeasible drift** (eq 20): `g_i>0` AND `g_{i-1}>0` AND `Dg_i >= 0` ⇒ shift the buffer *centre* inward,
  `CBV_{i+1} = CBV_i - (g(x_{i-1}) - LV)` (eq 21, index typo in print) — "moving the real boundaries deeper
  inside the feasible domain". Same schema restores `CBV` if correction is too strong; (22) = equality form.
## 5. Step size & convergence — Algorithm 1 p.1640, §4
Constant `alpha` (= max update magnitude, 0.15 / 0.5 mm) applied to the max-norm-normalized `s` ⇒ **alpha is literally a max-norm trust radius**. No line search. Stop on max iterations or <0.1% objective gain over the last 10 (p.1644). Loop: `f,g → UpdateBufferZones(14) → BufferCoeffs(12,13) → AdaptiveFns(18,21) → gradients → normalize → direction(17) → x += alpha*s`. They say plainly (p.1640, Conclusions p.1650) that the constant step is the method's weak point (analytic cases need 261/102/2766 iterations); line search is named as future work.
## 6. Tuning actually used, and sensitivity
Tables 4 (p.1642) / 5 (p.1649): step 0.15 / 0.5; **buffer scale factor (18) = 1, initial BSF = 2.0** — the *same* two knobs in both problems, vs GP's hand-tuned per-constraint correction coefficients (1.0 compliance, 0.05 packaging) where "with smaller coefficients … after some iterations, method diverges" (p.1641). Conclusions (p.1650): RGP "does not require accurate parameter set up". Payoff: over iterations 5–50 GP improved the objective 9.5%, RGP 14.4% — **1.7×** (p.1642) — at identical per-iteration cost (247 vs 250 s; total 14326 → 10750 s, Table 6 p.1649). From an *infeasible* start (§4.3) RGP wins because `w_max=2` caps how much of the step restoration may eat, while GP spends nearly the whole update correcting.
## 7. Single-constraint case — SRGP, eqs (23)(24), p.1639
With r = 1 the `(N^T N)^{-1}` "linear system" is a scalar — the projection is free, no solver. They also give a projection-free variant `s = -(1-w)*grad_f - w*grad_g`, `w = (g(x) - LBV)/(2*BS)` in [0,1] — the same buffer ramp reused as a **convex blend weight** (multi-constraint: `w = max_j w_j` or `mean_j w_j`, eq 25). ⚠ At the limit SRGP has `w = 0.5` and already points *into* the feasible domain — it retreats from the boundary instead of riding it; RGP at `w=1` rides it exactly. SRGP oscillates more and damps slower (p.1639).

# Mapping to `run_projected`
Ours: `lumopt2_design.py:2223 _proj_step` (3-way hard branch on measured `W` at `W_tgt ± marg/2`) and `:2253 run_projected`. Maximizing `T` ⇒ `f = -T`, `grad_f = -gT`; ceiling `g_hi = W - W_hi`, `grad_g = gW`; floor `g_lo = W_lo - W`, `grad_g = -gW`. Only one side is ever in its buffer ⇒ **r = 1: §7 applies, no linear solve.**

**R1 — replace the 3-way branch with one continuous formula (the whole recommendation).** CLIMB and RIDE are the `w_r=0` and `w_r=1` ends of eq (17); RESTORE becomes the additive `w_c` term. Keeping our `D` metric:
```
u    = D*gW ;  coef = (gT.u)/(gW.u)          # our existing ride coefficient
p    = D*gT - w_r*coef*u                     # w_r=0 -> climb, w_r=1 -> today's RIDE
s_h  = p/||p||_max - w_c * u/||u||_max       # correction rotates, does not replace
step = cap * s_h/||s_h||_max                 # cap = wgp_step_max_nm (their alpha)
```
At `w_r=1, w_c=0` this reproduces today's RIDE **exactly** and `gW·step = 0` stays exact (`w_c` is the only term that breaks it, and only once already violated), so `gates/gate_projection_local.py` stays valid as the `w_r=1` case — add a `w_r=0` case and a `w_c>0` sign case.

**R2 — size the buffer from our own history (eq 14).** `BS = BSF*max_k|W_k - W_{k-1}|`, `BSF_init = 2`, seeded at `max(1e-3 µm, 1% of marg)`. Our control drifted **+0.0110 / +0.0122 µm per iterate** (THEORY §11b, MEASURED) ⇒ `BS ≈ 0.022–0.025 µm` vs a `marg` of order 0.4 µm: relaxation engages only in the last couple of iterates before the ceiling — exactly our ceiling-riding regime.

**R3 — kill the pure-restore step.** Today's `restore` spends 100% of the move on `-(W-W_tgt)*gW/|gW|²` and zero on transmission — that *is* classical GP eq (11), whose cost §4.3 measures. With `w_max=2`, `BSF_init=2` the correction never exceeds ~2 units of a max-norm-1 direction, so **every** iterate keeps climbing T.

**R4 — add the two adaptation rules (18–21); ~15 lines, free.** We already log `W` per iterate in `<label>_proj.jsonl` (zigzag needs 3 stored `ΔW`, drift needs 2). Both symptoms are documented here — boundary thrashing (THEORY §4b Reason 2) and monotone infeasible drift (defect #19) — and rule (21), widen *plus* recentre inward, is the published answer to exactly "width creeps up every iterate".

**R5 — what stays.** The Fletcher–Leyffer filter + `alpha *= 0.5` halving and `_cap(alpha)` (the paper has **no** line search and calls that its main deficiency, so ours is strictly better); the `D = (bound half-range)²` metric; decisions on **MEASURED** `fwhm_env`; the λ-chain-corrected `gW` (RGP uses the same ∇g and would mis-measure identically — §10c prerequisite stands); bounds clip; resume; logging. Optionally add their stop rule.

**R6 — SRGP eq (23) as the degraded-`gW` fallback**, replacing today's silent fall-back to the fixed-λ `gW` on `λ-CHAIN SKIPPED`: `s = -(1-w)*gT_dir - w*gW_dir` needs no `1/|gW|²` and makes no null-space claim, so it degrades gracefully instead of projecting onto a wrong plane. Its `w=0.5`-at-the-limit bias pulls *off* the ceiling — fine in a fallback, wrong as the default.

**Not applicable:** the `(N^T N)^{-1}` solver discussion (r=1) and the multi-constraint `max_j/mean_j` rule (25). Equality form (13) applies only if we drop the deadband and target `W` exactly — then `w_r ≡ 1` with `w_c = BSF_init*|W - W_tgt|/BS` is the entire policy.
