"""gate_projection_local.py — LOCAL, ZERO-GPU gate for wg_project (gate P0 +
clip + restoration + legacy-off). Study: v2 width projection; 2026-08-25;
no jobs. Run AFTER applying patch_projection.diff:  python gate_projection_local.py
Exit 0 = all pass. Tests the REAL engine code (_proj_step, make_fct_v2,
CampaignSpec) with synthetic gradient vectors — no lumapi, no FDTD.
Section 11 (2026-10-05) drives the REAL run_projected through a fake project
(review fixes A1 noise retry, A4 ineligible rows, A5 filter band)."""
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")   # check names print λ (cp1252 console)
sys.path.insert(0, r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes"
                   r"\runners\lumopt2_design")
import lumopt2_design as eng  # noqa: E402

rng = np.random.default_rng(7)
N = 191
# D mimics the real conditioning: shift block 0-200 nm vs 1e-3 nm slivers
half = rng.uniform(0.5, 100.0, N)
half[50:75] = 100.0          # shift-like block, half-range 100 nm
half[75:100] = 1e-3          # frozen slivers
D = half ** 2
W_TGT, MARG, BIG = 18.613, 0.10, 1e9

fails = []
def check(name, ok, msg=""):
    print(("PASS " if ok else "FAIL ") + name + ((" " + msg) if msg else ""))
    if not ok:
        fails.append(name)

# ── 1) P0: null-space exactness to machine precision ──────────────────────
worst = 0.0
for _ in range(50):
    gT, gW = rng.standard_normal(N), rng.standard_normal(N)
    step, phase, _ = eng._proj_step(gT, gW, D, W_TGT - 0.01, W_TGT, MARG,
                                    alpha=0.3, step_max_nm=BIG)
    assert phase == "ride", phase
    worst = max(worst, abs(gW @ step) /
                (np.linalg.norm(gW) * np.linalg.norm(step)))
check("P0 gW.d == 0 (ride)", worst < 1e-12, f"worst rel {worst:.2e}")

# ── 2) scaled-space round-trip exact ──────────────────────────────────────
s = np.sqrt(D)
gT, gW = rng.standard_normal(N), rng.standard_normal(N)
step, _, _ = eng._proj_step(gT, gW, D, W_TGT, W_TGT, MARG, 0.3, BIG)
gTs, gWs = s * gT, s * gW
ref = 0.3 * s * (gTs - (gTs @ gWs) / (gWs @ gWs) * gWs)
check("scaled round-trip exact", np.allclose(step, ref, rtol=1e-12, atol=1e-15))

# ── 3) climb clip never predicts past the ceiling ─────────────────────────
worst = -np.inf
for _ in range(200):
    gT, gW = rng.standard_normal(N), rng.standard_normal(N)
    W = W_TGT - rng.uniform(0.06, 3.0)
    a = rng.uniform(0.01, 50.0)
    step, phase, _ = eng._proj_step(gT, gW, D, W, W_TGT, MARG, a, BIG)
    assert phase == "climb", phase
    worst = max(worst, W + float(gW @ step) - W_TGT)
check("clip: predicted W <= ceiling", worst <= 1e-9, f"worst over {worst:.2e}")
# unclipped when safe: tiny alpha reproduces alpha*D*gT exactly
step, _, _ = eng._proj_step(gT, gW, D, W_TGT - 2.0, W_TGT, MARG, 1e-9, BIG)
check("no clip when safe", np.allclose(step, 1e-9 * D * gT, rtol=1e-12))

# ── 4) restoration: first-order ΔW = −(W − W_tgt), MEASURED W in, out ─────
W = W_TGT + 0.15
step, phase, _ = eng._proj_step(gT, gW, D, W, W_TGT, MARG, 0.3, BIG)
check("restore phase fires", phase == "restore")
check("restore 1st-order exact", np.isclose(float(gW @ step), -0.15, rtol=1e-12))

# ── 5) step cap: inf-norm bounded, direction preserved (so gW.d=0 survives)
step_u, _, _ = eng._proj_step(gT, gW, D, W_TGT, W_TGT, MARG, 50.0, BIG)
step_c, _, _ = eng._proj_step(gT, gW, D, W_TGT, W_TGT, MARG, 50.0, 2.0)
k = np.max(np.abs(step_u)) / 2.0
check("cap inf-norm", np.max(np.abs(step_c)) <= 2.0 * (1 + 1e-12))
check("cap is scalar (parallel)", np.allclose(step_c * k, step_u, rtol=1e-12))

# ── degenerate: gW = 0 must not NaN ───────────────────────────────────────
step, _, lam = eng._proj_step(gT, np.zeros(N), D, W_TGT, W_TGT, MARG, 0.3, BIG)
check("gW=0 finite", np.all(np.isfinite(step)) and lam == 0.0)

# ── 6) legacy path untouched with wg_project=False ────────────────────────
check("CampaignSpec default OFF", eng.CampaignSpec().wg_project is False)
wl = list(np.linspace(1561.21, 1567.21, 301))
common = dict(width_grad=True, fwhm0_um=18.346,
              wg_anchor={"softw": 18.0, "fwhm": 18.4},
              wg_lam_hi=0.3, wg_lam_lo=0.1)
x = np.concatenate([rng.uniform(0.2, 0.9, 301), [18.9]])
base = eng.make_fct(wl)
f_leg = eng.make_fct_v2(wl, eng.CampaignSpec(**common))
pen = float(eng.width_band_penalty(eng.CampaignSpec(**common), x[301]))
check("legacy fct == base - penalty",
      np.isclose(float(f_leg(x)), float(base(x[:301])) - pen, rtol=1e-12))
f_prj = eng.make_fct_v2(wl, eng.CampaignSpec(wg_project=True,
                                             wgp_target_um=18.613, **common))
check("wg_project fct == pure T",
      np.isclose(float(f_prj(x)), float(base(x[:301])), rtol=1e-12))
import autograd  # noqa: E402
check("width jac exactly 0 under wg_project",
      float(autograd.jacobian(f_prj)(x)[301]) == 0.0)
# MixedFom override + optimizer dispatch are gated (source-level guard —
# instantiating MixedFom needs lumopt2; the gate stays dependency-free)
src = open(eng.__file__, encoding="utf-8").read()
check("MixedFom override gated on wg_project",
      'if not getattr(spec, "wg_project", False):' in src)
check("run_campaign dispatches on wg_project", "run_projected(" in src)

# ── 7) RGP checks REMOVED 2026-09-01 with the _rgp_step surgery (never
# adopted; superseded by ns2). History: git ≤744b4f1. Teeth check: the
# symbol must actually be GONE.
_dir = lambda v: v / np.linalg.norm(v)
check("RGP surgery complete: _rgp_step gone",
      not hasattr(eng, "_rgp_step")
      and not hasattr(eng.CampaignSpec(), "wgp_rgp"))

# ── 8) ns2: two-constraint null+range-space step (d1, 2026-08-30) ─────────
LAM_TGT, LAM_MARG = 1566.44, 0.05
# (a) both orthogonalities, machine precision, real D conditioning, mid-band
worst_w = worst_l = 0.0
for _ in range(50):
    gT8, gW8, gL8 = (rng.standard_normal(N) for _ in range(3))
    st8, ph8, dg8 = eng._ns2_step(gT8, gW8, gL8, D, W_TGT, W_TGT, MARG,
                                  LAM_TGT, LAM_TGT, LAM_MARG, 10.0)
    nrm = np.linalg.norm(st8)
    worst_w = max(worst_w, abs(gW8 @ st8) / (np.linalg.norm(gW8) * nrm))
    worst_l = max(worst_l, abs(gL8 @ st8) / (np.linalg.norm(gL8) * nrm))
check("ns2 gW.d == 0 (mid-band)", worst_w < 1e-9, f"worst rel {worst_w:.2e}")
check("ns2 gLam.d == 0 (mid-band)", worst_l < 1e-9, f"worst rel {worst_l:.2e}")
check("ns2 phase mid-band", ph8 == "ns2")
# (b) MUST-FAIL teeth: the old single-constraint ride direction does NOT
# satisfy gLam.d = 0 — the new property is non-trivial
s_old, _, _ = eng._proj_step(gT8, gW8, D, W_TGT, W_TGT, MARG, 0.3, 1e9)
check("teeth: old ride violates gLam.d=0",
      abs(gL8 @ s_old) / (np.linalg.norm(gL8) * np.linalg.norm(s_old)) > 1e-4)
# (c) coefficient independence: (gW + c·gLam)·d == gW·d for ANY c — the
# fitted wg_dwdlam is out of the feasible-direction math
for c in (0.1, 0.3655, 0.7, -0.5):
    check(f"ns2 chain-coef {c:+.3g} drops out",
          abs(float((gW8 + c * gL8) @ st8) - float(gW8 @ st8)) < 1e-9)
# (d) restoration first-order exact + deadbanded (gT=0 isolates xi_C)
sC, phC, dgC = eng._ns2_step(np.zeros(N), gW8, gL8, D, W_TGT + 0.15, W_TGT,
                             MARG, LAM_TGT + 0.20, LAM_TGT, LAM_MARG, 1e9)
check("ns2 restore phase", phC == "ns2+restore")
check("ns2 restore gW exact",
      np.isclose(float(gW8 @ sC), -(0.15 - MARG / 2.0), rtol=1e-9))
check("ns2 restore gLam exact",
      np.isclose(float(gL8 @ sC), -(0.20 - LAM_MARG), rtol=1e-9))
# inside both deadbands ⇒ zero restoration content
sZ, phZ, _ = eng._ns2_step(gT8, gW8, gL8, D, W_TGT + 0.03, W_TGT, MARG,
                           LAM_TGT + 0.02, LAM_TGT, LAM_MARG, 10.0)
check("ns2 deadband: no restore inside bands", phZ == "ns2"
      and abs(gW8 @ sZ) / (np.linalg.norm(gW8) * np.linalg.norm(sZ)) < 1e-9)
# (e) collinear degeneracy: must degrade to single-constraint, not NaN
sD, phD, dgD = eng._ns2_step(gT8, gW8, gW8 * 1.0001, D, W_TGT, W_TGT, MARG,
                             LAM_TGT, LAM_TGT, LAM_MARG, 10.0)
check("ns2 collinear degrades finite", np.all(np.isfinite(sD))
      and dgD["ns2_degraded"] and phD == "ns2_degraded")
check("ns2 degraded keeps gW.d == 0",
      abs(gW8 @ sD) / (np.linalg.norm(gW8) * np.linalg.norm(sD)) < 1e-9)
# (f) gLam=None ⇒ single-constraint; direction == _proj_step ride direction
sN, _, _ = eng._ns2_step(gT8, gW8, None, D, W_TGT, W_TGT, MARG,
                         None, None, LAM_MARG, 10.0)
sR, _, _ = eng._proj_step(gT8, gW8, D, W_TGT, W_TGT, MARG, 0.3, 1e9)
check("ns2 gLam=None == ride direction",
      np.allclose(_dir(sN), _dir(sR), atol=1e-9))
# (g) cap: inf-norm respected and mid-band direction cap-invariant
s10, _, _ = eng._ns2_step(gT8, gW8, gL8, D, W_TGT, W_TGT, MARG,
                          LAM_TGT, LAM_TGT, LAM_MARG, 10.0)
s02, _, _ = eng._ns2_step(gT8, gW8, gL8, D, W_TGT, W_TGT, MARG,
                          LAM_TGT, LAM_TGT, LAM_MARG, 2.0)
check("ns2 cap inf-norm", float(np.max(np.abs(s02))) <= 2.0 * (1 + 1e-12)
      and np.isclose(float(np.max(np.abs(s10))), 10.0, rtol=1e-9))
check("ns2 cap parallel (mid-band)", np.allclose(_dir(s10), _dir(s02),
                                                 atol=1e-9))
# (h) rho_T sane: in (0,1]; and == 1 when constraints are orthogonal to gT
check("ns2 rho_T in (0,1]", 0.0 < dg8["rho_T"] <= 1.0 + 1e-12)
# build a gT already D-orthogonal to both rows: project in the D metric
A8 = np.stack([gW8, gL8], axis=1)
M8 = A8.T @ (D[:, None] * A8)
gT_dperp = gT8 - A8 @ np.linalg.solve(M8, A8.T @ (D * gT8))
_, _, dgP = eng._ns2_step(gT_dperp, gW8, gL8, D, W_TGT, W_TGT, MARG,
                          LAM_TGT, LAM_TGT, LAM_MARG, 10.0)
check("ns2 rho_T == 1 for feasible gT", np.isclose(dgP["rho_T"], 1.0,
                                                   rtol=1e-9))
# (i) gW=0 must not raise/NaN
sF, phF, _ = eng._ns2_step(gT8, np.zeros(N), gL8, D, W_TGT, W_TGT, MARG,
                           LAM_TGT, LAM_TGT, LAM_MARG, 10.0)
check("ns2 gW=0 finite", np.all(np.isfinite(sF)))
# (j) optstate sidecar round-trip + defaults off
import json as _json, tempfile as _tf  # noqa: E402
with _tf.TemporaryDirectory() as td:
    state = {"cap_nm": 22.5, "wgain": 1.1, "dTp0": 2.2, "dwdlam": 0.31,
             "lam_tgt_nm": 1566.401, "n_acc": 7, "n_rej": 1}
    eng._save_opt_state(td, "gate", state)
    back = eng._load_opt_state(td, "gate")
    check("optstate round-trip", back == state)
    check("optstate missing -> {}", eng._load_opt_state(td, "nope") == {})
check("wgp_ns2 default OFF", eng.CampaignSpec().wgp_ns2 is False)
check("wgp_cap_adapt default OFF", eng.CampaignSpec().wgp_cap_adapt is False)

# -- 9) noise-slack RATCHET + reuse TRAVEL budget (2026-09-01 fixes) -------
src9 = open(eng.__file__, encoding="utf-8").read()
check("slack anchored to fom_best (not the moving acc)",
      'fom_ref = max(acc["fom"], fom_best)' in src9
      and "fom > fom_ref" in src9
      and "fom_best = max(fom_best, fom)" in src9)
check("reuse gate carries the travel budget",
      "reuse_travel + _cap(alpha)" in src9
      and "wgp_reuse_travel_nm" in src9
      and "reuse_travel = 0.0" in src9)
check("ns2 WidthTrip shrinks the cap, not corr_max",
      '_ost["cap_nm"] = max(_cap_now * 0.5, 2.0)' in src9)
check("new spec knobs default-safe",
      eng.CampaignSpec().wgp_reuse_travel_nm == 40.0
      and eng.CampaignSpec().wgp_reuse_k == 0)

# BEHAVIORAL: replay the filter predicate over a downhill drift. slack 1.5e-3,
# three steps each losing 1.0e-3. Anchored to fom_best the THIRD must reject;
# the old acc-anchored form accepts all three -- that is the ratchet, and it
# is the must-fail half: if both forms agreed the gate would have no teeth.
SLACK = 1.5e-3
def _replay(anchor_best):
    acc_fom, best_seen, verdicts = 0.7200, 0.7200, []
    for trial in (0.7190, 0.7180, 0.7170):
        ref = max(acc_fom, best_seen) if anchor_best else acc_fom
        ok_ = trial > ref - SLACK
        verdicts.append(ok_)
        if ok_:
            acc_fom = trial
            best_seen = max(best_seen, trial)
    return verdicts
v_fixed, v_old = _replay(True), _replay(False)
# anchored to best, drift is bounded to ONE slack below the best ever seen:
# the 1st dip (-1.0e-3) is inside the 1.5e-3 band, the 2nd (-2.0e-3 cumulative)
# is not -- so the sequence is [True, False, False], stricter than a
# per-step slack. That bound is the whole point of the fix.
check("ratchet FIX bounds drift to one slack below best",
      v_fixed == [True, False, False], f"{v_fixed}")
check("teeth: old acc-anchored form accepts all three (the ratchet)",
      v_old == [True, True, True], f"{v_old}")

# -- 10) optimizer upgrades U1-U4 (2026-10-04) ------------------------------
import os as _os, pickle as _pk  # noqa: E402
sp = eng.CampaignSpec()
check("U1-U4 spec knobs default OFF",
      sp.wgp_noise_freeze is False and sp.wgp_noise_stop == 3
      and sp.wgp_reuse_broyden is False and sp.wgp_mode_mac is None
      and sp.wgp_range_alpha == 1.0 and sp.wgp_range_cap_frac is None)

# U4 (Feppon et al. 2020 separate range cap). (a) defaults bit-identical to
# the PRE-edit _ns2_step: snapshots/ns2_step_ref.pkl was written by running
# the OLD function on 6 fixed regimes (mid-band, clamped/free/binding ξ_C
# restore, collinear degrade, gLam=None) before the edit.
_ref = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "snapshots", "ns2_step_ref.pkl")
with open(_ref, "rb") as f:
    ref_cases = _pk.load(f)
ok = len(ref_cases) == 6
for name, args, st_ref, ph_ref, dg_ref in ref_cases:
    for kw in ({}, {"range_alpha": 1.0, "range_cap_nm": None}):
        s_, p_, d_ = eng._ns2_step(*args, **kw)
        ok = ok and np.array_equal(s_, st_ref) and p_ == ph_ref and d_ == dg_ref
check("U4 defaults bit-identical to pre-edit _ns2_step (6 cases x 2)", ok)
# (b) range_alpha=0.5: gT=0 isolates ξ_C (ξ_J = 0) ⇒ the step IS ξ_C
a10 = (gW8, gL8, D, W_TGT + 0.15, W_TGT, MARG, LAM_TGT + 0.20, LAM_TGT,
       LAM_MARG)
xc1, _, _ = eng._ns2_step(np.zeros(N), *a10, 1e9)
xc5, _, _ = eng._ns2_step(np.zeros(N), *a10, 1e9, range_alpha=0.5)
check("U4 range_alpha=0.5 halves xi_C", np.allclose(xc5, 0.5 * xc1,
                                                    rtol=1e-12, atol=0.0))
check("U4 range_alpha=0.5 halves the restoration",
      np.isclose(float(gW8 @ xc5), -0.5 * (0.15 - MARG / 2.0), rtol=1e-9))
check("teeth: range_alpha=0.5 changes the step", not np.allclose(xc5, xc1))
# ξ_J part (full step minus its ξ_C) keeps BOTH orthogonalities
sh5, _, _ = eng._ns2_step(gT8, *a10, 10.0, range_alpha=0.5)
xj = sh5 - xc5
rw = abs(gW8 @ xj) / (np.linalg.norm(gW8) * np.linalg.norm(xj))
rl = abs(gL8 @ xj) / (np.linalg.norm(gL8) * np.linalg.norm(xj))
check("U4 alpha=0.5: gW.xi_J = gLam.xi_J = 0", max(rw, rl) < 1e-12,
      f"worst rel {max(rw, rl):.2e}")
# (c) range_cap_nm=1.0 bounds ‖ξ_C‖∞: rows scaled ×1e-3 ⇒ unclamped ξ_C ≫ 1
a10s = (gW8 * 1e-3, gL8 * 1e-3) + a10[2:]
xbig, _, _ = eng._ns2_step(np.zeros(N), *a10s, 1e9)
xcap, _, _ = eng._ns2_step(np.zeros(N), *a10s, 1e9, range_cap_nm=1.0)
check("teeth: unclamped xi_C exceeds 1 nm (cap test is live)",
      float(np.max(np.abs(xbig))) > 1.0, f"{np.max(np.abs(xbig)):.3g}")
check("U4 range_cap_nm=1.0 bounds |xi_C|_inf <= 1",
      float(np.max(np.abs(xcap))) <= 1.0 * (1 + 1e-12))
check("U4 range cap is scalar (parallel)", np.allclose(_dir(xcap), _dir(xbig),
                                                       atol=1e-12))
src10 = open(eng.__file__, encoding="utf-8").read()
check("U4 wired: _step_of threads range_alpha / range_cap_nm",
      "range_alpha=range_alpha" in src10
      and "_cap(a) * range_frac" in src10)

# U1 (noise-aware cap, Cao/Berahas/Scheinberg 2205.03667): the reject rule
SL = 1.5e-3
check("U1 noise reject keeps the cap",
      eng._reject_cap(20.0, 1.0e-3, SL, True) == (20.0, True))
check("U1 negative predicted gain counts by magnitude",
      eng._reject_cap(20.0, -1.0e-3, SL, True) == (20.0, True))
check("U1 non-noise reject halves",
      eng._reject_cap(20.0, 5.0e-3, SL, True) == (10.0, False))
check("U1 non-noise floor 2 nm", eng._reject_cap(3.0, 5.0e-3, SL, True)
      == (2.0, False))
check("teeth: freeze=False halves the noise case too",
      eng._reject_cap(20.0, 1.0e-3, SL, False) == (10.0, False)
      and eng._reject_cap(20.0, 5.0e-3, SL, False) == (10.0, False))
check("U1 wired: reject branch uses _reject_cap + stop rule + persistence",
      "_reject_cap(cap_state, dT_pred_trial, slack" in src10
      and "n_noise_rej >= noise_stop" in src10
      and '"n_noise_rej": n_noise_rej' in src10
      and 'int(ost.get("n_noise_rej", 0))' in src10
      and 'rec["dT_pred_trial"] = dT_pred_trial' in src10)

# U2 (Broyden secant on the reused row, Walther & Biegler)
g_true = rng.standard_normal(N)
g_old = g_true + 0.3 * rng.standard_normal(N)
dp10 = rng.standard_normal(N)
dW10 = float(g_true @ dp10)
g_new, info = eng._broyden_update(g_old, dp10, dW10, 0.01)
check("U2 secant: |g.dp - dW| ~ 0 after one update",
      abs(float(g_new @ dp10) - dW10) < 1e-12 * np.linalg.norm(dp10) ** 2,
      f"{abs(float(g_new @ dp10) - dW10):.2e}")
check("U2 logged residual is the pre-update one",
      np.isclose(info["broyden_dW_resid"], dW10 - float(g_old @ dp10),
                 rtol=1e-12))
v_perp = rng.standard_normal(N)
v_perp -= (v_perp @ dp10) / (dp10 @ dp10) * dp10
check("U2 unchanged orthogonal to dp",
      np.isclose(float(g_new @ v_perp), float(g_old @ v_perp), rtol=1e-10))
check("U2 error to g_true shrinks",
      np.linalg.norm(g_new - g_true) < np.linalg.norm(g_old - g_true))
g_sk, info_sk = eng._broyden_update(g_old, dp10, dW10, 0.06)
check("U2 skip when |dlam| > 0.05", np.array_equal(g_sk, g_old)
      and info_sk == {"broyden_skipped": "dlam"})
check("U2 skip when dlam unknown",
      eng._broyden_update(g_old, dp10, dW10, None)[1]
      == {"broyden_skipped": "dlam"})
check("teeth: no-update path leaves the residual unchanged (and large)",
      abs(float(g_sk @ dp10) - dW10) == abs(float(g_old @ dp10) - dW10)
      and abs(float(g_old @ dp10) - dW10) > 1e-3)
check("U2 wired: reuse branch applies _broyden_update on softW",
      "_broyden_update(" in src10 and 'float(sw) - acc["softw"]' in src10)

# U3 (MAC mode tracking, Kim & Kim 2000)
with _tf.TemporaryDirectory() as td:
    xg = np.linspace(-40.0, 40.0, 801)
    SIG = 2.0
    def _npz(name, x, I):
        pth = _os.path.join(td, name)
        np.savez_compressed(pth, x_um=x, I=I, lam_pk_nm=1566.4)
        return pth
    g0 = _npz("g0.npz", xg, np.exp(-xg ** 2 / (2 * SIG ** 2)))
    xg2 = np.linspace(-35.0, 35.0, 1001)        # different grid: interp path
    g0b = _npz("g0b.npz", xg2, np.exp(-xg2 ** 2 / (2 * SIG ** 2)))
    gs = _npz("gs.npz", xg, np.exp(-(xg - 3 * SIG) ** 2 / (2 * SIG ** 2)))
    two = _npz("two.npz", xg, np.exp(-(xg - 2 * SIG) ** 2 / (2 * SIG ** 2))
               + np.exp(-(xg + 2 * SIG) ** 2 / (2 * SIG ** 2)))
    m_self = eng.profile_mac(g0, g0)
    m_grid = eng.profile_mac(g0, g0b)
    m_shift = eng.profile_mac(g0, gs)
    m_two = eng.profile_mac(g0, two)
    check("U3 MAC self == 1", abs(m_self - 1.0) < 1e-12, f"{m_self!r}")
    check("U3 MAC self across grids ~ 1", abs(m_grid - 1.0) < 1e-4,
          f"{m_grid:.6f}")
    check("U3 MAC 3-width shift < 0.1", m_shift < 0.1, f"{m_shift:.4f}")
    check("U3 MAC two-lobe < 0.9", m_two < 0.9, f"{m_two:.4f}")
    check("U3 missing file -> None",
          eng.profile_mac(g0, _os.path.join(td, "nope.npz")) is None)
check("U3 wired: mode hop rejects exactly like the λ jump",
      "lam_jump or mode_hop or" in src10
      and 'acc[\'eval_num\']' in src10 and '"eval_num": int(it)' in src10)

# -- 11) DRIVER-LEVEL: the REAL run_projected through a FAKE project ---------
# (2026-10-05, GPT review A1/A4/A5). No Lumerical: the project returns
# scripted foms + fixed synthetic gradients; the fake callback writes the same
# <label>_evals.jsonl rows and profiles/<label>_ev####.npz the real CampaignLog
# writes. Spec = the REAL TE S1 campaign spec (296 params; ns2, cap_adapt,
# λ-chain, reuse_k 5, broyden, MAC 0.9, filter_band, noise_freeze all live).
# Mocked: project.compute_fom/compute_gradient, the gradient-from-fields
# conversion (returns gW / gTlo / gThi), the log callback, spec._wg_dTp = −1.
# Every eval sits at W = W_tgt, λ = λ_tgt unless stated (no restoration).
import contextlib as _cl, dataclasses as _dc, inspect as _insp  # noqa: E402
import io as _io, types as _ty  # noqa: E402
sys.path.insert(0, r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes")
from runners.lumopt2_design.campaign_te_s1 import SPEC as TE_S1  # noqa: E402

rng11 = np.random.default_rng(11)
S11 = _dc.replace(TE_S1, label="gate11", max_iter=10)
b11 = np.asarray(eng.param_bounds(S11), dtype=float)
D11 = ((b11[:, 1] - b11[:, 0]) / 2.0) ** 2
P0 = np.asarray(eng.seed_params(S11), dtype=float)
W0, LAM0 = float(S11.wgp_target_um), float(S11.scan_center_nm)
F0, SL11, CAP0 = 0.70, float(S11.wgp_fom_slack), float(S11.wgp_step_max_nm)
gW11, gLo11, gHi11 = (rng11.standard_normal(len(P0)) for _ in range(3))
gT_dir = rng11.standard_normal(len(P0))
# scale gT so the full-cap ns2 step predicts a gain of 0.4×slack (< 2×slack):
# the step direction is scale-free, gT·Δp is linear in |gT|
s_u, _, _ = eng._ns2_step(gT_dir, gW11, gHi11 - gLo11, D11, W0, W0,
                          S11.wgp_margin_um, LAM0, LAM0,
                          S11.wgp_lam_margin_nm, CAP0)
gT11 = gT_dir * (0.4 * SL11 / float(gT_dir @ s_u))
XG = np.linspace(-40.0, 40.0, 801)
GAUSS = np.exp(-XG ** 2 / 32.0)                                   # sigma 4 um
TWO = np.exp(-(XG - 8) ** 2 / 32.0) + np.exp(-(XG + 8) ** 2 / 32.0)  # mode hop


class _FakeProject:
    def __init__(self, script, gT=None):
        self.script, self.n = script, 0
        self.gT = gT11 if gT is None else gT
        self.fom = _ty.SimpleNamespace(gfields_W="W", gfields_Tlo="Tlo",
                                       gfields_Thi="Thi")
        self.fdtd_session = None
        g = {"W": gW11, "Tlo": gLo11, "Thi": gHi11}
        self.parametrization = _ty.SimpleNamespace(
            compute_gradient_from_fields=lambda f, sess, p: g[f].copy())

    def compute_fom(self, p):
        self.n += 1
        return self.script[self.n - 1]["fom"]

    def compute_gradient(self, p):
        return self.gT.copy()


class _FakeLog:
    def __init__(self, spec, out_dir, script):
        self.spec, self.out_dir, self.script = spec, out_dir, script

    def on_function_eval(self, project, it, p, fom):
        e, lab = self.script[it], self.spec.label
        # real callback: λ-stencil curvature; None = ineligible spectrum (T5)
        self.spec._wg_dTp = None if e.get("stale") else -1.0
        self.spec._wg_lam_idx = None if e.get("stale") else (0, 1)
        # softw_adj_um (the twin's sample) differs from softw_um by an
        # eval-DEPENDENT offset so T7 can tell which one the secant used
        with open(_os.path.join(self.out_dir, f"{lab}_evals.jsonl"), "a") as f:
            f.write(_json.dumps({"eval": int(it), "fom": float(fom),
                                 "params": [float(v) for v in p],
                                 "fwhm_env_um": e["W"], "lam_pk_nm": LAM0,
                                 "softw_um": e["W"] - 0.4,
                                 "softw_adj_um": e["W"] - 0.3 + 0.01 * it})
                    + "\n")
        pd = _os.path.join(self.out_dir, "profiles")
        _os.makedirs(pd, exist_ok=True)
        np.savez_compressed(_os.path.join(pd, f"{lab}_ev{int(it):04d}.npz"),
                            x_um=XG, I=e.get("prof", GAUSS), lam_pk_nm=LAM0)


def _drive(td, script, fn=None, gT=None, **kw):
    """Run the real (or a source-patched) run_projected in td; artefacts."""
    spec = _dc.replace(S11, **kw)
    buf = _io.StringIO()
    with _cl.redirect_stdout(buf):
        (fn or eng.run_projected)(spec, _FakeProject(script, gT),
                                  _FakeLog(spec, td, script), td, P0)
    rd = lambda n: [_json.loads(l) for l in open(_os.path.join(td, n))]
    return {"spec": spec, "out": buf.getvalue(),
            "ev": rd(f"{spec.label}_evals.jsonl"),
            "pr": rd(f"{spec.label}_proj.jsonl"),
            "ost": eng._load_opt_state(td, spec.label)}


def _trial_steps(r):
    acc_p = np.asarray(r["ev"][0]["params"])
    tr = [np.asarray(e["params"]) for e in r["ev"][1:4]]
    steps = [float(np.max(np.abs(t - acc_p))) for t in tr]
    gaps = [float(np.max(np.abs(tr[i] - tr[j])))
            for i, j in ((0, 1), (0, 2), (1, 2))]
    return steps, min(gaps)


# T1 (A1): flat-within-noise ⇒ 3 DIFFERENT trials at cap, cap/2, cap/4,
# persisted cap untouched, stop after exactly 3 via CONVERGED WITHIN NOISE
flat = [dict(fom=F0, W=W0)] + [dict(fom=F0 - 1.2 * SL11, W=W0)] * 9
with _tf.TemporaryDirectory() as td:
    r1 = _drive(td, flat)
st1, gap1 = _trial_steps(r1)
check("T1 precondition: predicted |gT.dp| < 2*slack on every trial",
      all(abs(r1["pr"][i]["dT_pred_trial"]) < 2 * SL11 for i in (1, 2, 3)),
      str([round(r1["pr"][i]["dT_pred_trial"], 6) for i in (1, 2, 3)]))
check("T1 three noise rejects, then stop (4 evals, marker fired)",
      len(r1["ev"]) == 4 and "CONVERGED WITHIN NOISE" in r1["out"]
      and all(r1["pr"][i].get("noise_reject") for i in (1, 2, 3)),
      f"evals {len(r1['ev'])}")
check("T1 trial steps = cap, cap/2, cap/4 (inf-norm, 1e-9)",
      np.allclose(st1, [CAP0, CAP0 / 2, CAP0 / 4], rtol=0, atol=1e-9),
      str([round(s, 12) for s in st1]))
check("T1 three trials are DIFFERENT vectors", gap1 > 1e-6, f"min gap {gap1:.3g}")
check("T1 persisted cap_nm unchanged", r1["ost"]["cap_nm"] == CAP0
      and r1["ost"]["n_noise_rej"] == 3, f"cap {r1['ost']['cap_nm']}")
# tooth: the PRE-FIX driver (retry_shrink never halves) — same source, one
# token changed, exec'd in a copy of the module namespace
src_rp = _insp.getsource(eng.run_projected)


def _patched(*pairs):
    """run_projected with source tokens replaced (the pre-fix behaviour)."""
    src = src_rp
    for old, new in pairs:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    ns = dict(vars(eng))
    exec(compile(src, eng.__file__, "exec"), ns)
    return ns["run_projected"]


# the duplicate-retry guard (T9) would mask the pre-fix rules the T1/T8
# teeth isolate, so those teeth switch it off too
NO_GUARD = ("while any(float(np.max(np.abs(p - r))) < 1e-9 "
            "for r in rej_trials):", "while False:")
pre_fix1 = _patched(("retry_shrink *= 0.5", "retry_shrink *= 1.0"), NO_GUARD)
with _tf.TemporaryDirectory() as td:
    r1t = _drive(td, flat, fn=pre_fix1)
st1t, gap1t = _trial_steps(r1t)
check("teeth: pre-fix driver proposes 3 IDENTICAL trials and 'converges'",
      src_rp.count("retry_shrink *= 0.5") == 1 and gap1t == 0.0
      and not gap1t > 1e-6 and "CONVERGED WITHIN NOISE" in r1t["out"],
      f"steps {[round(s, 9) for s in st1t]}")

# T2 (A1 2nd half): small prediction, LARGE observed loss ⇒ NOT noise, halve
for cap_in, cap_out in ((CAP0, CAP0 / 2), (3.0, 2.0)):          # floor 2 nm
    with _tf.TemporaryDirectory() as td:
        r2 = _drive(td, [dict(fom=F0, W=W0), dict(fom=F0 - 20 * SL11, W=W0)],
                    max_iter=2, wgp_step_max_nm=cap_in)
    row2 = r2["pr"][1]
    check(f"T2 cap {cap_in:g}: big loss is not noise, cap -> {cap_out:g}",
          row2["phase"].endswith("-retry") and "noise_reject" not in row2
          and abs(row2["dT_pred_trial"]) < 2 * SL11
          and r2["ost"]["cap_nm"] == cap_out and r2["ost"]["n_noise_rej"] == 0,
          f"cap {r2['ost']['cap_nm']} pred {row2['dT_pred_trial']:.2e}")
check("teeth: without the loss arm the same reject WOULD be noise",
      eng._reject_cap(CAP0, row2["dT_pred_trial"], SL11, True) == (CAP0, True))

# T3 (A4): mode-hop trial with the HIGHEST fom must not be the best row
hop = [dict(fom=F0, W=W0), dict(fom=F0 + 0.01, W=W0, prof=TWO),
       dict(fom=F0 + 4 * SL11, W=W0)]
with _tf.TemporaryDirectory() as td:
    r3 = _drive(td, hop[:2], max_iter=2)
    acc3 = [e for e, pr in zip(r3["ev"], r3["pr"])
            if not pr["phase"].endswith("-retry")]
    best_acc = max(acc3, key=lambda e: e["fom"])
    fb = {"fom": -np.inf, "params": P0}
    bp3, _ = eng._best_from_log(r3["spec"], td, fb)
    hop_p = np.asarray(r3["ev"][1]["params"])
    check("T3 precondition: hop trial rejected on MAC, highest fom in log",
          r3["pr"][1]["mac"] < 0.9 and r3["pr"][1]["phase"].endswith("-retry")
          and r3["ev"][1]["fom"] == max(e["fom"] for e in r3["ev"]),
          f"mac {r3['pr'][1]['mac']:.4f}")
    check("T3 hop key in ineligible list",
          r3["ost"]["ineligible"] == [eng._param_key(hop_p)])
    check("T3 _best_from_log returns the best ACCEPTED row",
          np.allclose(bp3, best_acc["params"]) and not np.allclose(bp3, hop_p),
          f"-> eval {best_acc['eval']} fom {best_acc['fom']}")
    o3 = dict(r3["ost"], ineligible=[])           # tooth: list cleared
    eng._save_opt_state(td, r3["spec"].label, o3)
    bp3t, _ = eng._best_from_log(r3["spec"], td, fb)
    check("teeth: cleared ineligible list -> the mode-hop row is returned",
          np.allclose(bp3t, hop_p))
# T3b (found by this gate 2026-10-05, fixed same day): a λ-jump / mode-hop
# REJECT must not raise fom_best — else the hop's fom becomes the acceptance
# bar, a genuine F0+4·slack improvement is rejected, and _best_from_log (the
# restart point) picks that driver-rejected row
def _ev_idx(r, p):
    return [e["eval"] for e in r["ev"] if np.allclose(p, e["params"])]
with _tf.TemporaryDirectory() as td:
    r3b = _drive(td, hop, max_iter=3)
    bp3b, _ = eng._best_from_log(r3b["spec"], td, fb)
check("T3b post-hop improvement (MAC 1) ACCEPTED, anchor not raised",
      not r3b["pr"][2]["phase"].endswith("-retry")
      and r3b["pr"][1]["fom_best"] == F0 and r3b["pr"][2]["mac"] > 0.999,
      f"{r3b['pr'][2]['phase']} fom_best {r3b['pr'][1]['fom_best']:.4f}")
check("T3b _best_from_log returns it", _ev_idx(r3b, bp3b) == [2],
      f"-> eval {_ev_idx(r3b, bp3b)}")
FB_FIX = "if not (lam_jump or mode_hop):\n            fom_best = max(fom_best"
ns3b = dict(vars(eng))
exec(compile(src_rp.replace(FB_FIX, "if True:\n            fom_best = max("
                            "fom_best"), eng.__file__, "exec"), ns3b)
with _tf.TemporaryDirectory() as td:
    r3bt = _drive(td, hop, max_iter=3, fn=ns3b["run_projected"])
check("teeth: old unconditional fom_best update REJECTS it",
      src_rp.count(FB_FIX) == 1 and r3bt["pr"][2]["phase"].endswith("-retry")
      and r3bt["pr"][1]["fom_best"] == F0 + 0.01, r3bt["pr"][2]["phase"])

# T5: stale stencil (callback cleared _wg_dTp/_wg_lam_idx to None on an
# ineligible spectrum) — no float(None); chain skipped, ns2 fallback, continue
stale = [dict(fom=F0, W=W0), dict(fom=F0, W=W0, stale=True), dict(fom=F0, W=W0)]
with _tf.TemporaryDirectory() as td:
    r5 = _drive(td, stale, max_iter=3)
check("T5 stale stencil: chain skipped + ns2 fallback logged, loop continues",
      len(r5["pr"]) == 3 and r5["pr"][1].get("lam_chain") == "skipped"
      and r5["pr"][1].get("ns2_fallback") is True
      and "ns2_fallback" not in r5["pr"][2],
      f"{[(q['phase'], q.get('lam_chain')) for q in r5['pr']]}")
DTP_FIX = 'float(getattr(spec, "_wg_dTp", 0.0) or 0.0)'
ns5 = dict(vars(eng))
exec(compile(src_rp.replace(DTP_FIX, 'float(getattr(spec, "_wg_dTp", 0.0))'),
             eng.__file__, "exec"), ns5)
try:
    with _tf.TemporaryDirectory() as td:
        _drive(td, stale, max_iter=3, fn=ns5["run_projected"])
    raised5 = False
except TypeError:
    raised5 = True
check("teeth: pre-fix float(_wg_dTp) raises TypeError on the stale eval",
      src_rp.count(DTP_FIX) == 1 and raised5)

# T6: _row_of_params is ABSOLUTE-tolerance only (rtol=0): at |p| ~ 1e4 the
# default np.allclose rtol 1e-5 would match a 0.05 nm different vector
with _tf.TemporaryDirectory() as td:
    pa = np.full(4, 10000.0)
    with open(_os.path.join(td, "gate6_evals.jsonl"), "w") as f:
        f.write(_json.dumps({"params": pa.tolist(), "fwhm_env_um": 19.0}) + "\n")
    s6 = _dc.replace(S11, label="gate6")
    check("T6 exact params match (control)",
          eng._row_of_params(s6, td, pa, tol=1e-6) is not None)
    check("T6 0.05 nm at magnitude 1e4 does NOT match (tol 1e-6)",
          eng._row_of_params(s6, td, pa + 0.05, tol=1e-6) is None)
check("teeth: np.allclose default rtol WOULD match them",
      np.allclose(pa, pa + 0.05, atol=1e-6))

# T7: Broyden secant on a REUSED width row reads softw_adj_um (the twin's
# sample), not softw_um. Two accepts at W = W_tgt ⇒ eval 1 reuses the row.
with _tf.TemporaryDirectory() as td:
    r7 = _drive(td, [dict(fom=F0, W=W0)] * 2, max_iter=2)
q7, e7 = r7["pr"][1], r7["ev"]
dp7 = np.asarray(e7[1]["params"]) - np.asarray(e7[0]["params"])
want_adj = (e7[1]["softw_adj_um"] - e7[0]["softw_adj_um"]) - float(gW11 @ dp7)
want_raw = (e7[1]["softw_um"] - e7[0]["softw_um"]) - float(gW11 @ dp7)
check("T7 eval 1 reused the width row and ran the secant",
      q7.get("gw_reused") == 1 and "broyden_dW_resid" in q7, str(
          {k: q7.get(k) for k in ("gw_reused", "broyden_skipped")}))
check("T7 secant residual == the softw_adj_um values",
      np.isclose(q7.get("broyden_dW_resid", np.nan), want_adj, rtol=0,
                 atol=1e-12), f"{q7.get('broyden_dW_resid')} vs {want_adj}")
check("teeth: the softw_um residual differs (T7 discriminates)",
      abs(want_adj - want_raw) > 1e-3)

# T4 (A5): both points in the deadband, trial closer to W_tgt but 5×slack
# worse ⇒ REJECT under filter_band; legacy (False) accepts it — the tooth
band = [dict(fom=F0, W=W0 + 0.04), dict(fom=F0 - 5 * SL11, W=W0 + 0.01)]
check("T4 precondition: both points inside marg/2",
      max(abs(e["W"] - W0) for e in band) < S11.wgp_margin_um / 2)
with _tf.TemporaryDirectory() as td:
    r4 = _drive(td, band, max_iter=2)
check("T4 filter_band=True rejects the in-band FOM loss",
      r4["spec"].wgp_filter_band and r4["pr"][1]["phase"].endswith("-retry"),
      r4["pr"][1]["phase"])
with _tf.TemporaryDirectory() as td:
    r4t = _drive(td, band, max_iter=2, wgp_filter_band=False)
check("teeth: filter_band=False ACCEPTS it (legacy distance-to-centre arm)",
      not r4t["pr"][1]["phase"].endswith("-retry"), r4t["pr"][1]["phase"])


def _deliv(r):
    """inf-norm displacement of every trial from the accepted eval 0."""
    a = np.asarray(r["ev"][0]["params"])
    return [float(np.max(np.abs(np.asarray(e["params"]) - a)))
            for e in r["ev"][1:]]


# T8 (F1a): noise, noise, ORDINARY reject from cap 10 — one effective radius,
# never enlarged by a reject: trials 10, 5, 2.5, then max(1.25, 2.0) = 2.0
seq8 = [dict(fom=F0, W=W0), dict(fom=F0 - 1.2 * SL11, W=W0),
        dict(fom=F0 - 1.2 * SL11, W=W0), dict(fom=F0 - 20 * SL11, W=W0),
        dict(fom=F0 - 1.2 * SL11, W=W0)]
with _tf.TemporaryDirectory() as td:
    r8 = _drive(td, seq8, max_iter=5)
d8 = _deliv(r8)
check("T8 trials 10, 5, 2.5, then 2.0 (<= 2.5) after the ordinary reject",
      np.allclose(d8, [10.0, 5.0, 2.5, 2.0], rtol=0, atol=1e-9)
      and "noise_reject" not in r8["pr"][3], str([round(v, 9) for v in d8]))
OLD8 = "cap_state * retry_shrink * 0.5, 2.0)"
with _tf.TemporaryDirectory() as td:
    r8t = _drive(td, seq8, max_iter=5,
                 fn=_patched((OLD8, "cap_state * 0.5, 2.0)"), NO_GUARD))
d8t = _deliv(r8t)
check("teeth: pre-fix (halve base, reset shrink) ENLARGES trial 4 to 5",
      abs(d8t[3] - 5.0) < 1e-9, str([round(v, 9) for v in d8t]))

# T9 (F1b): restoration-dominated retries — gT = 0, W just outside the
# deadband, so the delivered step is pure ξ_C, far BELOW every cap; shrinking
# the cap cannot change it. Every rejected trial must be a NEW geometry.
seq9 = [dict(fom=F0, W=W0 + 0.06)] + [dict(fom=F0 - 1.2 * SL11,
                                           W=W0 + 0.06)] * 7
z9 = np.zeros_like(gT11)
with _tf.TemporaryDirectory() as td:
    r9 = _drive(td, seq9, max_iter=8, gT=z9)
d9, ev9 = _deliv(r9), [np.asarray(e["params"]) for e in r9["ev"]]
gap9 = min(float(np.max(np.abs(ev9[i] - ev9[j])))     # incl. the accepted eval
           for i in range(len(ev9)) for j in range(i))
msg9 = (f"deliv {[f'{v:.4g}' for v in d9]} dup "
        f"{[int(bool(q.get('dup_retry'))) for q in r9['pr'][1:]]} "
        f"conv {'CONVERGED WITHIN NOISE' in r9['out']}")
check("T9 precondition: delivered step << every cap (restoration-only)",
      d9[0] < 1.0, msg9)
check("T9 trials s, s/2, s/4: no geometry evaluated twice",
      len(d9) == 3 and gap9 > 1e-9
      and np.allclose(d9, [d9[0], d9[0] / 2, d9[0] / 4], rtol=1e-6, atol=0),
      msg9 + f" min gap {gap9:.3g}")
check("T9 halved retries carry dup_retry; CONVERGED only after 3 distinct",
      r9["pr"][1].get("dup_retry") and r9["pr"][2].get("dup_retry")
      and len(r9["ev"]) == 4 and "CONVERGED WITHIN NOISE" in r9["out"]
      and not any(q.get("stalled") for q in r9["pr"]), msg9)
with _tf.TemporaryDirectory() as td:
    r9t = _drive(td, seq9, max_iter=8, gT=z9,
                 fn=_patched(NO_GUARD))
tr9t = [np.asarray(e["params"]) for e in r9t["ev"][1:]]
check("teeth: no duplicate guard -> 3 IDENTICAL trials, 'converged'",
      len(tr9t) == 3 and all(np.array_equal(t, tr9t[0]) for t in tr9t)
      and "CONVERGED WITHIN NOISE" in r9t["out"], f"evals {len(r9t['ev'])}")

# T11: collapse path. Same restoration-only setup with the violation shrunk
# so s ~ 1.5e-9 nm (ξ_C is linear in it): the 1st retry halves s -> s/2 (still
# within 1e-9 of the rejected trial) -> s/4 (< 1e-9 from the accepted point)
# ⇒ stalled=True and the resolution-limited STOP, never "CONVERGED".
dW11 = 0.05 + 0.01 * 1.5e-9 / d9[0]
seq11 = [dict(fom=F0, W=W0 + dW11)] + [dict(fom=F0 - 1.2 * SL11,
                                            W=W0 + dW11)] * 7
with _tf.TemporaryDirectory() as td:
    r11 = _drive(td, seq11, max_iter=8, gT=z9)
d11 = _deliv(r11)
check("T11 collapse: stalled row + STOPPED, no CONVERGED, 2 evals",
      len(r11["ev"]) == 2 and r11["pr"][1].get("stalled")
      and "[proj] STOPPED" in r11["out"]
      and "CONVERGED WITHIN NOISE" not in r11["out"],
      f"s {d11[0]:.3g} nm, evals {len(r11['ev'])}")
with _tf.TemporaryDirectory() as td:
    r11t = _drive(td, seq11, max_iter=3, gT=z9, fn=_patched(
        ('rec["stalled"] = True', 'pass')))
check("teeth: without the stall stop a geometry < 1e-9 from the accepted "
      "point is re-solved", len(r11t["ev"]) == 3 and _deliv(r11t)[1] < 1e-9
      and "[proj] STOPPED" not in r11t["out"],
      f"deliv {[f'{v:.3g}' for v in _deliv(r11t)]}")

# T10 (F1c): noise reject (shrink 0.5) then an ACCEPTED retry. The fake holds
# W and λ fixed, so the "held" growth test PASSES: base 10 -> 15, then the
# accepted radius 15 x 0.5 = 7.5 is adopted (pre-fix: snaps back to 15).
seq10 = [dict(fom=F0, W=W0), dict(fom=F0 - 1.2 * SL11, W=W0),
         dict(fom=F0, W=W0)]
with _tf.TemporaryDirectory() as td:
    r10 = _drive(td, seq10, max_iter=3)
check("T10 accepted retry adopts the radius used (held: 15 x 0.5 = 7.5)",
      r10["pr"][1].get("noise_reject") and abs(_deliv(r10)[1] - 5.0) < 1e-9
      and not r10["pr"][2]["phase"].endswith("-retry")
      and r10["ost"]["cap_nm"] == 7.5 and r10["ost"]["retry_shrink"] == 1.0,
      f"cap {r10['ost']['cap_nm']}")
with _tf.TemporaryDirectory() as td:
    r10t = _drive(td, seq10, max_iter=3,
                  fn=_patched(("cap_state = max(cap_state * retry_shrink, 2.0)",
                              "cap_state = cap_state")))
check("teeth: pre-fix accept snaps back to the (grown) base cap 15",
      r10t["ost"]["cap_nm"] == 15.0, f"cap {r10t['ost']['cap_nm']}")

print(("\nALL PASS" if not fails else f"\nFAILED: {fails}"))
sys.exit(1 if fails else 0)
