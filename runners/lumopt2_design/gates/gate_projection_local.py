"""gate_projection_local.py — LOCAL, ZERO-GPU gate for wg_project (gate P0 +
clip + restoration + legacy-off). Study: v2 width projection; 2026-08-25;
no jobs. Run AFTER applying patch_projection.diff:  python gate_projection_local.py
Exit 0 = all pass. Tests the REAL engine code (_proj_step, make_fct_v2,
CampaignSpec) with synthetic gradient vectors — no lumapi, no FDTD."""
import sys
import numpy as np

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

print(("\nALL PASS" if not fails else f"\nFAILED: {fails}"))
sys.exit(1 if fails else 0)
