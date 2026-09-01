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

print(("\nALL PASS" if not fails else f"\nFAILED: {fails}"))
sys.exit(1 if fails else 0)
