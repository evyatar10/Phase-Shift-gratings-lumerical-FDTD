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

# ── 7) RGP (spec.wgp_rgp, notes_relaxed_projection.md) ────────────────────
# endpoint identity + both limits, on a random well-conditioned instance
gT7 = rng.standard_normal(191); gW7 = rng.standard_normal(191)
D7 = rng.uniform(0.5, 2.0, 191) ** 2
W_tgt7, marg7, cap7 = 18.613, 0.746, 5.0
def _st(W0):
    return {"bsf": 2.0, "cbv_hi": W_tgt7 + marg7 / 2, "cbv_lo": W_tgt7 - marg7 / 2,
            "Wh": [W0], "w_prev": None}
_dir = lambda v: v / np.linalg.norm(v)
# (a) w=1 (W exactly at the ceiling): direction == today's RIDE direction
W_at = W_tgt7 + marg7 / 2
s_rgp, ph, lam_r, wc = eng._rgp_step(gT7, gW7, D7, W_at, W_tgt7, marg7, cap7,
                                     _st(W_at))
# RIDE direction is W-independent; evaluate _proj_step mid-band (its ride
# branch — at W_at itself a 1-ULP excess flips it into restore)
s_ride, _, lam_p = eng._proj_step(gT7, gW7, D7, W_tgt7, W_tgt7, marg7, 1.0, cap7)
check("RGP w=1 endpoint == RIDE direction",
      np.allclose(_dir(s_rgp), _dir(s_ride), atol=1e-9) and wc == 0.0)
check("RGP w=1 keeps gW.d = 0",
      abs(float(gW7 @ s_rgp)) / (np.linalg.norm(gW7) * np.linalg.norm(s_rgp))
      < 1e-9)
check("RGP shadow price == _proj_step's", np.isclose(lam_r, lam_p, rtol=1e-9))
# (b) w=0: mid-band, buffer tiny vs marg ⇒ pure climb direction D*gT
# (a point BELOW THE FLOOR correctly engages the floor correction instead —
# the width spec is two-sided)
W_in = W_tgt7
s0, ph0, _, wc0 = eng._rgp_step(gT7, gW7, D7, W_in, W_tgt7, marg7, cap7,
                                _st(W_in))
check("RGP w=0 endpoint == climb direction",
      np.allclose(_dir(s0), _dir(D7 * gT7), atol=1e-9) and wc0 == 0.0)
# (c) violated ceiling (w=2): correction engaged and pushes W DOWN
W_hi = W_tgt7 + marg7 / 2 + 1.0
st_v = _st(W_hi)
s2, ph2, _, wc2 = eng._rgp_step(gT7, gW7, D7, W_hi, W_tgt7, marg7, cap7, st_v)
check("RGP violated: wc > 0 and gW.step < 0",
      wc2 > 0.0 and float(gW7 @ s2) < 0.0)
# (d) max-norm radius respected
check("RGP step capped at cap_nm",
      float(np.max(np.abs(s2))) <= cap7 * (1 + 1e-12))
# (e) default OFF — existing specs bit-identical
check("wgp_rgp default OFF", eng.CampaignSpec().wgp_rgp is False)

print(("\nALL PASS" if not fails else f"\nFAILED: {fails}"))
sys.exit(1 if fails else 0)
