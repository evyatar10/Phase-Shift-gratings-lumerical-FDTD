"""Fit C_field for the width-gradient adjoint (the C-recipe, field-monitor path).

Study dir: runners/lumopt2_design/  |  Created 2026-08-21  |  Jobs: W3a/W3b
(validate_c325 tasks 12/13, first attempt 135971).
Purpose: the port-path C (skill item 6) does NOT transfer to the field-adjoint
path; this fits C_field = s*e^{i*phi} from one naive validate_gradient run
(FD + Re-adjoint, task 12) plus one adjoint-only quadrature run (Im-adjoint,
task 13):    FD_p  ~=  s*(cos(phi)*Re_p - sin(phi)*Im_p)
Paste the three printed vectors below (SAME param indices, SAME detune point),
run `python -m runners.lumopt2_design.fit_c_field`. Standing order: re-run
this fit on every Lumerical version bump before trusting width gradients.

PASS gates (V2_FWHM_PLAN.md W3): post-fit sign agreement on every param and
per-param |s*(cos*Re - sin*Im)/FD - 1| <= ~0.10; re-verify at a second
operating point before any campaign.
"""
import numpy as np

# ── paste measured vectors here (from the W3-GPU printouts) ─────────────────
# ★★2026-08-25 — THE PRODUCTION TILED CONFIG (wg_src_tiles=4, GPU, full
# region). Re = job 137003 task 37, Im = task 40, both exit 0, both with
# `[wg-tiles] live 4/4`. FD is the keep-forever CPU FieldRegion vector
# (job 136189) — comparable HERE because tiling reproduces the FULL-REGION
# source, which the cropped rungs did not.
FD = [-0.00365, 0.01825, 0.02026]        # job 136189 (CPU, full region)
RE = [-0.00652518, 0.03027003, 0.03658102]  # 137019 t42 PRODUCTION numerics
IM = [-0.00245131, 0.01082157, 0.01080934]  # 137019 t43 PRODUCTION numerics
# Tasks 20/21 print indices [0, SL_SHIFT.start, I_CAV] — 3 rows, not the old
# 5-row task-12 set (stale LABELS fixed 2026-08-23 review: zip() would have
# mislabeled shift_1 as "corr_25" and wcav as "avg_1").
LABELS = ["corr_1", "shift_1", "wcav"]
# ★PENALTY CORRECTION for vectors printed by jobs ≤136190 (engine fixed
# 2026-08-23, review): those adjoint prints came from the attach_penalty-
# WRAPPED gradient, i.e. raw − kappa_penalty_grad, while lumopt2's FD is
# penalty-free (fom.calculate_fom). At the detune-1 point (25 shifts of
# 20 nm → elong = 1000 nm > 120 deadband) the guard adds EXACTLY
# 2·BETA_ELONG·(1000−120)·2 = 2·1e-5·880·2 = 0.0352 to the shift row
# (corr/wcav rows: 0 — rho = 1.0 sits inside its deadband). Add it back to
# RE and IM before fitting. Set to zeros for vectors printed by the FIXED
# engine (which prints raw directly).
# ★ZERO for the 2026-08-25 tiled vectors: run_adjoint_only calls
# compute_gradient_raw (penalty-free) and wg_pure makes J = -softW alone, so
# these prints carry no penalty term to add back.
PEN_GRAD = [0.0, 0.0, 0.0]

def fit(fd, re, im):
    fd, re, im = (np.asarray(v, float) for v in (fd, re, im))
    best = None
    for phi in np.radians(np.arange(-90.0, 90.0, 0.05)):
        proj = np.cos(phi) * re - np.sin(phi) * im
        s = float(np.dot(proj, fd) / np.dot(proj, proj))
        r = float(np.linalg.norm(s * proj - fd) / np.linalg.norm(fd))
        if best is None or r < best[2]:
            best = (phi, s, r)
    return best

def main():
    assert FD and RE and IM and len(FD) == len(RE) == len(IM) == len(LABELS), \
        "paste the three measured vectors first (same indices, same detune)"
    re = np.asarray(RE, float) + np.asarray(PEN_GRAD, float)
    im = np.asarray(IM, float) + np.asarray(PEN_GRAD, float)
    if any(PEN_GRAD):
        print(f"[pen-corrected] added PEN_GRAD {PEN_GRAD} to RE and IM "
              f"(pre-fix engine prints; see comment)")
    phi, s, r = fit(FD, re, im)
    c = s * np.exp(1j * phi)
    # ★THE ENGINE APPLIES a*RE + b*IM (lumopt2_design MixedFom phasing), while
    # the fit above solves FD ~= s*(cos*RE - sin*IM). So the coefficients to
    # STORE are (s*cos, -s*sin) = the CONJUGATE of s*e^{i*phi}. Printing
    # s*e^{i*phi} directly caused a 15% magnitude error that passed every
    # SIGN check (2026-08-25, C_field). Print BOTH, and label which to store.
    print(f"fit form   C = {c.real:.4f}{c.imag:+.4f}i  (s {s:.4f}, phi "
          f"{np.degrees(phi):+.2f} deg)  vec residual {r:.3f}")
    print(f">>> STORE THIS -> adj_fix_field_re/im = ({c.real:.4f}, {-c.imag:+.4f})"
          f"   [conjugate: the engine applies a*RE + b*IM]")
    proj = s * (np.cos(phi) * re - np.sin(phi) * im)
    ok = True
    for name, f, p in zip(LABELS, FD, proj):
        rr = p / f - 1.0
        sgn = np.sign(p) == np.sign(f)
        ok &= sgn and abs(rr) <= 0.10
        print(f"  {name:8s} fd {f:+.4e}  fitted {p:+.4e}  resid {rr:+7.1%} "
              f"sign {'ok' if sgn else 'FLIP'}")
    print("W3 " + ("PASS — set adj_fix_field_re/im to the C above; re-verify "
                   "at a 2nd operating point before any campaign"
                   if ok else "FAIL — do NOT campaign on this gradient"))

if __name__ == "__main__":
    main()
