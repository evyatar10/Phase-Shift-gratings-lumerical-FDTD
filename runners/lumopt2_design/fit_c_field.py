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

# ── paste measured vectors here (from the W3a/W3b printouts) ────────────────
FD = []          # task 12: validate_gradient fd vector (FD FIRST in printout)
RE = []          # task 12: its adjoint vector (naive C_field=(1,0))
IM = []          # task 13: adjoint-only vector at C_field=(0,1)
LABELS = ["corr_1", "corr_25", "avg_1", "shift_1", "wcav"]

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
    assert FD and RE and IM and len(FD) == len(RE) == len(IM), \
        "paste the three measured vectors first (same indices, same detune)"
    phi, s, r = fit(FD, RE, IM)
    c = s * np.exp(1j * phi)
    print(f"C_field = {c.real:.4f}{c.imag:+.4f}i  (s {s:.4f}, phi {np.degrees(phi):+.2f} deg)"
          f"  vec residual {r:.3f}")
    proj = s * (np.cos(phi) * np.asarray(RE) - np.sin(phi) * np.asarray(IM))
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
