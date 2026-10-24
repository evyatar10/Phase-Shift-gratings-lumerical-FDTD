# postarray_nearfield.py (study: runners/sweeps/farfield_sph_15um_postarray_overshoot variant nfctrl, Athena 269582; 2026-10-24)
# DIRECT in-situ post drive from the z=0 XY monitor (field_2d_xy_band_m) instead of back-propagating the tube side plane, so posts can sit
# inside y < 0.9 um near the cavity (local tooth wall 0.48-0.8 um at |x| < 10 um). Same units/convention as postarray_born.posts_ff
# (raw field x NORM, NORM = conj(D)/|D|/sqrt(U) from the same run). Spec + gates: postarray_work_dy29/fable_alpha/RESULTS.md section 3.
#   CTRL_MAT=<nfctrl result .mat> PA_WORK=<work dir> KCUT_PER_UM=7.5 python postarray_nearfield.py      (runs the gates)
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postarray_born as dp
C0 = dp.C0; m = dp.m
xy = m["field_xy"]; X = np.ravel(xy.x); Y = np.ravel(xy.y); LXY = np.ravel(xy.lambda_3d)
E = np.asarray(xy.E_res).reshape(X.size, Y.size, -1, LXY.size, 3)[:, :, 0]      # (nx, ny, nf, 3), z = 0 plane
lam_ff = float(np.ravel(m["surface_top"].lam)[0]); i = int(np.argmin(np.abs(LXY - lam_ff)))
om, kap, t, Ec, comp = dp.g["trace_fit"](m["trace_time_cavity"])
D = lambda lam: np.trapezoid(Ec * np.exp(1j * 2 * np.pi * C0 / lam * t), t)
FAC = D(lam_ff) / D(LXY[i])                     # single-mode ring-down: move the XY bin to the far-field frequency
EXY = E[:, :, i, :] * FAC                       # raw units, at lam_ff
print(f"XY map: {X.size} x {Y.size} pts, x {X[0]*1e6:.2f}..{X[-1]*1e6:.2f} um, y {Y[0]*1e6:.3f}..{Y[-1]*1e6:.3f} um, max dy {np.diff(Y).max()*1e9:.1f} nm; "
      f"bin {LXY[i]*1e9:.4f} nm vs far field {lam_ff*1e9:.4f} nm, |FAC| {abs(FAC):.3f} arg {np.degrees(np.angle(FAC)):+.1f} deg")
from scipy.interpolate import RegularGridInterpolator as RGI
_I = [RGI((X, Y), EXY[:, :, c], bounds_error=True) for c in range(3)]
def drive_xy(x, y):
    """Raw in-situ field (3,) at (x, y, z=0) [m] from the XY map (y >= 0 half; mirror rule applied by the caller)."""
    return np.array([_I[c]((x, y)) for c in range(3)])
def posts_ff_direct(xn, yn, rn):
    """As postarray_born.posts_ff (mirrored posts, same NORM), drive from the XY map."""
    Ef = np.zeros((dp.nhat.shape[0], 3), complex)
    for x, y, r in zip(xn, yn, rn):
        e = drive_xy(x, y)
        for yy, ee in ((y, e), (-y, dp.s_y * e * np.array([1, -1, 1]))):
            p = dp.k ** 2 / (4 * np.pi) * np.pi * r ** 2 * dp.H_POST * dp.GAM * ee
            Ef += (p[None] - dp.nhat * (dp.nhat @ p)[:, None]) * np.exp(-1j * dp.k * (dp.nhat @ np.array([x, yy, 0.0])))[:, None]
    Eth = np.sum(dp.th * Ef, -1).reshape(dp.xg.size, dp.phig.size); Eph = np.sum(dp.ph * Ef, -1).reshape(dp.xg.size, dp.phig.size)
    return np.stack([Eth, Eph]) * dp.NORM
if __name__ == "__main__":
    xs_g = dp.xs[np.abs(dp.xs) <= 30e-6]
    # gate (a): XY map at the tube plane vs the stored side-monitor field (same run, same frequency)
    if Y[-1] >= dp.y0 - 1e-9:
        es = np.array([[np.interp(x, dp.xs, dp.Es[:, np.argmin(abs(dp.zs)), c].real) + 1j * np.interp(x, dp.xs, dp.Es[:, np.argmin(abs(dp.zs)), c].imag) for c in range(3)] for x in xs_g])
        ex = np.array([drive_xy(x, dp.y0) for x in xs_g]); z = np.vdot(es, ex) / np.vdot(es, es)
        print(f"GATE a (y {dp.y0*1e6:.3f}): |E_xy|/|E_side| fit {abs(z):.3f} arg {np.degrees(np.angle(z)):+.1f} deg, overlap {abs(np.vdot(es, ex))**2/(np.vdot(es,es).real*np.vdot(ex,ex).real):.4f}")
    else:
        print(f"GATE a skipped: map ends at y {Y[-1]*1e6:.3f} < tube plane {dp.y0*1e6:.3f}")
    # gate (b): XY map vs back-propagated drive where both are valid
    for y in (1.2e-6, 1.5e-6, 1.8e-6):
        ebp = dp.drive_plane(y); ebp = np.array([[np.interp(x, dp.xs, ebp[:, c].real) + 1j * np.interp(x, dp.xs, ebp[:, c].imag) for c in range(3)] for x in xs_g])
        ex = np.array([drive_xy(x, y) for x in xs_g]); z = np.vdot(ebp, ex) / np.vdot(ebp, ebp)
        print(f"GATE b y {y*1e6:.1f}: fit {abs(z):.3f} arg {np.degrees(np.angle(z)):+.1f} deg, overlap {abs(np.vdot(ebp, ex))**2/(np.vdot(ebp,ebp).real*np.vdot(ex,ex).real):.4f}, "
              f"|E| ratio max-region {np.abs(ex).max()/np.abs(ebp).max():.3f}")
