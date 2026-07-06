"""
DERIVED BOUNDARY PROFILE ("golden shape") — first-order boundary-perturbation
solve for the sidewall profile delta(x) that cancels the device's residual
in-cone (radiating) field. User-selected route 2026-07-06.

Physics (assumptions stated honestly):
  * Shifting-boundary perturbation (Johnson et al. 2002): moving a sidewall
    outward by delta(x) adds a polarization line source ~ delta(x) * dEps *
    E_par(x, y_wall). For device-TM the dominant field E_z is PARALLEL to the
    sidewall -> the simple dEps*E_par form applies (no D-normal term).
  * Radiated in-cone amplitude of that source: dA(kx) ~ C * FFT[delta * E_wall]
    restricted to |kx| < kc. C = one complex constant absorbing dEps, the
    radiation Green factor, and the wall->axis measure. C is CALIBRATED, not
    assumed: fit |C| to ONE measured FDTD point (gap-shift +20 on W800), then
    VALIDATE the kernel against the rest of the measured dose curve
    (+-10/20/30) — sign asymmetry and shape must come out right or we stop.
  * Loss change model: dLoss/Loss = (int |A0 + dA|^2 / int |A0|^2) - 1 over
    the light cone (radiated power ~ in-cone weight; supported by the k-space
    diagnostic's attribution matching the polarimetry).
  * Wall field from axis field: E_wall(x) = E_axis(x) * t(W_local(x)) with
    t(W) = cos(kappa_lat * W / 2) from the EIM lateral mode (debug mode A,
    using the W800 k-space dataset). Mode B (after tm_field_export lands)
    samples the true complex E(x, y) at the wall — no EIM factor.

Deliverables printed:
  1. KERNEL VALIDATION vs measured gap-shift dose curve (W800 base).
  2. Constrained least-squares delta(x) over the inner +-N_SEG half-pitches
     (segment basis = per-tooth/per-gap edges -> directly fabricable),
     constraints: [a] resonance preserved (zeroth |E|^2-weighted moment = 0),
     [b] kappa / mode width preserved (2*beta Fourier component = 0).
  3. Predicted in-cone reduction of the optimal profile, and its DECOMPOSITION
     onto the already-used knob patterns (uniform gap shift / tooth see-saw /
     uniform tooth width) vs ORTHOGONAL remainder — the decision variable:
     large orthogonal content with real predicted gain => a genuinely new
     shape exists; tiny orthogonal content => the frontier is fundamental for
     local boundary perturbations (question closed by math).

Run:  python python_tools/derive_boundary_profile.py
"""

import os
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lateral_radiation_theory import slab_neff, N_CORE, N_CLAD, H, LAM  # noqa: E402

ROOT = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes"
KSPACE = os.path.join(ROOT, "results_from_athena", "radiation_kspace_diag",
                      "kspace_diag_N80_TM.mat")

PITCH = 0.51683          # um
HALF = PITCH / 2
W_WIDE, W_NARROW = 1.000, 0.600   # um (W800 baseline)
# Measured anchors (job 117927, W800 base, accurate): gap-shift dose curve.
MEAS_SHIFT_NM = np.array([-30, -20, -10, 10, 20, 30])
MEAS_DLOSS_REL = np.array([+12.01, +10.11, +2.67, -2.11, -8.75, -11.49]) / 117.4
CAL_POINT = (20, -8.75 / 117.4)   # calibrate |C| on this one


# ---------------------------------------------------------------- load field
m = loadmat(KSPACE, squeeze_me=True)
x = m["x_um"]                       # uniform grid (um)
Ez = m["Ez_axis_re"] + 1j * m["Ez_axis_im"]
kc, beta = float(m["kc_um"]), float(m["beta_um"])
dx = float(x[1] - x[0])
k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(x), dx))
IN = np.abs(k) < kc

F = lambda f: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f)))
A0 = F(Ez)[IN]                      # residual in-cone amplitude (relative units)
P0 = float(np.sum(np.abs(A0) ** 2))


# --------------------------------------------------- local width + wall field
def width_profile(xs):
    """W(x) of the W800 baseline (um): cavity |x|<HALF/... geometry from the
    builder: cavity(avg 0.8, len HALF) centered, then narrow/wide half-pitches."""
    W = np.full_like(xs, W_NARROW)
    ax = np.abs(xs)
    W[ax <= HALF / 2] = 0.8                       # cavity segment (avg width)
    # teeth: wide segments at [HALF/2 + (j)(HALF)] alternating narrow/wide,
    # left arm ends wide adjacent to cavity, right arm starts narrow — for the
    # kernel we use the SYMMETRIZED |x| layout (fields are ~symmetric).
    seg = np.floor((ax - HALF / 2) / HALF).astype(int)
    wide_mask = (ax > HALF / 2) & (seg % 2 == 0)  # first segment out = wide
    W[wide_mask] = W_WIDE
    return W


n_v = slab_neff(N_CORE, N_CLAD, H, LAM, "TM")     # vertical EIM index


def wall_factor(W_um):
    """t(W) = cos(kappa_lat*W/2) of the EIM lateral fundamental mode."""
    ne = slab_neff(n_v, N_CLAD, W_um * 1e-6, LAM, "TE")
    kap = 2 * np.pi / (LAM * 1e6) * np.sqrt(max(n_v ** 2 - ne ** 2, 1e-12))
    return np.cos(kap * W_um / 2)


Wx = width_profile(x)
tW = np.array([wall_factor(w) for w in np.unique(Wx)])
tmap = dict(zip(np.unique(Wx), tW))
E_wall = Ez * np.array([tmap[w] for w in Wx])


def dA_of(delta):
    """In-cone amplitude added by sidewall profile delta(x) (um units)."""
    return F(delta * E_wall)[IN]


def rel_dloss(delta, C):
    return float(np.sum(np.abs(A0 + C * dA_of(delta)) ** 2)) / P0 - 1.0


# --------------------------------------- knob patterns as delta(x) profiles
def deposit(d, x0, length_um, height):
    """Area-correct sub-cell deposit of a thin strip [x0, x0+length] of the
    given delta height, x-symmetric (+|x| and -|x|). A 10-30 nm strip is far
    below the 26 nm grid: only its integrated area matters in-cone, so spread
    area = height*length onto the nearest cell(s) with fractional weights."""
    for sgn in (+1, -1):
        a, b = sorted((sgn * x0, sgn * (x0 + length_um)))
        lo = np.searchsorted(x, a) - 1
        hi = np.searchsorted(x, b) + 1
        for i in range(max(lo, 0), min(hi, len(x) - 1)):
            cell_a, cell_b = x[i] - dx / 2, x[i] + dx / 2
            ov = max(0.0, min(b, cell_b) - max(a, cell_a))
            d[i] += height * ov / dx
    return d


def gap_shift_pattern(s_nm):
    """Tooth-1 gap shift of s (nm; + = tooth toward cavity, lengthen_cavity ON).
    Boundary picture per wall (delta = OUTWARD wall displacement / 2 walls,
    but we work single-wall and symmetric — the constant absorbs the 2):
      * tooth-1 inner edge at HALF/2: tooth (wide, W_wide) recedes -> the
        CAVITY (width 0.8) extends: delta = (0.8 - W_wide)/2 over strip s
        ... net edge move: material at the boundary changes from W_wide to
        cavity width => delta = (0.8 - W_WIDE)/2 * strip(s) at HALF/2.
      * tooth-1 outer edge at HALF/2+HALF: tooth extends into the narrow gap:
        delta = (W_WIDE - W_NARROW)/2 * strip(s).
    (positive s: both strips sit inside the region the tooth vacates/claims;
    negative s handled by sign of the area.)"""
    d = np.zeros_like(x)
    s_um = abs(s_nm) * 1e-3
    sg = np.sign(s_nm)
    # inner edge: cavity (0.8) replaces tooth (1.0) when tooth moves inward?
    # tooth moves toward cavity => tooth occupies [HALF/2 - s, ...]: at the
    # inner edge WIDE replaces CAVITY width: delta = +(W_WIDE-0.8)/2 * sgn
    deposit(d, HALF / 2 - s_um, s_um, sg * (W_WIDE - 0.8) / 2)
    # outer edge: gap (narrow) replaces tooth: delta = -(W_WIDE-W_NARROW)/2*sgn
    deposit(d, HALF / 2 + HALF - s_um, s_um, -sg * (W_WIDE - W_NARROW) / 2)
    # lengthen_cavity: cavity grows by 2s total => the far cavity boundary
    # against tooth±1 on each side moves out by s... on the symmetrized layout
    # the cavity edge IS the tooth inner edge — already covered above.
    return d


def seesaw_pattern(d1_nm, d2_nm):
    d = np.zeros_like(x)
    t1 = (np.abs(x) > HALF / 2) & (np.abs(x) <= HALF / 2 + HALF)
    t2 = (np.abs(x) > HALF / 2 + 2 * HALF) & (np.abs(x) <= HALF / 2 + 3 * HALF)
    d[t1] = d1_nm * 1e-3 / 2
    d[t2] = d2_nm * 1e-3 / 2
    return d


# ------------------------------------------------------------- calibration
s_cal, meas_cal = CAL_POINT
best = None
for phase in np.linspace(0, 2 * np.pi, 73):
    for mag in np.geomspace(1e-3, 1e3, 121):
        C = mag * np.exp(1j * phase)
        pred = rel_dloss(gap_shift_pattern(s_cal), C)
        if best is None or abs(pred - meas_cal) < best[0]:
            best = (abs(pred - meas_cal), C, pred)
C = best[1]
print(f"calibrated C = {abs(C):.3g} * exp(i {np.angle(C):+.2f}); "
      f"pred@cal {best[2]*100:+.1f}% vs meas {meas_cal*100:+.1f}%")

print("\nKERNEL VALIDATION — gap-shift dose curve (W800, measured vs predicted):")
ok = True
for s, meas in zip(MEAS_SHIFT_NM, MEAS_DLOSS_REL):
    pred = rel_dloss(gap_shift_pattern(s), C)
    flag = "OK" if np.sign(pred) == np.sign(meas) else "SIGN-FAIL"
    ok &= (flag == "OK")
    print(f"  shift {s:+3d} nm: meas {meas*100:+6.1f}%   pred {pred*100:+6.1f}%   {flag}")
print("VALIDATION:", "PASSED (signs reproduce)" if ok else
      "FAILED — kernel not trustworthy, STOP before deriving shapes")

if not ok:
    sys.exit(1)

# ------------------------------------------- constrained least-squares solve
N_SEG = 14                     # half-pitch segments per side over inner ~3.6 um
segs = []
for j in range(N_SEG):
    lo = HALF / 2 + j * HALF
    sel = (np.abs(x) > lo) & (np.abs(x) <= lo + HALF)
    segs.append(sel.astype(float))
segs.insert(0, (np.abs(x) <= HALF / 2).astype(float))   # cavity segment
B = np.array(segs)             # (nb, nx) basis (x-symmetric by construction)

# columns of the in-cone response matrix
M = np.array([C * dA_of(b) for b in B]).T          # (n_in, nb) complex
# constraints: [a] resonance: sum(delta * |E|^2) = 0 ; [b] 2*beta component
w_res = np.abs(Ez) ** 2
c_res = B @ w_res
c_2b = B @ (w_res * np.cos(2 * beta * x))
c_2bs = B @ (w_res * np.sin(2 * beta * x))
# fwhm guard: the mode-width response is driven by the slowly-varying part of
# the effective-potential change ~ delta(x)*|E|^2; kill its curvature (x^2)
# moment too, else the solver sneaks onto the delocalization manifold via a
# smooth inner->outer ramp (observed without this constraint).
c_x2 = B @ (w_res * (x ** 2) / max(np.max(x ** 2), 1e-30))
Cons = np.vstack([c_res, c_2b, c_2bs, c_x2])       # (4, nb)

# solve min ||A0 + M d||^2 + ridge, s.t. Cons d = 0  (real d, complex system)
Mr = np.vstack([M.real, M.imag])
A0r = np.concatenate([A0.real, A0.imag])
ridge = 1e-3 * np.linalg.norm(Mr, ord=2) ** 2
H_ = Mr.T @ Mr + ridge * np.eye(B.shape[0])
g = -Mr.T @ A0r
# KKT
nb, nc = B.shape[0], Cons.shape[0]
KKT = np.block([[H_, Cons.T], [Cons, np.zeros((nc, nc))]])
rhs = np.concatenate([g, np.zeros(nc)])
sol = np.linalg.solve(KKT, rhs)
d_opt = sol[:nb]

pred_opt = rel_dloss(B.T @ d_opt, C)
print(f"\nOPTIMAL constrained profile: predicted relative loss change "
      f"{pred_opt*100:+.1f}% (residual in-cone)")
print("segment amplitudes (nm, cavity first then per half-pitch outward):")
print("  " + "  ".join(f"{v*1e3:+.1f}" for v in d_opt))

# ---------------------------------------------------- knob decomposition
knobs = {
    "uniform-gap-shift": gap_shift_pattern(20.0),
    "see-saw(+20,-20)": seesaw_pattern(20.0, -20.0),
    "uniform-tooth-width": (lambda d=np.zeros_like(x): (
        [d.__setitem__((np.abs(x) > HALF / 2), 0.010), d][1]))(),
}
d_x = B.T @ d_opt
proj_total = 0.0
nrm = np.linalg.norm(d_x)
print("\nDECOMPOSITION of the optimal profile onto known knobs:")
for name, pat in knobs.items():
    p = np.dot(d_x, pat) / (np.linalg.norm(pat) * nrm + 1e-30)
    proj_total += p ** 2
    print(f"  {name:22s}: cos-similarity {p:+.2f}")
print(f"  ORTHOGONAL remainder: {max(0.0, 1 - proj_total)*100:.0f}% of the profile norm")
print("\nInterpretation: large predicted gain AND large orthogonal remainder")
print("=> a genuinely new shape; otherwise the frontier is fundamental for")
print("local boundary perturbations. (Debug mode A: W800 axis field + EIM")
print("wall factor. Rerun in mode B with tm_field_export for the stack.)")
