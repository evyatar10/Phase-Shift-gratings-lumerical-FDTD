"""
Phase-0 theory for the TM loss program (2026-07-05): where can the remaining
~8% resonant loss go, and which levers are worth GPU time?

Zero-GPU, zero-license: analytic slab modes + effective-index method (EIM) +
1D transfer-matrix models. Anchors against measured facts:
  - Bloch index n = lambda_res/(2*pitch) = 1558.6/1033.66 = 1.5078
    (matches the k-space diagnostic's n_eff 1.508)
  - cavity-width ladder: loss max ~W800 (control), min 1050-1075 (champion),
    worse again by 1400 (+7% vs control at coarse mesh)
  - radiation is in-plane; k-space: ~30% of radiating weight within +-3
    pitches of the defect, ~70% along the arms.

Sections:
  A. n_eff(W) table via EIM  -> is there a guided lateral channel? (verdict:
     lateral "leakage" in the slab-mode sense needs a slab; fully-etched ridge
     in uniform oxide has none -> radiation goes to the 3D oxide continuum)
  B. Two-edge interference discriminator: can propagating sidewall-pair
     interference explain the cavity-width ladder period? Registered
     prediction either way (second minimum near W ~ 2 um = the cheap test).
  C. Near-cavity side-reflector feasibility (user request): normal-incidence
     SiN strip DBR in oxide for the broadside in-plane channel; grazing
     Fresnel for the near-axial arm channel; d-scan phase periods vs the
     converged y=6.8 um box; evanescent-drain estimate.
  D. SSH gap-dimerization registered prediction: 1D TMM of the pi-shift
     stack, envelope light-cone weight, uniform vs dimerized inner gaps.

Run:  python python_tools/lateral_radiation_theory.py
"""

import numpy as np

# ---------------------------------------------------------------- constants
LAM = 1558.6e-9          # resonance (optimization-mesh anchor)
N_CORE = 1.97
N_CLAD = 1.444
H = 350e-9               # core height
PITCH = 516.83e-9
W_WIDE, W_NARROW = 1000e-9, 600e-9
K0 = 2 * np.pi / LAM
KC = N_CLAD * K0         # light-cone edge (oxide), rad/m
N_BLOCH_MEAS = 1558.6 / (2 * 516.83)   # 1.5078


# ------------------------------------------------- symmetric-slab mode solver
def slab_neff(n1, n2, d, lam, pol):
    """Fundamental even mode of a symmetric slab (core n1, clad n2, thickness d).
    pol='TE': E parallel to interfaces; 'TM': E normal to interfaces.
    Returns n_eff, or n2 if the mode is at/below cutoff (fundamental never is,
    but EIM lateral solves can get arbitrarily close)."""
    k0 = 2 * np.pi / lam

    def resid(neff):
        kap = k0 * np.sqrt(max(n1**2 - neff**2, 0.0))
        gam = k0 * np.sqrt(max(neff**2 - n2**2, 0.0))
        if pol == "TE":
            return kap * np.tan(kap * d / 2) - gam
        return (kap / n1**2) * np.tan(kap * d / 2) - gam / n2**2

    # fundamental even mode: kap*d/2 in (0, pi/2). resid > 0 at neff -> n2
    # (gam=0, tan>0) and resid < 0 at neff -> n1 (kap=0, gam>0): decreasing.
    lo = max(n2 + 1e-9, np.sqrt(max(n1**2 - (np.pi / (k0 * d))**2, n2**2)) + 1e-9)
    hi = n1 - 1e-9
    if resid(lo) < 0:
        return n2   # no root on the fundamental branch (never for symmetric slab)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ridge_neff_tm(width):
    """Device-TM (E along height z) ridge n_eff via EIM:
    vertical slab solve is TM (E normal to the z-interfaces), lateral solve is
    TE (E parallel to the y-interfaces) with the vertical n_eff as core."""
    n_v = slab_neff(N_CORE, N_CLAD, H, LAM, "TM")
    return slab_neff(n_v, N_CLAD, width, LAM, "TE"), n_v


# ------------------------------------------------------------- 1D TMM helpers
def tmm_stack(n_list, L_list, lam):
    """Total transfer matrix (field convention M @ [E+, E-]) of a layer stack
    embedded between semi-infinite media equal to the first/last given index.
    Returns 2x2 complex M and per-layer interface amplitudes for field maps."""
    k0 = 2 * np.pi / lam
    M = np.eye(2, dtype=complex)
    mats = []
    for i in range(len(n_list)):
        # propagation in layer i
        ph = k0 * n_list[i] * L_list[i]
        P = np.array([[np.exp(-1j * ph), 0], [0, np.exp(1j * ph)]])
        M = P @ M
        mats.append(M.copy())
        if i < len(n_list) - 1:
            n_a, n_b = n_list[i], n_list[i + 1]
            r = (n_a - n_b) / (n_a + n_b)
            t = 2 * np.sqrt(n_a * n_b) / (n_a + n_b)  # power-symmetric convention
            I = (1 / t) * np.array([[1, r], [r, 1]], dtype=complex)
            M = I @ M
            mats.append(M.copy())
    return M, mats


def stack_transmission(n_list, L_list, lam):
    M, _ = tmm_stack(n_list, L_list, lam)
    # incidence from the left: [t, 0]^T = M [1, r]^T
    r = -M[1, 0] / M[1, 1]
    t = M[0, 0] + M[0, 1] * r
    return abs(t) ** 2, abs(r) ** 2


def build_grating(n_hi, n_lo, n_cav, n_side, cav_len, gap_len_list=None):
    """pi-shift stack matching bragg_device geometry: the cavity segment
    REPLACES the innermost narrow slot (half-period pattern replacement, NOT a
    quarter-wave insertion — the merged wide|cavity block is what creates the
    mid-gap state). From the cavity outward:
      left arm : wide(1), narrow(1), wide(2), ...   (wide adjacent to cavity)
      right arm: narrow(1), wide(1), narrow(2), ... (narrow adjacent to cavity)
    gap_len_list overrides the innermost narrow-gap lengths per side."""
    half = PITCH / 2

    def gap(j):
        if gap_len_list is None or j >= len(gap_len_list):
            return half
        return gap_len_list[j]

    left_out, left_L = [], []     # from cavity outward
    right_out, right_L = [], []
    for j in range(n_side):       # j=0 innermost
        left_out += [n_hi, n_lo]
        left_L += [half, gap(j)]
        right_out += [n_lo, n_hi]
        right_L += [gap(j), half]
    n = [n_hi] + left_out[::-1] + [n_cav] + right_out + [n_hi]
    L = [2e-6] + left_L[::-1] + [cav_len] + right_L + [2e-6]
    return n, L


def field_profile(n_list, L_list, lam):
    """|E(x)| sampled on a uniform grid, incidence 1 from the left."""
    M, _ = tmm_stack(n_list, L_list, lam)
    r = -M[1, 0] / M[1, 1]
    k0 = 2 * np.pi / lam
    # forward/backward amplitudes at each layer entrance, marching left->right
    amps = [np.array([1, r], dtype=complex)]
    for i in range(len(n_list) - 1):
        ph = k0 * n_list[i] * L_list[i]
        P = np.array([[np.exp(-1j * ph), 0], [0, np.exp(1j * ph)]])
        a = P @ amps[-1]
        n_a, n_b = n_list[i], n_list[i + 1]
        rr = (n_a - n_b) / (n_a + n_b)
        tt = 2 * np.sqrt(n_a * n_b) / (n_a + n_b)
        I = (1 / tt) * np.array([[1, rr], [rr, 1]], dtype=complex)
        amps.append(I @ a)
    xs, Es = [], []
    x0 = 0.0
    for i, (nn, LL) in enumerate(zip(n_list, L_list)):
        xl = np.linspace(0, LL, max(3, int(LL / 20e-9)), endpoint=False)
        a = amps[i]
        Es.append(a[0] * np.exp(1j * k0 * nn * xl) + a[1] * np.exp(-1j * k0 * nn * xl))
        xs.append(x0 + xl)
        x0 += LL
    return np.concatenate(xs), np.concatenate(Es)


def envelope_fwhm(x, E):
    """FWHM of the |E|^2 envelope (coarse boxcar over ~one period)."""
    xu = np.linspace(x[0], x[-1], 1 << 13)
    P = np.interp(xu, x, np.abs(E) ** 2)
    nbox = max(3, int(PITCH / (xu[1] - xu[0])))
    ker = np.ones(nbox) / nbox
    Ps = np.convolve(P, ker, mode="same")
    half = Ps.max() / 2
    above = np.where(Ps >= half)[0]
    return xu[above[-1]] - xu[above[0]]


def lightcone_weight(x, E):
    """Fraction of |FFT(E)|^2 inside |k| < KC on a uniform resample of x."""
    xu = np.linspace(x[0], x[-1], 1 << 14)
    Eu = np.interp(xu, x, np.real(E)) + 1j * np.interp(xu, x, np.imag(E))
    win = np.hanning(len(Eu))
    F = np.fft.fftshift(np.fft.fft(Eu * win))
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(xu), xu[1] - xu[0]))
    P = np.abs(F) ** 2
    return P[np.abs(k) < KC].sum() / P.sum()


def find_resonance(n_list, L_list, lam_lo, lam_hi, npts=1201):
    lams = np.linspace(lam_lo, lam_hi, npts)
    Ts = np.array([stack_transmission(n_list, L_list, l)[0] for l in lams])
    i = int(np.argmax(Ts))
    return lams[i], Ts[i]


def arm_reflection(n_arm, L_arm, n_in, lam):
    """Complex reflection of one mirror arm seen from the cavity side
    (incidence medium n_in, arm exits into its own last layer index)."""
    n = [n_in] + list(n_arm) + [n_arm[-1]]
    L = [0.0] + list(L_arm) + [0.0]
    M, _ = tmm_stack(n, L, lam)
    return -M[1, 0] / M[1, 1]


def find_defect_resonance(n_list, L_list, lam_guess, span=16e-9):
    """pi-shift defect resonance via the smooth Fabry-Perot phase condition
    (the T=1 peak is a few pm wide -- brute-force scans miss it). Cavity layer
    = the middle one; resonance where the cavity round trip closes:
    2*phi_cav + arg(r_left) + arg(r_right) = 2*pi*m."""
    mid = len(n_list) // 2
    n_cav, L_cav = n_list[mid], L_list[mid]
    nR, LR = list(n_list[mid + 1:-1]), list(L_list[mid + 1:-1])
    nL = list(n_list[1:mid])[::-1]      # from cavity outward
    LL = list(L_list[1:mid])[::-1]

    def phi(lam):
        k0 = 2 * np.pi / lam
        return (2 * k0 * n_cav * L_cav
                + np.angle(arm_reflection(nR, LR, n_cav, lam))
                + np.angle(arm_reflection(nL, LL, n_cav, lam)))

    lams = np.linspace(lam_guess - span / 2, lam_guess + span / 2, 801)
    ph = np.unwrap([phi(l) for l in lams])
    # crossings of ANY multiple of 2*pi (the resonance branch is not known)
    c = np.floor(ph / (2 * np.pi))
    idx = np.where(np.diff(c) != 0)[0]
    if len(idx) == 0:
        return None, 0.0
    # take the crossing nearest lam_guess, then bisect its 2*pi level
    i = idx[np.argmin(np.abs(lams[idx] - lam_guess))]
    m = 2 * np.pi * max(c[i], c[i + 1])
    lo, hi = lams[i], lams[i + 1]
    f_lo = ph[i] - m
    for _ in range(60):
        mid_l = 0.5 * (lo + hi)
        # phi() is wrapped; m is an exact 2*pi multiple, so the wrapped angle
        # itself crosses zero at the resonance
        fm = phi(mid_l)
        if np.sign(fm) == np.sign(f_lo):
            lo, f_lo = mid_l, fm
        else:
            hi = mid_l
    lam_r = 0.5 * (lo + hi)
    return lam_r, stack_transmission(n_list, L_list, lam_r)[0]


# ============================================================== A. n_eff table
print("=" * 72)
print("A. EIM n_eff(W) — device-TM, h=350 nm, n 1.97/1.444, lam 1558.6 nm")
print("=" * 72)
n_v_tm = slab_neff(N_CORE, N_CLAD, H, LAM, "TM")
n_v_te = slab_neff(N_CORE, N_CLAD, H, LAM, "TE")
print(f"vertical slab n_eff:  TM {n_v_tm:.4f}   (TE {n_v_te:.4f})")
widths = [500, 600, 700, 800, 900, 1000, 1050, 1100, 1200, 1400]
neffs = {}
for w in widths:
    ne, _ = ridge_neff_tm(w * 1e-9)
    neffs[w] = ne
    margin = ne - N_CLAD
    print(f"  W = {w:5d} nm : n_eff = {ne:.4f}   (margin over n_clad {margin:+.4f})")
n_hi, n_lo = neffs[1000], neffs[600]
print(f"\nBloch check: mean(n_eff 600,1000) = {(n_hi + n_lo)/2:.4f} "
      f"vs measured lambda/(2*pitch) = {N_BLOCH_MEAS:.4f}")
print("VERDICT A: fully-etched ridge in uniform oxide -> no guided slab channel;")
print("  'lateral leakage' exists only as radiation into the 3D oxide continuum")
print("  (light-cone content). Magic-width bound-channel cancellation N/A as-is;")
print("  what CAN exist is directional two-edge interference (section B).")

# ================================================== B. two-edge discriminator
print()
print("=" * 72)
print("B. Two-edge interference vs the measured cavity-width ladder")
print("=" * 72)
# measured ladder (coarse mesh, loss rel. control 0.110): 895.5:-17% 950:-24%
# 1050:-30% 1150:-26% 1250:-14% 1400:+7%  -> max near W~800, min 1050,
# rising through 1400. If the two cavity sidewalls (y=+-W/2) radiate as an
# in-phase pair into in-plane angle theta (ky = KC*sin(theta_from_axis)),
# far-field ~ cos^2(ky*W/2): min-to-min period dW = 2*pi/ky.
for per_nm in (500.0, 1000.0, 1100.0, 1200.0):
    ky = 2 * np.pi / (per_nm * 1e-9)
    s = ky / KC
    ang = np.degrees(np.arcsin(min(s, 1.0))) if s <= 1 else float("nan")
    print(f"  ladder period {per_nm:6.0f} nm -> ky = {ky*1e-6:5.2f} rad/um "
          f"(ky/kc = {s:.2f}; theta from axis = {ang:.0f} deg)" +
          ("  [outside light cone!]" if s > 1 else ""))
print(f"  light-cone max ky = {KC*1e-6:.2f} rad/um")
print("Observed max(800)->min(1050)->rising(1400): period ~1000-1100 nm ->")
print("  ky ~ 5.7-6.3 rad/um ~ kc -> BROADSIDE in-plane radiation (theta ~ 90")
print("  deg from the guide axis) is marginally consistent; near-axial (10 deg)")
print("  would need period ~6 um and is firmly excluded as the ladder driver.")
print("REGISTERED DISCRIMINATOR: two-edge model predicts a SECOND loss minimum")
print(f"  near W ~ {1050 + 1050:.0f}-{1050 + 1150:.0f} nm; a pure moment-null predicts monotonic")
print("  worsening past 1400. One cheap FDTD row at W~2100 decides.")

# =========================================== C. near-cavity reflector numbers
print()
print("=" * 72)
print("C. Near-cavity side reflectors (SiN strips in oxide, same litho layer)")
print("=" * 72)
# broadside channel: normal incidence on a strip parallel to the guide
qw = LAM / (4 * N_CORE)
print(f"quarter-wave SiN strip width = {qw*1e9:.0f} nm")


def dbr_R(n_strips, w_strip, gap):
    n = [N_CLAD]
    L = [0.5e-6]
    for _ in range(n_strips):
        n += [N_CORE, N_CLAD]
        L += [w_strip, gap]
    n += [N_CLAD]
    L += [0.5e-6]
    _, R = stack_transmission(n, L, LAM)
    return R


qgap = LAM / (4 * N_CLAD)
for ns in (1, 2, 3, 4, 6):
    print(f"  normal incidence, {ns} strip(s) ({qw*1e9:.0f} nm) + {qgap*1e9:.0f} nm gaps: "
          f"R = {dbr_R(ns, qw, qgap):.3f}")
# d-scan phase period for the broadside channel
print(f"broadside d-scan period pi/ky = {np.pi/KC*1e6:.2f} um  -> fits the box easily")
print(f"near-axial (10 deg) d-scan period = {np.pi/(KC*np.sin(np.radians(10)))*1e6:.2f} um"
      "  -> exceeds usable offsets in the y=6.8 um box (phase must come from strip width)")

# grazing incidence (near-axial arm radiation) Fresnel at a single oxide->SiN face
for th_deg in (60, 70, 80, 85):
    th = np.radians(th_deg)
    n1, n2 = N_CLAD, N_CORE
    ct1 = np.cos(th)
    st2 = n1 * np.sin(th) / n2
    ct2 = np.sqrt(1 - st2**2)
    rs = (n1 * ct1 - n2 * ct2) / (n1 * ct1 + n2 * ct2)
    rp = (n2 * ct1 - n1 * ct2) / (n2 * ct1 + n1 * ct2)
    print(f"  grazing theta_i={th_deg} deg: single-interface R_s={rs**2:.3f}  R_p={rp**2:.3f}")

# evanescent drain of the GUIDED mode into a strip at offset d
gam = K0 * np.sqrt(N_BLOCH_MEAS**2 - N_CLAD**2)
print(f"guided-mode lateral decay gamma = {gam*1e-6:.2f} 1/um; tail amplitude at "
      f"d=1.0/1.5/2.0 um: {np.exp(-gam*1.0e-6):.3f} / {np.exp(-gam*1.5e-6):.3f} / "
      f"{np.exp(-gam*2.0e-6):.3f}")
print("GATE C: >=0.2 reflectance within <=4 strips is met at NORMAL incidence")
print("  (broadside channel) by 2 strips; keep strips d>=1.2 um to bound drain.")

# =============================================== D. SSH dimerization TMM model
print()
print("=" * 72)
print("D. SSH gap-dimerization — envelope light-cone weight (registered pred.)")
print("=" * 72)
N_SIDE = 80
half = PITCH / 2
# CALIBRATION (EIM contrast is too weak vs the real 3D device): pin the mean
# index to the measured Bloch index (fixes the resonance at 1558.6) and pick
# the hi/lo split to reproduce the measured spatial mode width fwhm_m=15.4 um.
FWHM_MEAS = 15.4e-6


def model_at(dn):
    nh, nl = N_BLOCH_MEAS + dn / 2, N_BLOCH_MEAS - dn / 2
    nn, LL = build_grating(nh, nl, nh, N_SIDE, half)
    lr, tr = find_defect_resonance(nn, LL, LAM)
    xx, EE = field_profile(nn, LL, lr)
    return nh, nl, lr, tr, xx, EE


lo_dn, hi_dn = 0.01, 0.25
for _ in range(30):
    dn = 0.5 * (lo_dn + hi_dn)
    _, _, _, _, xx, EE = model_at(dn)
    if envelope_fwhm(xx, EE) > FWHM_MEAS:
        lo_dn = dn
    else:
        hi_dn = dn
n_hi_c, n_lo_c, lam_r, T_r, x, E = model_at(dn)
w0 = lightcone_weight(x, E)
fw0 = envelope_fwhm(x, E)
print(f"calibrated: dn = {dn:.4f} (n {n_lo_c:.4f}/{n_hi_c:.4f}), resonance "
      f"{lam_r*1e9:.2f} nm (T={T_r:.3f})")
print(f"uniform stack: light-cone weight = {w0:.3e}, envelope fwhm = "
      f"{fw0*1e6:.2f} um (target {FWHM_MEAS*1e6:.1f})")

# --- model VALIDATION against a known FDTD outcome: distributing the pi-shift
# over the 4 innermost gaps/side made loss +21..39% WORSE in FDTD. The model
# must show increased light-cone weight there or its predictions are worthless.
gaps_dist = [half + half / 8] * 4
n_d, L_d = build_grating(n_hi_c, n_lo_c, n_hi_c, N_SIDE, half, gaps_dist)
# distributed variant: shrink the lumped cavity extra by what the gaps added
L_d[len(L_d) // 2] = half * 0 + 1e-12  # remove the lumped extra half-period
lr_d, tr_d = find_defect_resonance(n_d, L_d, lam_r, span=28e-9)
x_d, E_d = field_profile(n_d, L_d, lr_d)
w_d = lightcone_weight(x_d, E_d)
print(f"VALIDATION distributed pi-shift (4 gaps/side): res {lr_d*1e9:.2f} nm  "
      f"T={tr_d:.3f}  lc-weight {w_d:.3e} ({(w_d/w0-1)*100:+.0f}% vs uniform; "
      f"FDTD found +21..39% loss -> model must be positive here)")

for K in (4, 8, 16):
    for dnm in (10.0, 20.0, 40.0):
        d = dnm * 1e-9
        gaps = [half + (d / 2 if (j % 2 == 0) else -d / 2) for j in range(K)]
        n_l, L_l = build_grating(n_hi_c, n_lo_c, n_hi_c, N_SIDE, half, gaps)
        lr, tr = find_defect_resonance(n_l, L_l, lam_r)
        xd, Ed = field_profile(n_l, L_l, lr)
        wd = lightcone_weight(xd, Ed)
        fwd = envelope_fwhm(xd, Ed)
        print(f"  K={K:2d} delta={dnm:4.0f} nm: res {lr*1e9:8.2f} nm  T={tr:.3f}  "
              f"lc-weight {wd:.3e} ({(wd/w0-1)*100:+.1f}%)  "
              f"fwhm {fwd*1e6:6.2f} um ({(fwd/fw0-1)*100:+.1f}%)")
print("(1D proxy: light-cone weight of the internal field envelope stands in for")
print(" radiated power, same proxy as the k-space diagnostic. GO for the FDTD")
print(" dimerization study ONLY if some row cuts the weight >~10% at |fwhm")
print(" change| <~1%; otherwise paper-kill, do not dispatch.)")
