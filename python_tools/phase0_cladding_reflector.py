"""
PHASE-0 THEORY GATE (zero-GPU, 2026-07-08) — CLADDING REFLECTOR for the lateral
leak: air trench (TIR) vs 1D SiN/oxide Bragg mirror (DBR) vs 2D photonic-crystal
bandgap. Answers the user's questions before any FDTD:
  Q1 does a mirror in the cladding even reflect light coming FROM the cladding? yes
     (a dielectric mirror is reciprocal; it reflects the outbound leak back).
  Q2 why does a Bragg mirror NOT reflect like air, but a PhC bandgap might?
  Q3 didn't the sparse 'comb' scatterers already do this? (no -- wrong regime.)

KEY PHYSICS the numbers below quantify:
  The lateral leak hits a wall at y=d with a conserved tangential wavevector kx.
  Incidence angle on the wall: sin(theta) = |kx|/kc  (kc = n_clad*k0).
  * TIR (air trench): total reflection only for theta > theta_crit, i.e.
    |kx| > n_low*k0. So TIR catches only the GRAZING (high-kx) part of the leak
    and LETS THROUGH the more-normal (low-kx) part. Broadband in wavelength,
    but ANGLE-LIMITED to super-critical angles.
  * 1D Bragg/DBR (SiN/oxide layers parallel to the guide, periodic in y): reflects
    by interference within a STOPBAND of angles around its design point -- NO TIR
    (both indices >= incidence oxide, so kz is real in every layer; reflection is
    purely Bragg). Angle/wavelength selective; can be TUNED to a chosen band.
  * 2D PhC complete bandgap: if the leak frequency is in the gap, R=1 for ALL kx
    -> catches the WHOLE lateral leak, including the low-kx part TIR misses. THAT
    is the advantage over both TIR and the 1D DBR -- omnidirectional reflection.

This computes, from the MEASURED leak flux(kx), the reflected fraction of the
lateral leak for: air-TIR, PDMS-TIR, a quarter-wave SiN/oxide DBR (optimally
angle-tuned, N periods, exact TMM for p-pol/TM), and the PhC ceiling (=1.0).

Run:  python python_tools/phase0_cladding_reflector.py
"""

import os
import numpy as np
from scipy.io import loadmat

ROOT = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes"
MAT = os.path.join(ROOT, "results_from_athena", "tm_field_export", "results",
                   "result_N80_TM_W1050_dsh2S40s20_ptw2W1040to980_Ybox6p8_Zbox8p8_EZSLICE.mat")
N_CLAD = 1.444
N_SIN = 1.97
LOSS0 = 0.0545
INPLANE = 0.58        # measured in-plane share of the loss
NX = 1 << 13

m = loadmat(MAT, squeeze_me=True)
x = np.asarray(m["x"], float) * 1e6
y = np.asarray(m["y"], float) * 1e6
E = np.asarray(m["Ez_re"]) + 1j * np.asarray(m["Ez_im"])
lam = float(m["lam_used_nm"]) * 1e-3
k0 = 2 * np.pi / lam
kc = N_CLAD * k0

xu = np.linspace(x.min(), x.max(), NX)
dx = xu[1] - xu[0]
kxg = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, dx))
FT = lambda f: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f))) * dx
win = np.hanning(NX)


def leak_flux(y0):
    j = int(np.argmin(np.abs(y - y0)))
    f = np.interp(xu, x, E[:, j].real) + 1j * np.interp(xu, x, E[:, j].imag)
    A = FT(win * f)
    inc = np.abs(kxg) < kc
    ky = np.sqrt(np.maximum(kc**2 - kxg[inc]**2, 0.0))
    return kxg[inc], (ky / kc) * np.abs(A[inc])**2


y_edge = min(2.6, 0.85 * y.max())
kx1, fl1 = leak_flux(+y_edge)
kx2, fl2 = leak_flux(-y_edge)
kx = kx1
flux = 0.5 * (fl1 + fl2)
tot = float(np.sum(flux))
akx = np.abs(kx)


def caught(Rfun):
    R = Rfun(akx)
    return float(np.sum(R * flux)) / tot


# --- TIR reflectance (step at critical) ---
def tir(n_low):
    kcrit = n_low * k0
    return lambda a: (a > kcrit).astype(float)


# --- exact p-pol (TM) TMM for a quarter-wave SiN/oxide DBR embedded in oxide ---
def dbr_reflectance(N_periods, kx_design):
    """Return R(akx) for a stack (SiN,oxide)xN, quarter-wave at design angle kx_design,
    incidence & exit medium = oxide, p-polarization."""
    def kz(n, a):
        return np.sqrt((n * k0)**2 - a**2 + 0j)
    # quarter-wave physical thickness at the design tangential wavevector
    dH = np.pi / 2 / np.real(kz(N_SIN, kx_design))
    dL = np.pi / 2 / np.real(kz(N_CLAD, kx_design))
    layers = [(N_SIN, dH), (N_CLAD, dL)] * N_periods

    def eta_p(n, a):                      # p-pol tilted admittance ~ n^2 k0 / kz
        return (n**2) * k0 / kz(n, a)

    def R_of(a):
        a = np.atleast_1d(a).astype(float)
        out = np.empty_like(a)
        for i, av in enumerate(a):
            M = np.eye(2, dtype=complex)
            for (n, d) in layers:
                kzl = kz(n, av)
                dl = kzl * d
                eta = eta_p(n, av)
                Ml = np.array([[np.cos(dl), 1j * np.sin(dl) / eta],
                               [1j * eta * np.sin(dl), np.cos(dl)]])
                M = M @ Ml
            eta0 = eta_p(N_CLAD, av)       # oxide incidence
            etas = eta_p(N_CLAD, av)       # oxide exit (DBR embedded in oxide)
            B = M[0, 0] + M[0, 1] * etas
            C = M[1, 0] + M[1, 1] * etas
            r = (eta0 * B - C) / (eta0 * B + C)
            out[i] = np.abs(r)**2
        return out
    return R_of


print(f"lam={lam*1e3:.1f} nm  n_clad={N_CLAD}  n_SiN={N_SIN}  kc={kc:.3f}/um")
print(f"theta_crit(air)={np.degrees(np.arcsin(1.0/N_CLAD)):.1f} deg  "
      f"(TIR needs |kx|/kc > {1.0/N_CLAD:.3f})\n")

f_air = caught(tir(1.0))
f_pdms = caught(tir(1.40))
print("Reflected fraction of the LATERAL leak, and the loss it could recover")
print("(ceiling: assumes perfect re-coupling; in-plane share {:.0%}):".format(INPLANE))


def report(name, frac):
    dloss = LOSS0 * INPLANE * frac
    print(f"   {name:34s}: reflects {frac*100:5.1f}%  ->  dLoss up to +{dloss:.4f} "
          f"(loss {LOSS0} -> ~{LOSS0-dloss:.4f})")


report("air trench (TIR)", f_air)
report("PDMS-filled trench (TIR)", f_pdms)

# DBR: scan design angle to MAXIMISE caught flux, for two period counts
for Nper in (8, 16):
    best = (-1, None)
    for kd in np.linspace(0.05 * kc, 0.99 * kc, 40):
        fr = caught(dbr_reflectance(Nper, kd))
        if fr > best[0]:
            best = (fr, kd)
    report(f"1D SiN/oxide DBR (N={Nper}, best-tuned)", best[0])
    print(f"        (best design at |kx|/kc = {best[1]/kc:.2f})")

report("2D PhC COMPLETE bandgap (ceiling)", 1.0)

print(f"""
READING / VERDICTS:
 * Air-TIR catches only the GRAZING share (|kx|/kc > {1.0/N_CLAD:.2f}); the rest of the
   lateral leak passes THROUGH the trench. That grazing share is what the trench's
   measured -0.012 came from. TIR is broadband in wavelength but angle-limited.
 * A 1D DBR reflects a tunable stopband. If tuned to the near-normal (low-kx) leak
   it grabs exactly the part TIR MISSES -> DBR and trench are COMPLEMENTARY, not
   redundant. But one DBR can't cover all angles at once (its number above is the
   best single design).
 * A 2D PhC with a COMPLETE gap reflects ALL angles -> the full lateral channel
   ceiling (loss {LOSS0} -> as low as {LOSS0*(1-INPLANE):.4f} if it ALL re-coupled). That is why a
   PhC can beat both the 1D Bragg mirror and the air trench: omnidirectional.
 * HONEST CAVEATS: (i) reflected-FLUX is an UPPER BOUND; the loss drop depends on
   RE-COUPLING back into the localized mode, which is kx-WEIGHTED toward GRAZING
   (the grazing leak sits closest to the mode carrier beta, so it re-enters best).
   The trench realized 0.012 of its 0.0146 ceiling (~82%) BECAUSE it catches exactly
   that grazing part. A DBR/PhC that reflects more TOTAL flux but at low kx may
   re-couple that extra flux POORLY (low-kx returns are far from beta) -> the net
   can be less than the flux number suggests. This kx-weighting is the crux the
   FDTD must settle. (ii) a COMPLETE 2D gap usually needs n-ratio >~2; SiN/oxide is
   only 1.36 -> expect a PARTIAL / directional gap (a wide angular cone, not all
   angles) -> real PhC sits BETWEEN the DBR and the ceiling; (iii) the sparse 'comb'
   scatterers we tried were WEAK & dilute (no gap) -> that is why they failed, and
   why a DENSE high-contrast PhC is a different experiment.
""")
