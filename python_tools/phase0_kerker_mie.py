"""
PHASE-0 (4.2) — Kerker/Huygens directional-scatterer Mie analysis. Zero GPU.

Question: can a SiN scatterer (n=1.97, in oxide n=1.444, at lam~1556.6 nm)
be sized so it scatters DIRECTIONALLY (generalized Kerker: interference of
its multipoles), with contrast >= 3:1, at scattering strength comparable to
the validated plain cylinders (r = 100/150/200 nm, tm_scatterer_scan)?

Model hierarchy (honest about what each captures):
  * 2D infinite-cylinder Mie, normal incidence — the RIGHT in-plane model:
    the pillar is vertical, the leak travels in-plane. Device-TM leak keeps
    E || z (measured f_TE ~ 0) = E parallel to the cylinder axis -> Case I
    (monopole b0 + dipole b1 + ... interference). The TE-device leak has E
    in-plane -> Case II (a_n).  Coefficients derived from Ez/Hz + tangential
    continuity; validated below against the Rayleigh limits and the 2D
    lossless unitarity relation |b|^2 = -Re(b).
  * 3D sphere Mie dipoles a1/b1 — compact-particle cross-check (the first
    Kerker condition a1=b1 in its native form).
  * Finite height (350 nm) truncation shifts everything -> the FDTD radius
    scan brackets the model radius; the model sets the EXISTENCE and the
    bracket, not the final nm.

Run:  python python_tools/phase0_kerker_mie.py
"""

import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp

LAM = 1.5566        # um
N_HOST = 1.444
N_CYL = 1.97
M = N_CYL / N_HOST
K = 2 * np.pi * N_HOST / LAM
NMAX = 14


def coeffs_caseI(x, m):
    """E parallel to cylinder axis (TM_z). total = incident + scattered."""
    n = np.arange(0, NMAX + 1)
    num = m * jvp(n, m * x) * jv(n, x) - jv(n, m * x) * jvp(n, x)
    den = jv(n, m * x) * h1vp(n, x) - m * jvp(n, m * x) * hankel1(n, x)
    return num / den


def coeffs_caseII(x, m):
    """E perpendicular to axis (Hz formulation)."""
    n = np.arange(0, NMAX + 1)
    num = jvp(n, m * x) * jv(n, x) / m - jv(n, m * x) * jvp(n, x)
    den = jv(n, m * x) * h1vp(n, x) - jvp(n, m * x) * hankel1(n, x) / m
    return num / den


def pattern(b, theta):
    """far-field amplitude f(theta), theta=0 = forward."""
    f = b[0] * np.ones_like(theta, dtype=complex)
    for n in range(1, len(b)):
        f += 2 * b[n] * np.cos(n * theta)
    return f


def qsca(b, x):
    s = np.abs(b[0])**2 + 2 * np.sum(np.abs(b[1:])**2)
    return 4 * s / x


# ------------------------------------------------------------- validation
print("=" * 74)
print("Coefficient validation (small x = Rayleigh; lossless unitarity)")
print("=" * 74)
x_t = 0.05
bI = coeffs_caseI(x_t, M)
bII = coeffs_caseII(x_t, M)
b0_ray = -1j * np.pi * x_t**2 * (M**2 - 1) / 4
a1_ray = -1j * np.pi * x_t**2 / 4 * (M**2 - 1) / (M**2 + 1)
print(f"Case I  b0: {bI[0]:.3e}  vs Rayleigh {b0_ray:.3e}")
print(f"Case II a1: {bII[1]:.3e}  vs Rayleigh {a1_ray:.3e}")
x_t = 2.0
for tag, b in (("I", coeffs_caseI(x_t, M)), ("II", coeffs_caseII(x_t, M))):
    u = np.max(np.abs(np.abs(b[:6])**2 + np.real(b[:6])))
    print(f"Case {tag:2s} unitarity max| |b|^2 + Re b | = {u:.2e} (lossless -> ~0)")

# ------------------------------------------------------------- radius scan
th = np.array([0.0, np.pi])
radii = np.arange(0.04, 0.85, 0.005)
print()
print("=" * 74)
print("Radius scan — directionality (F=forward, B=backward) and strength")
print("=" * 74)
for tag, fn, pol in (("Case I  (device-TM in-plane leak, E||axis)", coeffs_caseI, "TM"),
                     ("Case II (device-TE in-plane leak, E-perp)", coeffs_caseII, "TE")):
    rows = []
    for a in radii:
        x = K * a
        b = fn(x, M)
        f = pattern(b, th)
        F, B = np.abs(f[0])**2, np.abs(f[1])**2
        rows.append((a, qsca(b, x), F, B))
    rows = np.array(rows)
    a_, q_, F_, B_ = rows.T
    q150 = q_[np.argmin(np.abs(a_ - 0.150))] * 0.150   # strength ~ Qsca * a
    print(f"\n--- {tag}")
    for crit, name in ((B_ / np.maximum(F_, 1e-30), "max BACKWARD:FORWARD"),
                       (F_ / np.maximum(B_, 1e-30), "max FORWARD:BACKWARD")):
        i = int(np.argmax(crit))
        rel = (q_[i] * a_[i]) / q150
        print(f"  {name}: {crit[i]:6.2f}:1  at r = {a_[i]*1e3:4.0f} nm "
              f"(x={K*a_[i]:.2f}, Qsca={q_[i]:.3f}, strength/plain-150 = {rel:.2f})")
    # contrast achievable with strength >= 0.5x plain-150 (usable amplitude)
    strong = (q_ * a_) >= 0.5 * q150
    bf = np.where(strong, B_ / np.maximum(F_, 1e-30), 0)
    i = int(np.argmax(bf))
    print(f"  best B:F with strength >= 0.5x plain-150: {bf[i]:.2f}:1 at "
          f"r = {a_[i]*1e3:.0f} nm (Qsca {q_[i]:.3f})")
    # reference values at the plain radii
    for r0 in (0.100, 0.150, 0.200):
        i = int(np.argmin(np.abs(a_ - r0)))
        print(f"  plain r={r0*1e3:.0f} nm: Qsca {q_[i]:.3f}  F/B {F_[i]/max(B_[i],1e-30):.2f}"
              f"  B/F {B_[i]/max(F_[i],1e-30):.2f}")
    if pol == "TM":
        a_tm, q_tm, F_tm, B_tm = a_, q_, F_, B_

# multipole anatomy at the best backward point (Case I)
print()
print("=" * 74)
print("Multipole anatomy near the Case-I backward optimum")
print("=" * 74)
bf = B_tm / np.maximum(F_tm, 1e-30)
i = int(np.argmax(bf))
b = coeffs_caseI(K * a_tm[i], M)
print(f"r = {a_tm[i]*1e3:.0f} nm: |b0| {abs(b[0]):.3f}  |b1| {abs(b[1]):.3f}  "
      f"|b2| {abs(b[2]):.3f}  |b3| {abs(b[3]):.3f}")
print(f"  arg(b1/b0) = {np.degrees(np.angle(b[1]/b[0])):+.0f} deg  "
      f"arg(b2/b0) = {np.degrees(np.angle(b[2]/b[0])):+.0f} deg")
print("  (2D generalized Kerker: b0 vs b1 anti-phase kills one direction)")
# angular pattern quality: power fraction into the backward half-plane
tha = np.linspace(0, np.pi, 361)
f = pattern(b, tha)
Pb = np.trapezoid(np.abs(f[tha > np.pi / 2])**2, tha[tha > np.pi / 2])
Pt = np.trapezoid(np.abs(f)**2, tha)
print(f"  backward-half-plane power fraction at optimum: {Pb/Pt*100:.0f}%")

# ------------------------------------------------------------- sphere check
print()
print("=" * 74)
print("3D sphere dipole cross-check (first Kerker condition a1 = b1)")
print("=" * 74)


def mie_ab1(x, m):
    """sphere dipole coefficients a1 (ED), b1 (MD)."""
    mx = m * x
    psi = lambda n, z: z * np.sqrt(np.pi / (2 * z)) * jv(n + 0.5, z)
    xi = lambda n, z: z * np.sqrt(np.pi / (2 * z)) * (jv(n + 0.5, z) + 1j * (-1)**1 *
                                                      np.sqrt(1) * 0)  # placeholder
    # do it with explicit spherical bessels
    from scipy.special import spherical_jn, spherical_yn
    jn = lambda n, z: spherical_jn(n, z)
    jnp = lambda n, z: spherical_jn(n, z, derivative=True)
    yn = lambda n, z: spherical_yn(n, z)
    ynp = lambda n, z: spherical_yn(n, z, derivative=True)
    hn = lambda n, z: jn(n, z) + 1j * yn(n, z)
    hnp = lambda n, z: jnp(n, z) + 1j * ynp(n, z)
    psi_ = lambda n, z: z * jn(n, z)
    psip = lambda n, z: jn(n, z) + z * jnp(n, z)
    xi_ = lambda n, z: z * hn(n, z)
    xip = lambda n, z: hn(n, z) + z * hnp(n, z)
    n = 1
    a1 = (m * psi_(n, mx) * psip(n, x) - psi_(n, x) * psip(n, mx)) / \
         (m * psi_(n, mx) * xip(n, x) - xi_(n, x) * psip(n, mx))
    b1 = (psi_(n, mx) * psip(n, x) - m * psi_(n, x) * psip(n, mx)) / \
         (psi_(n, mx) * xip(n, x) - m * xi_(n, x) * psip(n, mx))
    return a1, b1


best_k = None
for a in np.arange(0.05, 0.75, 0.005):
    x = K * a
    a1, b1 = mie_ab1(x, M)
    F = abs(a1 + b1)**2
    B = abs(a1 - b1)**2
    r = F / max(B, 1e-30)
    if best_k is None or r > best_k[0]:
        best_k = (r, a, a1, b1)
r, a, a1, b1 = best_k
print(f"max dipole F:B = {r:.1f}:1 at sphere r = {a*1e3:.0f} nm "
      f"(|a1| {abs(a1):.3f}, |b1| {abs(b1):.3f})   [and B:F by phase flip at "
      f"other radii]")

print("""
NOTES / honesty:
 * Case I is the device-TM in-plane physics (E||z). Directionality there
   comes from b0-b1 (monopole-dipole) interference — the 2D analog of
   Kerker. Case II (TE device) has the a1-a2 route.
 * Finite 350-nm height truncates the cylinder -> resonance positions shift;
   treat the model radius as a BRACKET CENTER for the FDTD radius scan
   (+-30% span), not a prediction in nm.
 * Directionality only re-aims the secondary source: the CEILING stays the
   placement-map number from phase0_greens_overlap.py (one pair ~ +0.008 T,
   two pairs ~ +0.016 T on the stack). Kerker's value = approaching that
   ceiling with FEWER parasitic side effects (less forward drain).
""")
