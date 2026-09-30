"""Vector-spherical-harmonic (multipole) content of a stored FDTD far field.

Study: runners/sweeps/farfield_sph_20um.py   |   Created 2026-09-29
Usage:  python python_tools/farfield_multipole.py result_..._ff.mat [more.mat ...]
        [--lmax 200] [--csv]     (--csv writes <mat>_multipoles.csv next to each file)

Input: a result .mat with farfield_top / farfield_side (complex Ex_c/Ey_c/Ez_c on
a direction-cosine grid ux/uy, Lumerical farfieldvector3d at 1 m). Only the
resonance-projected far field is meaningful (farfield_freq_points > 1 runs).

Method
  1. Full sphere from the two half-space monitors. A planar monitor's projection
     is valid only in the half space in front of it, so each direction n takes
     the monitor whose normal is closest: top (+z) where |nz| >= |ny|, side (+y)
     elsewhere. The device is mirror-symmetric in y and z, so the -z / -y halves
     are the mirror images E(Mn) = s * M E(n), M = diag(1,1,-1) or diag(1,-1,1);
     the sign s (mode parity) is READ from the data, not assumed (top monitor
     gives s_y, side monitor gives s_z). Directions near the +-x axis (both |ny|
     and |nz| small) are in neither monitor's good half space -> reported as the
     "grazing band" power fraction, never hidden.
  2. Transverse vector-harmonic projection (Jackson 9.7 basis):
        E(n) = sum_lm  a_M(l,m) X_lm(n) + a_E(l,m) n x X_lm(n),
        X_lm = L Y_lm / sqrt(l(l+1)),   a = <basis, E> over the sphere,
     orthonormal, so the power fraction of one multipole is
        (|a_E|^2 + |a_M|^2) / sum_all.   (Parseval vs int|E|^2 dOmega is printed.)
     Fully-normalized associated Legendre functions by the standard three-term
     recurrence, phi by FFT; l up to --lmax (kR of the radiating region: ~100 for
     a 20 um mode at 1.56 um, so 200 is generous -- the tail past l~150 is the
     convergence check).
Output per file: Parseval ratio, grazing-band fraction, the parity signs, power
vs l (E and M), top multipoles, and cumulative fractions for l <= 1, 2, 5, 10.
"""

import argparse
import os
import sys

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator

# ── sphere sampling ────────────────────────────────────────────────────────────


def sphere_grid(lmax):
    """Gauss-Legendre in cos(theta) x uniform phi; exact for products of degree 2*lmax."""
    n_th = 2 * lmax + 2
    x, w = np.polynomial.legendre.leggauss(n_th)          # x = cos(theta)
    n_ph = 2 * lmax + 2
    phi = np.arange(n_ph) * 2 * np.pi / n_ph
    return x, w, phi


def _interp(grid, ux_q, uy_q):
    """Complex bilinear interpolation of the three components at query (ux, uy)."""
    out = []
    for key in ("Ex_c", "Ey_c", "Ez_c"):
        f = np.asarray(grid[key])
        re = RegularGridInterpolator((grid["ux"], grid["uy"]), f.real, bounds_error=False, fill_value=0.0)
        im = RegularGridInterpolator((grid["ux"], grid["uy"]), f.imag, bounds_error=False, fill_value=0.0)
        pts = np.stack([ux_q, uy_q], axis=-1)
        out.append(re(pts) + 1j * im(pts))
    return np.stack(out, axis=-1)                          # (..., 3) global (x, y, z)


def parity_sign(grid, flip_axis):
    """Mirror parity of a monitor's own data: E(u_a -> -u_a) vs M E. Returns +1/-1."""
    u1, u2 = np.asarray(grid["ux"]), np.asarray(grid["uy"])
    E = np.stack([np.asarray(grid[k]) for k in ("Ex_c", "Ey_c", "Ez_c")], axis=-1)   # (n1, n2, 3)
    if flip_axis == 1:                                     # flip second grid axis
        Ef = E[:, ::-1, :] if np.allclose(u2, -u2[::-1]) else None
    else:
        Ef = E[::-1, :, :] if np.allclose(u1, -u1[::-1]) else None
    if Ef is None:
        raise ValueError("far-field grid is not symmetric about 0 — cannot read parity")
    return E, Ef


def full_sphere_field(top, side, lmax):
    """Sample E on the quadrature sphere from the two monitors (+ mirror symmetry).

    top : Z-normal monitor, ux -> x, uy -> y, nz = +sqrt(1-ux^2-uy^2)
    side: Y-normal monitor, ux -> x, uy -> z, ny = +sqrt(1-ux^2-uz^2)   (Lumerical)
    """
    x, w, phi = sphere_grid(lmax)
    ct = x[:, None]; st = np.sqrt(1 - x**2)[:, None]
    nx = st * np.cos(phi)[None, :]; ny = st * np.sin(phi)[None, :]; nz = ct * np.ones_like(phi)[None, :]

    # parity signs from each monitor's own symmetric half: s_y from top (uy -> -uy), s_z from side (uz -> -uz)
    def _sign(grid, axis, M):
        E, Ef = parity_sign(grid, axis)
        ME = Ef * np.asarray(M)[None, None, :]
        num = np.sum(np.conj(E) * ME); den = np.sum(np.abs(E) ** 2)
        return num / den
    cy = _sign(top, 1, [1, -1, 1]); cz = _sign(side, 1, [1, 1, -1])
    s_y = float(np.sign(cy.real)); s_z = float(np.sign(cz.real))

    use_top = np.abs(nz) >= np.abs(ny)
    E = np.zeros(nx.shape + (3,), dtype=complex)
    # top monitor: query (ux, uy) = (nx, |ny|*sign) with nz mirrored to +
    Et = _interp(top, nx, ny)
    Et_m = Et * np.array([1, 1, -1])[None, None, :] * s_z                       # for nz < 0
    E[use_top & (nz >= 0)] = Et[use_top & (nz >= 0)]
    E[use_top & (nz < 0)] = Et_m[use_top & (nz < 0)]
    # side monitor: query (ux, uz) with ny mirrored to +
    Es = _interp(side, nx, nz)
    Es_m = Es * np.array([1, -1, 1])[None, None, :] * s_y                       # for ny < 0
    E[~use_top & (ny >= 0)] = Es[~use_top & (ny >= 0)]
    E[~use_top & (ny < 0)] = Es_m[~use_top & (ny < 0)]

    n_vec = np.stack([nx, ny, nz], axis=-1)
    transv = np.sum(np.abs(np.sum(n_vec * E, axis=-1)) ** 2) / np.sum(np.abs(E) ** 2)   # should be ~0
    # spherical components
    th_hat = np.stack([ct * np.cos(phi)[None, :], ct * np.sin(phi)[None, :], -st * np.ones_like(phi)[None, :]], -1)
    ph_hat = np.stack([-np.sin(phi)[None, :] * np.ones_like(ct), np.cos(phi)[None, :] * np.ones_like(ct), np.zeros_like(nx)], -1)
    E_th = np.sum(th_hat * E, -1); E_ph = np.sum(ph_hat * E, -1)
    grazing = (np.abs(ny) < 0.1) & (np.abs(nz) < 0.1)
    dOmega = w[:, None] * (2 * np.pi / phi.size)
    P_tot = np.sum((np.abs(E_th) ** 2 + np.abs(E_ph) ** 2) * dOmega)
    P_graz = np.sum(((np.abs(E_th) ** 2 + np.abs(E_ph) ** 2) * dOmega)[grazing])
    info = dict(s_y=s_y, s_z=s_z, parity_purity=(abs(cy), abs(cz)), transversality=transv,
                P_total=P_tot, grazing_fraction=P_graz / P_tot)
    return x, w, phi, E_th, E_ph, info


# ── normalized associated Legendre + multipole projection ─────────────────────


def legendre_norm(lmax, m, x):
    """Fully normalized P_lm(x) for l = m..lmax (Condon-Shortley phase), shape (lmax-m+1, len(x)).
    int |P_lm(cos th) e^{im phi}|^2 dOmega = 1."""
    x = np.asarray(x); sx = np.sqrt(1 - x**2)
    P = np.zeros((lmax - m + 1, x.size))
    pmm = np.full(x.size, 1 / np.sqrt(4 * np.pi))
    for k in range(1, m + 1):
        pmm = -np.sqrt((2 * k + 1) / (2 * k)) * sx * pmm
    P[0] = pmm
    if lmax > m:
        P[1] = np.sqrt(2 * m + 3) * x * pmm
    for l in range(m + 2, lmax + 1):
        a = np.sqrt((4 * l * l - 1) / (l * l - m * m))
        b = np.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
        P[l - m] = a * (x * P[l - m - 1] - b * P[l - m - 2])
    return P


def multipoles(x, w, phi, E_th, E_ph, lmax):
    """a_E[l, m], a_M[l, m] (arrays (lmax+1, 2*lmax+1), m index = m + lmax)."""
    n_ph = phi.size
    F_th = np.fft.fft(E_th, axis=1) * (2 * np.pi / n_ph)   # int E_th e^{-i m phi} dphi, m = fft freq
    F_ph = np.fft.fft(E_ph, axis=1) * (2 * np.pi / n_ph)
    sx = np.sqrt(1 - x**2); cot = x / sx
    aE = np.zeros((lmax + 1, 2 * lmax + 1), dtype=complex)
    aM = np.zeros_like(aE)
    for m_abs in range(0, lmax + 1):
        P = legendre_norm(lmax, m_abs, x)                              # l = m_abs..lmax
        P1 = np.zeros_like(P)
        if m_abs + 1 <= lmax:                                          # P_{l, m+1} for the derivative
            Pn = legendre_norm(lmax, m_abs + 1, x)
            P1[1:] = Pn
        ls = np.arange(m_abs, lmax + 1)[:, None]
        # d/dtheta P_lm = m cot P_lm + sqrt((l-m)(l+m+1)) P_{l,m+1}   (normalized, CS phase)
        dP = m_abs * cot[None, :] * P + np.sqrt((ls - m_abs) * (ls + m_abs + 1)) * P1
        for m in ({m_abs, -m_abs}):
            sgn = 1.0 if m >= 0 else (-1.0) ** m_abs                   # Y_{l,-m} = (-1)^m conj(Y_lm)
            Pm, dPm = sgn * P, sgn * dP                                # real functions of theta
            fth = F_th[:, m % n_ph]; fph = F_ph[:, m % n_ph]           # (n_th,)
            mm = m
            # a_M = int conj(X_lm).E,  X = [-(m/sin) Y th - i dY ph]/sqrt(l(l+1))
            # a_E = int conj(n x X).E, n x X = [ i dY th - (m/sin) Y ph]/sqrt(l(l+1))
            norm = 1 / np.sqrt(ls * (ls + 1)).ravel() if m_abs > 0 else np.r_[np.inf, 1 / np.sqrt(ls[1:] * (ls[1:] + 1)).ravel()]
            norm = np.where(np.isfinite(norm), norm, 0.0)
            aM_l = norm * np.sum(w[None, :] * (-(mm / sx)[None, :] * Pm * fth[None, :] + 1j * dPm * fph[None, :]), axis=1)
            aE_l = norm * np.sum(w[None, :] * (-1j * dPm * fth[None, :] - (mm / sx)[None, :] * Pm * fph[None, :]), axis=1)
            aM[m_abs:, m + lmax] = aM_l
            aE[m_abs:, m + lmax] = aE_l
    return aE, aM


def analyze(mat_path, lmax=200, write_csv=False):
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    def _grid(s):
        return {k: np.asarray(getattr(s, k)) for k in ("ux", "uy", "Ex_c", "Ey_c", "Ez_c")}
    top, side = _grid(m["farfield_top"]), _grid(m["farfield_side"])
    lam_ff = float(m["farfield_top"].lam) * 1e9
    lam_res = float(m["resonance_wavelength_nm"]); fwhm = abs(float(m["spectral_fwhm_nm"]))
    x, w, phi, E_th, E_ph, info = full_sphere_field(top, side, lmax)
    aE, aM = multipoles(x, w, phi, E_th, E_ph, lmax)
    P = np.abs(aE) ** 2 + np.abs(aM) ** 2
    P_l = P.sum(axis=1); PE_l = (np.abs(aE) ** 2).sum(1); PM_l = (np.abs(aM) ** 2).sum(1)
    tot = P.sum()
    ls = np.arange(lmax + 1)
    print(f"\n=== {os.path.basename(mat_path)}")
    print(f"  far field at {lam_ff:.4f} nm | resonance {lam_res:.4f} nm | offset {abs(lam_ff-lam_res)/fwhm*100:.0f}% of the {fwhm:.3f} nm linewidth")
    print(f"  parity s_y={info['s_y']:+.0f} (purity {info['parity_purity'][0]:.3f})  s_z={info['s_z']:+.0f} (purity {info['parity_purity'][1]:.3f})"
          f"  transversality |n.E|^2/|E|^2 = {info['transversality']:.2e}")
    print(f"  Parseval sum|a|^2 / int|E|^2 = {tot/info['P_total']:.4f}   grazing band (|ny|,|nz|<0.1) = {info['grazing_fraction']*100:.1f}% of power")
    print(f"  electric {PE_l.sum()/tot*100:.1f}%  magnetic {PM_l.sum()/tot*100:.1f}%   power-weighted <l> = {np.sum(ls*P_l)/tot:.1f}"
          f"   tail l>{int(0.75*lmax)}: {P_l[int(0.75*lmax):].sum()/tot*100:.2f}%")
    for L in (1, 2, 5, 10, 20, 50, 100):
        if L <= lmax:
            print(f"  l <= {L:3d}: {P_l[:L+1].sum()/tot*100:6.2f}%")
    order = np.argsort(P.ravel())[::-1][:12]
    print("  top multipoles (type l m  %):")
    for k in order:
        l, mi = divmod(k, 2 * lmax + 1); mm = mi - lmax
        t = "E" if abs(aE[l, mi]) >= abs(aM[l, mi]) else "M"
        print(f"    {t} {l:3d} {mm:+4d}  {P[l, mi]/tot*100:6.2f}%")
    if write_csv:
        out = os.path.splitext(mat_path)[0] + "_multipoles.csv"
        with open(out, "w") as fh:
            fh.write("l,m,frac_E,frac_M\n")
            for l in range(lmax + 1):
                for mi in range(2 * lmax + 1):
                    if P[l, mi] / tot > 1e-6:
                        fh.write(f"{l},{mi-lmax},{abs(aE[l,mi])**2/tot:.6e},{abs(aM[l,mi])**2/tot:.6e}\n")
        print(f"  csv -> {out}")
    return dict(aE=aE, aM=aM, P_l=P_l, info=info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mats", nargs="+")
    ap.add_argument("--lmax", type=int, default=200)
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()
    for p in a.mats:
        analyze(p, a.lmax, a.csv)
