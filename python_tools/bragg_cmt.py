"""1D piecewise coupled-mode transfer-matrix engine for pi-shift Bragg gratings.

Method: Erdogan, "Fiber grating spectra", JLT 15, 1277 (1997) — piecewise-uniform
2x2 transfer matrices in the rotating (Bragg-synchronous) frame. Math ported from
the user's MATLAB @BraggGrating project (reimplemented fresh; that project's
loss/hybrid paths carry known bugs). New-engine justification (CLAUDE.md par.10):
no python TMM in this repo handles kappa(z) apodization + phase plates +
z-dependent loss + resonance envelopes (lateral_radiation_theory.py is a
replacement-topology slab model validated as a sign-oracle only).

Conventions (gated by _selftest below — run this file directly to verify):
- z increases left->right. R = forward field, S = backward field.
- delta(lambda) = 2*pi*n_eff/lambda - pi/Lambda   (first-order Bragg detuning)
- sigma = complex per-segment DC term added to delta; Im(sigma) > 0 == LOSS.
- Segment matrix maps [R;S] at the segment's left end to its right end;
  chain accumulates M_total = M_n @ ... @ M_1.
- r = -M21/M22, t = det(M)/M22 (det==1 for lossless segments).
- The pi shift is a physical extra half-period of unshaped waveguide at the
  cavity center: plate phase phi = (delta + sigma)*Lc + pi*Lc/Lambda.

CMT validity here: kappa*Lambda ~ 0.02 << 1 (slowly varying), first-order Bragg.
No dispersion: n_g == n_eff (same limitation as the MATLAB engine).
"""

import numpy as np

# -------------------------------------------------------------------------
# core segment math
# -------------------------------------------------------------------------

def segment_matrices(kappa, sigma_hat, dz):
    """2x2 transfer matrices for uniform grating segments, vectorized.

    kappa, sigma_hat, dz broadcast together; sigma_hat = delta + sigma is
    complex. Returns array (..., 2, 2). gamma^2 = kappa^2 - sigma_hat^2,
    branch-safe (cosh/sinh(x)/x are even / odd-analytic in gamma).
    """
    kappa = np.asarray(kappa, complex)
    sig = np.asarray(sigma_hat, complex)
    dz = np.asarray(dz, float)
    g = np.sqrt(kappa ** 2 - sig ** 2 + 0j)
    gdz = g * dz
    c = np.cosh(gdz)
    # sinh(g*dz)/g, stable at g->0
    small = np.abs(gdz) < 1e-9
    s_over_g = np.where(small, dz * (1.0 + gdz ** 2 / 6.0), np.sinh(gdz) / np.where(g == 0, 1, g))
    M = np.empty(np.broadcast(c, s_over_g).shape + (2, 2), complex)
    M[..., 0, 0] = c + 1j * sig * s_over_g
    M[..., 0, 1] = 1j * kappa * s_over_g
    M[..., 1, 0] = -1j * kappa * s_over_g
    M[..., 1, 1] = c - 1j * sig * s_over_g
    return M


def plate_matrix(phi):
    """Grating-free propagation plate: diag(exp(i*phi), exp(-i*phi))."""
    phi = np.asarray(phi, complex)
    P = np.zeros(phi.shape + (2, 2), complex)
    P[..., 0, 0] = np.exp(1j * phi)
    P[..., 1, 1] = np.exp(-1j * phi)
    return P


# -------------------------------------------------------------------------
# device description -> spectrum / envelope
# -------------------------------------------------------------------------
# A device is a list of elements, each ("seg", kappa_1_per_m, sigma_complex, length_m)
# or ("plate", extra_phase_const, length_m). Plates add phase
# (delta+sigma)*L + extra_phase_const at each wavelength (extra = pi*L/Lambda
# carrier mismatch for physical unshaped sections).

def pi_shift_device(kappa, n_periods, pitch, sigma=0.0, seg_periods=2,
                    kappa_profile=None):
    """Standard symmetric pi-shift grating: N periods each side + half-period
    cavity. kappa_profile: optional callable d -> kappa multiplier for tooth
    d = 1..N counted OUTWARD from the cavity on each side (apodization)."""
    elems = []
    Lc = 0.5 * pitch
    # left arm: teeth N..1 (outermost first, innermost last)
    for d_lo in range(n_periods, 0, -seg_periods):
        dd = list(range(d_lo, max(d_lo - seg_periods, 0), -1))
        k_eff = kappa * np.mean([kappa_profile(d) if kappa_profile else 1.0 for d in dd])
        elems.append(("seg", k_eff, sigma, len(dd) * pitch))
    elems.append(("plate", np.pi * Lc / pitch, Lc))
    for d_lo in range(1, n_periods + 1, seg_periods):
        dd = list(range(d_lo, min(d_lo + seg_periods, n_periods + 1)))
        k_eff = kappa * np.mean([kappa_profile(d) if kappa_profile else 1.0 for d in dd])
        elems.append(("seg", k_eff, sigma, len(dd) * pitch))
    return elems


def spectrum(elems, lam, n_eff, pitch):
    """t(lambda), r(lambda) for a device element list. lam in meters (array)."""
    lam = np.atleast_1d(np.asarray(lam, float))
    delta = 2 * np.pi * n_eff / lam - np.pi / pitch
    M = np.broadcast_to(np.eye(2, dtype=complex), lam.shape + (2, 2)).copy()
    for el in elems:
        if el[0] == "seg":
            _, k, sig, L = el
            F = segment_matrices(k, delta + sig, L)
        else:
            _, extra, L = el
            F = plate_matrix((delta + 0j) * L + extra)
        M = F @ M
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    t = det / M[..., 1, 1]
    r = -M[..., 1, 0] / M[..., 1, 1]
    return t, r


def envelope(elems, lam0, n_eff, pitch, points_per_seg=40):
    """Intensity envelope |R|^2+|S|^2 vs z at one wavelength, incident power 1.
    Returns (z, intensity)."""
    delta = 2 * np.pi * n_eff / lam0 - np.pi / pitch
    t, r = spectrum(elems, np.array([lam0]), n_eff, pitch)
    f = np.array([1.0 + 0j, complex(r[0])])
    zs, Is = [0.0], [abs(f[0]) ** 2 + abs(f[1]) ** 2]
    z = 0.0
    for el in elems:
        if el[0] == "seg":
            _, k, sig, L = el
            n_sub = max(2, int(points_per_seg))
            for _ in range(n_sub):
                F = segment_matrices(k, delta + sig, L / n_sub)
                f = F @ f
                z += L / n_sub
                zs.append(z)
                Is.append(abs(f[0]) ** 2 + abs(f[1]) ** 2)
        else:
            _, extra, L = el
            F = plate_matrix(delta * L + extra)
            f = F @ f
            z += L
            zs.append(z)
            Is.append(abs(f[0]) ** 2 + abs(f[1]) ** 2)
    return np.asarray(zs), np.asarray(Is)


def find_resonance(elems, n_eff, pitch, lam_lo, lam_hi, n_grid=513, rounds=14,
                   min_rel_span=None):
    """Iteratively zoom on the transmission peak; returns (lam0, T0, fwhm_m_spectral).
    Zooms until the FWHM is resolved by >= 15 grid points."""
    lo, hi = float(lam_lo), float(lam_hi)
    for _ in range(rounds):
        lam = np.linspace(lo, hi, n_grid)
        t, _ = spectrum(elems, lam, n_eff, pitch)
        T = np.abs(t) ** 2
        j = int(np.argmax(T))
        # FWHM on this grid (interpolated crossings), if resolved
        half = T[j] / 2.0
        above = T >= half
        if 0 < j < n_grid - 1 and above.sum() >= 15 and above[0] == False and above[-1] == False:
            l = j
            while l > 0 and T[l - 1] >= half:
                l -= 1
            rgt = j
            while rgt < n_grid - 1 and T[rgt + 1] >= half:
                rgt += 1
            x1 = np.interp(half, [T[l - 1], T[l]], [lam[l - 1], lam[l]])
            x2 = np.interp(half, [T[rgt + 1], T[rgt]], [lam[rgt + 1], lam[rgt]])
            return lam[j], T[j], abs(x2 - x1)
        span = (hi - lo) / 8.0
        lo, hi = lam[j] - span / 2, lam[j] + span / 2
    return lam[j], T[j], np.nan


def fwhm_of(z, I):
    """Interpolated FWHM (half of peak) of a sampled profile."""
    j = int(np.argmax(I))
    half = I[j] / 2.0
    l = j
    while l > 0 and I[l] >= half:
        l -= 1
    r = j
    while r < len(I) - 1 and I[r] >= half:
        r += 1
    if l == 0 or r == len(I) - 1:
        return np.nan  # truncated at device edge
    z1 = np.interp(half, [I[l], I[l + 1]], [z[l], z[l + 1]])
    z2 = np.interp(half, [I[r], I[r - 1]], [z[r], z[r - 1]])
    return abs(z2 - z1)


def alpha_from_qi(q_i, lam0, n_eff):
    """Uniform FIELD loss sigma (imag part) equivalent to intrinsic Q_i.
    Power loss alpha_p = 2*pi*n_g/(lambda*Q_i), n_g == n_eff here; field decay
    = alpha_p/2."""
    return 0.5 * 2 * np.pi * n_eff / (lam0 * q_i)


# -------------------------------------------------------------------------
# selftest gates — every gate can fail; deliberate-failure checks included
# -------------------------------------------------------------------------

def _selftest():
    rng = dict(n_eff=1.52, pitch=516.83e-9, kappa=0.0353e6)  # corr-325-like
    n_eff, pitch, kappa = rng["n_eff"], rng["pitch"], rng["kappa"]
    lamD = 2 * n_eff * pitch
    lam = np.linspace(lamD - 8e-9, lamD + 8e-9, 1601)

    # G1: lossless unitarity |t|^2+|r|^2 == 1
    dev = pi_shift_device(kappa, 80, pitch)
    t, r = spectrum(dev, lam, n_eff, pitch)
    u = np.abs(t) ** 2 + np.abs(r) ** 2
    assert np.max(np.abs(u - 1)) < 1e-9, f"G1 unitarity: {np.max(np.abs(u-1)):.2e}"

    # G2: subdivision invariance on an ASYMMETRIC device (catches ordering bugs)
    def apod(d):
        return 0.3 + 0.7 * min(d, 10) / 10.0
    L = 40 * pitch
    one = [("seg", kappa, 0.0, L), ("seg", 0.6 * kappa, 0.0, 2 * L)]
    many = [("seg", kappa, 0.0, L / 8)] * 8 + [("seg", 0.6 * kappa, 0.0, L / 4)] * 8
    t1, r1 = spectrum(one, lam, n_eff, pitch)
    t2, r2 = spectrum(many, lam, n_eff, pitch)
    assert np.max(np.abs(t1 - t2)) < 1e-9 and np.max(np.abs(r1 - r2)) < 1e-9, "G2 subdivision"

    # G2b: deliberate failure — reversed accumulation must NOT match on the
    # asymmetric device (a gate that cannot fail proves nothing)
    def spectrum_reversed(elems, lam):
        delta = 2 * np.pi * n_eff / lam - np.pi / pitch
        M = np.broadcast_to(np.eye(2, dtype=complex), lam.shape + (2, 2)).copy()
        for el in elems:
            F = segment_matrices(el[1], delta + el[2], el[3])
            M = M @ F  # WRONG order
        det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
        return det / M[..., 1, 1], -M[..., 1, 0] / M[..., 1, 1]
    tw, rw = spectrum_reversed(one, lam)
    assert np.max(np.abs(rw - r1)) > 1e-6, "G2b reversed-order check is toothless"

    # G3: loss sign — uniform alpha must LOWER peak T and keep T+R < 1
    dev_lossy = pi_shift_device(kappa, 80, pitch, sigma=1j * 500.0)  # 1000/m power loss
    tl, rl = spectrum(dev_lossy, lam, n_eff, pitch)
    Tl, Rl = np.abs(tl) ** 2, np.abs(rl) ** 2
    T0 = np.abs(t) ** 2
    assert Tl.max() < T0.max() and np.all(Tl + Rl < 1 + 1e-12), "G3 loss sign"
    # G3b: deliberate failure — flipped sign must ADD energy (gain)
    dev_gain = pi_shift_device(kappa, 80, pitch, sigma=-1j * 500.0)
    tg, rg = spectrum(dev_gain, lam, n_eff, pitch)
    assert (np.abs(tg) ** 2 + np.abs(rg) ** 2).max() > 1 + 1e-6, "G3b gain check toothless"

    # G4: lossless symmetric pi-shift resonance has T == 1 inside the gap
    lam0, Tpk, fw = find_resonance(dev, n_eff, pitch, lamD - 4e-9, lamD + 4e-9)
    assert Tpk > 1 - 1e-6, f"G4 pi-shift T=1: {Tpk}"
    assert abs(lam0 - lamD) < 0.2e-9, f"G4 resonance near lamD: {(lam0-lamD)*1e9:.3f} nm"

    # G5: Q_c growth ratio -> exp(2*kappa*pitch*dN) ASYMPTOTICALLY (large kappa*L;
    # at small kappa*L the full TMM is right and the exponential law is not —
    # measured ladders live in the crossover, which is exactly why the engine
    # beats the bare law there)
    q = {}
    for N in (200, 210):
        d = pi_shift_device(kappa, N, pitch)
        l0, Tp, fwq = find_resonance(d, n_eff, pitch, lamD - 4e-9, lamD + 4e-9)
        q[N] = l0 / fwq
    ratio = q[210] / q[200]
    expect = np.exp(2 * kappa * pitch * 10)
    assert abs(ratio / expect - 1) < 0.01, f"G5 Qc ratio {ratio:.3f} vs {expect:.3f}"

    # G6: envelope symmetric, decays, FWHM ~ ln2/kappa for large-N uniform
    dev120 = pi_shift_device(kappa, 120, pitch)
    l0, _, _ = find_resonance(dev120, n_eff, pitch, lamD - 4e-9, lamD + 4e-9)
    z, I = envelope(dev120, l0, n_eff, pitch)
    zc = z[int(np.argmax(I))]
    Ltot = z[-1]
    assert abs(zc - Ltot / 2) < 2 * pitch, "G6 envelope peak at center"
    fw_env = fwhm_of(z, I)
    expect_fw = np.log(2) / kappa
    assert abs(fw_env / expect_fw - 1) < 0.10, f"G6 envelope FWHM {fw_env*1e6:.2f} vs ln2/k {expect_fw*1e6:.2f} um"

    # G7: reciprocity — t identical for the flipped asymmetric device
    one_flip = one[::-1]
    t3, _ = spectrum(one_flip, lam, n_eff, pitch)
    assert np.max(np.abs(t3 - t1)) < 1e-9, "G7 reciprocity"

    print("bragg_cmt selftest: all gates pass",
          f"(G5 Qc ratio {ratio:.4f} vs {expect:.4f}; G6 env FWHM {fw_env*1e6:.3f} um)")


if __name__ == "__main__":
    _selftest()
