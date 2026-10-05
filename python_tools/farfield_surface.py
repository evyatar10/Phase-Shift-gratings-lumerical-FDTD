"""Far field of a waveguide device from E and H on ONE closed tube around it.

Created 2026-10-05. Replaces the two independent planar projections (top / side
monitor, Lumerical farfieldvector3d) that disagreed where both should be valid
(seam at 45 deg: power ratio 1.3 TE / 2.3 TM, docs/radiation_cancellation_review_2026-10-05.pdf):
a planar projection assumes the monitor catches all the light of its half space,
and ours sit ~2 wavelengths from the axis with a +-3 wavelength aperture.

Input: a result .mat with surface_top / surface_side (complex E, H on the top
(z = z0) and side (y = y0) monitors at the resonance; FarFieldConfig.save_surface_eh).
Method: the two monitors cross; the parts inside the crossing, |y| <= y0 on the
top and |z| <= z0 on the side, plus their mirror images (-z, -y; parity read from
the data) form a rectangular tube around the waveguide: closed in cross-section,
OPEN at its two x ends (a tapered open lateral surface, not a closed box). Surface equivalence in the uniform cladding (time dependence e^{-iwt}):
    J = n x H,  M = -n x E,   N, L = int (J, M) exp(-i k r_hat . r') dS
    E_far(r_hat) = (i k / 4 pi) [ eta (N - r_hat (r_hat . N)) - r_hat x L ]   (x e^{ikr}/r)
One coherent sum over the four faces -> no stitching seam. The guided wave still
crosses the open x ends: a cosine taper over the last TAPER_UM of each end keeps
its k_x = beta > k content outside the light cone (vary TAPER_UM to see what the
grazing band is worth). Directions within ~atan(y0 / half x-span) of the x axis
leave through the open ends and are not captured.

Checks printed per file: Poynting flux through the tube vs integrated far-field
power (agree to ~1 % for a mild taper; not an exact identity once tapered) and both
vs the port loss 1 - T - R (what leaves outside the tube's x span is not in either).
Valid for a uniform cladding of index N_CLAD and a y/z mirror-symmetric device.
Self-test (analytic dipole, no FDTD):  python python_tools/farfield_surface.py --selftest
"""

import sys

import numpy as np
import scipy.io as sio

N_CLAD   = 1.444
ETA0     = 376.730313668
TAPER_UM = 15.0            # cosine taper at each x end of the tube
X_STEP   = 2               # use every X_STEP-th x sample (dx 50 nm -> 100 nm, k_x Nyquist 31 /um)


def _trapz_w(t):
    w = np.zeros_like(t); d = np.diff(t)
    w[:-1] += d / 2; w[1:] += d / 2
    return w


def _taper(x, taper_m):
    """1 in the middle, cos^2 roll-off to 0 over taper_m at both ends."""
    if taper_m <= 0:
        return np.ones_like(x)
    d = np.minimum(x - x[0], x[-1] - x) / taper_m
    return np.where(d < 1, np.sin(0.5 * np.pi * np.clip(d, 0, 1)) ** 2, 1.0)


def _parity(F, t, M):
    """Sign s with F(x, -t) = s * M F(x, t) for a polar vector field sampled on (x, t)."""
    Fm = np.stack([np.stack([np.interp(-t, t, F[i, :, c].real) + 1j * np.interp(-t, t, F[i, :, c].imag)
                             for c in range(3)], -1) for i in range(0, F.shape[0], 8)])
    Fo = F[::8] * np.asarray(M)[None, None, :]
    c = np.sum(np.conj(Fo) * Fm).real / np.sum(np.abs(Fo) ** 2)
    return float(np.sign(c)), abs(c)


def tube_faces(top, side):
    """Four faces of the tube as dicts(x, t, E, H, n, axis). top/side: x, y|z (m), E, H (nx, nt, 3), pos (m)."""
    z0, y0 = float(top["pos"]), float(side["pos"])
    s_y, pur_y = _parity(top["E"], top["t"], [1, -1, 1])      # y -> -y, read on the top monitor
    s_z, pur_z = _parity(side["E"], side["t"], [1, 1, -1])    # z -> -z, read on the side monitor
    assert min(pur_y, pur_z) > 0.9, f"mirror parity not clean (y {pur_y:.2f}, z {pur_z:.2f}): the -y / -z faces cannot be rebuilt"
    assert abs(top["t"]).max() >= y0 and abs(side["t"]).max() >= z0, "monitors do not cross: no closed cross-section"
    faces = []
    for mon, half, ax_t, ax_n, s, Mdiag in ((top, y0, 1, 2, s_z, [1, 1, -1]), (side, z0, 2, 1, s_y, [1, -1, 1])):
        # crop to the crossing, with the two end samples INTERPOLATED onto the corner (+-half):
        # keeping only existing samples leaves a strip open at each corner (1.6 % in the field)
        k = np.abs(mon["t"]) < half * (1 - 1e-9)
        t = np.r_[-half, mon["t"][k], half]
        x = mon["x"][::X_STEP]
        E, H = (np.stack([np.stack([np.interp(t, mon["t"], F[i, :, c].real) + 1j * np.interp(t, mon["t"], F[i, :, c].imag)
                                    for c in range(3)], -1) for i in range(0, F.shape[0], X_STEP)])
                for F in (mon["E"], mon["H"]))
        n = np.zeros(3); n[ax_n] = 1.0
        faces.append(dict(x=x, t=t, E=E, H=H, n=n, ax_t=ax_t, ax_n=ax_n, pos=float(mon["pos"])))
        # mirror image face: polar E -> s M E, axial H -> -s M H; outward normal flips
        Md = np.asarray(Mdiag, float)
        faces.append(dict(x=x, t=t, E=s * E * Md, H=-s * H * Md, n=-n, ax_t=ax_t, ax_n=ax_n, pos=-float(mon["pos"])))
    return faces, dict(s_y=s_y, s_z=s_z, parity_purity=(pur_y, pur_z), y0=y0, z0=z0)


def tube_flux(faces, taper_m=0.0):
    """Outward Poynting flux through the tube (W)."""
    P = 0.0
    for f in faces:
        S = 0.5 * np.real(np.cross(f["E"], np.conj(f["H"])) @ f["n"])
        P += np.sum(S * (_trapz_w(f["x"]) * _taper(f["x"], taper_m))[:, None] * _trapz_w(f["t"])[None, :])
    return P


def tube_farfield(faces, dirs, lam_m, taper_m=TAPER_UM * 1e-6, n_clad=N_CLAD):
    """Complex far-field vector E_far * r * e^{-ikr} (V) for unit directions dirs (D, 3)."""
    k = 2 * np.pi * n_clad / lam_m; eta = ETA0 / n_clad
    D = dirs.shape[0]
    N = np.zeros((D, 3), complex); L = np.zeros((D, 3), complex)
    for f in faces:
        J = np.cross(f["n"], f["H"]); M = -np.cross(f["n"], f["E"])
        wx = _trapz_w(f["x"]) * _taper(f["x"], taper_m); wt = _trapz_w(f["t"])
        Pt = (np.exp(-1j * k * np.outer(f["t"], dirs[:, f["ax_t"]])) * wt[:, None])          # (nt, D)
        ph = np.exp(-1j * k * dirs[:, f["ax_n"]] * f["pos"])                                   # (D,)
        for lo in range(0, D, 2048):                                                          # chunk the directions
            sl = slice(lo, lo + 2048)
            Px = np.exp(-1j * k * np.outer(f["x"], dirs[sl, 0])) * wx[:, None]                # (nx, d)
            for src, acc in ((J, N), (M, L)):
                T = np.einsum("itc,td->icd", src, Pt[:, sl], optimize=True)                    # (nx, 3, d)
                acc[sl] += (np.einsum("icd,id->dc", T, Px, optimize=True) * ph[sl, None])
    Nperp = N - dirs * np.sum(dirs * N, -1, keepdims=True)
    return (1j * k / (4 * np.pi)) * (eta * Nperp - np.cross(dirs, L))


def load_surfaces(mat):
    """surface_top / surface_side structs of a result .mat -> the dicts tube_faces() expects."""
    out = []
    for key, tname in (("surface_top", "y"), ("surface_side", "z")):
        s = mat[key]
        out.append(dict(x=np.asarray(s.x, float), t=np.asarray(getattr(s, tname), float),
                        E=np.asarray(s.E), H=np.asarray(s.H), pos=float(s.pos), lam=float(s.lam),
                        source_power=float(s.source_power)))
    return out


def sphere_field(mat, lmax, taper_m=TAPER_UM * 1e-6):
    """Same return as farfield_multipole.full_sphere_field, from the tube surfaces of a loaded .mat."""
    from farfield_multipole import sphere_grid
    top, side = load_surfaces(mat)
    faces, info = tube_faces(top, side)
    x, w, phi = sphere_grid(lmax)
    ct = x[:, None]; st = np.sqrt(1 - x**2)[:, None]; one = np.ones((x.size, phi.size))
    n = np.stack([st * np.cos(phi)[None, :], st * np.sin(phi)[None, :], ct * one], -1)
    E = tube_farfield(faces, n.reshape(-1, 3), top["lam"], taper_m).reshape(n.shape)
    th = np.stack([ct * np.cos(phi)[None, :], ct * np.sin(phi)[None, :], -st * one], -1)
    ph = np.stack([-np.sin(phi)[None, :] * one, np.cos(phi)[None, :] * one, 0 * one], -1)
    E_th = np.sum(th * E, -1); E_ph = np.sum(ph * E, -1)
    dO = w[:, None] * (2 * np.pi / phi.size) * one
    I = np.abs(E_th) ** 2 + np.abs(E_ph) ** 2
    eta = ETA0 / N_CLAD
    P_far = np.sum(I * dO) / (2 * eta); P_tube = tube_flux(faces, taper_m); P_src = top["source_power"]
    half_x = 0.5 * (top["x"][-1] - top["x"][0])
    end_cone = np.abs(n[..., 0]) > np.cos(np.arctan(max(info["y0"], info["z0"]) / half_x))
    info.update(transversality=np.sum(np.abs(np.sum(n * E, -1)) ** 2) / np.sum(np.abs(E) ** 2),
                P_total=np.sum(I * dO), grazing_fraction=np.sum((I * dO)[end_cone]) / np.sum(I * dO),
                P_far_over_tube=P_far / P_tube, tube_loss=P_tube / P_src, far_loss=P_far / P_src)
    return x, w, phi, E_th, E_ph, info


# ── self-test: analytic dipole in the uniform cladding ────────────────────────


def _dipole_fields(r, r0, p, k, n_clad=N_CLAD):
    """Exact E, H of an electric dipole p (C m) at r0, e^{-iwt}, medium index n_clad (Jackson 9.18)."""
    eps = 8.8541878128e-12 * n_clad**2; c = 299792458.0 / n_clad
    R = r - r0; d = np.linalg.norm(R, axis=-1, keepdims=True); nh = R / d
    g = np.exp(1j * k * d)
    nxp = np.cross(nh, p)
    E = (k**2 * np.cross(nxp, nh) * g / d + (3 * nh * np.sum(nh * p, -1, keepdims=True) - p) * (1 / d**3 - 1j * k / d**2) * g) / (4 * np.pi * eps)
    H = c * k**2 / (4 * np.pi) * nxp * g / d * (1 - 1 / (1j * k * d))
    return E, H


def selftest():
    lam = 1.56e-6; k = 2 * np.pi * N_CLAD / lam
    eps = 8.8541878128e-12 * N_CLAD**2
    x = np.arange(-40e-6, 40e-6 + 1e-12, 50e-9); t = np.arange(-3.4e-6, 3.4e-6 + 1e-12, 50e-9)
    z0 = y0 = 2.1737e-6        # deliberately between grid samples (the real monitors are: 2.156 vs 2.121 um)
    rng = np.random.default_rng(0); v = rng.normal(size=(400, 3)); dirs = v / np.linalg.norm(v, axis=1)[:, None]
    dirs = dirs[np.abs(dirs[:, 0]) < 0.9]
    ok = True
    # mirror-symmetric sources (the tool rebuilds the -y / -z faces by parity): a y-dipole at the
    # centre, a y-dipole pair shifted in x, and a z-dipole (other parity class), like TE / TM
    for label, srcs in (("p_y at origin", [((0, 0, 0), (0, 1, 0))]),
                        ("p_y pair at x = +0.7 / -1.3 um", [((0.7e-6, 0, 0), (0, 1, 0)), ((-1.3e-6, 0, 0), (0, 0.6, 0))]),
                        ("p_z at x = 0.4 um", [((0.4e-6, 0, 0), (0, 0, 1))])):
        X, Y = np.meshgrid(x, t, indexing="ij")
        rt = np.stack([X, Y, z0 + 0 * X], -1); rs = np.stack([X, y0 + 0 * X, Y], -1)
        Et = Ht = Es = Hs = 0; exact = 0
        for r0, p in srcs:
            r0 = np.array(r0, float); p = np.array(p, float)
            e, h = _dipole_fields(rt, r0, p, k); Et = Et + e; Ht = Ht + h
            e, h = _dipole_fields(rs, r0, p, k); Es = Es + e; Hs = Hs + h
            exact = exact + k**2 / (4 * np.pi * eps) * np.cross(np.cross(dirs, p), dirs) * np.exp(-1j * k * dirs @ r0)[:, None]
        faces, info = tube_faces(dict(x=x, t=t, E=Et, H=Ht, pos=z0), dict(x=x, t=t, E=Es, H=Hs, pos=y0))
        got = tube_farfield(faces, dirs, lam, taper_m=10e-6)
        err = np.sqrt(np.sum(np.abs(got - exact) ** 2) / np.sum(np.abs(exact) ** 2))
        c = np.sum(np.conj(exact) * got) / np.sum(np.abs(exact) ** 2)
        print(f"  {label:32s} relative error {err:.4f}  complex scale {c.real:+.4f}{c.imag:+.4f}j  parities s_y={info['s_y']:+.0f} s_z={info['s_z']:+.0f}")
        ok &= err < 0.03
    # known-bad form must fail: flipping the sign of M (wrong equivalence) cannot reproduce the dipole
    for f in faces:
        f["E"] = -f["E"]
    bad = tube_farfield(faces, dirs, lam, taper_m=10e-6)
    err_bad = np.sqrt(np.sum(np.abs(bad - exact) ** 2) / np.sum(np.abs(exact) ** 2))
    print(f"  known-bad (E sign flipped)       relative error {err_bad:.2f}  (must be large)")
    ok &= err_bad > 0.5
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    for path in sys.argv[1:]:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        wl = np.asarray(m["wl_nm"], float); order = np.argsort(wl)
        loss_ports = float(np.interp(float(m["surface_top"].lam) * 1e9, wl[order], np.asarray(m["loss"], float)[order]))   # 1-T-R at the far-field wavelength
        for tap in (5.0, 10.0, 15.0, 20.0):
            *_, info = sphere_field(m, 40, tap * 1e-6)
            print(f"{path}\n  taper {tap:4.1f} um: far/tube power {info['P_far_over_tube']:.3f} | loss: tube {info['tube_loss']*100:.2f}%  far {info['far_loss']*100:.2f}%  ports {loss_ports*100:.2f}%"
                  f" | end-cone share {info['grazing_fraction']*100:.1f}%  s_y={info['s_y']:+.0f} s_z={info['s_z']:+.0f}")
