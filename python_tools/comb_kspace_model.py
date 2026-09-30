"""k-space interference model of the cladding post comb on the TM pi-shift Bragg grating.

Study: results_from_athena/comb_physics_rethink/  (2026-09-11, physics-first rethink).
Purpose: ONE semi-analytic model that (a) reads the device's radiation leak from stored
near-field planes (unclipped, unlike the far-field monitors), (b) models the comb as a
row of z-dipoles driven by the measured cladding tail, (c) predicts dT for every stored
comb row, and only then (d) designs the next comb.  Zero GPU.

Conventions: x = propagation, y = lateral (in-plane), z = vertical.  Far-field direction
u = (ux, uy, uz), |k| = kc = n_clad*k0.  Azimuth psi around the x-axis: psi = 0 is
in-plane (+y), psi = 90 deg is vertical (+z).  For the grazing cone k_perp = kc*sqrt(1-ux^2).

Data (all MEASURED, produced by scratch extract_planes.py + the Opus FF extraction):
  data/*_PLANES_RES.npz   resonance slice of the 3D field planes (scat_i_fieldmaps job 123991)
  data/*.npz + manifest.csv   far fields + scalars of every comb row
"""
import os, glob
import numpy as np

# ---------------------------------------------------------------- constants (knobs)
LAM_NM = 1558.61                     # TM corr-400 N80 W800 resonance, box16 (job 123563)
N_CLAD, N_CORE = 1.444, 1.97
K0 = 2 * np.pi / (LAM_NM * 1e-3)     # rad/um
KC = N_CLAD * K0                     # light-cone edge, rad/um
CTRL_T, CTRL_LOSS = 0.8851, 0.1110   # stored control (job 123563)
FLOOR = 0.0018                       # opt-mesh jitter floor on T
DATA = os.path.join('results_from_athena', 'comb_physics_rethink', 'data')
CTRL_PLANES = 'result_N80_TM_avg_Ybox16p0_Zbox8p8_PLANES_RES.npz'
Y_LINES_UM = (1.2, 1.6, 2.0, 2.5, 3.0)   # cladding lines for the in-plane leak spectrum
Z_LINES_UM = (0.6, 0.9, 1.2, 1.6, 2.0)   # cladding lines above the core for the top spectrum
TIP_Y_UM = 0.5                            # wide-tooth tip (corr 400: wide 1000 nm)


def load_planes(name=CTRL_PLANES):
    return dict(np.load(os.path.join(DATA, name)))


def line_field(planes, plane, coord_um):
    """Complex Ez along x on the line ax2 = coord (both signs averaged: mode is even)."""
    ax1 = planes[plane + '_ax1'] * 1e6; ax2 = planes[plane + '_ax2'] * 1e6
    Ez = planes[plane + '_Ez']
    j = np.argmin(np.abs(ax2 - coord_um)); k = np.argmin(np.abs(ax2 + coord_um))
    return ax1, 0.5 * (Ez[:, j] + Ez[:, k]), 0.5 * (ax2[j] - ax2[k])


def resample_uniform(x, f, n=None):
    """Lumerical monitor grids are non-uniform (CLAUDE.md gotcha): resample before any FFT."""
    n = n or len(x)
    xu = np.linspace(x.min(), x.max(), n)
    return xu, np.interp(xu, x, f.real) + 1j * np.interp(xu, x, f.imag)


def spectrum(x, f, window=True):
    """Angular spectrum F(kx) = int f(x) e^{-i kx x} dx  (Tukey window on the outer 15%)."""
    xu, fu = resample_uniform(x, f)
    if window:
        w = np.ones_like(xu); m = 0.15 * (xu.max() - xu.min()); e = xu.min() + m
        sel = xu < e; w[sel] = 0.5 * (1 - np.cos(np.pi * (xu[sel] - xu.min()) / m))
        sel = xu > xu.max() - m; w[sel] = 0.5 * (1 - np.cos(np.pi * (xu.max() - xu[sel]) / m))
        fu = fu * w
    dx = xu[1] - xu[0]
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(xu), dx))
    F = np.fft.fftshift(np.fft.fft(fu)) * dx * np.exp(-1j * kx * xu.min())
    return kx, F


def k_perp(kx):
    return np.sqrt(np.clip(KC ** 2 - kx ** 2, 0, None))


def leak_spectra(planes, y_lines=Y_LINES_UM, z_lines=Z_LINES_UM):
    """In-cone leak amplitude referenced to the guide axis, from several cladding lines.

    On a homogeneous-cladding line at distance s from the axis every propagating kx
    component carries e^{+i k_perp s}; the guided tail (|kx|>kc) decays instead.  Back-
    propagating each line and comparing across lines separates the two: the propagating
    part must be line-independent.  Returns dict with kx grid, per-line back-propagated
    spectra for psi=0 (xy plane, lines y=s) and psi=90 (xz plane, lines z=s)."""
    out = {'kx': None, 'inplane': {}, 'top': {}}
    for plane, lines, key in (('xy', y_lines, 'inplane'), ('xz', z_lines, 'top')):
        for s in lines:
            x, f, s_act = line_field(planes, plane, s)
            kx, F = spectrum(x, f)
            out['kx'] = kx
            out[key][s_act] = F * np.exp(-1j * k_perp(kx) * s_act)
    return out


def in_cone_power(kx, F, ux_min=0.0, ux_max=1.0):
    """Radiated power proxy int |F|^2 * (k_perp/kc) dkx over a |ux| band (Englund-type k_z weight)."""
    u = kx / KC; m = (np.abs(u) >= ux_min) & (np.abs(u) < ux_max)
    return np.trapezoid(np.abs(F[m]) ** 2 * k_perp(kx[m]) / KC, kx[m])


if __name__ == '__main__':
    P = load_planes()
    print('ctrl planes: T %.4f slice %.3f nm (res %.3f), fwhm %.2f um' % (P['T'], P['lam_slice_nm'], P['lam_res_nm'], P['fwhm_um']))
    x = P['xy_ax1'] * 1e6; dx = np.diff(x)
    print('x grid: %d pts, dx min/median/max = %.4f/%.4f/%.4f um (uniform? %s)' % (len(x), dx.min(), np.median(dx), dx.max(), dx.max() - dx.min() < 1e-6))
    S = leak_spectra(P)
    kx = S['kx']; u = kx / KC
    print('kx bin %.4f rad/um = %.4f in ux; carrier beta=%.3f is %.2f bins from the cone edge' % (kx[1] - kx[0], (kx[1] - kx[0]) / KC, 1.5078 * K0, (1.5078 * K0 - KC) / (kx[1] - kx[0])))
    for key in ('inplane', 'top'):
        print('--- %s: back-propagated in-cone spectra, per line' % key)
        lines = sorted(S[key]); ref = S[key][lines[1]]
        for s in lines:
            F = S[key][s]
            tot = in_cone_power(kx, F)
            print('  line %.2f um: P_incone(rel to line %.2f) %.3f | share |ux|>0.9: %.3f  >0.95: %.3f  >0.977: %.3f | corr with ref (0.5<|ux|<0.95): %.3f' % (
                s, lines[1], tot / in_cone_power(kx, ref), in_cone_power(kx, F, 0.9) / tot, in_cone_power(kx, F, 0.95) / tot, in_cone_power(kx, F, 0.977) / tot,
                abs(np.vdot(ref[(np.abs(u) > 0.5) & (np.abs(u) < 0.95)], F[(np.abs(u) > 0.5) & (np.abs(u) < 0.95)])) / np.sqrt(np.vdot(ref[(np.abs(u) > 0.5) & (np.abs(u) < 0.95)], ref[(np.abs(u) > 0.5) & (np.abs(u) < 0.95)]).real * np.vdot(F[(np.abs(u) > 0.5) & (np.abs(u) < 0.95)], F[(np.abs(u) > 0.5) & (np.abs(u) < 0.95)]).real)))
        # profile of the middle line
        F = S[key][lines[2]]; m = (u > 0) & (u < 1)
        prof = np.abs(F[m]) ** 2 * k_perp(kx[m]) / KC
        print('  |ux| profile (line %.2f), normalised to max: ' % lines[2] + ' '.join('%.2f:%.3f' % (uu, pp) for uu, pp in zip(u[m][::6], prof[::6] / prof.max())))
    # in-plane vs top comparison at the middle lines
    Fi = S['inplane'][sorted(S['inplane'])[2]]; Ft = S['top'][sorted(S['top'])[2]]
    m = (np.abs(u) > 0.85) & (np.abs(u) < 0.99)
    print('top/in-plane amplitude ratio in 0.85<|ux|<0.99: median %.2f ; phase diff median %+.0f deg' % (
        np.median(np.abs(Ft[m]) / np.abs(Fi[m])), np.degrees(np.median(np.angle(Ft[m] / Fi[m])))))


# ================================================================ interference model
# Two channels radiate into the same cladding light cone: the device's own leak L(kx, psi)
# and the comb's beam C(kx, psi).  Radiated power P = int |L + C|^2 dOmega, so the change of
# the radiative decay rate is  dgamma/gamma = (2 Re<L,C> + <C,C>) / P_total, and the peak
# transmission follows from the symmetric two-port CMT  T = (1 - rho)^2, rho = gamma/(2kappa+gamma).
RHO_CTRL = 1 - np.sqrt(CTRL_T)             # 0.0593: gives R 0.0035, loss 0.1115 (both measured)
N_PSI = 72
DEPS_SIN, DEPS_AIR_IN_OX, DEPS_OX_IN_SIN, DEPS_AIR_IN_SIN = N_CORE**2 - N_CLAD**2, 1 - N_CLAD**2, N_CLAD**2 - N_CORE**2, 1 - N_CORE**2


def carrier_field(planes, y_um):
    """Guided-carrier part (|kx| > kc) of Ez on the cladding line y = y_um: the comb drive.
    Returns a callable Ez_drive(x_um) (complex), built from the stored control plane."""
    x, f, _ = line_field(planes, 'xy', y_um)
    xu, fu = resample_uniform(x, f)
    F = np.fft.fft(fu); kx = 2 * np.pi * np.fft.fftfreq(len(xu), xu[1] - xu[0])
    fc = np.fft.ifft(np.where(np.abs(kx) > KC, F, 0))
    return lambda xq: np.interp(xq, xu, fc.real) + 1j * np.interp(xq, xu, fc.imag)


def leak_source_spectrum(planes):
    """Needle proxy: FT of the axis field (volume-current formulation, no k_perp weight).
    Uniform in azimuth (the source is sub-wavelength in y, z).  Complex, on the FFT grid."""
    x, f, _ = line_field(planes, 'xy', 0.0)
    kx, F = spectrum(x, f)
    m = np.abs(kx) < KC
    return kx[m], F[m]


def comb_spectrum(kx, psi, sites, drive, alpha):
    """C(kx, psi) for posts at sites = [(x_um, y_um, weight)], weight ~ r^2 relative to 1.
    Each post is a z-dipole driven by the carrier at (x, y) (rows at +y and -y both drawn),
    radiating with the propagation phase e^{-i k_y y}, k_y = k_perp cos(psi)."""
    kp = k_perp(kx)[:, None]; cpsi = np.cos(psi)[None, :]
    C = np.zeros((len(kx), len(psi)), complex)
    for (xs, ys, w) in sites:
        a = alpha * w * drive(xs) * np.exp(-1j * kx * xs)
        C += a[:, None] * 2 * np.cos(kp * ys * cpsi)      # mirror rows +/-y, even mode
    return C


def dgamma_over_gamma(kx, L, C, f_needle):
    """(2 Re<L,C> + <C,C>) / P_total with P_total = P_needle / f_needle; dOmega = dkx dpsi / kc."""
    PL = np.sum(np.abs(L) ** 2) * (2 * np.pi)           # L uniform in psi
    X = 2 * np.sum((np.conj(L)[:, None] * C).real) * (2 * np.pi / N_PSI)
    PC = np.sum(np.abs(C) ** 2) * (2 * np.pi / N_PSI)
    return (X + PC) * f_needle / PL, X * f_needle / PL, PC * f_needle / PL


def T_from_dgamma(x, T0=CTRL_T):
    rho = 1 - np.sqrt(T0); rho2 = rho * (1 + x) / (1 + rho * x)
    return (1 - rho2) ** 2


def sites_from_row(row):
    """Post sites from a manifest row: x0..x1 step Lambda per row, rows at each y in y_list."""
    ys = [float(v) * 1e-3 for v in str(row['y_list']).split(';')]
    n_per = int(row['n_posts']) // len(ys)
    xs = np.linspace(float(row['x0_nm']), float(row['x1_nm']), n_per) * 1e-3
    r = row['r_nm']
    if 'to' in str(r):            # radius-apodized comb: r_j = r_max exp(-kappa |x_j| / 2)
        rmin, rmax = [float(v) for v in str(r).split('to')]
        w = (rmax * np.exp(-0.0446 * np.abs(xs) / 2)) ** 2 / 110.0 ** 2
    else:
        w = np.full(len(xs), (float(r) / 110.0) ** 2)
    return [(x, y, wi) for y in ys for x, wi in zip(xs, w)]


def predict_row(row, planes, alpha, f_needle, kxL=None, L=None, deps_scale=1.0):
    kxL, L = (kxL, L) if L is not None else leak_source_spectrum(planes)
    psi = np.linspace(0, 2 * np.pi, N_PSI, endpoint=False)
    ys = sorted(set(float(v) * 1e-3 for v in str(row['y_list']).split(';')))
    drives = {y: carrier_field(planes, y) for y in ys}
    sites = sites_from_row(row)
    C = np.zeros((len(kxL), N_PSI), complex)
    for y in ys:
        C += comb_spectrum(kxL, psi, [s for s in sites if abs(s[1] - y) < 1e-9], drives[y], alpha * deps_scale)
    x, X, PC = dgamma_over_gamma(kxL, L, C, f_needle)
    return T_from_dgamma(x), X, PC


# ---------------------------------------------------------------- leak azimuth profile
# The uniform-in-psi needle is refuted by the phase circles (the mirror-row factor
# J0(k_perp d) would flip the optimal phase between Lambda 536 and 545; measured: 270 deg
# for both).  The unclipped top line carries ~2x the in-plane amplitude in the wedge.
# So the needle gets an azimuth profile |L(psi)|^2 ~ (1 + b sin^2 psi), b fitted.
def leak_top_spectrum(planes, s_um=1.6):
    """Alternative needle proxy: the top-going radiated field on the line z = s (xz plane),
    back-propagated to z = 0 (k_perp weight belongs to line data: amplitude x sqrt(k_perp/kc))."""
    x, f, s_act = line_field(planes, 'xz', s_um)
    kx, F = spectrum(x, f)
    m = np.abs(kx) < KC
    return kx[m], F[m] * np.exp(-1j * k_perp(kx[m]) * s_act) * np.sqrt(k_perp(kx[m]) / KC)


def leak_with_profile(L, b):
    """L(kx) -> L(kx, psi) with |L|^2 ~ (1 + b sin^2 psi), normalised so the psi-mean is |L|^2."""
    psi = np.linspace(0, 2 * np.pi, N_PSI, endpoint=False)
    prof = np.sqrt((1 + b * np.sin(psi) ** 2) / (1 + b / 2))
    return L[:, None] * prof[None, :]


def dgamma_over_gamma_2d(L2, C, f_needle):
    PL = np.sum(np.abs(L2) ** 2) * (2 * np.pi / N_PSI)
    X = 2 * np.sum((np.conj(L2) * C).real) * (2 * np.pi / N_PSI)
    PC = np.sum(np.abs(C) ** 2) * (2 * np.pi / N_PSI)
    return (X + PC) * f_needle / PL, X * f_needle / PL, PC * f_needle / PL


def predict_row2(row, planes, alpha, f_needle, b, kxL, L, drives=None, deps_scale=1.0):
    psi = np.linspace(0, 2 * np.pi, N_PSI, endpoint=False)
    ys = sorted(set(float(v) * 1e-3 for v in str(row['y_list']).split(';')))
    drives = drives or {}
    for y in ys:
        if y not in drives: drives[y] = carrier_field(planes, y)
    sites = sites_from_row(row)
    C = np.zeros((len(kxL), N_PSI), complex)
    for y in ys:
        C += comb_spectrum(kxL, psi, [s for s in sites if abs(s[1] - y) < 1e-9], drives[y], alpha * deps_scale)
    x, X, PC = dgamma_over_gamma_2d(leak_with_profile(L, b), C, f_needle)
    return T_from_dgamma(x), X, PC


# ---------------------------------------------------------------- fast evaluation
# Precompute per row the psi-resolved overlaps once; then any (alpha, f_needle, b) is instant.
def row_basis(row, planes, kxL, L, drives, deps_scale=1.0):
    psi = np.linspace(0, 2 * np.pi, N_PSI, endpoint=False)
    ys = sorted(set(float(v) * 1e-3 for v in str(row['y_list']).split(';')))
    for y in ys:
        if y not in drives: drives[y] = carrier_field(planes, y)
    sites = sites_from_row(row)
    C = np.zeros((len(kxL), N_PSI), complex)
    for y in ys:
        C += comb_spectrum(kxL, psi, [s for s in sites if abs(s[1] - y) < 1e-9], drives[y], deps_scale)
    return {'a': np.sum(np.conj(L)[:, None] * C, axis=0),      # <L, C>(psi)  (alpha = 1)
            'p': np.sum(np.abs(C) ** 2, axis=0),                 # <C, C>(psi)
            'PL': np.sum(np.abs(L) ** 2)}


def eval_basis(B, alpha, f_needle, b, T0=CTRL_T):
    psi = np.linspace(0, 2 * np.pi, N_PSI, endpoint=False)
    prof = np.sqrt((1 + b * np.sin(psi) ** 2) / (1 + b / 2))
    PL = B['PL'] * np.sum(prof ** 2) * (2 * np.pi / N_PSI)
    X = 2 * np.sum((alpha * B['a'] * prof).real) * (2 * np.pi / N_PSI)
    PC = abs(alpha) ** 2 * np.sum(B['p']) * (2 * np.pi / N_PSI)
    x = (X + PC) * f_needle / PL
    return T_from_dgamma(x, T0), X * f_needle / PL, PC * f_needle / PL


def fit(bases, T_meas, b_fixed=None, T0=CTRL_T):
    """Fit |alpha|, arg(alpha), f_needle (and b) to measured T's.  Returns (alpha, f, b, rms)."""
    from scipy.optimize import minimize
    def cost(p):
        a = p[0] * np.exp(1j * p[1]); b = b_fixed if b_fixed is not None else abs(p[3])
        return sum((eval_basis(B, a, abs(p[2]), b, T0)[0] - t) ** 2 for B, t in zip(bases, T_meas))
    best = None
    for a0 in (0.03, 0.1, 0.3, 1.0, 3.0):
        for ph0 in np.linspace(-np.pi, np.pi, 6, endpoint=False):
            for b0 in ((0.5, 3.0, 20.0) if b_fixed is None else (None,)):
                x0 = [a0, ph0, 0.6] + ([] if b0 is None else [b0])
                res = minimize(cost, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-12, 'maxiter': 4000})
                if best is None or res.fun < best.fun: best = res
    p = best.x
    return p[0] * np.exp(1j * p[1]), abs(p[2]), (b_fixed if b_fixed is not None else abs(p[3])), np.sqrt(best.fun / len(bases))


# ---------------------------------------------------------------- fine-grid needle
# An 84-um window has a kx bin of 0.013 in ux, but the needle (Lorentzian tail of the
# e^{-kappa|x|} envelope, kappa = 0.045/um) is NARROWER than one bin at the horizon, which is
# where the comb beam sits.  Fix: keep the measured field inside the device, continue it
# outside with the pure guided wave (no in-cone content), and evaluate the Fourier sum
# directly on a fine kx grid.
L_DEVICE_UM = 2 * 80 * 0.51683 + 0.51683 / 2      # N=80/side + cavity: 83.0 um
BETA = 1.5078 * K0                                # carrier (kspace diag), rad/um
X_EXTEND_UM = 400.0
DU_FINE = 0.001


def fine_kx():
    u = np.arange(-1 + DU_FINE / 2, 1, DU_FINE)
    return u * KC


def continued_field(x, f, beta=BETA, L=L_DEVICE_UM, x_ext=X_EXTEND_UM):
    """Measured field inside |x| < L/2, guided plane-wave continuation outside, smooth
    taper at |x| ~ x_ext.  Returns (x, f) on a uniform grid."""
    xu, fu = resample_uniform(x, f)
    dx = xu[1] - xu[0]
    inside = np.abs(xu) <= L / 2
    xr, xl = xu[inside][-1], xu[inside][0]
    fr, fl = fu[inside][-1], fu[inside][0]
    x_right = np.arange(xr + dx, x_ext, dx); x_left = np.arange(xl - dx, -x_ext, -dx)[::-1]
    f_right = fr * np.exp(1j * beta * (x_right - xr)); f_left = fl * np.exp(-1j * beta * (x_left - xl))
    X = np.concatenate([x_left, xu[inside], x_right]); F = np.concatenate([f_left, fu[inside], f_right])
    taper = np.ones_like(X); m = np.abs(X) > 0.6 * x_ext
    taper[m] = 0.5 * (1 + np.cos(np.pi * (np.abs(X[m]) - 0.6 * x_ext) / (0.4 * x_ext)))
    return X, F * taper


def dft(x, f, kx):
    dx = x[1] - x[0]
    out = np.empty(len(kx), complex)
    for i in range(0, len(kx), 200):
        out[i:i + 200] = (np.exp(-1j * np.outer(kx[i:i + 200], x)) @ f) * dx
    return out


def leak_fine(planes, kxf=None, source='axis', s_um=1.6):
    """Needle proxy on the fine grid.  source='axis': FT of the axis field (volume-current
    picture); source='top': the top-going line at z = s, back-propagated (line picture)."""
    kxf = fine_kx() if kxf is None else kxf
    if source == 'axis':
        x, f, _ = line_field(planes, 'xy', 0.0); X, F = continued_field(x, f)
        return kxf, dft(X, F, kxf)
    x, f, s_act = line_field(planes, 'xz', s_um); X, F = continued_field(x, f)
    return kxf, dft(X, F, kxf) * np.exp(-1j * k_perp(kxf) * s_act) * np.sqrt(k_perp(kxf) / KC)


# ---------------------------------------------------------------- validation over stored rows
MANIFEST = os.path.join(DATA, 'manifest.csv')
T0_BY_STUDY = {'scat_air_comb': 0.8864, 'scat_x_incore': 0.8864, 'scat_q_r80phase': 0.8864}   # IGUM ctrl
T0_ACCURATE = 0.8786                     # scat_t accurate-mesh control (dx~35 nm)
CAL_STUDIES = ('scat_p_antineedle', 'scat_r_aim536')
VALID_STUDIES = ('scat_p_antineedle', 'scat_r_aim536', 'scat_s_refine', 'scat_w_dscan', 'scat_y_polish',
                 'scat_t_confirm', 'scat_air_comb', 'scat_x_incore', 'scat_q_r80phase', 'scat_o_comb1800')


def load_rows():
    import csv
    rows = list(csv.DictReader(open(MANIFEST)))
    out = []
    for r in rows:
        if r['study'] not in VALID_STUDIES or not r['r_nm'] or 'RECT' in r['r_nm']: continue
        note = r['note']
        if 'flush' in note or 'H12000' in note or 'W1050' in note or 'H4000' in note or 'H6000' in note: continue
        if int(r['n_posts']) <= 2: continue
        y = float(str(r['y_list']).split(';')[0]) * 1e-3
        r['T0'] = T0_BY_STUDY.get(r['study'], CTRL_T)
        r['deps'] = 1.0
        if 'hole' in note and y > 1.0: r['deps'] = DEPS_AIR_IN_OX / DEPS_SIN          # air post in oxide
        if 'hole' in note and y < 0.4: r['deps'] = (DEPS_OX_IN_SIN if r['study'] == 'scat_x_incore' else DEPS_AIR_IN_SIN) / DEPS_SIN
        r['accurate'] = (r['study'] == 'scat_t_confirm' and float(r['T']) < 0.8905 and r['dx_nm'] == '398.0' and r['n_posts'] == '31' and r['r_nm'] == '110')
        if r['accurate']: r['T0'] = T0_ACCURATE
        r['tag'] = 'L%s dx%s r%s n%s y%s%s' % (r['Lambda_nm'].replace('.0', ''), r['dx_nm'].replace('.0', ''), r['r_nm'], r['n_posts'], r['y_list'], (' ' + note.replace('Ybox16p0', '').strip(';')) if note.replace('Ybox16p0', '').strip(';') else '')
        out.append(r)
    return out


def validate(planes, source='axis', b_fixed=None, fit_studies=CAL_STUDIES, verbose=True):
    kxf, L = leak_fine(planes, source=source)
    rows = load_rows(); drives = {}
    bases = [row_basis(r, planes, kxf, L, drives, deps_scale=r['deps']) for r in rows]
    cal = [i for i, r in enumerate(rows) if r['study'] in fit_studies and r['r_nm'] == '110' and r['Lambda_nm'] in ('545.0', '536.0') and r['dx_nm'] != '929.0']
    alpha, f, b, rms_cal = fit([bases[i] for i in cal], [float(rows[i]['T']) for i in cal], b_fixed)
    table = []
    for i, (r, B) in enumerate(zip(rows, bases)):
        Tm, X, PC = eval_basis(B, alpha, f, b, r['T0'])
        table.append(dict(study=r['study'], tag=r['tag'], cal=(i in cal), T=float(r['T']), T0=r['T0'], dT_meas=float(r['T']) - r['T0'], dT_model=Tm - r['T0'], X=X, PC=PC, resid=Tm - float(r['T'])))
    pred = [t for t in table if not t['cal']]
    rms_pred = np.sqrt(np.mean([t['resid'] ** 2 for t in pred]))
    if verbose:
        print('source=%s  fit: |alpha| %.4g arg %+.1f deg  f_needle %.3f  b %.2f | cal rms %.4f (n=%d) | PREDICTION rms %.4f (n=%d), within 2x floor: %d/%d' % (
            source, abs(alpha), np.degrees(np.angle(alpha)), f, b, rms_cal, len(cal), rms_pred, len(pred), sum(abs(t['resid']) < 2 * FLOOR for t in pred), len(pred)))
        for t in table:
            print('  %s %-18s %-46s meas %+.4f model %+.4f resid %+.4f  (X %+.3f PC %.3f)' % ('C' if t['cal'] else ' ', t['study'], t['tag'][:46], t['dT_meas'], t['dT_model'], t['resid'], t['X'], t['PC']))
    return dict(alpha=alpha, f=f, b=b, rms_cal=rms_cal, rms_pred=rms_pred, table=table, kxf=kxf, L=L)


# ---------------------------------------------------------------- smooth analytic needle
# The continued-field proxy carries sign-flipping ripples (device-end radiation) that the
# comb beam averages over in reality but that kill a point-wise overlap.  Cleaner: the mode
# is A(x) [c+ e^{i beta x} + c- e^{-i beta x}] with A = A0 e^{-kappa|x|} (measured), whose
# in-cone Fourier tail is an exact Lorentzian.  c+/- are read from the axis field at x = 0.
def needle_lorentz(planes, kxf=None, kappa=None):
    kxf = fine_kx() if kxf is None else kxf
    x, f, _ = line_field(planes, 'xy', 0.0)
    xu, fu = resample_uniform(x, f)
    F = np.fft.fft(fu); kx = 2 * np.pi * np.fft.fftfreq(len(xu), xu[1] - xu[0])
    cp = np.fft.ifft(np.where(kx > KC, F, 0)); cm = np.fft.ifft(np.where(kx < -KC, F, 0))   # +beta / -beta carriers
    env = np.abs(cp) + np.abs(cm)
    if kappa is None:                                            # fit e^{-kappa|x|} on 2..12 um
        m = (np.abs(xu) > 2) & (np.abs(xu) < 12)
        kappa = -np.polyfit(np.abs(xu[m]), np.log(env[m]), 1)[0]
    i0 = np.argmin(np.abs(xu)); cplus, cminus = cp[i0], cm[i0]
    L = cplus * 2 * kappa / (kappa ** 2 + (kxf - BETA) ** 2) + cminus * 2 * kappa / (kappa ** 2 + (kxf + BETA) ** 2)
    return kxf, L, kappa, cplus, cminus


def validate2(planes, needle='lorentz', b_fixed=None, cal_sel=None, verbose=True):
    """Fit on cal_sel (function row -> bool) and predict the rest; needle in {'lorentz','axis','top'}."""
    if needle == 'lorentz':
        kxf, L, kappa, cplus, cminus = needle_lorentz(planes)
    else:
        kxf, L = leak_fine(planes, source=needle)
    rows = load_rows(); drives = {}
    bases = [row_basis(r, planes, kxf, L, drives, deps_scale=r['deps']) for r in rows]
    cal_sel = cal_sel or (lambda r: r['study'] in CAL_STUDIES and r['r_nm'] == '110' and r['Lambda_nm'] in ('545.0', '536.0'))
    cal = [i for i, r in enumerate(rows) if cal_sel(r)]
    alpha, f, b, rms_cal = fit([bases[i] for i in cal], [float(rows[i]['T']) for i in cal], b_fixed)
    table = []
    for i, (r, B) in enumerate(zip(rows, bases)):
        Tm, X, PC = eval_basis(B, alpha, f, b, r['T0'])
        eta = abs(np.sum(B['a'])) / np.sqrt(B['PL'] * np.sum(B['p']))
        table.append(dict(study=r['study'], tag=r['tag'], cal=(i in cal), T=float(r['T']), T0=r['T0'], dT_meas=float(r['T']) - r['T0'], dT_model=Tm - r['T0'], X=X, PC=PC, resid=Tm - float(r['T']), eta=eta, y=float(str(r['y_list']).split(';')[0]) * 1e-3))
    clad = [t for t in table if t['y'] > 1.0]
    pred = [t for t in clad if not t['cal']]
    rms_pred = np.sqrt(np.mean([t['resid'] ** 2 for t in pred]))
    if verbose:
        print('needle=%s  fit: |alpha| %.4g arg %+.1f deg  f_needle %.3f  b %.2f | cal rms %.4f (n=%d) | cladding-comb PREDICTION rms %.4f (n=%d), within 2x floor: %d/%d, within 3x: %d/%d' % (
            needle, abs(alpha), np.degrees(np.angle(alpha)), f, b, rms_cal, len(cal), rms_pred, len(pred), sum(abs(t['resid']) < 2 * FLOOR for t in pred), len(pred), sum(abs(t['resid']) < 3 * FLOOR for t in pred), len(pred)))
        for t in table:
            print('  %s %-16s %-44s meas %+.4f model %+.4f resid %+.4f  X %+.3f PC %.3f eta %.2f' % ('C' if t['cal'] else ' ', t['study'][:16], t['tag'][:44], t['dT_meas'], t['dT_model'], t['resid'], t['X'], t['PC'], t['eta']))
    return dict(alpha=alpha, f=f, b=b, rms_cal=rms_cal, rms_pred=rms_pred, table=table, kxf=kxf, L=L)


# ---------------------------------------------------------------- design scan
# For a geometry (Lambda, dx, N, d) the comb field scales linearly with the post amplitude
# a ~ r^2, so dgamma/gamma = (a X1 + a^2 PC1) f / PL has its optimum at a* = -X1/(2 PC1) with
# value -X1^2 f/(4 PC1 PL): the best T per geometry and the radius that reaches it, in one shot.
R_MIN_NM, R_MAX_NM = 55.0, 150.0          # mesh floor (dx 50 nm) / measured r>110 degradation


def geometry_row(Lambda_nm, dx_nm, n_posts, d_um, r_nm=110.0):
    half = (n_posts - 1) / 2
    return {'x0_nm': -half * Lambda_nm + dx_nm, 'x1_nm': half * Lambda_nm + dx_nm, 'n_posts': n_posts,
            'y_list': '%.0f' % (d_um * 1e3), 'r_nm': '%g' % r_nm, 'Lambda_nm': '%g' % Lambda_nm, 'dx_nm': '%g' % dx_nm}


def best_for_geometry(row, planes, kxf, L, drives, alpha, f, b, T0=CTRL_T):
    B = row_basis(row, planes, kxf, L, drives)
    _, X1, PC1 = eval_basis(B, alpha, f, b, T0)          # at r = 110 (a = 1)
    if X1 >= 0 or PC1 <= 0: return None
    a = -X1 / (2 * PC1); r = 110.0 * np.sqrt(a)
    r_c = min(max(r, R_MIN_NM), R_MAX_NM); a_c = (r_c / 110.0) ** 2
    x = a_c * X1 + a_c ** 2 * PC1
    return dict(r_opt=r, r=r_c, dT=T_from_dgamma(x, T0) - T0, dT_unclamped=T_from_dgamma(-X1 ** 2 / (4 * PC1), T0) - T0, X=a_c * X1, PC=a_c ** 2 * PC1)


def design_scan(planes, alpha, f, b, kxf, L, Lambdas=(524, 526, 528, 530, 531, 532, 534, 536), phases_deg=range(0, 360, 15),
                Ns=(21, 31, 41, 51, 61, 81, 101), ds=(1.0, 1.2, 1.4, 1.6, 1.8, 2.1)):
    drives = {}; out = []
    for d in ds:
        for N in Ns:
            for Lam in Lambdas:
                for ph in phases_deg:
                    row = geometry_row(Lam, ph / 360.0 * Lam, N, d)
                    res = best_for_geometry(row, planes, kxf, L, drives, alpha, f, b)
                    if res: out.append(dict(Lambda=Lam, phase=ph, N=N, d=d, **res))
    return out
