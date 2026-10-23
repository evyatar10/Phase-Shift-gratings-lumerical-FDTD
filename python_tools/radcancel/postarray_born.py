# postarray_born.py (study: runners/sweeps/farfield_sph_15um_postarray_overshoot, 2026-10-23; doc docs/postarray_radiation_cancellation_2026-10-23.md)
# Born/dipole model of side posts. Drive = control field back-propagated from the tube side plane (y0 = 2.104 um) to the
# post plane; post = point dipole (cylinder polarisability); far field = free-oxide dipole radiation; same normalisation as the control C.
# Gate: predicted B vs the 8 measured comb rows (overlap, amplitude, P/Pc). Then the "any-dipole" ceiling over a candidate region.
import sys, os, numpy as np, scipy.io as sio
sys.path.insert(0, r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/python_tools")
from farfield_multipole import sphere_grid
S = os.path.dirname(os.path.abspath(__file__)); ctrl = os.environ.get("CTRL_MAT", r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/results_from_athena/comb_overshoot/control_nt80x_T340.mat")
WORK = os.environ.get("PA_WORK", r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/results_from_athena/comb_overshoot/postarray_work"); os.makedirs(WORK, exist_ok=True)
C0 = 299792458.0; N_OX, N_P = 1.444, 1.97; EO, EP = N_OX**2, N_P**2; H_POST = 350e-9
src = open(r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/python_tools/radcancel/solve_v4.py", encoding="utf-8").read()
m = sio.loadmat(ctrl, squeeze_me=True, struct_as_record=False)
SHORT = "trace_time_cavity" in m          # short ring-down run (overshoot) vs full steady run (plain TE controls, 2026-10-23)
sys.argv = ["x"] + (["--short"] if SHORT else []) + ["--energy", ctrl]; g = {"__name__": "x"}; exec(src.split("rows = [load(p)")[0], g)
lam = float(np.ravel(m["surface_top"].lam)[0]); w_ff = 2 * np.pi * C0 / lam; k = 2 * np.pi * N_OX / lam
U = np.trapezoid(np.ravel(m["field_energy_density_1D"]), np.ravel(m["field_x"]))
if SHORT:
    om, kap, t, Ec, comp = g["trace_fit"](m["trace_time_cavity"]); D = np.trapz(Ec * np.exp(1j * w_ff * t), t)
    NORM = np.conj(D) / abs(D) / np.sqrt(U)
else:   # same factor solve_v4.load applies to the steady far field: (1 + i det) / sqrt(U (1 + det_fp^2))
    lam0 = float(m["resonance_wavelength_nm"]) * 1e-9; Qs = lam0 / abs(float(m["spectral_fwhm_nm"]) * 1e-9); det = 2 * Qs * (lam - lam0) / lam0
    wl = np.ravel(m["wl_m"]); f_fp = np.linspace(C0 / wl.max(), C0 / wl.min(), 501); f_fp = f_fp[np.argmin(np.abs(f_fp - C0 / lam0))]
    det_fp = 2 * Qs * (C0 / f_fp - lam0) / lam0; NORM = (1 + 1j * det) / np.sqrt(U * (1 + det_fp ** 2))
side = m["surface_side"]; xs0 = np.ravel(side.x); zs0 = np.ravel(side.z); y0 = float(side.pos); Es0 = np.asarray(side.E)   # (nx, nz, 3)
# resample onto a UNIFORM (x, z) grid before the FFT (GPT 01a165e7: stored grids are non-uniform; 0.5-0.8 % drive error)
xs = np.linspace(xs0[0], xs0[-1], xs0.size); zs = np.linspace(zs0[0], zs0[-1], zs0.size)
from scipy.interpolate import RegularGridInterpolator as _RGI
_XZ = np.stack(np.meshgrid(xs, zs, indexing="ij"), -1)
Es = np.stack([_RGI((xs0, zs0), Es0[:, :, c].real)(_XZ) + 1j * _RGI((xs0, zs0), Es0[:, :, c].imag)(_XZ) for c in range(3)], -1)
# parity y -> -y from the top monitor (same rule as farfield_surface.tube_faces)
from farfield_surface import _parity
top = m["surface_top"]; s_y, pur = _parity(np.asarray(top.E), np.ravel(top.y), [1, -1, 1]); print(f"y parity s_y {s_y:+.0f} (purity {pur:.3f})")
KCUT = float(os.environ.get("KCUT_PER_UM", "9")) * 1e6   # 1/m: drop |k_t| > 9 /um before the evanescent back-propagation (higher Bloch orders are ~e^-17 at the post plane)
def drive_plane(y):
    """Control E(x, y, z=0) on the side-plane x grid, from angular-spectrum back-propagation y0 -> y (outgoing in +y)."""
    nx, nz = Es.shape[:2]; dx = np.mean(np.diff(xs)); dz = np.mean(np.diff(zs))
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx); kz = 2 * np.pi * np.fft.fftfreq(nz, dz); KX, KZ = np.meshgrid(kx, kz, indexing="ij")
    ky = np.sqrt((k**2 - KX**2 - KZ**2).astype(complex)); ky = np.where(ky.imag < 0, -ky, ky)
    P = np.exp(1j * ky * (y - y0)) * (np.hypot(KX, KZ) < KCUT)
    Ez0 = []
    for c in range(3):
        F = np.fft.ifft2(np.fft.fft2(Es[:, :, c]) * P)
        Ez0.append(np.array([np.interp(0.0, zs, F[i].real) + 1j * np.interp(0.0, zs, F[i].imag) for i in range(nx)]))
    return np.stack(Ez0, -1)   # (nx, 3)
xg, wg, phig = sphere_grid(150); ct = xg[:, None]; st = np.sqrt(1 - xg**2)[:, None]; one = np.ones((xg.size, phig.size))
nhat = np.stack([st * np.cos(phig)[None], st * np.sin(phig)[None], ct * one], -1).reshape(-1, 3)
th = np.stack([ct * np.cos(phig)[None], ct * np.sin(phig)[None], -st * one], -1).reshape(-1, 3)
ph = np.stack([-np.sin(phig)[None] * one, np.cos(phig)[None] * one, 0 * one], -1).reshape(-1, 3)
GAM = np.array([2 * (EP - EO) / (EP + EO)] * 2 + [(EP - EO) / EO])     # cylinder (axis z) polarisability / (eps_ox V)
_cache = {}
def drive(y):
    if y not in _cache: _cache[y] = drive_plane(y)
    return _cache[y]
def posts_ff(xn, yn, rn, scale=1.0):
    """Normalised far field (2, nth, nph) of mirrored posts (lists, m) in the same units as the control C."""
    E = np.zeros((nhat.shape[0], 3), complex)
    for x, y, r in zip(xn, yn, rn):
        Ed = drive(y); e = np.array([np.interp(x, xs, Ed[:, c].real) + 1j * np.interp(x, xs, Ed[:, c].imag) for c in range(3)])
        for yy, ee in ((y, e), (-y, s_y * e * np.array([1, -1, 1]))):
            p = scale * k**2 / (4 * np.pi) * np.pi * r**2 * H_POST * GAM * ee          # = k^2/(4 pi eps) * alpha E, alpha = eps V GAM
            E += (p[None] - nhat * (nhat @ p)[:, None]) * np.exp(-1j * k * (nhat @ np.array([x, yy, 0.0])))[:, None]
    Eth = np.sum(th * E, -1).reshape(xg.size, phig.size); Eph = np.sum(ph * E, -1).reshape(xg.size, phig.size)
    return np.stack([Eth, Eph]) * NORM
def control_C():
    """Energy-normalised control far field C and its quadrature weights dW (cached in WORK)."""
    f = os.path.join(WORK, "control_C_" + os.path.splitext(os.path.basename(ctrl))[0] + ".npz")   # keyed by the control file (GPT 01a1688x)
    if not os.path.exists(f):
        c = g["load"](ctrl); np.savez(f, C=c["E"], dW=c["dW"])
    d = np.load(f); return d["C"], d["dW"]
if __name__ == "__main__":   # gate: model vs the 8 measured production-mesh comb rows (expect overlap ~0.6, |B|^2 within ~10 %)
    C, dW = control_C(); names_mat = r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/results_from_athena/comb_overshoot/"
    ip = lambda a, b: np.sum(np.conj(a) * b * dW); cc = ip(C, C).real
    sys.path.insert(0, r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes")
    os.environ["FF15_ROUND"] = "C_NT80"; os.environ["FF15_C_STEP"] = "PROD"
    from runners.sweeps.farfield_sph_15um_comb_overshoot import ROWS
    names = ["comb_ph0", "comb_ph120", "comb_ph240", "comb_single_ph51_r63", "comb_pair_ph51_r63", "combL_ph0_r63", "combL_ph120_r63", "combL_ph240_r63"]
    for j, (nm, row) in enumerate(zip(names, ROWS[:8])):
        xn = np.array(row[1]) * 1e-9; Bp = posts_ff(xn, [1.6e-6] * len(xn), [row[2] * 1e-9] * len(xn)); Bm = g["load"](names_mat + nm + ".mat")["E"] - C
        z = ip(Bp, Bm) / ip(Bp, Bp)
        print(f"{nm:22s} pred |B|^2 {ip(Bp,Bp).real/cc:.4f} meas {ip(Bm,Bm).real/cc:.4f} | overlap {abs(ip(Bp,Bm))**2/(ip(Bp,Bp).real*ip(Bm,Bm).real):.3f} "
              f"fit scale {abs(z):.2f} arg {np.degrees(np.angle(z)):+.0f} | P/Pc pred {ip(C+Bp,C+Bp).real/cc:.4f} meas {ip(C+Bm,C+Bm).real/cc:.4f}")
