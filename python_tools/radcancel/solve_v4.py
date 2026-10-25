# scratch 2026-10-20 v4 (after the Fable bug-hunt): per-row CAVITY-TRACE reference for short runs, s=-1 Lorentzian for steady runs,
# bounded weighted least squares (lsq_linear), eta^2 reported at the chosen alpha.
#   python solve_v4.py [--short] [--pair i j] [--exclude k,..] [--amax 1] [--slam 0.02] [--sW 0.1] control.mat knob.mat ...
import sys, numpy as np, scipy.io as sio
from scipy.signal import hilbert
from scipy.optimize import lsq_linear
sys.path.insert(0, r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/python_tools")
from farfield_surface import sphere_field
from farfield_multipole import multipoles
C0 = 299792458.0; LMAX = 150
argv = sys.argv[1:]; opts = {"amax": 1.0, "slam": 0.02, "sW": 0.1, "lcut": 0.0}
short = "--short" in argv; energy = "--energy" in argv; pair = None; excl = []
def take(flag, n=1):
    i = argv.index(flag); v = argv[i + 1:i + 1 + n]; del argv[i:i + 1 + n]; return v
if short: argv.remove("--short")
if energy: argv.remove("--energy")   # 2026-10-21 (GPT audit): divide by the stored-energy proxy ∫|E|^2 dx (field_energy_density_1D), not one point
if "--pair" in argv: pair = tuple(int(v) for v in take("--pair", 2))
if "--exclude" in argv: excl = [int(v) for v in take("--exclude")[0].split(",")]
for k in list(opts):
    if f"--{k}" in argv: opts[k] = float(take(f"--{k}")[0])
files = argv
t_start_fit = 15e-12
def trace_fit(trace, t_start=t_start_fit):
    t = np.asarray(trace.t, float).ravel(); E = np.asarray(trace.E, float).reshape(t.size, -1)
    late = t >= t_start; comp = int(np.argmax(np.sum(E[late] ** 2, axis=0))); z = hilbert(E[:, comp])
    t_end = t[-1] - 0.05 * (t[-1] - t_start); k = (t >= t_start) & (t <= t_end)
    kappa = -np.polyfit(t[k], np.log(np.abs(z[k])), 1)[0]; omega = abs(np.polyfit(t[k], np.unwrap(np.angle(z[k])), 1)[0])
    return omega, kappa, t, E[:, comp], comp
def load(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    x, w, phi, E_th, E_ph, info = sphere_field(m, LMAX)
    lam_ff = float(np.ravel(m["surface_top"].lam)[0]); w_ff = 2 * np.pi * C0 / lam_ff
    lam_port = float(m["resonance_wavelength_nm"]) * 1e-9; i = int(np.argmin(np.abs(np.ravel(m["wl_m"]) - lam_port))); T_port = float(np.ravel(m["T"])[i])
    E = np.stack([E_th, E_ph]); dW = (w[:, None] * (2 * np.pi / len(phi)))[None] * np.ones_like(E, dtype=float)
    if short:
        omega0, kappa, t, Ec, comp = trace_fit(m["trace_time_cavity"]); Q = omega0 / (2 * kappa); lam0 = 2 * np.pi * C0 / omega0
        t_exc = t[np.argmax(np.abs(Ec) * (t < t_start_fit))]   # ring-down starts at the excitation (envelope peak ~0.4 ps), NOT 5 ps (GPT 01a170xx audit, 2026-10-25: -5 ps over-corrected T by ~1 %)
        g2 = 1 / (1 - np.exp(-kappa * (t[-1] - t_exc))) ** 2; T = T_port * g2
        D = np.trapz(Ec * np.exp(1j * w_ff * t), t)                 # cavity DFT at the far-field frequency (e^{-iwt} field convention -> e^{+iwt} kernel)
        En = E / D; ref = "cavity-trace DFT"; det = (w_ff - omega0) / kappa
    else:
        lam0 = lam_port; Q = lam0 / abs(float(m["spectral_fwhm_nm"]) * 1e-9); T = T_port; omega0 = 2 * np.pi * C0 / lam0
        det = 2 * Q * (lam_ff - lam0) / lam0; En = E * (1 + 1j * det) / (np.sqrt(Q) * T ** 0.25); ref = "Lorentzian s=-1"
    if energy:
        # |E|^2 at z=0 integrated over y (field_profile, 501 points uniform in f over the source window), integrated over x.
        # Steady: undo the far field's Lorentzian (1+i det) as before and put the energy at resonance by (1+det_fp^2).
        # Short: keep the cavity-DFT phase for coherent rows, take the magnitude from the energy.
        U = np.trapezoid(np.ravel(m["field_energy_density_1D"]), np.ravel(m["field_x"]))
        wl = np.ravel(m["wl_m"]); f_fp = np.linspace(C0 / wl.max(), C0 / wl.min(), 501)
        f_fp = f_fp[np.argmin(np.abs(f_fp - C0 / lam0))]; det_fp = 2 * Q * (C0 / f_fp - lam0) / lam0
        if short: En = E * (np.conj(D) / abs(D)) / np.sqrt(U)
        else: En = E * (1 + 1j * det) / np.sqrt(U * (1 + det_fp ** 2))
        ref += f" -> ENERGY (int|E|^2 dx, det_fp {det_fp:+.3f})"
    return dict(E=En, dW=dW, det=det, lam=lam0 * 1e9, T=T, Q=Q, W=float(m["fwhm_m"]) * 1e6, Qi=Q / (1 - np.sqrt(T)), lam_ff=lam_ff * 1e9, ref=ref)
rows = [load(p) for p in files]; c = rows[0]; R = rows[1:]
if opts["lcut"] > 0:
    L = int(opts["lcut"]); LM = 40
    xg, wg, phig, _, _, _ = sphere_field(sio.loadmat(files[0], squeeze_me=True, struct_as_record=False), LMAX)
    def harm(E):
        aE, aM = multipoles(xg, wg, phig, E[0], E[1], LM)
        v = np.concatenate([aE[:L + 1].ravel(), aM[:L + 1].ravel()])
        return v[None, :, None]                      # shape-compatible with the (2, n_th, n_ph) handling below (dW = 1)
    for r in rows:
        r["E"] = harm(r["E"]); r["dW"] = np.ones_like(r["E"], dtype=float)
    c = rows[0]; R = rows[1:]
    print(f"OBJECTIVE restricted to harmonics l <= {L} (converged physical part); scores are per stored energy in that subspace")
def ip(a, b): return np.sum(np.conj(a) * b * c["dW"])
Cn = c["E"]; cc = ip(Cn, Cn).real; B = [r["E"] - Cn for r in R]
print(f"normalisation: {c['ref']}; control lam {c['lam']:.4f} (ff {c['lam_ff']:.4f}, det {c['det']:+.2f}) T {c['T']:.4f} Q_L {c['Q']:.0f} Q_i {c['Qi']:.3g} W {c['W']:.3f}")
for j, r in enumerate(R):
    ang = np.degrees(np.angle(ip(Cn, r["E"])))
    print(f"row {j:2d}: dlam {r['lam']-c['lam']:+.4f} det {r['det']:+.2f} T {r['T']:.4f} Q_L {r['Q']:.0f} Q_i {r['Qi']:.3g} dW {r['W']-c['W']:+.3f} | arg<c,c_j> {ang:+6.1f} deg | |b|^2/|c|^2 {ip(B[j],B[j]).real/cc:.4f} P/Pc {ip(r['E'],r['E']).real/cc:.4f} eta1 {ip(B[j],Cn).real**2/(ip(B[j],B[j]).real*cc):.4f}")
if pair:
    i, j = pair; bp, bm = B[i], B[j]
    print(f"PAIR {i}/{j}: |b+ + b-|/|b+ - b-| = {np.sqrt(abs(ip(bp+bm,bp+bm))/abs(ip(bp-bm,bp-bm))):.3f} (noise/nonlinearity floor); using central difference")
    B[i] = (bp - bm) / 2; R[i] = dict(R[i]); R[i]["lam"] = c["lam"] + (R[i]["lam"] - R[j]["lam"]) / 2; R[i]["W"] = c["W"] + (R[i]["W"] - R[j]["W"]) / 2; excl.append(j)
keep = [j for j in range(len(B)) if j not in excl]; B = [B[j] for j in keep]; R = [R[j] for j in keep]
n = len(B); scale = 1.0 / np.sqrt(cc)            # field rows in units of |c| so the lambda/W sigma-rows have a defined weight
A = np.concatenate([np.stack([(b * np.sqrt(c["dW"]) * scale).ravel().real for b in B], 1), np.stack([(b * np.sqrt(c["dW"]) * scale).ravel().imag for b in B], 1),
                    np.array([[(r["lam"] - c["lam"]) / opts["slam"] for r in R]]), np.array([[(r["W"] - c["W"]) / opts["sW"] for r in R]])])
y = np.concatenate([(Cn * np.sqrt(c["dW"]) * scale).ravel().real, (Cn * np.sqrt(c["dW"]) * scale).ravel().imag, [0.0, 0.0]])
def report(tag, a):
    res = Cn + sum(ai * bi for ai, bi in zip(a, B)); P = ip(res, res).real / cc
    dl = sum(ai * (r["lam"] - c["lam"]) for ai, r in zip(a, R)); dw = sum(ai * (r["W"] - c["W"]) for ai, r in zip(a, R))
    print(f"{tag}: P/Pc {P:.3f} (eta^2 {1-P:.3f}), half step {ip(Cn+sum(ai/2*bi for ai,bi in zip(a,B)), Cn+sum(ai/2*bi for ai,bi in zip(a,B))).real/cc:.3f}; alpha {np.round(a,3)}; dlam {dl:+.3f} nm dW {dw:+.3f} um")
    return a
a_free = lsq_linear(A[:-2], -y[:-2]).x; report("FREE (no bounds, no lambda/W rows)", a_free)
a_b = lsq_linear(A, -y, bounds=(-opts["amax"], opts["amax"])).x; report(f"BOUNDED |alpha|<={opts['amax']}, weighted lambda ({opts['slam']} nm) / W ({opts['sW']} um) rows", a_b)
np.save(files[0] + ".alpha_v4.npy", a_b); print("rows kept:", keep)
