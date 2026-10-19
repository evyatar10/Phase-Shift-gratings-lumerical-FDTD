"""Loss budget + harmonics of a HIGH-Q SHORT RUN (FarFieldConfig.short_run; skill highq-short-run).

Study: runners/sweeps/farfield_sph_15um | Created 2026-10-19, rewritten 2026-10-20 | Gates: Athena 265002/265023
Usage:  python python_tools/ringdown_budget.py <result.mat> [--gate-ql Q] [--lmax 150]

Method (validated, gate B2 on the stored 20-um overshoot vs its full 150-ps run):
  Q_L      from the time_cavity trace: amplitude ~ exp(-w t / 2Q_L)  (fit of ln|envelope|; exact, 0.0 %)
  harmonic fractions + mode width: shape of the mode -> correct from ~1.5-2 decay times (0.01-0.03 pt)
  T, tube power: source-normalized, multiplied by gamma^2 = 1/(1 - exp(-w T_run / 2Q_L))^2
           (Ansys "correcting field amplitudes for high-Q cavities"); tube 0.4 %, T ~0.006 low at 2 decay times.
  loss = 2t(1-t), Q_i = Q_L/(1-t), Q_c = Q_L/t (t = sqrt T; symmetric two-port, no direct transmission).
NOT usable from a short run: the spectral FWHM, R, and tube/end-cap flux RATIOS (the DFT fields are the driven
response, not a pure decaying mode -- the GPU solver drops monitor time windows).
"""
import argparse
import os
import sys

import numpy as np
import scipy.io as sio
from scipy.signal import hilbert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import farfield_surface as fs          # noqa: E402
import farfield_multipole as fm        # noqa: E402

C0 = 299792458.0


def decay_q(trace, lam_m, t_start_s):
    """Q_L from ln|analytic envelope| of the strongest field component after t_start."""
    t = np.asarray(trace.t, float).ravel()
    E = np.asarray(trace.E, float)
    E = E.reshape(t.size, -1)
    late = t >= t_start_s
    comp = int(np.argmax(np.sum(E[late] ** 2, axis=0)))     # strongest component AFTER the window, not the transient
    env = np.abs(hilbert(E[:, comp]))
    w = 2 * np.pi * C0 / lam_m
    out = {}
    t_end = t[-1] - 0.05 * (t[-1] - t_start_s)          # drop the Hilbert edge at the end
    for name, a, b in (("full", t_start_s, t_end),
                       ("first half", t_start_s, 0.5 * (t_start_s + t_end)),
                       ("second half", 0.5 * (t_start_s + t_end), t_end)):
        k = (t >= a) & (t <= b) & (env > 0)
        slope = np.polyfit(t[k], np.log(env[k]), 1)[0]
        out[name] = -w / (2 * slope) if slope < 0 else np.nan     # non-decaying fit -> reject
    return out, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat")
    ap.add_argument("--ref", help="stored steady-state _multipoles.csv to compare harmonic fractions")
    ap.add_argument("--lmax", type=int, default=150)
    ap.add_argument("--gate-ql", type=float, help="expected Q_L (PASS within 5 %%)")
    ap.add_argument("--fit-from-ps", type=float, default=15.0, help="decay fit starts here (after the pulse)")
    a = ap.parse_args()
    m = sio.loadmat(a.mat, squeeze_me=True, struct_as_record=False)
    lam = float(m["resonance_wavelength_nm"]) * 1e-9
    q, comp = decay_q(m["trace_time_cavity"], lam, a.fit_from_ps * 1e-12)
    Q_L = q["full"]
    halves = abs(q["first half"] / q["second half"] - 1)
    T_run = float(np.asarray(m["trace_time_cavity"].t).ravel()[-1])
    g2 = 1 / (1 - np.exp(-2 * np.pi * C0 / lam * T_run / (2 * Q_L))) ** 2
    print(f"=== {os.path.basename(a.mat)}\n  run {T_run*1e12:.0f} ps = {T_run/(2*Q_L/(2*np.pi*C0/lam)):.2f} amplitude decay times | "
          f"lambda {lam*1e9:.4f} nm | component {'xyz'[comp]}")
    print("  Q_L decay: " + ", ".join(f"{k} {v:.0f}" for k, v in q.items())
          + f"  (halves agree {halves*100:.1f} %: {'OK' if halves < 0.05 else 'FAIL: beating / second mode'})")
    i = int(np.argmin(np.abs(np.asarray(m["wl_nm"]) - lam * 1e9)))
    T = float(m["T"][i]) * g2; t = np.sqrt(T); loss = 2 * t * (1 - t)
    x, w, phi, Et, Ep, info = fs.sphere_field(m, 2)
    rad = info["tube_loss"] * g2
    Q_i, Q_c = Q_L / (1 - t), Q_L / t
    print(f"  gamma^2 {g2:.3f} | T {T:.3f} (DERIVED, ~-0.006 at 2 decay times) | loss 2t(1-t) {loss:.3f} | "
          f"radiated (tube) {rad:.3f} = {rad/loss*100:.0f} % of loss")
    print(f"  Q_i {Q_i:.3g}  Q_c {Q_c:.3g}  Q_rad {Q_i*loss/rad:.3g}  | mode width {float(m['fwhm_m'])*1e6:.2f} um")
    if a.gate_ql:
        print(f"  GATE Q_L {Q_L:.0f} vs {a.gate_ql:.0f}: {'PASS' if abs(Q_L/a.gate_ql-1) <= 0.05 else 'FAIL'}")

    res = fm.analyze(a.mat, a.lmax)
    if a.ref:
        P = np.abs(res["aE"]) ** 2 + np.abs(res["aM"]) ** 2
        P /= P.sum()
        ref = np.loadtxt(a.ref, delimiter=",", skiprows=1)
        rows = sorted(ref, key=lambda r: -(r[2] + r[3]))[:10]
        worst = max(abs(P[int(l), int(mm) + a.lmax] - (fe + fmg)) for l, mm, fe, fmg in rows)
        print(f"  harmonics vs ref: max |diff| over top-10 = {worst*100:.2f} pt  ({'PASS' if worst <= 0.005 else 'FAIL'} at 0.5 pt)")


if __name__ == "__main__":
    main()
