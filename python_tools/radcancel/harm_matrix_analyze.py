# scratch 2026-10-21 harm_matrix: per-knob harmonic response matrix + bounded multi-harmonic solves (zero GPU).
# Rows from build_cache.py (solve_v4 --short normalisation, common phase reference). |.|^2 in units of parent |c|^2 (whole field, lmax 150).
import os, glob, numpy as np
from scipy.optimize import lsq_linear, minimize
HERE = os.path.dirname(os.path.abspath(__file__)); LM = 40; NOISE = 0.012
rows = {os.path.basename(f)[4:-4]: np.load(f) for f in sorted(glob.glob(HERE + "/row_*.npz"))}
def ipf(u, v, dW): return np.sum(np.conj(u) * v * dW[None]).real
lab = np.array([f"{t}({l},{m:+d})" for t in "EM" for l in range(LM + 1) for m in range(-LM, LM + 1)]).reshape(2, LM + 1, 2 * LM + 1)
lmask = np.zeros((2, LM + 1, 2 * LM + 1), bool); lmask[:, :6, LM - 5:LM + 6] = True     # l <= 5 block

def analyse(parent, knobs, units, tag, do_multi):
    c = rows[parent]; dW = c["dW"]; Cn = c["E"]; cc = ipf(Cn, Cn, dW); ac = c["a"]
    Pc_h = np.abs(ac) ** 2 / cc; top = np.argsort(Pc_h.ravel())[::-1][:12]
    print(f"\n#### {tag}: parent {parent}  |c|^2 (lmax150) = 1; harmonics l<=40 hold {Pc_h.sum():.3f}, l<=5 block {Pc_h[lmask].sum():.3f}")
    print("top-12 parent harmonics (fraction of |c|^2):", ", ".join(f"{lab.ravel()[k]} {Pc_h.ravel()[k]:.3f}" for k in top))
    B = {}; A = {}; Pb = {}
    for k in knobs:
        if k not in rows: print("  MISSING", k); continue
        if isinstance(knobs[k], tuple):                     # central difference (plus, minus)
            kp, km = knobs[k]; B[k] = (rows[kp]["E"] - rows[km]["E"]) / 2; A[k] = (rows[kp]["a"] - rows[km]["a"]) / 2
        else:
            B[k] = rows[k]["E"] - Cn; A[k] = rows[k]["a"] - ac
        Pb[k] = ipf(B[k], B[k], dW) / cc
    names = list(B)
    print(f"{'knob':26s} {'unit':>6s} {'|b|^2':>7s} {'eta1^2':>7s} | per top harmonic: dA/A (|.|, deg) , selectivity |dA|^2/|b|^2")
    for k in names:
        e1 = ipf(B[k], Cn, dW) ** 2 / (ipf(B[k], B[k], dW) * cc)
        cells = []
        for h in top[:6]:
            da, a = A[k].ravel()[h], ac.ravel()[h]; sel = abs(da) ** 2 / cc / Pb[k]
            cells.append(f"{abs(da/a):5.2f}@{np.degrees(np.angle(da/a)):+4.0f} s{sel:.2f}")
        flag = " <NOISE" if Pb[k] < NOISE else ""
        Prow = ipf(Cn + B[k], Cn + B[k], dW) / cc
        print(f"{k:26s} {units[k]:>6s} {Pb[k]:7.4f} {e1:7.3f} P{Prow:5.3f} | " + " ".join(cells) + flag)
    print("single-knob first-order zeroing (real amplitude): harmonic | best knob | alpha (knob units) = amplitude | fraction of harmonic removed (phase-match cos^2) | P/Pc at alpha")
    for h in top:
        best = None
        for k in names:
            da, a = A[k].ravel()[h], ac.ravel()[h]
            if abs(da) == 0: continue
            al = -(np.conj(da) * a).real / abs(da) ** 2; rem = (np.conj(da) * a).real ** 2 / (abs(da) ** 2 * abs(a) ** 2)
            res = Cn + al * B[k]; P = ipf(res, res, dW) / cc
            score = rem if abs(al) <= 2 else rem / (abs(al) / 2) ** 2
            if best is None or score > best[0]: best = (score, k, al, rem, P, abs(da) ** 2 / cc / Pb[k])
        s, k, al, rem, P, sel = best
        print(f"  {lab.ravel()[h]:9s} {Pc_h.ravel()[h]:.3f} | {k:24s} | alpha {al:+6.2f} x {units[k]:>5s} | removes {rem:.2f} | P/Pc {P:.3f} | sel {sel:.2f}{' <NOISE-ROW' if Pb[k] < NOISE else ''}")
    if not do_multi: return
    # ---- multi-knob bounded solves on real alpha, CLEAN basis (one row per knob, rows above the noise floor)
    names = [k for k in names if k in do_multi]; print("  clean basis:", names)
    n = len(names); s = 1 / np.sqrt(cc)
    Bf = np.stack([(B[k] * np.sqrt(dW)[None] * s).ravel() for k in names], 1); cf = (Cn * np.sqrt(dW)[None] * s).ravel()
    Gf = np.concatenate([Bf.real, Bf.imag]); gf = np.concatenate([cf.real, cf.imag])            # P = |gf + Gf a|^2
    Ah = np.stack([A[k].ravel() * s for k in names], 1); ah = ac.ravel() * s
    Gl = np.concatenate([Ah[lmask.ravel()].real, Ah[lmask.ravel()].imag]); gl = np.concatenate([ah[lmask.ravel()].real, ah[lmask.ravel()].imag])
    Ptot = lambda a: np.sum((gf + Gf @ a) ** 2); Plow = lambda a: np.sum((gl + Gl @ a) ** 2)
    def row(tag_, a, extra=""):
        print(f"  {tag_:34s} P/Pc {Ptot(a):.3f}  l<=5 block {Plow(a):.3f}  |alpha|max {np.abs(a).max():.2f} |alpha|2 {np.linalg.norm(a):.2f}  half-step P {Ptot(a/2):.3f}{extra}")
    a0 = lsq_linear(Gf, -gf, bounds=(-2, 2)).x; row("(c) min TOTAL, |a|<=2", a0); Pmin = Ptot(a0)
    print("      alpha:", {k: round(float(v), 2) for k, v in zip(names, a0) if abs(v) > 0.05})
    for K in (1, 2, 4, 8):
        hk = top[:K]; Gk = np.concatenate([Ah[hk].real, Ah[hk].imag]); gk = np.concatenate([ah[hk].real, ah[hk].imag])
        Wt = 1e3; x = lsq_linear(np.concatenate([Gf, Wt * Gk]), -np.concatenate([gf, Wt * gk]), bounds=(-2, 2)).x
        resid = np.sum((gk + Gk @ x) ** 2) / np.sum(gk ** 2)
        row(f"(a) zero top-{K} (w=1e3), min total", x, f"  residual on those {resid:.1e}" + ("  NOT REACHED (bounds)" if resid > 1e-2 else ""))
    r = minimize(Plow, a0, jac=lambda a: 2 * Gl.T @ (gl + Gl @ a), bounds=[(-2, 2)] * n, method="SLSQP",
                 constraints=[{"type": "ineq", "fun": lambda a: Pmin * 1.02 - Ptot(a), "jac": lambda a: -2 * Gf.T @ (gf + Gf @ a)}], options={"maxiter": 500, "ftol": 1e-12})
    row("(b) min l<=5 block s.t. P<=Pmin+2%", r.x, "" if r.success else "  " + r.message)
    print("      alpha:", {k: round(float(v), 2) for k, v in zip(names, r.x) if abs(v) > 0.05})
    al = lsq_linear(Gl, -gl, bounds=(-2, 2)).x; row("(b') min l<=5 block, no P constraint", al)

bumps = [k for k in rows if k.startswith("bump_") and k not in ("bump_mean_g0203_m5nm", "bump_mean_g0203_p2.5nm")]
knobs = {k: k for k in bumps}; knobs["bump_mean_g0203_p5nm"] = ("bump_mean_g0203_p5nm", "bump_mean_g0203_m5nm")
units = {k: "5nm" for k in bumps}
for k in rows:
    if k.startswith("contrast_"): knobs[k] = k; units[k] = k.split("_p")[-1]
    if k.startswith("subteeth_"): knobs[k] = k; units[k] = "d" + k.split("_d")[-1]
# linearity checks
c = rows["ctrl"]; dW = c["dW"]; cc = ipf(c["E"], c["E"], dW)
if "bump_mean_g0203_m5nm" in rows:
    bp, bm = rows["bump_mean_g0203_p5nm"]["E"] - c["E"], rows["bump_mean_g0203_m5nm"]["E"] - c["E"]
    print(f"pair residual mean_g0203 |b+ + b-|/|b+ - b-| = {np.sqrt(ipf(bp+bm,bp+bm,dW)/ipf(bp-bm,bp-bm,dW)):.3f};", end=" ")
if "bump_mean_g0203_p2.5nm" in rows:
    b2 = rows["bump_mean_g0203_p2.5nm"]["E"] - c["E"]; print(f"||b5 - 2 b2.5||/||b5|| = {np.sqrt(ipf(bp-2*b2,bp-2*b2,dW)/ipf(bp,bp,dW)):.3f}")
clean = [k for k in knobs if k.startswith("bump_") and k not in ("bump_diff_g0607_p5nm", "bump_diff_g1011_p5nm", "bump_diff_g1214_p5nm")] + ["contrast_diff_g0001_p10nm", "contrast_diff_g0003_p10nm", "subteeth_A_L578_d5nm"]
analyse("ctrl", knobs, units, "CONTROL BASIS", clean)
if "contrast_diff_g0001_p20nm" in rows:
    tabs = {k: k for k in rows if k.startswith("tab_")}; tu = {k: "d" + k.split("_d")[-1] for k in tabs}
    analyse("contrast_diff_g0001_p20nm", tabs, tu, "+20 PARENT, TAB ROWS", ["tab_r0_d5", "tab_rh_d5", "tab_r0_d5_R", "tab_rh_d5_R", "tab_r0_M_d2p5"])
