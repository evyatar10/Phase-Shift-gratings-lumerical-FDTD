# postarray_dwba.py (study: runners/sweeps/farfield_sph_15um_postarray_overshoot, 2026-10-23; handoff docs/HANDOFF_postarray_2026-10-23.md R1)
# MEASURED-RESIDUAL (Gauss-Newton / DWBA-1) target for the next post design: the model far field of a design A is
#     E(A) = C' + s B A,   C' = E_meas - s B A_cur
# i.e. the MEASURED far field of the current device with the model's own estimate of the current posts removed, so the model error of
# the current posts is carried as part of the target. Writes a Gram-compatible npz (G from the arch Gram, b' = Re<C', B_i>, cc' = |C'|^2,
# cc_ctrl; G stays in the arch Gram file named inside) -> pa2d.Gram with b, cc replaced. First run gates the site basis: b recomputed for the CONTROL must equal the stored Gram b.
#   CTRL_MAT=<dy29 control> PA_WORK=<work dir> KCUT_PER_UM=7.5 python postarray_dwba.py <gram.npz> <device.mat> <design.npz|json:name> <out.npz> [s]
import os, sys, json, numpy as np
ARGS = sys.argv[1:]                                  # postarray_born overwrites sys.argv at import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postarray_born as dp
gram_f, dev_f, des_f, out_f = ARGS[:4]; S = float(ARGS[4]) if len(ARGS) > 4 else 0.70; R_REF = 63e-9
gr = np.load(gram_f); xs, ys = gr["xs"], gr["ys"]; G = gr["G"]; nx = len(xs)
sites = [(x, y) for y in ys for x in xs]; n = len(sites)
C, dW = dp.control_C(); cc = float(np.sum(np.abs(C) ** 2 * dW))
Em = dp.g["load"](dev_f)["E"]
pos, pvec = [], []
for x, y in sites:
    Ed = dp.drive(y); e = np.array([np.interp(x, dp.xs, Ed[:, c].real) + 1j * np.interp(x, dp.xs, Ed[:, c].imag) for c in range(3)])
    for yy, ee in ((y, e), (-y, dp.s_y * e * np.array([1, -1, 1]))):
        pos.append([x, yy, 0.0]); pvec.append(dp.k ** 2 / (4 * np.pi) * np.pi * R_REF ** 2 * dp.H_POST * dp.GAM * ee)
pos = np.array(pos); pvec = np.array(pvec)
nd = dp.nhat.shape[0]; tg = {"ctrl": C, "meas": Em}; bb = {k: np.zeros(n) for k in tg}; CH = 1500
for lo in range(0, nd, CH):
    sl = slice(lo, min(lo + CH, nd))
    E = np.exp(-1j * dp.k * (dp.nhat[sl] @ pos.T)).astype(np.complex64)
    for comp, basis in ((0, dp.th), (1, dp.ph)):
        F = E * (basis[sl] @ pvec.T).astype(np.complex64); F = (F[:, 0::2] + F[:, 1::2]) * dp.NORM      # mirror pairs, (ch, n)
        w = dW[comp].ravel()[sl]
        for k, t in tg.items(): bb[k] += np.real(F.conj().T @ (t[comp].ravel()[sl] * w))
rel = np.abs(bb["ctrl"] - gr["b"]).max() / np.abs(gr["b"]).max()
print(f"GATE site basis: recomputed control b vs stored Gram b, max rel diff {rel:.2e} (cc {cc:.4e} vs stored {float(gr['cc']):.4e})")
assert rel < 2e-3 and abs(cc / float(gr["cc"]) - 1) < 1e-6, "site basis / normalisation mismatch -> wrong Gram for this control"
# current design -> site amplitudes (bilinear in x and y, same spreading as pa2d)
if des_f.startswith("json:"):
    D = json.load(open(r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/runners/sweeps/postarray_overshoot_designs.json"))[des_f[5:]]
    x, y, r = (np.array(D[k], float) for k in ("x_nm", "y_nm", "r_nm"))
else:
    d = np.load(des_f); x, y, r = (d[k] if k in d.files else d[k + "_nm"] for k in ("x", "y", "r"))
sys.path.insert(0, os.path.join(os.path.dirname(gram_f), "twod")); import pa2d
gm = pa2d.Gram(gram_f, s=S); A = gm.spread(np.asarray(x, float), np.asarray(y, float), pa2d.amp(r))
assert abs(A.sum() - pa2d.amp(r).sum()) < 1e-6 * A.sum(), "posts outside the Gram site range"
GA = G @ A; bm = bb["meas"]; ccm = float(np.sum(np.abs(Em) ** 2 * dW))
# fitted complex response scale of the CURRENT posts: (E_meas - C) ~ z * B A  ->  z = <BA, E_meas - C> / <BA, BA>  (Re part from b's)
zr = (bm - bb["ctrl"]) @ A / (A @ GA)
P_model = (cc + 2 * S * bb["ctrl"] @ A + S * S * A @ GA) / cc
b_new = bm - S * GA; cc_new = ccm - 2 * S * bm @ A + S * S * A @ GA
print(f"device {os.path.basename(dev_f)[:60]}: N {len(x)} | P_meas/Pc {ccm/cc:.4f} | model P(s={S}) {P_model:.4f} | Re fitted scale {zr:.3f}")
print(f"corrected target C': |C'|^2/cc {cc_new/cc:.4f} (control 1.0000) -> model of the CURRENT design on C' reproduces P_meas by construction")
np.savez(out_f, gram=gram_f, b=b_new, cc=cc_new, cc_ctrl=cc, xs=xs, ys=ys, A_cur=A, s=S, P_meas=ccm / cc, P_model=P_model)
