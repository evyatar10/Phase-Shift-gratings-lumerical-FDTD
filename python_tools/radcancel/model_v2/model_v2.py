# model_v2.py (2026-10-25) — MODEL v2 for SiN post arrays beside the TE overshoot pi-shift grating (Nt80x, 96 periods/side, dy29 numerics).
#   from model_v2 import predict
#   out = predict({"x_nm": [...], "y_nm": [...], "r_nm": [...]})        # one side (+y); the -y mirror is implied
#   out["Qi_gain"], out["P_tube"], out["terms"]
# Configs (env MODEL_V2_CONFIG), refit 2026-10-25 on 34 devices incl. ext42_hx / uni_sub500 / c2d_full:
#   model_v2_final.json (= model_v2_ddaq.json) = v2.1 DEFAULT: DDA drive + direct + strip re-radiation + mode reshaping h(|x|) fitted on Q_i;
#       held-out log-rms 0.037, 2/10 families (ext42, uniform); clouds family error rms 6.4 % in gain. Use this to design.
#   model_v2_primary.json = 5c (no reshaping; 0.045, 2/10); model_v2_height.json = 5c + mesh + finite-height (use for post-height studies);
#   model_v2_qresh.json = reshaping without DDA (0.037). None passes the full gate. Fast design: design_v2.py (Gram form, exact vs predict).
# Physics (each term tested on the held-out-family gate, modelv2/RESULTS.md):
#   E = C + sum_c a_c(Omega) B_c + sum_c rho_c(Omega) I_c [+ alpha C]
#   B_c   Born field of the posts (point dipoles integrated over the post height, drive = control field at the post: nfctrl z=0 map |x| <= 35.95 um,
#         side-plane back-propagation beyond), optional mesh-strength factor and coupled-dipole (DDA) local field -- per FINAL["variant"]
#   I_c   strip re-radiation: each post's field reaches the SiN strip (2-D cylindrical wave H0(k_t y_j)) and is re-radiated from the core line
#         with the post's axial phase exp(-i k u_x x_j)
#   alpha optional mode-reshaping term (sum_j r_j^2 |e_j|^2 h(|x_j|)) C
#   a, rho, h fitted on the 33 measured dy29 devices (Q_i from repo solve_v4 as of 2026-10-25 08:23); tube P -> Q_i map 1/gain = f P + c.
# VALIDITY: TE, post height 350 nm, THIS device (control far field + drive maps are of the 350-nm Nt80x device at dy29 numerics). Changing the core
# height changes the control itself (new C, new drive maps) -> predict() is only valid for a different POST height at a fixed device, and even that
# is unvalidated (needs >= 2-3 FDTD post-array runs at the new post height). TM needs a TM control run (far field + XY map + side plane) and
# TM post-array runs to refit a/rho; the TM code path (p_z, axial polarisability GAM[2], opposite y-mirror parity) is implemented but unvalidated.
import os, json, numpy as np
from scipy.special import hankel1
D = os.environ.get("MODEL_V2_HOME", r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/results_from_athena/comb_overshoot/model_v2_work"); DATA = os.path.join(D, "data")   # weights/configs/data live in the results tree (2026-10-25)
FINAL = json.load(open(os.path.join(D, os.environ.get("MODEL_V2_CONFIG", "model_v2_final.json"))))   # alternative: model_v2_qresh.json
_c = np.load(os.path.join(DATA, "control.npz")); C, dW, NHAT, TH, PH = (_c[q] for q in ("C", "dW", "nhat", "th", "ph"))
K = float(_c["k"]); NORM = complex(_c["NORM"]); S_Y = float(_c["s_y"]); GAM = _c["GAM"]; NTH, NPH = C.shape[1:]; CC = float(np.sum(np.abs(C) ** 2 * dW))
_g = np.load(os.path.join(DATA, os.environ.get("DRIVE_GRID", "drive_grid.npz"))); _W = np.load(os.path.join(D, FINAL["weights"]))   # DRIVE_GRID=drive_grid_y28.npz: side-plane drive to y 2.8 um
QI_C = json.load(open(os.path.join(DATA, "qi.json")))["control"]["Qi"]
UX, UZ = np.real(NHAT[:, 0]), np.real(NHAT[:, 2]).reshape(NTH, NPH)[:, 0]
VALID = {("TE", 350.0)}
XMAP, H0_NM = 35.95e-6, 350.0
from scipy.interpolate import RegularGridInterpolator as _RGI
_IM = [_RGI((_g["X"], _g["Y"]), _g["EXY"][:, :, c]) for c in range(3)]
def _side(x, y):
    """side-plane back-propagated drive (NZ, 3) at (x, y) [m], corrected to the map by zc(y); linear in y between grid rows."""
    YB, xs, bp, zc = _g["YB"], _g["xs"], _g["bp"], _g["zc"]
    j = int(np.clip(np.searchsorted(YB, y) - 1, 0, YB.size - 2)); t = (y - YB[j]) / (YB[j + 1] - YB[j])
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, xs.size - 2)); u = (x - xs[i]) / (xs[i + 1] - xs[i])
    v = lambda jj: ((1 - u) * bp[i, jj] + u * bp[i + 1, jj]) * zc[jj]
    return (1 - t) * v(j) + t * v(j + 1)
def drive(x_nm, y_nm):
    """control drive e(z=0) (N,3) and z-profile ratio e(z)/e(0) on ZG (N, NZ, 3)."""
    e0, prof = [], []
    i0 = np.argmin(np.abs(_g["ZG"]))
    for x, y in zip(np.asarray(x_nm) * 1e-9, np.asarray(y_nm) * 1e-9):
        s = _side(x, y)
        e = np.array([_IM[c]((x, y)) for c in range(3)]) if (abs(x) <= XMAP and y <= 2.29e-6) else s[i0]   # map ends at y 2.3 um
        e0.append(e); prof.append(s / np.where(np.abs(s[i0]) > 0, s[i0], 1)[None])
    return np.array(e0), np.array(prof)
def _mesh_amp(r, y):
    mp = np.load(r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/results_from_athena/post_mesh_probe/post_mesh_probe.npz")
    out = np.ones((len(r), 3))
    for jc, pol in ((0, "Ey"), (1, "Ey"), (2, "Ez")):
        tab = {m_: np.array([np.sqrt(float(mp["sigma"][(mp["mesh"] == m_) & (mp["pol"] == pol) & (mp["r"] == rr)][0]) /
                                     float(mp["sigma"][(mp["mesh"] == 12.5) & (mp["pol"] == pol) & (mp["r"] == rr)][0])) for rr in (35, 50, 75, 100)]) for m_ in (50, 25)}
        t = np.where(np.asarray(y) < 1080.0, (50 - np.sqrt(51.4 * 28.8)) / 25.0, 0.0)
        out[:, jc] = (1 - t) * np.interp(r, [35, 50, 75, 100], tab[50]) + t * np.interp(r, [35, 50, 75, 100], tab[25])
    return out
def _dda(x, y, r, e, h_nm, NZ=3):
    """coupled-dipole local/incident field ratio (N,2), in-plane comps (validated vs measured in-situ drive, corr 0.78-0.95)."""
    H = h_nm * 1e-9; N = len(x); zs = (np.arange(NZ) + 0.5) / NZ * H - H / 2
    P = np.array([[xi * 1e-9, yi * 1e-9, z] for xi, yi in zip(x, y) for z in zs]); pid = np.repeat(np.arange(N), NZ)
    q = np.repeat(np.pi * (np.asarray(r) * 1e-9) ** 2 * H / NZ, NZ); M = np.diag([1.0, -1.0, 1.0]) * S_Y
    def kern(R):
        rr = np.linalg.norm(R, axis=1); n = R / rr[:, None]; nn = np.einsum("ni,nj->nij", n, n); I = np.eye(3)[None]
        return (np.exp(1j * K * rr) / (4 * np.pi))[:, None, None] * (K ** 2 * (I - nn) / rr[:, None, None] + (3 * nn - I) * (1 / rr ** 3 - 1j * K / rr ** 2)[:, None, None])
    n = N * NZ; A = np.zeros((n, 2, n, 2), complex)
    for i in range(n):
        Kd = kern(np.where((pid == pid[i])[:, None], 1.0, P[i] - P)); Kd[pid == pid[i]] = 0
        T = (Kd + kern(P[i] - P * np.array([1, -1, 1])) @ M)[:, :2, :2] * (q[:, None] * GAM[None, :2])[:, None, :]
        A[i] = -np.transpose(T, (1, 0, 2))
    El = np.linalg.solve(A.reshape(2 * n, 2 * n) + np.eye(2 * n), np.repeat(e[:, :2], NZ, 0).reshape(-1)).reshape(N, NZ, 2).mean(1)
    return El / np.where(np.abs(e[:, :2]) > 0, e[:, :2], 1)
def _zfac(h_nm):
    h = h_nm * 1e-9; z = np.linspace(-h / 2, h / 2, 201); w = np.full(z.size, z[1] - z[0]); w[[0, -1]] /= 2
    ph_ = np.exp(-1j * K * np.outer(UZ, z)); return np.array([(ph_ * (z ** n * w)[None]).sum(1) / h for n in range(3)])
def predict(posts, h_nm=350.0, pol="TE", allow_unvalidated=False, return_field=False):
    """posts: dict with x_nm, y_nm, r_nm (+y side; optional r2_nm ellipse -> equal-area circle). Returns dict:
    Qi_gain (vs the bare control, Q_i,ctrl = 5.555e5 with the 2026-10-25 solve_v4), P_tube (radiated tube power / control), terms, flags."""
    if (pol, float(h_nm)) not in VALID and not allow_unvalidated:
        raise ValueError(f"model v2 is validated only for {sorted(VALID)}; ({pol}, {h_nm}) needs: "
                         + ("a TM control run (far field + z=0 XY map + side plane) and >= 3 TM post-array runs to refit a/rho" if pol != "TE"
                            else f">= 2-3 FDTD post-array runs with {h_nm}-nm posts on this device (a different CORE height changes the control itself)")
                         + ". Pass allow_unvalidated=True to get the extrapolation.")
    if pol == "TM": raise NotImplementedError("TM drive tables do not exist yet (need a TM control run); TE weights do not transfer to p_z dipoles")
    x = np.asarray(posts["x_nm"], float); y = np.asarray(posts["y_nm"], float); r = np.asarray(posts["r_nm"], float)
    if posts.get("r2_nm"): r = np.array([np.sqrt(a * (b if b is not None else a)) for a, b in zip(r, posts["r2_nm"])])
    flags = []
    if np.abs(x).max() > 49.9e3: flags.append("posts beyond |x| 49.9 um: outside the side-plane drive (clamped)")
    if y.min() < 700 or y.max() > 2100: flags.append("y outside 0.70-2.10 um: z-profile clamped")
    if r.min() < 35 or r.max() > 100: flags.append("radius outside the mesh-probe table 35-100 nm (clamped)")
    if x.size > 450 and FINAL["variant"].get("dda"): flags.append("DDA cost grows as N^3 (N*3 slices)")
    if float(h_nm) != H0_NM and FINAL["variant"].get("height") != "zdrive": flags.append("this config has no finite-height integration: use MODEL_V2_CONFIG=model_v2_height.json for height studies")
    var = FINAL["variant"]; e0, prof = drive(x, y)
    amp = _mesh_amp(r, y).astype(complex) if var.get("mesh") else np.ones((x.size, 3), complex)
    if var.get("dda"): amp[:, :2] *= np.nan_to_num(_dda(x, y, r, e0, h_nm), nan=1.0)
    pref = K ** 2 / (4 * np.pi) * np.pi * (r * 1e-9) ** 2 * (h_nm * 1e-9)
    # height: z-profile of the drive over the post (side-plane back-propagation), integrated against exp(-i k u_z z)
    ZG = _g["ZG"]; m = np.abs(ZG) <= h_nm * 1e-9 / 2 + 1e-12; Vd = np.vander(ZG[m], 3, increasing=True)
    if var.get("height") == "zdrive":
        coef = np.stack([np.linalg.lstsq(Vd, prof[:, m, c].T, rcond=None)[0] for c in range(2)], -1)          # (3 powers, N, 2)
        F = _zfac(h_nm)
    else:
        coef = np.zeros((3, x.size, 2), complex); coef[0] = 1; F = np.array([np.ones(NTH), np.zeros(NTH), np.zeros(NTH)])
    B = np.zeros((2, 2, NHAT.shape[0]), complex)            # (comp, pol, nang)
    for i0 in range(0, x.size, 48):
        sl = slice(i0, i0 + 48)
        for sgn in (1, -1):
            ph_ = np.exp(-1j * K * (np.outer(x[sl] * 1e-9, NHAT[:, 0]) + np.outer(sgn * y[sl] * 1e-9, NHAT[:, 1])))
            for c in (0, 1):
                mir = (S_Y * (1 if c == 0 else -1)) if sgn < 0 else 1.0
                for nz in range(3):
                    p = pref[sl] * GAM[c] * e0[sl, c] * amp[sl, c] * coef[nz, sl, c] * mir
                    fz = np.repeat(F[nz], NPH)
                    sp = p @ ph_
                    B[c, 0] += sp * TH[:, c] * fz; B[c, 1] += sp * PH[:, c] * fz
    B *= NORM
    I = np.zeros((2, NHAT.shape[0]), complex); KT = K * np.sqrt(np.maximum(1 - UX ** 2, 0.05 ** 2))
    pimg = pref[:, None] * GAM[None, :2] * e0[:, :2] * amp[:, :2]
    for yv in np.unique(y):
        mm = y == yv; Hy = hankel1(0, KT * yv * 1e-9); ph_ = np.exp(-1j * K * np.outer(x[mm] * 1e-9, UX))
        for c in (0, 1): I[c] += Hy * (pimg[mm, c] @ ph_)
    I *= NORM
    W, pinv = _W["W"], _W["pinv"]; NB = FINAL["NB"]
    Xb = [B[0], B[1]] + ([np.stack([I[0], I[0]]), np.stack([I[1], I[1]])] if NB == 4 else [])
    R = np.zeros((2, NHAT.shape[0]), complex)
    for k_, Xk in enumerate(Xb):
        for pl in range(2): R[pl] += Xk[pl] * W[pl][pinv][:, k_]
    terms = {}
    if FINAL.get("resh"):
        KN = _W["KNOTS"]; wj = (r * 1e-3) ** 2 * np.sum(np.abs(e0[:, :2]) ** 2, 1) * (h_nm / H0_NM)
        A = np.array([np.interp(np.abs(x) / 1e3, KN, np.eye(KN.size)[q]) for q in range(KN.size)]) @ wj / float(_W["An"])
        alpha = A @ _W["hk"]; R += alpha * C.reshape(2, -1); terms["alpha"] = complex(alpha)
    E = C + R.reshape(2, NTH, NPH); P = float(np.sum(np.abs(E) ** 2 * dW) / CC)
    f, c0 = FINAL["map"]; Pq = f * P + c0
    terms.update(P_born_only=float(np.sum(np.abs(C + (B[0] + B[1]).reshape(2, NTH, NPH)) ** 2 * dW) / CC))
    out = dict(Qi_gain=1 / Pq, P_tube=P, P_loss=Pq, Qi=QI_C / Pq, terms=terms, flags=flags, validated=(pol, float(h_nm)) in VALID)
    if return_field: out.update(R=R, A=(A if FINAL.get("resh") else None))
    return out
