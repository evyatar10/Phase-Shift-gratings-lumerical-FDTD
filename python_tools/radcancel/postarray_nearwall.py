# postarray_nearwall.py (study: runners/sweeps/farfield_sph_15um_postarray_overshoot, 2026-10-24): a wall-following row of SiN posts close to
# the cavity (edge >= CLEAR from the LOCAL tooth wall, |x| <= X_LIM), driven by the DIRECT in-situ XY map (postarray_nearfield, Athena 269582),
# scored with the ARR model P = |T|^2/cc + 2 Re<T, B a>/cc + ZETA |B a|^2/cc (Fable 2026-10-24: plain Born s=1, ZETA 0.815, LOO rms 0.028),
# T = target far field: the control C (standalone calibration row) or a MEASURED device far field whose posts are kept (add-on step).
#   CTRL_MAT=<nfctrl .mat> PA_WORK=<dir> KCUT_PER_UM=7.5 python postarray_nearwall.py <out.npz> [<device.mat> <design json name>]
import os, sys, json, numpy as np
from scipy.optimize import lsq_linear
ARGS = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postarray_nearfield as nf
dp = nf.dp
sys.path.insert(0, r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes"); os.environ["FF15_ROUND"] = "C_NT80"; os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um import ROWS
_, PITCH, _, _, CAV, (NAR, WID), *_ = ROWS[0]
CLEAR, R_MIN, R_MAX, GAP, X_LIM, ZETA, R_REF = 250.0, 50.0, 80.0, 100.0, float(os.environ.get("X_LIM_NM", "10000")), 0.815, 63.0
edges = CAV / 2 + PITCH * np.arange(len(NAR) + 1)
def wall(x, reach):
    """Local wall half-width (nm) seen by a post at |x| with radius+margin 'reach' (max over every period it overlaps, and the cavity)."""
    w = [max(NAR[i], WID[i]) / 2 for i in range(len(NAR)) if edges[i] - reach <= abs(x) <= edges[i + 1] + reach]
    return max(w + [CAV / 2])
C, dW = dp.control_C(); cc = float(np.sum(np.abs(C) ** 2 * dW)); ip = lambda a, b: np.sum(np.conj(a) * b * dW)
T, keep_x, keep_y, keep_r = C, np.zeros(0), np.zeros(0), np.zeros(0)
if len(ARGS) > 1:                               # add-on step: target = the MEASURED far field of a device whose posts are kept
    # "self": CTRL_MAT is that device's own run (its far field AND its in-situ XY map = DWBA drive); P is then relative to the device
    T = C if ARGS[1] == "self" else dp.g["load"](ARGS[1])["E"]; D = json.load(open(r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/runners/sweeps/postarray_overshoot_designs.json"))[ARGS[2]]
    keep_x, keep_y, keep_r = (np.array(D[k], float) for k in ("x_nm", "y_nm", "r_nm"))
print(f"target |T|^2/cc {ip(T, T).real / cc:.4f}; kept posts {len(keep_x)}")
# candidate sites: 50 nm in x, 3 wall offsets; centre y = wall + offset + R_MAX (edge clearance >= offset for any r <= R_MAX); clear kept posts
OFFS = [CLEAR, CLEAR + 150, CLEAR + 300]
xs = np.tile(np.arange(-X_LIM, X_LIM + 1, 50.0), len(OFFS)); ys = np.concatenate([[wall(x, R_MAX + 100) + d + R_MAX for x in np.arange(-X_LIM, X_LIM + 1, 50.0)] for d in OFFS])
if len(keep_x):
    ok = np.array([np.min(np.hypot(keep_x - x, keep_y - y) - keep_r) - R_MAX >= GAP for x, y in zip(xs, ys)]); xs, ys = xs[ok], ys[ok]
B = np.array([nf.posts_ff_direct([x * 1e-9], [y * 1e-9], [R_REF * 1e-9]).astype(np.complex64) for x, y in zip(xs, ys)])   # (n, 2, nth, nph), a = 1
n = len(xs); Bw = (B * np.sqrt(dW)[None].astype(np.float32)).reshape(n, -1); del B; Tw = (T * np.sqrt(dW)).ravel().astype(np.complex64)
b = (Bw.conj() @ Tw).real.astype(float); G = (Bw.conj() @ Bw.T).real.astype(float); del Bw
print(f"{n} candidate sites, y {ys.min():.0f}..{ys.max():.0f} nm")
def P_of(idx, a): return (ip(T, T).real + 2 * b[idx] @ a + ZETA * a @ G[np.ix_(idx, idx)] @ a) / cc
def solve(idx, lo, hi):                          # min 2 b.a + ZETA a.G.a, bounds  -> lsq form
    lam, V = np.linalg.eigh(ZETA * G[np.ix_(idx, idx)]); keep = lam > 1e-9 * lam.max()       # PSD up to round-off (float32 far fields)
    L = np.sqrt(lam[keep])[:, None] * V[:, keep].T; t = -(V[:, keep].T @ b[idx]) / np.sqrt(lam[keep])   # |L a - t|^2 = a.G.a + 2 b.a + const
    return lsq_linear(L, t, bounds=(lo, hi)).x
amp = lambda r: (np.asarray(r) / R_REF) ** 2; rad = lambda a: R_REF * np.sqrt(a)
a_inf = solve(np.arange(n), 0.0, amp(R_MAX)); print(f"site-grid relaxation (any a <= r80): P {P_of(np.arange(n), a_inf):.4f}")
# discrete greedy insertion (exact on the site grid): best feasible amplitude per site, insert the best, update the residual drive;
# then re-fit all amplitudes in [r50, r80] and delete posts pinned at r50 that still want to shrink; repeat until no gain
def feasible_r(i, xp, yp, rp):
    if not len(xp): return R_MAX
    d = np.hypot(np.asarray(xp) - xs[i], np.asarray(yp) - ys[i]) - np.asarray(rp) - GAP
    return min(R_MAX, d.min())
pick, a = [], np.zeros(0)
for it in range(400):
    g = b + ZETA * (G[:, pick] @ a if pick else 0)
    best, bi, ba = 0.0, None, None
    xp = list(xs[pick]) + list(keep_x); yp = list(ys[pick]) + list(keep_y); rp = list(rad(a)) + list(keep_r)
    astar = np.clip(-g / (ZETA * np.diag(G)), amp(R_MIN), amp(R_MAX)); gain = 2 * astar * g + ZETA * astar ** 2 * np.diag(G)
    for i in np.argsort(gain)[:200]:
        if gain[i] >= best or i in pick: continue
        rmax_i = feasible_r(i, xp, yp, rp)
        if rmax_i < R_MIN: continue
        ai = min(astar[i], amp(rmax_i)); gi = 2 * ai * g[i] + ZETA * ai ** 2 * G[i, i]
        if gi < best: best, bi, ba = gi, i, ai
    if bi is None or best > -1e-6 * cc: break
    pick.append(bi); a = np.append(a, ba)
    if it % 10 == 9:                              # re-fit amplitudes; radii may only shrink (keeps gaps)
        a = solve(pick, amp(R_MIN), np.maximum(a, amp(R_MIN) + 1e-9))
pick = np.array(pick); a = solve(pick, amp(R_MIN), np.maximum(a, amp(R_MIN) + 1e-9)); r = rad(a); x, y = xs[pick], ys[pick]
gp = np.hypot(x[:, None] - x[None], y[:, None] - y[None]) - r[:, None] - r[None]; np.fill_diagonal(gp, 1e9)
clr = np.array([yy - rr - wall(xx, rr + 100) for xx, yy, rr in zip(x, y, r)])
print(f"DESIGN: {len(x)} posts, r {r.min():.0f}-{r.max():.0f}, x {x.min():.0f}..{x.max():.0f}, y {y.min():.0f}..{y.max():.0f}, min gap {gp.min():.0f}, "
      f"min wall clearance {clr.min():.0f} nm | ARR P/Pc {P_of(pick, a):.4f} (gain x{1/P_of(pick, a):.3f}); x1 {b[pick] @ a / cc:+.4f} |B|^2 {a @ G[np.ix_(pick, pick)] @ a / cc:.4f}")
assert gp.min() >= GAP - 1e-6 and clr.min() >= CLEAR - 1e-6
np.savez(ARGS[0], x=x, y=y, r=r, P_arr=P_of(pick, a), keep_x=keep_x, keep_y=keep_y, keep_r=keep_r, clear=CLEAR, zeta=ZETA)
