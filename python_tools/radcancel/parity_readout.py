# scratch 2026-10-21: x-parity readout of the off-lattice tab rows on the Nt80x +20 parent.
# usage: python parity_readout.py parent.mat r0_d5.mat rh_d5.mat r0_d2p5.mat [r0_d5_R.mat rh_d5_R.mat]
# Fields normalised exactly as solve_v4 --short (cavity-trace DFT, common reference, no per-row phase alignment).
# Parity about the leak's own mirror plane x0 = +0.195 um (Fable review a373bcb3: 0.895 even there vs 0.433 about x = 0).
import sys, io, contextlib, numpy as np, scipy.io as sio
from scipy.optimize import lsq_linear
import os
HERE = os.path.dirname(os.path.abspath(__file__))           # solve_v4.py sits next to this file
files = sys.argv[1:]
ns = {}
sys.argv = ["solve_v4.py", "--short"] + files
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(HERE + "/solve_v4.py", encoding="utf-8").read(), ns)
rows, ip, Cn, cc = ns["rows"], ns["ip"], ns["Cn"], ns["cc"]
B = [r["E"] - Cn for r in rows[1:]]
x, w, phi, *_ = ns["sphere_field"](sio.loadmat(files[0], squeeze_me=True, struct_as_record=False), ns["LMAX"])
n = phi.size; kf = (n // 2 - np.arange(n)) % n
nx = np.sqrt(1 - x ** 2)[:, None] * np.cos(phi)[None, :]
k = 2 * np.pi * 1.444 / (rows[0]["lam_ff"] * 1e-9); X0 = 0.195e-6
ph = np.exp(1j * k * X0 * nx)


def split(E):
    """(even, odd) about x0: shift the origin to x0, apply the vector mirror E_th(kf), -E_ph(kf), shift back."""
    A = E * ph[None]
    M = np.stack([A[0][:, kf], -A[1][:, kf]])
    return (A + M) / 2 / ph[None], (A - M) / 2 / ph[None]


ce, co = split(Cn)
print(f"parent leak about x0 = +0.195 um: even {ip(ce, ce).real / cc:.3f}  odd {ip(co, co).real / cc:.3f}")
names = ["r0_d5", "rh_d5", "r0_d2p5", "r0_d5_R", "rh_d5_R"][:len(B)]
for nm, b in zip(names, B):
    be, bo = split(b)
    print(f"{nm:8s} |b|^2/|c|^2 {ip(b, b).real / cc:.4f} (even {ip(be, be).real / cc:.4f} odd {ip(bo, bo).real / cc:.4f}) | "
          f"Re<c,b>/|c|^2 {ip(Cn, b).real / cc:+.4f} (even {ip(ce, be).real / cc:+.4f} odd {ip(co, bo).real / cc:+.4f})")
if len(B) >= 3:
    print(f"linearity r0: ||b5 - 2 b2.5|| / ||b5|| = {np.sqrt(ip(B[0] - 2 * B[2], B[0] - 2 * B[2]).real / ip(B[0], B[0]).real):.3f} (gate <= 0.25)")
if len(B) >= 5:                                     # arms independent: b_L = b_both - b_R per registration
    K = {"r0_R": B[3], "r0_L": B[0] - B[3], "rh_R": B[4], "rh_L": B[1] - B[4]}
    for nm, b in K.items():
        print(f"  knob {nm}: |b|^2/|c|^2 {ip(b, b).real / cc:.4f}  Re<c,b>/|c|^2 {ip(Cn, b).real / cc:+.4f}")
    names_k = list(K); Bk = [K[s] for s in names_k]; dW = rows[0]["dW"]; s = 1 / np.sqrt(cc)
    A = np.concatenate([np.stack([(b * np.sqrt(dW) * s).ravel().real for b in Bk], 1), np.stack([(b * np.sqrt(dW) * s).ravel().imag for b in Bk], 1)])
    y = np.concatenate([(Cn * np.sqrt(dW) * s).ravel().real, (Cn * np.sqrt(dW) * s).ravel().imag])
    a = lsq_linear(A, -y, bounds=(0, 1)).x
    res = Cn + sum(ai * bi for ai, bi in zip(a, Bk))
    print(f"NONNEGATIVE 4-knob solve (0 <= alpha <= 1, alpha = fraction of the 5-nm depth): P/P_parent {ip(res, res).real / cc:.3f}; "
          f"alpha {dict(zip(names_k, np.round(a, 3)))}; half step {ip(Cn + sum(ai / 2 * bi for ai, bi in zip(a, Bk)), Cn + sum(ai / 2 * bi for ai, bi in zip(a, Bk))).real / cc:.3f}")
