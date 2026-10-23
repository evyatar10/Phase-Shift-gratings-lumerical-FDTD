import numpy as np, scipy.linalg as sl
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reach_map.py")).read().split('print("=== gate')[0])   # reuse definitions only
a = 0.514/4; Y0 = 0.475
def feat(xi, comp, sx):   # x-pair at +-xi with x-sign sx, each with its y-image
    return images_y(comp, [xi, Y0, 0]) + images_y(list(np.array(comp)*sx), [-xi, Y0, 0])
def coef(src):
    aE, aM = multipoles(x, w, phi, *farfield(src), LMAX); return aE, aM
def share_table(ws, feats):
    src = []
    for wi, f in zip(ws, feats):
        src += [(wi*p, r) for p, r in f]
    global spectrum_cache
    aE, aM, P = spectrum(src); spectrum_cache = (aE, aM); Pl = P.sum(1)
    return cum(Pl,2), cum(Pl,5), top(P, 3)

print("=== (4) measured leak match: per-feature-class spectra at wall steps (1a..9a) ===")
for cls, comp, sx in [("p_y x-odd", [0,1,0], -1), ("p_x x-even", [1,0,0], +1)]:
    for n in [1,3,5,7,9]:
        c2, c5, t = share_table([1.0], [feat(n*a, comp, sx)])
        print(f"  {cls:10s} step {n}a={n*a:.3f}um  l<=2 {c2*100:4.1f}% l<=5 {c5*100:4.1f}%  " + ", ".join(f"{nm} {v*100:.1f}%" for v, nm in t))

print("\n=== (2a) max single-harmonic share: Rayleigh quotient over real weights on wall features ===")
targets = {"B": ["M(1,0)", "E(2,+-2)", "M(3,0)", "E(4,+-4)", "M(3,+-2)"], "A": ["E(1,+-1)", "E(3,+-3)", "M(2,+-1)", "E(3,+-1)", "M(4,+-1)"]}
classes = {"B": [("p_y", [0,1,0], -1), ("p_x", [1,0,0], +1)], "A": [("p_y", [0,1,0], +1), ("p_x", [1,0,0], -1)]}
def pick(aE, aM, name):
    kind, rest = name[0], name[2:-1]; l, m = rest.split(","); l = int(l); m = int(m.replace("+-", ""))
    A = aE if kind == "E" else aM
    v = [A[l, m+LMAX]] + ([A[l, -m+LMAX]] if m else [])
    return np.array(v)
for cls in "BA":
    for pos in [[1,3], [1,3,5], [1,3,5,7]]:
        feats, labels = [], []
        for nm, comp, sx in classes[cls]:
            for n in pos:
                feats.append(feat(n*a, comp, sx)); labels.append(f"{nm}@{n}a")
        C = [coef(f) for f in feats]
        tot = np.array([np.r_[c[0].ravel(), c[1].ravel()] for c in C])      # (nf, N)
        B = (tot @ tot.conj().T).real
        for tg in targets[cls]:
            T = np.array([pick(c[0], c[1], tg) for c in C]); A = (T @ T.conj().T).real
            ev, V = sl.eigh(A, B); wv = V[:, -1]; wv /= np.abs(wv).max()
            c2, c5, t = share_table(wv, feats)
            print(f"  class {cls} feats {pos}: {tg:9s} max share {ev[-1]*100:5.1f}%  weights " +
                  " ".join(f"{l}:{wi:+.2f}" for l, wi in zip(labels, wv)) + f"   | l<=2 {c2*100:.0f}% l<=5 {c5*100:.0f}%")

print("\n=== (2b/2c) null-space designs: zero chosen harmonics with 3-4 wall features (class B, p_y x-odd) ===")
for pos, kill in [([1,3,5], ["M(1,0)"]), ([1,3,5], ["E(2,+-2)"]), ([1,3,5,7], ["M(1,0)", "E(2,+-2)"]), ([1,3,5,7], ["E(4,+-4)"]), ([1,3,5,7], ["M(3,0)"])]:
    feats = [feat(n*a, [0,1,0], -1) for n in pos]; C = [coef(f) for f in feats]
    M = np.array([np.concatenate([pick(c[0], c[1], k) for k in kill]) for c in C]).T   # (ncons, nf)
    M = np.vstack([M.real, M.imag]); ns = sl.null_space(M)
    if ns.shape[1] == 0: print(f"  pos {pos} kill {kill}: no null space"); continue
    wv = ns[:, 0]; wv /= np.abs(wv).max()
    c2, c5, t = share_table(wv, feats)
    print(f"  pos {pos} kill {kill}: weights " + " ".join(f"{n}a:{wi:+.2f}" for n, wi in zip(pos, wv)) +
          f"  -> l<=2 {c2*100:.0f}% l<=5 {c5*100:.0f}% top " + ", ".join(f"{nm} {v*100:.1f}%" for v, nm in t))

print("\n=== Bessel-zero check: on-axis hole pair x-odd (p_y), l<=2 share vs x_h ===")
for xh in np.arange(0.5, 1.2, 0.05):
    c2, c5, t = share_table([1.0], [[(np.array([0,1.,0]), np.array([xh,0,0])), (np.array([0,-1.,0]), np.array([-xh,0,0]))]])
    print(f"  x_h={xh:.2f} um (k x_h={K*xh:.2f})  l<=2 {c2*100:5.1f}%  l<=5 {c5*100:5.1f}%  top {t[0][1]} {t[0][0]*100:.0f}%")
