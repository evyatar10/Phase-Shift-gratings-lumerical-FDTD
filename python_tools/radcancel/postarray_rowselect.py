# postarray_rowselect.py (study 2026-10-23, doc docs/HANDOFF_postarray_2026-10-23.md): which post ROWS (distances from the guide) and how
# long each row should be, from a saved Gram system ceiling_G{TAG}.npz (G, b, cc, xs, ys; written by postarray_ceiling.py).
# Greedy: best single row, then the row whose addition removes most leak, ... (non-negative amplitudes, any post size = model ceiling).
# Per chosen row: x-extent holding 90 % of the optimal amplitude. Model numbers are CONDITIONAL (calibration regime, CLAUDE.md 5).
#   python postarray_rowselect.py <ceiling_G file> [max_rows]
import sys
import numpy as np

d = np.load(sys.argv[1]); G, b, cc, xs, ys = d["G"], d["b"], float(d["cc"]), d["xs"] * 1e6, d["ys"] * 1e6   # um
MAX_ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
nx, nr = len(xs), len(ys)


def sites(rows):
    return np.concatenate([np.arange(r * nx, (r + 1) * nx) for r in rows])


def ceiling(rows, it=1500):
    """min (cc + 2 b.a + a G a)/cc over a >= 0 on the given rows (FISTA); returns amplitudes and leak ratio."""
    s = sites(rows); Gs, bs = G[np.ix_(s, s)], b[s]
    L = 2 * np.linalg.eigvalsh(Gs)[-1]; a = np.zeros(len(s)); z = a.copy(); t = 1.0
    for _ in range(it):
        a_new = np.maximum(z - (2 * Gs @ z + 2 * bs) / L, 0.0); t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = a_new + (t - 1) / t_new * (a_new - a); a, t = a_new, t_new
    return a, (cc + 2 * bs @ a + a @ Gs @ a) / cc


print("single rows:")
for r in range(nr):
    _, P = ceiling([r]); print(f"  y {ys[r]:.2f} um: removes {1 - P:.3f}  (Q_i x{1 / P:.2f} if loss = radiation)")
chosen = []
for _ in range(min(MAX_ROWS, nr)):
    best = None
    for r in range(nr):
        if r in chosen:
            continue
        a, P = ceiling(sorted(chosen + [r]))
        if best is None or P < best[0]:
            best = (P, sorted(chosen + [r]), a)
    P, chosen, a = best
    ext = []
    for i, r in enumerate(chosen):
        ai = a[i * nx:(i + 1) * nx]; order = np.argsort(np.abs(xs)); cum = np.cumsum(ai[order]) / max(ai.sum(), 1e-30)
        ext.append(f"y {ys[r]:.2f}: 90 % within |x| <= {np.abs(xs[order])[np.searchsorted(cum, 0.9)]:.0f} um")
    print(f"best {len(chosen)} rows {[round(ys[r], 2) for r in chosen]}: removes {1 - P:.3f} (x{1 / P:.2f}) | " + "; ".join(ext))
