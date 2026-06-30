"""
Supermode-splitting analysis for the TM side-by-side grid. TM analog of
plot_te_300nm_splitting.py.

When the two cavities are close they couple and the single defect resonance splits
into TWO supermodes. The single-value resonance_transmission map
(plot_tm_400nm_map.py) reports only one of them and is misleading for small gaps.
This script detects both peaks per config and maps:
  - splitting Δλ between the two supermodes  (≈ coupling strength)
  - the higher of the two supermode transmissions

    python -m runners.side_by_side.analysis.plot_tm_400nm_splitting
"""

import glob
import os

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RESULTS_DIR = os.path.join(ROOT, "results_from_athena", "side_by_side_tm_400nm", "results")
OUT_DIR = os.path.dirname(RESULTS_DIR)


# Search only the defect band (±SEARCH_HALF_NM around the cavity resonance) for the
# supermode peaks. Without this, the TM 400 nm window also contains Bragg band-edge
# ripples (~15+ nm from the defect) that the simple top-2 picker mistakes for a
# second supermode → spurious 17 nm "splitting" at gap 1.4 µm.
SEARCH_HALF_NM = 10.0


def two_peaks(wl, T, res_nm):
    if not np.isfinite(res_nm):
        res_nm = wl[int(np.argmax(T))]
    sel = np.abs(wl - res_nm) <= SEARCH_HALF_NM
    w, t = wl[sel], T[sel]
    if w.size < 5:
        return []
    pk, _ = find_peaks(t, height=0.03, distance=20)
    pk = sorted(pk, key=lambda p: -t[p])[:2]
    pk = sorted(pk, key=lambda p: w[p])
    peaks = [(float(w[p]), float(t[p])) for p in pk]
    return peaks


def load():
    rows = []
    for fn in glob.glob(os.path.join(RESULTS_DIR, "result_*.mat")):
        m = sio.loadmat(fn)
        wl = np.asarray(m["wl_nm"]).ravel(); T = np.asarray(m["T"]).ravel()
        o = np.argsort(wl); wl, T = wl[o], T[o]
        g = round(float(np.asarray(m["device_gap_m"]).ravel()[0]) * 1e9)
        s = round(float(np.asarray(m["device_stagger_m"]).ravel()[0]) * 1e9)
        res_nm = float(np.asarray(m["resonance_wavelength_nm"]).ravel()[0]) if "resonance_wavelength_nm" in m else np.nan
        pks = two_peaks(wl, T, res_nm)
        split = abs(pks[1][0] - pks[0][0]) if len(pks) == 2 else 0.0
        tmax = max(p[1] for p in pks) if pks else np.nan
        rows.append(dict(gap=g, stag=s, npk=len(pks), split=split, tmax=tmax, peaks=pks))
    return rows


def grid(rows, gaps, stags, key):
    G = np.full((len(gaps), len(stags)), np.nan)
    for r in rows:
        G[gaps.index(r["gap"]), stags.index(r["stag"])] = r[key]
    return G


def heatmap(ax, M, gaps, stags, title, cmap, fmt, cbar):
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(stags))); ax.set_xticklabels([f"{s/1000:g}" for s in stags])
    ax.set_yticks(range(len(gaps)));  ax.set_yticklabels([f"{g/1000:g}" for g in gaps])
    ax.set_xlabel("Δx stagger (µm)"); ax.set_ylabel("Δy gap (µm)"); ax.set_title(title)
    for i in range(len(gaps)):
        for j in range(len(stags)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar)


def main():
    rows = load()
    gaps = sorted({r["gap"] for r in rows}); stags = sorted({r["stag"] for r in rows})
    S = grid(rows, gaps, stags, "split")
    Tm = grid(rows, gaps, stags, "tmax")

    # splitting vs gap, averaged over stagger (it's gap-driven)
    print("gap (nm)  mean split (nm)  #two-peak")
    for g in gaps:
        gr = [r for r in rows if r["gap"] == g]
        sp = np.mean([r["split"] for r in gr if r["npk"] == 2]) if any(r["npk"] == 2 for r in gr) else 0
        n2 = sum(1 for r in gr if r["npk"] == 2)
        print(f"  {g:5d}     {sp:6.2f}        {n2}/{len(gr)}")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    heatmap(axs[0], S, gaps, stags, "Supermode splitting Δλ (nm)  [0 = single peak]",
            "inferno", "{:.1f}", "Δλ (nm)")
    heatmap(axs[1], Tm, gaps, stags, "Best supermode peak T",
            "viridis", "{:.3f}", "max peak T")
    fig.suptitle("TM side-by-side — supermode splitting & transmission (N=80, 400 nm, n=1.97)")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "map_splitting_TM.png")
    fig.savefig(p, dpi=140); print("wrote", p)


if __name__ == "__main__":
    main()
