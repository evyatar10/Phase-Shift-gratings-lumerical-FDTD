"""
TM side-by-side — the COUPLING / RADIATION channel (vs the splitting channel).

Maps device-2 capture (coupling_total = |S31|²+|S41|² at the device-1 resonance) and
the device-1 radiated-but-not-captured loss (loss_4port) over (gap, stagger), plus a
coupling-vs-stagger line cut per gap. The near-field (evanescent) coupling peaks at
Δx=0 and decays with gap; the RADIATIVE coupling shows up as a bump near the TM
forward lobe Δx ≈ 5.7·gap (paper_8) — in range only for gap ≈ 1 µm in this grid.

    python -m runners.side_by_side.analysis.plot_tm_400nm_coupling
"""
import glob, os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RESULTS_DIR = os.path.join(ROOT, "results_from_athena", "side_by_side_tm_400nm", "results")
OUT_DIR = os.path.dirname(RESULTS_DIR)


def _s(m, k):
    return float(np.asarray(m[k]).ravel()[0]) if k in m else np.nan


def load():
    rows = []
    for fn in glob.glob(os.path.join(RESULTS_DIR, "result_*.mat")):
        m = sio.loadmat(fn)
        wl = np.asarray(m["wl_nm"]).ravel()
        res = _s(m, "resonance_wavelength_nm"); i = int(np.argmin(np.abs(wl - res)))
        ct = np.asarray(m["coupling_total"]).ravel() if "coupling_total" in m else np.full_like(wl, np.nan)
        l4 = np.asarray(m["loss_4port"]).ravel() if "loss_4port" in m else np.full_like(wl, np.nan)
        rows.append(dict(gap=round(_s(m, "device_gap_m")*1e9), stag=round(_s(m, "device_stagger_m")*1e9),
                         C=float(ct[i]), L=float(l4[i])))
    return rows


def grid(rows, gaps, stags, key):
    G = np.full((len(gaps), len(stags)), np.nan)
    for r in rows:
        G[gaps.index(r["gap"]), stags.index(r["stag"])] = r[key]
    return G


def heat(ax, M, gaps, stags, title, cmap, cbar):
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(stags))); ax.set_xticklabels([f"{s/1000:g}" for s in stags])
    ax.set_yticks(range(len(gaps)));  ax.set_yticklabels([f"{g/1000:g}" for g in gaps])
    ax.set_xlabel("Δx stagger (µm)"); ax.set_ylabel("Δy gap (µm)"); ax.set_title(title)
    for i in range(len(gaps)):
        for j in range(len(stags)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.5*np.nanmax(M) else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar)


def main():
    rows = load()
    gaps = sorted({r["gap"] for r in rows}); stags = sorted({r["stag"] for r in rows})
    C = grid(rows, gaps, stags, "C"); L = grid(rows, gaps, stags, "L")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    heat(axs[0], C, gaps, stags, "Coupling → device 2 at resonance  (|S31|²+|S41|²)", "viridis", "coupling")
    heat(axs[1], L, gaps, stags, "Device-1 radiated, NOT captured (loss_4port)", "inferno", "loss_4port")
    fig.suptitle("TM side-by-side — coupling / radiation channel (evanescent peaks Δx=0; TM lobe ≈ 5.7·gap)")
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "map_coupling_TM.png"); fig.savefig(p1, dpi=140); print("wrote", p1)

    # line cuts: coupling vs stagger per gap, with the 5.7·gap lobe marker per gap
    fig2, ax = plt.subplots(figsize=(10, 6))
    for gi, g in enumerate(gaps):
        line, = ax.plot(np.array(stags)/1000, C[gi, :], "-o", label=f"gap {g/1000:g} µm")
        lobe = 5.7 * g / 1000
        if lobe <= max(stags)/1000 + 0.3:
            ax.axvline(lobe, ls=":", lw=1, color=line.get_color())
    ax.set_xlabel("Δx stagger (µm)"); ax.set_ylabel("coupling → device 2 at resonance")
    ax.set_title("Coupling vs stagger (dotted = TM forward lobe Δx = 5.7·gap; only gap 1 µm reaches it here)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(OUT_DIR, "coupling_vs_stagger_TM.png"); fig2.savefig(p2, dpi=140); print("wrote", p2)


if __name__ == "__main__":
    main()
