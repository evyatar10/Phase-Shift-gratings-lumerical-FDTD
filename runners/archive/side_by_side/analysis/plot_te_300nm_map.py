"""
Build the device-1 deliverable maps for the clean TE side-by-side scan
(runners/side_by_side/side_by_side_te_300nm.py).

Figure of merit = device-1 input->output PEAK TRANSMISSION at the pi-shift defect
resonance (resonance_transmission, found by find_bragg_resonance). Plotted as a 2D
heatmap over (Δy gap, Δx stagger). Q (= λ_res / |spectral_fwhm_nm|) and the spectral
FWHM and resonance wavelength are plotted alongside as records.

Reads every result_*.mat in the results dir; geometry (gap, stagger) is read from
the .mat itself (not parsed from the filename). Robust to missing tasks.

    python -m runners.side_by_side.analysis.plot_te_300nm_map
"""

import glob
import os

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RESULTS_DIR = os.path.join(ROOT, "results_from_athena", "side_by_side_te_300nm", "results")
OUT_DIR = os.path.dirname(RESULTS_DIR)


def _scalar(m, key, default=np.nan):
    return float(np.asarray(m[key]).ravel()[0]) if key in m else default


def load_points():
    rows = []
    for fn in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*.mat"))):
        m = sio.loadmat(fn)
        gap_nm = round(_scalar(m, "device_gap_m") * 1e9)
        stag_nm = round(_scalar(m, "device_stagger_m") * 1e9)
        Tres = _scalar(m, "resonance_transmission")
        lam = _scalar(m, "resonance_wavelength_nm")
        fwhm = abs(_scalar(m, "spectral_fwhm_nm"))           # stored negative
        Q = lam / fwhm if fwhm > 0 else np.nan
        rows.append(dict(gap=gap_nm, stag=stag_nm, T=Tres, lam=lam, fwhm=fwhm, Q=Q, file=os.path.basename(fn)))
    return rows


def grid(rows, gaps, stags, key):
    G = np.full((len(gaps), len(stags)), np.nan)
    for r in rows:
        if r["gap"] in gaps and r["stag"] in stags:
            G[gaps.index(r["gap"]), stags.index(r["stag"])] = r[key]
    return G


def heatmap(ax, M, gaps, stags, title, cmap, fmt="{:.3f}", cbar_label=""):
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
                   extent=[-0.5, len(stags) - 0.5, -0.5, len(gaps) - 0.5])
    ax.set_xticks(range(len(stags))); ax.set_xticklabels([f"{s/1000:g}" for s in stags])
    ax.set_yticks(range(len(gaps)));  ax.set_yticklabels([f"{g/1000:g}" for g in gaps])
    ax.set_xlabel("Δx stagger (µm)"); ax.set_ylabel("Δy gap (µm)")
    ax.set_title(title)
    for i in range(len(gaps)):
        for j in range(len(stags)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=7, color="black")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label)


def main():
    rows = load_points()
    if not rows:
        print(f"No result_*.mat found in {RESULTS_DIR}")
        return
    gaps = sorted({r["gap"] for r in rows})
    stags = sorted({r["stag"] for r in rows})
    print(f"Loaded {len(rows)} results — gaps={gaps} nm, staggers={stags} nm")

    T = grid(rows, gaps, stags, "T")
    Q = grid(rows, gaps, stags, "Q")
    F = grid(rows, gaps, stags, "fwhm")
    L = grid(rows, gaps, stags, "lam")

    # best device-1 transmission
    bi = np.unravel_index(np.nanargmax(T), T.shape)
    print(f"MAX device-1 peak T = {T[bi]:.4f}  at gap={gaps[bi[0]]} nm, stagger={stags[bi[1]]} nm")
    print(f"MIN device-1 peak T = {np.nanmin(T):.4f}")

    # ── Headline: device-1 peak transmission map (standalone) ────────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    heatmap(ax, T, gaps, stags, "Device-1 PEAK TRANSMISSION at resonance (TE, 300 nm corr)",
            "viridis", "{:.3f}", "peak T = |S21|²")
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "map_transmission_TE.png")
    fig.savefig(p1, dpi=140); print("wrote", p1)

    # ── Records: Q, FWHM, λ_res ───────────────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    heatmap(axs[0], Q, gaps, stags, "Q factor (= λ / |FWHM|)", "plasma", "{:.0f}", "Q")
    heatmap(axs[1], F, gaps, stags, "Spectral FWHM (nm)", "cividis", "{:.2f}", "FWHM (nm)")
    heatmap(axs[2], L, gaps, stags, "Resonance wavelength (nm)", "magma", "{:.1f}", "λ_res (nm)")
    fig.suptitle("Device-1 resonance records — TE, 300 nm corrugation, N=80")
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "map_records_TE.png")
    fig.savefig(p2, dpi=140); print("wrote", p2)


if __name__ == "__main__":
    main()
