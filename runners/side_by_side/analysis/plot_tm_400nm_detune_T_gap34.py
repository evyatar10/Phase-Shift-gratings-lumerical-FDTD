"""
TM side-by-side — device-1 PEAK TRANSMISSION vs device-2 corrugation, gaps 3 & 4 only.

Same data/extraction as plot_tm_400nm_detune.py (resonance_transmission at the stored
resonance_wavelength_nm, never argmax(T)), but a single figure: peak T = |S21|^2 vs the
full device-2 corrugation scan (360..650 nm), for gap Dy = 3 and 4 um only.

    python -m runners.side_by_side.analysis.plot_tm_400nm_detune_T_gap34
"""

import glob
import os
import re

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RESULTS_DIR = os.path.join(ROOT, "results_from_athena", "side_by_side_tm_detune_400nm", "results")
OUT_DIR = os.path.dirname(RESULTS_DIR)

GAPS_NM = [3000, 4000]   # only Dy = 3 and 4 um

_GAP_RE = re.compile(r"_Ygap(\d+)nm")
_CORR2_RE = re.compile(r"_corr2(\d+)nm")


def _s(m, k, default=np.nan):
    return float(np.asarray(m[k]).ravel()[0]) if k in m else default


def load_points():
    rows = []
    for fn in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*.mat"))):
        base = os.path.basename(fn)
        gm = _GAP_RE.search(base)
        cm = _CORR2_RE.search(base)
        m = sio.loadmat(fn)
        gap_nm = int(gm.group(1)) if gm else round(_s(m, "device_gap_m") * 1e9)
        if gap_nm not in GAPS_NM:
            continue
        corr2_nm = int(cm.group(1)) if cm else round(_s(m, "corrugation_depth_2") * 1e9)
        rows.append(dict(gap=gap_nm, corr2=corr2_nm, T=_s(m, "resonance_transmission")))
    return rows


def main():
    rows = load_points()
    if not rows:
        print(f"No matching result_*.mat in {RESULTS_DIR}")
        return
    corrs = sorted({r["corr2"] for r in rows})
    print("gap(nm) corr2(nm)  peakT")
    for g in GAPS_NM:
        for c in corrs:
            r = next((r for r in rows if r["gap"] == g and r["corr2"] == c), None)
            if r:
                print(f" {g:5d}   {c:5d}    {r['T']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for g in GAPS_NM:
        ys = [next((r["T"] for r in rows if r["gap"] == g and r["corr2"] == c), np.nan)
              for c in corrs]
        ax.plot(corrs, ys, "-o", label=f"gap {g/1000:g} µm")
    ax.axvline(400, ls=":", lw=1, color="0.5")   # matched reference (device-1 = 400 nm)
    ax.set_xlabel("device-2 corrugation (nm)")
    ax.set_ylabel("peak T = |S21|²")
    ax.set_title("TM side-by-side — device-1 peak transmission vs device-2 corrugation\n"
                 "(device-1 = 400 nm, pitch 516.83, N=80, Δx=0; gap 3 & 4 µm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "detune_transmission_gap34_TM.png")
    fig.savefig(p, dpi=140)
    print("wrote", p)


if __name__ == "__main__":
    main()
