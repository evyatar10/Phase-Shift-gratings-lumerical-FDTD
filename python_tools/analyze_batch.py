"""
Batch result reader for the BIC/Kerker/CD program. Loads every result_*.mat in
a study's results/ dir and prints a correctness-checked table:

  resonance_wavelength_nm (finder value, NOT argmax T), peak T, loss at
  resonance (1 - T - R there), Q = lam/|spectral_fwhm_nm|, fwhm_m (spatial mode
  width, um), n_peaks in window (single-resonance check for FW-BIC), and the
  CLAUDE.md sanity flags (in-window? peak above the dead floor 0.0008?).

Usage:
  python python_tools/analyze_batch.py <results_dir> [--tsv out.tsv]
  # defaults to results_from_athena/<study>/results if a bare study name given

The table is sorted by filename. Loss delta vs a control is left to the reader
(the study docstrings name the control rows) — this tool reports absolutes +
flags, honestly, per row.
"""

import os
import sys
import glob

import numpy as np
from scipy.io import loadmat

DEAD_FLOOR = 0.02      # peak T below this = suspect dead/off-resonance (TM healthy ~0.83+)


def count_peaks(wl, T, rel_prom=0.15, min_sep_nm=1.0):
    """Number of T maxima above rel_prom*max within the window, separated by
    >= min_sep_nm. Coarse single-resonance check for the FW-BIC scan."""
    T = np.asarray(T, float)
    wl = np.asarray(wl, float)
    thr = rel_prom * np.nanmax(T)
    peaks = []
    for i in range(1, len(T) - 1):
        if T[i] > T[i - 1] and T[i] >= T[i + 1] and T[i] > thr:
            if peaks and abs(wl[i] - wl[peaks[-1]]) < min_sep_nm:
                if T[i] > T[peaks[-1]]:
                    peaks[-1] = i
                continue
            peaks.append(i)
    return len(peaks), [round(float(wl[i]), 2) for i in peaks]


def analyze_one(path):
    m = loadmat(path, squeeze_me=True)
    wl = np.asarray(m["wl_nm"], float)
    T = np.asarray(m["T"], float)
    R = np.asarray(m.get("R", np.zeros_like(T)), float)
    lam = float(m["resonance_wavelength_nm"])
    peakT = float(m["resonance_transmission"])
    fwhm_nm = abs(float(m["spectral_fwhm_nm"]))
    fwhm_m = float(m.get("fwhm_m", np.nan))
    Q = lam / fwhm_nm if fwhm_nm > 0 else np.nan
    # loss at resonance
    j = int(np.argmin(np.abs(wl - lam)))
    loss_res = float(np.asarray(m["loss"], float)[j]) if "loss" in m else (1.0 - T[j] - R[j])
    in_window = wl.min() <= lam <= wl.max()
    npk, pk_wl = count_peaks(wl, T)
    flags = []
    if not in_window:
        flags.append("OFF-WINDOW")
    if peakT < DEAD_FLOOR:
        flags.append("DEAD?")
    if not np.isfinite(lam):
        flags.append("NO-RES")
    return dict(name=os.path.basename(path), lam=lam, peakT=peakT, loss=loss_res,
                Q=Q, fwhm_um=fwhm_m * 1e6, npk=npk, pk_wl=pk_wl,
                flag=",".join(flags) if flags else "ok")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    arg = sys.argv[1]
    if os.path.isdir(arg):
        rdir = arg
    else:
        rdir = os.path.join("results_from_athena", arg, "results")
    paths = sorted(glob.glob(os.path.join(rdir, "result_*.mat")))
    if not paths:
        print(f"no result_*.mat in {rdir}")
        sys.exit(1)
    rows = []
    for p in paths:
        try:
            rows.append(analyze_one(p))
        except Exception as e:
            rows.append(dict(name=os.path.basename(p), lam=np.nan, peakT=np.nan,
                             loss=np.nan, Q=np.nan, fwhm_um=np.nan, npk=-1,
                             pk_wl=[], flag=f"READ-ERR:{e}"))
    print(f"{len(rows)} results in {rdir}\n")
    hdr = f"{'file':<62} {'lam':>8} {'peakT':>7} {'loss':>7} {'Q':>6} {'fwhmum':>7} {'npk':>3} {'flag'}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name'][:62]:<62} {r['lam']:>8.2f} {r['peakT']:>7.4f} "
              f"{r['loss']:>7.4f} {r['Q']:>6.0f} {r['fwhm_um']:>7.2f} "
              f"{r['npk']:>3d} {r['flag']}")
    # optional TSV
    if "--tsv" in sys.argv:
        out = sys.argv[sys.argv.index("--tsv") + 1]
        with open(out, "w") as f:
            f.write("file\tlam_nm\tpeakT\tloss\tQ\tfwhm_um\tnpk\tpk_wl\tflag\n")
            for r in rows:
                f.write(f"{r['name']}\t{r['lam']:.3f}\t{r['peakT']:.5f}\t{r['loss']:.5f}\t"
                        f"{r['Q']:.1f}\t{r['fwhm_um']:.3f}\t{r['npk']}\t"
                        f"{'|'.join(map(str, r['pk_wl']))}\t{r['flag']}\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
