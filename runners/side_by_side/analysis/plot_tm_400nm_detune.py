"""
Device-2 corrugation-DETUNE deliverable for the TM side-by-side scan
(runners/side_by_side/side_by_side_tm_detune_400nm.py).

Same criteria as the gap x stagger maps (plot_tm_400nm_map / _splitting / _coupling),
but the swept axis here is device-2 CORRUGATION (360..440 nm) at fixed stagger 0, for
three gaps (Dy = 1, 2, 3 um). With only three gaps and one stagger, a heatmap is
wasteful — so each metric is drawn as a LINE vs corrugation-2 with one curve per gap.

Metrics (identical extraction to the map scripts):
  - device-1 PEAK transmission at the defect resonance (resonance_transmission)
  - Q = resonance_wavelength_nm / |spectral_fwhm_nm|
  - supermode splitting Dlambda (two-peak search within +-10 nm of the resonance)
  - coupling into device 2 at resonance  (coupling_total = |S31|^2 + |S41|^2)
  - device-1 radiated-but-not-captured loss at resonance (loss_4port)

Geometry per file: device-2 corrugation and gap are parsed from the filename tag
(..._Ygap{gap}nm_..._corr2{d2}nm), which generate_file_tag always writes; the gap is
cross-checked against device_gap_m when present. The resonance index uses the stored
resonance_wavelength_nm (never argmax(T)).

    python -m runners.side_by_side.analysis.plot_tm_400nm_detune
"""

import glob
import os
import re

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RESULTS_DIR = os.path.join(ROOT, "results_from_athena", "side_by_side_tm_detune_400nm", "results")
OUT_DIR = os.path.dirname(RESULTS_DIR)

# Same defect-band window as plot_tm_400nm_splitting.py — keeps the top-2 picker off
# the Bragg band-edge ripples (~15+ nm away) that masquerade as a second supermode.
SEARCH_HALF_NM = 10.0

_GAP_RE = re.compile(r"_Ygap(\d+)nm")
_CORR2_RE = re.compile(r"_corr2(\d+)nm")


def _s(m, k, default=np.nan):
    return float(np.asarray(m[k]).ravel()[0]) if k in m else default


def two_peaks(wl, T, res_nm):
    """Both supermode peaks within +-SEARCH_HALF_NM of the cavity resonance."""
    if not np.isfinite(res_nm):
        res_nm = wl[int(np.argmax(T))]
    sel = np.abs(wl - res_nm) <= SEARCH_HALF_NM
    w, t = wl[sel], T[sel]
    if w.size < 5:
        return []
    pk, _ = find_peaks(t, height=0.03, distance=20)
    pk = sorted(pk, key=lambda p: -t[p])[:2]
    pk = sorted(pk, key=lambda p: w[p])
    return [(float(w[p]), float(t[p])) for p in pk]


def load_points():
    rows = []
    for fn in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*.mat"))):
        base = os.path.basename(fn)
        m = sio.loadmat(fn)

        gm = _GAP_RE.search(base)
        cm = _CORR2_RE.search(base)
        gap_nm = int(gm.group(1)) if gm else round(_s(m, "device_gap_m") * 1e9)
        corr2_nm = int(cm.group(1)) if cm else round(_s(m, "corrugation_depth_2") * 1e9)

        wl = np.asarray(m["wl_nm"]).ravel()
        T = np.asarray(m["T"]).ravel()
        o = np.argsort(wl); wl, T = wl[o], T[o]

        res = _s(m, "resonance_wavelength_nm")
        i = int(np.argmin(np.abs(wl - res))) if np.isfinite(res) else int(np.argmax(T))

        Tres = _s(m, "resonance_transmission")
        fwhm = abs(_s(m, "spectral_fwhm_nm"))                 # stored negative
        Q = res / fwhm if (np.isfinite(res) and fwhm > 0) else np.nan

        pks = two_peaks(wl, T, res)
        split = abs(pks[1][0] - pks[0][0]) if len(pks) == 2 else 0.0

        ct = np.asarray(m["coupling_total"]).ravel() if "coupling_total" in m else None
        l4 = np.asarray(m["loss_4port"]).ravel() if "loss_4port" in m else None
        C = float(ct[o][i]) if ct is not None else np.nan
        L = float(l4[o][i]) if l4 is not None else np.nan

        rows.append(dict(gap=gap_nm, corr2=corr2_nm, T=Tres, Q=Q, fwhm=fwhm,
                         lam=res, split=split, npk=len(pks), C=C, L=L, file=base))
    return rows


def _lines(ax, rows, gaps, corrs, key, ylabel, title):
    for g in gaps:
        ys = [next((r[key] for r in rows if r["gap"] == g and r["corr2"] == c), np.nan)
              for c in corrs]
        ax.plot(corrs, ys, "-o", label=f"gap {g/1000:g} µm")
    ax.axvline(400, ls=":", lw=1, color="0.5")   # matched reference (device-1 = 400 nm)
    ax.set_xlabel("device-2 corrugation (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def main():
    rows = load_points()
    if not rows:
        print(f"No result_*.mat found in {RESULTS_DIR}")
        return
    gaps = sorted({r["gap"] for r in rows})
    corrs = sorted({r["corr2"] for r in rows})
    print(f"Loaded {len(rows)} results — gaps={gaps} nm, device-2 corr={corrs} nm")
    print("gap(nm) corr2(nm)  peakT     Q    split(nm)  coupling  loss4")
    for g in gaps:
        for c in corrs:
            r = next((r for r in rows if r["gap"] == g and r["corr2"] == c), None)
            if r:
                print(f" {g:5d}   {c:5d}    {r['T']:.4f}  {r['Q']:6.0f}   {r['split']:5.2f}    "
                      f"{r['C']:.4f}   {r['L']:.4f}   ({r['file']})")

    # Figure A — device-1 / resonance criteria vs device-2 corrugation
    figA, axs = plt.subplots(1, 3, figsize=(18, 5))
    _lines(axs[0], rows, gaps, corrs, "T", "peak T = |S21|²",
           "Device-1 peak transmission at resonance")
    _lines(axs[1], rows, gaps, corrs, "Q", "Q = λ / |FWHM|", "Device-1 Q factor")
    _lines(axs[2], rows, gaps, corrs, "split", "Δλ (nm)  [0 = single peak]",
           "Supermode splitting")
    figA.suptitle("TM side-by-side — detune device-2 corrugation (device-1 = 400 nm, "
                  "pitch 516.83, N=80, Δx=0)")
    figA.tight_layout()
    pA = os.path.join(OUT_DIR, "detune_records_TM.png")
    figA.savefig(pA, dpi=140); print("wrote", pA)

    # Figure B — coupling / radiation channel vs device-2 corrugation
    figB, axs = plt.subplots(1, 2, figsize=(13, 5))
    _lines(axs[0], rows, gaps, corrs, "C", "coupling → device 2  (|S31|²+|S41|²)",
           "Coupling into device 2 at resonance")
    _lines(axs[1], rows, gaps, corrs, "L", "loss_4port",
           "Device-1 radiated, NOT captured")
    figB.suptitle("TM side-by-side — coupling / radiation vs device-2 corrugation (Δx=0)")
    figB.tight_layout()
    pB = os.path.join(OUT_DIR, "detune_coupling_TM.png")
    figB.savefig(pB, dpi=140); print("wrote", pB)


if __name__ == "__main__":
    main()
