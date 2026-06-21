"""
Interactive (MATLAB-style) viewer for the TM period-match study.

Unlike plot_tm_periods_match_te.py (which forces the headless 'Agg' backend to
write PNGs), this script uses the default GUI backend (TkAgg on this machine) and
plt.show(), so you get a real window with the navigation toolbar: rubber-band
zoom, pan, home/back, live cursor x/y readout, and save — just like a MATLAB
figure. A mouse-hover annotation prints the nearest (lambda, T) on each curve.

Run locally (NOT on the headless server):
    python runners/sweeps/view_tm_match_interactive.py
    python runners/sweeps/view_tm_match_interactive.py --all     # overlay every TM N
    python runners/sweeps/view_tm_match_interactive.py --dir <folder>

Default folder: results_from_athena/tm_match_bisect/results
"""

import argparse
import glob
import os
import re

import numpy as np
import scipy.io as sio

# Prefer the Qt backend: its toolbar has the "Edit axis, curve and image
# parameters" button (a MATLAB-style property editor for line color/style/marker/
# label, axis titles/limits/scale, and the legend). The Tk backend can only
# zoom/pan/save. Fall back to whatever interactive backend exists if Qt is absent.
import matplotlib
for _bk in ("QtAgg", "TkAgg"):
    try:
        matplotlib.use(_bk)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt


def _load(path):
    d = sio.loadmat(path)
    wl = np.asarray(d["wl_nm"]).squeeze().astype(float)
    T = np.asarray(d["T"]).squeeze().astype(float)
    o = np.argsort(wl)
    wl, T = wl[o], T[o]
    lam = float(np.asarray(d["resonance_wavelength_nm"]).squeeze())
    fwhm = abs(float(np.asarray(d["spectral_fwhm_nm"]).squeeze()))  # stored sign-flipped
    Q = lam / fwhm if fwhm > 0 else float("nan")
    return wl, T, lam, Q


def main():
    ap = argparse.ArgumentParser(description="Interactive TE@80 vs period-matched TM viewer.")
    ap.add_argument("--dir", default=os.path.join("results_from_athena", "tm_match_bisect", "results"))
    ap.add_argument("--all", action="store_true", help="overlay every TM period count, not just the match")
    ap.add_argument("--match-N", type=int, default=132, help="matched TM period count to highlight")
    args = ap.parse_args()

    te_files = [p for p in glob.glob(os.path.join(args.dir, "result_*.mat"))
                if re.search(r"_te(_|\.)", os.path.basename(p)) and "summary" not in p]
    tm_files = [p for p in glob.glob(os.path.join(args.dir, "result_*.mat"))
                if re.search(r"_tm(_|\.)", os.path.basename(p)) and "summary" not in p]
    if not te_files or not tm_files:
        raise SystemExit(f"No TE/TM result_*.mat found in {args.dir}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    curves = []  # (label, line)

    wl, T, lam, Q = _load(te_files[0])
    (ln,) = ax.plot(wl, T, color="C1", lw=1.8, label=f"TE  N=80  (Q~{Q:.0f})")
    curves.append((ln.get_label(), ln))

    tm_by_N = {}
    for p in tm_files:
        m = re.search(r"_?N(\d+)", os.path.basename(p))
        if m:
            tm_by_N[int(m.group(1))] = p

    if args.all:
        for n in sorted(tm_by_N):
            wl, T, lam, Q = _load(tm_by_N[n])
            (ln,) = ax.plot(wl, T, lw=1.2, alpha=0.85,
                            label=f"TM  N={n}  (Q~{Q:.0f})")
            curves.append((ln.get_label(), ln))
    else:
        p = tm_by_N.get(args.match_N) or tm_by_N[max(tm_by_N)]
        wl, T, lam, Q = _load(p)
        (ln,) = ax.plot(wl, T, color="C0", lw=1.8,
                        label=f"TM  N={args.match_N}  (Q~{Q:.0f})")
        curves.append((ln.get_label(), ln))

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Transmission, T")
    ax.set_title("Combined transmission: TE@80 vs period-matched TM  (interactive)")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best")

    # Hover readout: nearest point on the nearest curve.
    annot = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w", alpha=0.9),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            if annot.get_visible():
                annot.set_visible(False); fig.canvas.draw_idle()
            return
        best = None
        for label, ln in curves:
            xd, yd = ln.get_xdata(), ln.get_ydata()
            i = int(np.argmin(np.abs(xd - event.xdata)))
            d = abs(yd[i] - event.ydata)
            if best is None or d < best[0]:
                best = (d, xd[i], yd[i], label)
        if best:
            _, x, y, label = best
            annot.xy = (x, y)
            annot.set_text(f"{label.split('(')[0].strip()}\nλ={x:.3f} nm\nT={y:.4f}")
            annot.set_visible(True); fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.tight_layout()
    print(f"[viewer] backend = {plt.get_backend()} — close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
