"""
Summary plots for the innermost-tooth-shift sweep (runners/sweeps/tm_te_shift.py).

Standalone — imports only glob/scipy/matplotlib (NO lumapi, NO project config),
so it runs in a lightweight CPU job on Athena (athena/jobs/run_shift_summary.sh)
right after the sweep array, and the PNGs it writes download with `--results-no-fsp`.
It can also be run locally on a folder of downloaded result_*.mat files.

For every result_*.mat in --results-dir it parses:
    _S<n>  -> shift in nm   (no _S token  => 0, i.e. no shift)
    _TM    -> TM            (otherwise TE)
and reads `resonance_transmission` (peak T) and `fwhm_m` (spatial mode width).
It writes into the same folder:
    transmission_vs_shift.png   peak T vs shift (TE & TM)
    modewidth_vs_shift.png      mode width FWHM [um] vs shift (TE & TM)
    shift_summary.csv           pol,shift_nm,T_peak,mode_width_um,lambda_nm

Usage:
    python runners/sweeps/plot_tm_te_shift.py [--results-dir DIR]
"""

import argparse
import glob
import os
import re

import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt


FONT_SIZE = 13


def _scalar(d, key):
    """Return a float for a scipy-loaded scalar field, or None if absent/empty."""
    if key not in d:
        return None
    try:
        return float(np.asarray(d[key]).squeeze())
    except (TypeError, ValueError):
        return None


def _parse(fname):
    """(shift_nm, is_tm) from a result filename."""
    m = re.search(r"_S(\d+)", fname)
    shift_nm = int(m.group(1)) if m else 0
    is_tm = "_TM" in fname
    return shift_nm, is_tm


def collect(results_dir):
    """Read every result_*.mat into a list of row dicts."""
    rows = []
    for fp in sorted(glob.glob(os.path.join(results_dir, "result_*.mat"))):
        fname = os.path.basename(fp)
        shift_nm, is_tm = _parse(fname)
        try:
            d = sio.loadmat(fp)
        except Exception as e:  # noqa: BLE001 — skip unreadable files, keep going
            print(f"  WARN: could not read {fname}: {e}")
            continue

        T_peak = _scalar(d, "resonance_transmission")
        if T_peak is None and "T" in d and "wl_nm" in d and "resonance_wavelength_nm" in d:
            wl = np.asarray(d["wl_nm"]).squeeze()
            T = np.asarray(d["T"]).squeeze()
            lam = _scalar(d, "resonance_wavelength_nm")
            if wl.size and T.size and lam is not None:
                T_peak = float(T[int(np.argmin(np.abs(wl - lam)))])

        fwhm_m = _scalar(d, "fwhm_m")
        width_um = fwhm_m * 1e6 if fwhm_m is not None else np.nan
        lam_nm = _scalar(d, "resonance_wavelength_nm")

        rows.append({
            "fname": fname,
            "shift_nm": shift_nm,
            "pol": "TM" if is_tm else "TE",
            "T_peak": T_peak if T_peak is not None else np.nan,
            "width_um": width_um,
            "lambda_nm": lam_nm if lam_nm is not None else np.nan,
        })
        print(f"  {fname:48s}  pol={rows[-1]['pol']}  shift={shift_nm:3d}nm  "
              f"T={rows[-1]['T_peak']:.4f}  width={width_um:.3f}um")
    return rows


def _series(rows, pol, key):
    """Sorted (shift, value) arrays for one polarization."""
    pts = sorted((r["shift_nm"], r[key]) for r in rows if r["pol"] == pol)
    if not pts:
        return np.array([]), np.array([])
    x, y = zip(*pts)
    return np.array(x, float), np.array(y, float)


def _plot(rows, key, ylabel, title, out_png):
    te_x, te_y = _series(rows, "TE", key)
    tm_x, tm_y = _series(rows, "TM", key)
    fig, ax = plt.subplots()
    if te_x.size:
        ax.plot(te_x, te_y, "o-", lw=1.6, ms=7, mfc="w", label="TE")
    if tm_x.size:
        ax.plot(tm_x, tm_y, "s-", lw=1.6, ms=7, mfc="w", label="TM")
    ax.set_xlabel("Innermost-tooth shift [nm]", fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)
    all_x = np.unique(np.concatenate([te_x, tm_x])) if (te_x.size or tm_x.size) else None
    if all_x is not None and all_x.size:
        ax.set_xticks(all_x)
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def write_csv(rows, out_csv):
    order = sorted(rows, key=lambda r: (r["pol"], r["shift_nm"]))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("pol,shift_nm,T_peak,mode_width_um,lambda_nm\n")
        for r in order:
            f.write(f"{r['pol']},{r['shift_nm']},{r['T_peak']:.6f},"
                    f"{r['width_um']:.6f},{r['lambda_nm']:.4f}\n")
    print(f"  wrote {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Summary plots for the tooth-shift sweep.")
    ap.add_argument("--results-dir", default=os.path.join("results", "tm_te_shift"),
                    help="Folder containing result_*.mat (default: results/tm_te_shift).")
    args = ap.parse_args()

    results_dir = args.results_dir
    print(f"[plot_tm_te_shift] scanning {results_dir}")
    rows = collect(results_dir)
    if not rows:
        print("  No result_*.mat found — nothing to plot.")
        return

    _plot(rows, "T_peak", "Peak transmission (T)",
          "Resonance transmission vs. tooth shift",
          os.path.join(results_dir, "transmission_vs_shift.png"))
    _plot(rows, "width_um", r"Mode width, FWHM [$\mu$m]",
          "Spatial mode width vs. tooth shift",
          os.path.join(results_dir, "modewidth_vs_shift.png"))
    write_csv(rows, os.path.join(results_dir, "shift_summary.csv"))


if __name__ == "__main__":
    main()
