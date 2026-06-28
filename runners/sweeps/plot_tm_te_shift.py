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
import scipy.integrate
import scipy.signal
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt


FONT_SIZE = 13


# ── in-plane (yx) vs out-of-plane (zx) mode width ─────────────────────────────
# Recompute the longitudinal cavity FWHM (along x) from the 2D field monitors in
# both planes, on an identical footing: |E|^2 integrated over the transverse axis
# at the resonance wavelength, cropped to the grating extent, peak-envelope, then
# FWHM relative to the floor. Mirrors sim_helpers.extract_and_process_field_profile.
# Self-contained (scipy/numpy only) so this script stays a lightweight CPU job.

def _fwhm_relative(x, y):
    ymax, ymin = float(np.max(y)), float(np.min(y))
    tgt = ymin + 0.5 * (ymax - ymin)
    zc = np.where(np.diff(np.sign(y - tgt)))[0]
    if len(zc) < 2:
        return 0.0
    def cross(i):
        s = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        return x[i] if s == 0 else x[i] + (tgt - y[i]) / s
    return cross(zc[-1]) - cross(zc[0])


def _envelope(x, y):
    peaks, _ = scipy.signal.find_peaks(y)
    if len(peaks) < 2:
        return y
    return interp1d(x[peaks], y[peaks], kind="cubic", bounds_error=False,
                    fill_value=(y[peaks][0], y[peaks][-1]))(x)


def _plane_width(struct, transverse, res_wl_m, x_limit):
    """FWHM [um] of the x-envelope from one 2D monitor struct, or NaN."""
    if struct is None:
        return np.nan
    x = np.squeeze(np.asarray(struct.x))
    E = np.asarray(struct.E_res)               # (x, y, z, lambda, component)
    lam = np.squeeze(np.asarray(struct.lambda_3d))
    idx = int(np.argmin(np.abs(lam - res_wl_m)))
    Ei = E[:, :, :, idx, :]                     # (x, y, z, comp)
    I = np.sum(np.abs(Ei) ** 2, axis=-1)        # (x, y, z)
    if transverse == "y":
        coord = np.squeeze(np.asarray(struct.y))
        Ix = scipy.integrate.trapezoid(I[:, :, 0], coord, axis=1)
    else:
        coord = np.squeeze(np.asarray(struct.z))
        Ix = scipy.integrate.trapezoid(I[:, 0, :], coord, axis=1)
    m = np.abs(x) <= x_limit
    xc, Ic = x[m], Ix[m]
    if xc.size < 3:
        return np.nan
    return _fwhm_relative(xc, _envelope(xc, Ic)) * 1e6


def plane_widths(d):
    """(h_um, v_um) from a loaded result dict: in-plane (yx) / out-of-plane (zx)."""
    if "resonance_wavelength_nm" not in d or "L_device" not in d:
        return np.nan, np.nan
    res_wl_m = float(np.squeeze(d["resonance_wavelength_nm"])) * 1e-9
    x_limit = float(np.squeeze(d["L_device"])) / 2.0
    xy = d["field_xy"][0, 0] if "field_xy" in d else None
    xz = d["field_xz_side"][0, 0] if "field_xz_side" in d else None
    return (_plane_width(xy, "y", res_wl_m, x_limit),
            _plane_width(xz, "z", res_wl_m, x_limit))


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
            # struct_as_record=False so the 2D field monitors load as objects
            # with .x/.y/.z/.E_res/.lambda_3d attributes for plane_widths().
            d = sio.loadmat(fp, struct_as_record=False, squeeze_me=False)
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

        # In-plane (yx) and out-of-plane (zx) mode widths from the 2D monitors.
        width_h_um, width_v_um = plane_widths(d)
        # Stored 1D-monitor fwhm_m is the horizontal fallback when 2D fields are
        # absent (e.g. the 0-shift baselines), and keeps the legacy CSV column.
        fwhm_m = _scalar(d, "fwhm_m")
        width_um = fwhm_m * 1e6 if fwhm_m is not None else np.nan
        if np.isnan(width_h_um):
            width_h_um = width_um
        lam_nm = _scalar(d, "resonance_wavelength_nm")

        # Relative shift = shift as % of the half-pitch, so points at different
        # pitches (e.g. TE@500 vs TM@518.3) are directly comparable.
        pitch_m = _scalar(d, "pitch_m")
        pitch_nm = pitch_m * 1e9 if pitch_m else np.nan
        rel_pct = (shift_nm / (pitch_nm / 2.0) * 100.0) if pitch_m else np.nan

        rows.append({
            "fname": fname,
            "shift_nm": shift_nm,
            "rel_pct": rel_pct,
            "pitch_nm": pitch_nm,
            "pol": "TM" if is_tm else "TE",
            "T_peak": T_peak if T_peak is not None else np.nan,
            "width_um": width_um,
            "width_h_um": width_h_um,
            "width_v_um": width_v_um,
            "lambda_nm": lam_nm if lam_nm is not None else np.nan,
        })
        print(f"  {fname:48s}  pol={rows[-1]['pol']}  shift={shift_nm:3d}nm "
              f"({rel_pct:4.1f}%)  T={rows[-1]['T_peak']:.4f}  "
              f"wH={width_h_um:.3f}um  wV={width_v_um:.3f}um")
    return rows


def _series(rows, pol, xkey, ykey):
    """Sorted (x, y) arrays for one polarization."""
    pts = sorted((r[xkey], r[ykey]) for r in rows if r["pol"] == pol)
    if not pts:
        return np.array([]), np.array([])
    x, y = zip(*pts)
    return np.array(x, float), np.array(y, float)


def _plot(rows, ykey, ylabel, title, out_png, xkey, xlabel):
    te_x, te_y = _series(rows, "TE", xkey, ykey)
    tm_x, tm_y = _series(rows, "TM", xkey, ykey)
    fig, ax = plt.subplots()
    if te_x.size:
        ax.plot(te_x, te_y, "o-", lw=1.6, ms=7, mfc="w", label="TE")
    if tm_x.size:
        ax.plot(tm_x, tm_y, "s-", lw=1.6, ms=7, mfc="w", label="TM")
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)
    all_x = np.unique(np.concatenate([te_x, tm_x])) if (te_x.size or tm_x.size) else None
    if all_x is not None and all_x.size:
        # For the relative axis, TE (e.g. 20.0%) and TM (e.g. 20.07%, from the
        # integer-nm filename tag) round to the same tick — dedupe so ticks read
        # 0/20/40/60/80 instead of doubling up.
        ax.set_xticks(np.unique(np.round(all_x).astype(int)) if xkey == "rel_pct" else all_x)
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def _plot_planes(rows, out_png, xkey, xlabel):
    """Mode width vs shift in BOTH planes: in-plane (yx, solid) and
    out-of-plane (zx, dashed), for TE and TM."""
    fig, ax = plt.subplots()
    style = {"TE": ("o", "C0"), "TM": ("s", "C1")}
    all_x = []
    for pol, (mk, col) in style.items():
        hx, hy = _series(rows, pol, xkey, "width_h_um")
        vx, vy = _series(rows, pol, xkey, "width_v_um")
        if hx.size:
            ax.plot(hx, hy, mk + "-", lw=1.6, ms=7, mfc="w", color=col,
                    label=f"{pol} in-plane (yx)")
            all_x.append(hx)
        if vx.size and not np.all(np.isnan(vy)):
            ax.plot(vx, vy, mk + "--", lw=1.6, ms=7, mfc=col, color=col,
                    label=f"{pol} out-of-plane (zx)")
            all_x.append(vx)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(r"Mode width, FWHM [$\mu$m]", fontsize=FONT_SIZE)
    ax.set_title("Spatial mode width vs. shift — in-plane (yx) vs out-of-plane (zx)",
                 fontsize=FONT_SIZE)
    if all_x:
        ux = np.unique(np.concatenate(all_x))
        ax.set_xticks(np.unique(np.round(ux).astype(int)) if xkey == "rel_pct" else ux)
    ax.grid(True)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def write_csv(rows, out_csv):
    order = sorted(rows, key=lambda r: (r["pol"], r["shift_nm"]))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("pol,shift_nm,rel_pct_halfpitch,pitch_nm,T_peak,"
                "mode_width_um,mode_width_inplane_um,mode_width_outofplane_um,lambda_nm\n")
        for r in order:
            f.write(f"{r['pol']},{r['shift_nm']},{r['rel_pct']:.2f},{r['pitch_nm']:.2f},"
                    f"{r['T_peak']:.6f},{r['width_um']:.6f},"
                    f"{r['width_h_um']:.6f},{r['width_v_um']:.6f},{r['lambda_nm']:.4f}\n")
    print(f"  wrote {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Summary plots for the tooth-shift sweep.")
    ap.add_argument("--results-dir", default=os.path.join("results", "tm_te_shift"),
                    help="Folder containing result_*.mat (default: results/tm_te_shift).")
    ap.add_argument("--x", choices=["absolute", "relative"], default="absolute",
                    help="x-axis: 'absolute' tooth shift [nm] (default) or 'relative' "
                         "shift as %% of half-pitch (compares across pitches).")
    args = ap.parse_args()

    if args.x == "relative":
        xkey, xlabel, sub = "rel_pct", "Innermost-tooth shift [% of half-pitch]", "tooth shift (relative)"
    else:
        xkey, xlabel, sub = "shift_nm", "Innermost-tooth shift [nm]", "tooth shift"

    results_dir = args.results_dir
    print(f"[plot_tm_te_shift] scanning {results_dir}  (x-axis: {args.x})")
    rows = collect(results_dir)
    if not rows:
        print("  No result_*.mat found — nothing to plot.")
        return

    _plot(rows, "T_peak", "Peak transmission (T)",
          f"Resonance transmission vs. {sub}",
          os.path.join(results_dir, "transmission_vs_shift.png"), xkey, xlabel)
    # Mode width vs shift — now in BOTH planes (in-plane yx + out-of-plane zx).
    _plot_planes(rows, os.path.join(results_dir, "modewidth_vs_shift.png"), xkey, xlabel)
    write_csv(rows, os.path.join(results_dir, "shift_summary.csv"))


if __name__ == "__main__":
    main()
