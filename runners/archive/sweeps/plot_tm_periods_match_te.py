"""
Analysis for the TM period-count study (runners/sweeps/tm_periods_match_te.py):
find the TM period count N whose resonance peak transmission matches the fixed
TE@80 reference, then compare the Q factor of TE@80 vs the matched-N TM.

Standalone — imports only glob/scipy/numpy/matplotlib (NO lumapi, NO project
config), so it runs anywhere on a folder of downloaded result_*.mat files.

For every TM result_*.mat in --tm-dir it parses N{n} from the filename and reads
`resonance_transmission` (peak T), `resonance_wavelength_nm`, `spectral_fwhm_nm`,
and the wl_nm/T spectrum. It reads the TE@80 reference (a result_*_te.mat) from
--te-dir. The spectral Q is computed as Q = resonance_wavelength_nm / spectral_fwhm_nm
(NOT the buggy MATLAB window-edge FWHM).

It writes into --tm-dir:
    peakT_vs_periods.png                       TM peak T vs N, TE@80 line + crossing
    Q_vs_periods.png                           TM Q vs N, TE@80 Q line
    combined_transmission_TE80_vs_TMmatched.png overlay of TE@80 and matched-N TM T(lambda)
    tm_periods_summary.csv                     N,peakT,lambda_nm,fwhm_nm,Q

and prints the matched N (nearest integer), the interpolated crossing, the residual
dT, and which polarization has the higher Q.

Usage:
    python runners/sweeps/plot_tm_periods_match_te.py \
        [--tm-dir results_from_athena/tm_periods_match_te/results] \
        [--te-dir results_from_athena/tm_te/results]
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
    """Float for a scipy-loaded scalar field, or None if absent/empty."""
    if key not in d:
        return None
    try:
        return float(np.asarray(d[key]).squeeze())
    except (TypeError, ValueError):
        return None


def _spectrum(d):
    """(wl_nm, T) 1-D arrays from a result .mat, or (None, None)."""
    if "wl_nm" not in d or "T" not in d:
        return None, None
    wl = np.asarray(d["wl_nm"]).squeeze()
    T = np.asarray(d["T"]).squeeze()
    if wl.size == 0 or T.size == 0:
        return None, None
    return wl, T


def _peak_T(d):
    """Resonance peak T — prefer stored scalar, else read off the spectrum."""
    T_peak = _scalar(d, "resonance_transmission")
    if T_peak is not None:
        return T_peak
    wl, T = _spectrum(d)
    lam = _scalar(d, "resonance_wavelength_nm")
    if wl is not None and lam is not None:
        return float(T[int(np.argmin(np.abs(wl - lam)))])
    return None


def _q_factor(d):
    """Spectral Q = lambda_res / |spectral_fwhm_nm| (None if FWHM unavailable).

    spectral_fwhm_nm is stored sign-flipped (negative) because post_processing
    multiplies the scipy peak width by dw < 0 (descending wavelength axis), so
    take abs() to recover the linewidth.
    """
    lam = _scalar(d, "resonance_wavelength_nm")
    fwhm = _scalar(d, "spectral_fwhm_nm")
    if lam is None or fwhm is None or fwhm == 0:
        return None
    return lam / abs(fwhm)


def collect_tm(tm_dir):
    """One row per TM result_*.mat, keyed by period count N."""
    rows = []
    for fp in sorted(glob.glob(os.path.join(tm_dir, "result_*.mat"))):
        fname = os.path.basename(fp)
        # skip summaries and the TE reference (polarization token is _te_/_tm_,
        # possibly followed by backend tags like _smp before .mat).
        if "summary" in fname or re.search(r"_te(_|\.)", fname):
            continue
        if not re.search(r"_tm(_|\.)", fname):
            continue
        m = re.search(r"_?N(\d+)", fname)
        if not m:
            print(f"  WARN: no N token in {fname} — skipped")
            continue
        n = int(m.group(1))
        try:
            d = sio.loadmat(fp)
        except Exception as e:  # noqa: BLE001 — skip unreadable, keep going
            print(f"  WARN: could not read {fname}: {e}")
            continue
        T_peak = _peak_T(d)
        lam = _scalar(d, "resonance_wavelength_nm")
        fwhm = _scalar(d, "spectral_fwhm_nm")
        Q = _q_factor(d)
        wl, T = _spectrum(d)
        rows.append({
            "fname": fname, "N": n,
            "T_peak": T_peak if T_peak is not None else np.nan,
            "lambda_nm": lam if lam is not None else np.nan,
            "fwhm_nm": fwhm if fwhm is not None else np.nan,
            "Q": Q if Q is not None else np.nan,
            "wl": wl, "T": T,
        })
        print(f"  {fname:52s}  N={n:4d}  T={rows[-1]['T_peak']:.4f}  "
              f"lam={rows[-1]['lambda_nm']:.3f}nm  Q={rows[-1]['Q']:.1f}")
    rows.sort(key=lambda r: r["N"])
    return rows


def load_te_ref(te_dir):
    """The TE@80 reference row from a result_*_te.mat in te_dir (prefer N80)."""
    cands = [p for p in sorted(glob.glob(os.path.join(te_dir, "result_*.mat")))
             if re.search(r"_te(_|\.)", os.path.basename(p))
             and "summary" not in os.path.basename(p)]
    if not cands:
        raise FileNotFoundError(
            f"no result_*_te*.mat in {te_dir} — run the TE@80 reference "
            f"(bash athena/deploy_athena.sh --option2 --run=run_te) and download it.")
    # Prefer an explicit N80 file if several TE files exist.
    pick = next((p for p in cands if re.search(r"_?N80", os.path.basename(p))), cands[0])
    d = sio.loadmat(pick)
    wl, T = _spectrum(d)
    row = {
        "fname": os.path.basename(pick),
        "N": int((re.search(r"_?N(\d+)", os.path.basename(pick)) or [0, 80])[1]),
        "T_peak": _peak_T(d), "lambda_nm": _scalar(d, "resonance_wavelength_nm"),
        "fwhm_nm": _scalar(d, "spectral_fwhm_nm"), "Q": _q_factor(d),
        "wl": wl, "T": T,
    }
    print(f"  TE ref: {row['fname']}  N={row['N']}  T={row['T_peak']:.4f}  "
          f"lam={row['lambda_nm']:.3f}nm  Q={row['Q']:.1f}")
    return row


def find_crossing(rows, te_T):
    """Interpolated N where TM peak T crosses te_T, plus nearest-grid match.

    Returns (n_cross_float | None, nearest_row). n_cross is found from the first
    sign change of (T_tm - te_T) along increasing N; None if no crossing in range.
    """
    valid = [r for r in rows if np.isfinite(r["T_peak"])]
    nearest = min(valid, key=lambda r: abs(r["T_peak"] - te_T)) if valid else None
    n_cross = None
    for a, b in zip(valid, valid[1:]):
        da, db = a["T_peak"] - te_T, b["T_peak"] - te_T
        if da == 0:
            n_cross = float(a["N"]); break
        if da * db < 0:  # straddles the target
            frac = da / (da - db)
            n_cross = a["N"] + frac * (b["N"] - a["N"])
            break
    return n_cross, nearest


def _plot_vs_N(rows, key, ylabel, title, out_png, te_val=None, te_label=None,
               cross_n=None):
    xs = [r["N"] for r in rows if np.isfinite(r[key])]
    ys = [r[key] for r in rows if np.isfinite(r[key])]
    fig, ax = plt.subplots()
    ax.plot(xs, ys, "s-", lw=1.6, ms=7, mfc="w", label="TM")
    if te_val is not None and np.isfinite(te_val):
        ax.axhline(te_val, color="C1", ls="--", lw=1.6, label=te_label)
    if cross_n is not None:
        ax.axvline(cross_n, color="0.4", ls=":", lw=1.4,
                   label=f"crossing N~{cross_n:.1f}")
    ax.set_xlabel("TM periods per side, N", fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)
    if xs:
        ax.set_xticks(xs)
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def plot_combined(te_row, tm_row, out_png):
    """Overlay TE@80 and matched-N TM transmission spectra."""
    fig, ax = plt.subplots()
    if te_row["wl"] is not None:
        ax.plot(te_row["wl"], te_row["T"], "-", lw=1.6, color="C1",
                label=f"TE  N={te_row['N']}  (Q~{te_row['Q']:.0f})")
    if tm_row["wl"] is not None:
        ax.plot(tm_row["wl"], tm_row["T"], "-", lw=1.6, color="C0",
                label=f"TM  N={tm_row['N']}  (Q~{tm_row['Q']:.0f})")
    ax.set_xlabel("Wavelength [nm]", fontsize=FONT_SIZE)
    ax.set_ylabel("Transmission, T", fontsize=FONT_SIZE)
    ax.set_title("Combined transmission: TE@80 vs period-matched TM",
                 fontsize=FONT_SIZE)
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def write_csv(rows, out_csv):
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("N,peakT,lambda_nm,fwhm_nm,Q\n")
        for r in rows:
            f.write(f"{r['N']},{r['T_peak']:.6f},{r['lambda_nm']:.4f},"
                    f"{r['fwhm_nm']:.6f},{r['Q']:.4f}\n")
    print(f"  wrote {out_csv}")


def main():
    ap = argparse.ArgumentParser(
        description="Match TM period count to TE@80 peak T; compare Q.")
    ap.add_argument("--tm-dir",
                    default=os.path.join("results_from_athena", "tm_periods_match_te", "results"),
                    help="Folder with the TM period-sweep result_*.mat.")
    ap.add_argument("--te-dir",
                    default=os.path.join("results_from_athena", "tm_te", "results"),
                    help="Folder with the TE@80 reference result_*_te.mat.")
    args = ap.parse_args()

    print(f"[plot_tm_periods_match_te] TM dir: {args.tm_dir}")
    tm_rows = collect_tm(args.tm_dir)
    if not tm_rows:
        print("  No TM result_*.mat found — nothing to do.")
        return
    print(f"[plot_tm_periods_match_te] TE dir: {args.te_dir}")
    te = load_te_ref(args.te_dir)

    cross_n, nearest = find_crossing(tm_rows, te["T_peak"])

    _plot_vs_N(tm_rows, "T_peak", "Peak transmission, T",
               "TM peak transmission vs period count",
               os.path.join(args.tm_dir, "peakT_vs_periods.png"),
               te_val=te["T_peak"], te_label=f"TE@{te['N']}  T={te['T_peak']:.4f}",
               cross_n=cross_n)
    _plot_vs_N(tm_rows, "Q", "Spectral Q = $\\lambda/\\Delta\\lambda$",
               "TM Q factor vs period count",
               os.path.join(args.tm_dir, "Q_vs_periods.png"),
               te_val=te["Q"], te_label=f"TE@{te['N']}  Q={te['Q']:.0f}")
    if nearest is not None:
        plot_combined(te, nearest,
                      os.path.join(args.tm_dir, "combined_transmission_TE80_vs_TMmatched.png"))
    write_csv(tm_rows, os.path.join(args.tm_dir, "tm_periods_summary.csv"))

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"TE@{te['N']} reference:  peak T = {te['T_peak']:.4f}   Q = {te['Q']:.1f}")
    if cross_n is not None:
        print(f"TM peak T crosses TE value at N ~ {cross_n:.1f} (interpolated).")
    else:
        print("TM peak T did NOT cross the TE value within the swept range — "
              "extend n_periods_each_side in tm_periods_match_te.py.")
    if nearest is not None:
        dT = nearest["T_peak"] - te["T_peak"]
        print(f"Nearest grid point: N = {nearest['N']}  "
              f"(T = {nearest['T_peak']:.4f}, dT = {dT:+.4f})  Q = {nearest['Q']:.1f}")
        if np.isfinite(nearest["Q"]) and np.isfinite(te["Q"]):
            higher = "TE@{}".format(te["N"]) if te["Q"] > nearest["Q"] else f"TM@{nearest['N']}"
            print(f"Higher Q: {higher}  "
                  f"(TE {te['Q']:.1f} vs TM {nearest['Q']:.1f})")
    print("=" * 64)


if __name__ == "__main__":
    main()
