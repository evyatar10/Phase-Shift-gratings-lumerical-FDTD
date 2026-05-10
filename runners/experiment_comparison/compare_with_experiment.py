"""
Sim-vs-experiment comparison for IT11 fabricated devices.

Reads:
  - The cards manifest JSON written by build_sweep_list.py at submission time
    (default: results_from_athena/_cards_manifest.json)
  - Per-card simulation results pulled from Athena/DGX into
    <results-dir>/<sweep_label>/<device_label>/results/result_*.mat
  - Matching experiment .mat from <record.mat_dir>/<device_id>_2026*.mat

Writes (one per device):
  - <out-dir>/<device_label>.png   — sim S21(dB) overlaid on experiment loss(dB)
  - <out-dir>/summary.csv          — one row per device with Q/ER/lambda

Usage:
  python -m runners.experiment_comparison.compare_with_experiment \\
      --results-dir results_from_athena/it11_devices_500 \\
      --manifest    results_from_athena/_cards_manifest.json \\
      --pitch-only  500

Processing order: 500 nm devices before 516 nm.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


# Pitch -> wavelength window (nm) — mirrors analyze_IT11.m:561-562.
PITCH_WINDOW_NM = {
    500: (1530.0, 1546.0),
    516: (1572.0, 1582.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_sim_mat(results_dir: str, device_label: str) -> Optional[str]:
    """Locate the result_*.mat file for a given device label.

    Looks for: <results_dir>/<device_label>/results/result_*.mat   (athena layout)
        or:    <results_dir>/results/<device_label>/result_*.mat   (alt)
    """
    candidates = (
        os.path.join(results_dir, device_label, "results", "result_*.mat"),
        os.path.join(results_dir, "results", device_label, "result_*.mat"),
        os.path.join(results_dir, device_label, "result_*.mat"),
    )
    for pattern in candidates:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def _find_exp_mat(mat_dir: str, device_id: str) -> Optional[str]:
    """Locate the latest experiment .mat for a given device id (e.g. '1A0_2026*.mat')."""
    hits = sorted(glob.glob(os.path.join(mat_dir, f"{device_id}_2026*.mat")))
    return hits[-1] if hits else None


def _load_sim_spectrum(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wl_nm, S21_dB) from a simulation result .mat."""
    m = loadmat(path, squeeze_me=True, simplify_cells=True)
    wl_nm = np.asarray(m["wl_nm"], dtype=float).ravel()
    # Prefer S21_complex if present; fall back to T (transmission magnitude squared).
    if "S21_complex" in m:
        s21 = np.asarray(m["S21_complex"]).ravel()
        s21_db = 20.0 * np.log10(np.clip(np.abs(s21), 1e-30, None))
    else:
        T = np.asarray(m["T"], dtype=float).ravel()
        s21_db = 10.0 * np.log10(np.clip(T, 1e-30, None))
    return wl_nm, s21_db


def _load_exp_spectrum(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wl_nm, loss_dB) from an experiment .mat (lda in m, loss in W).

    Convention from analyze_IT11.m:580: loss_dB = 10*log10(1000*loss).
    """
    m = loadmat(path, squeeze_me=True, simplify_cells=True)
    lda  = np.asarray(m["lda"],  dtype=float).ravel()
    loss = np.asarray(m["loss"], dtype=float).ravel()
    wl_nm   = lda * 1e9
    loss_dB = 10.0 * np.log10(np.clip(1000.0 * np.abs(loss), 1e-30, None))
    return wl_nm, loss_dB


# ─────────────────────────────────────────────────────────────────────────────
# Q / ER metrics (consistent across sim and experiment)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PeakMetrics:
    lambda_nm: Optional[float]
    q:         Optional[float]
    er_db:     Optional[float]


def _movmean(y: np.ndarray, window: int = 11) -> np.ndarray:
    if window <= 1 or window > len(y):
        return y
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def _fwhm_3db(wl: np.ndarray, y_db: np.ndarray, peak_idx: int) -> Optional[float]:
    """Linear-interp 3-dB FWHM around peak_idx, or None if window doesn't bracket."""
    target = y_db[peak_idx] - 3.0
    # Walk left
    li = peak_idx
    while li > 0 and y_db[li] >= target:
        li -= 1
    if y_db[li] >= target:
        return None
    # Walk right
    ri = peak_idx
    while ri < len(y_db) - 1 and y_db[ri] >= target:
        ri += 1
    if y_db[ri] >= target:
        return None
    # Interpolate
    def _x_at(i_lo, i_hi):
        x0, x1 = wl[i_lo], wl[i_hi]
        y0, y1 = y_db[i_lo], y_db[i_hi]
        if y1 == y0:
            return 0.5 * (x0 + x1)
        return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    wl_lo = _x_at(li, li + 1)
    wl_hi = _x_at(ri - 1, ri)
    return abs(wl_hi - wl_lo)


def compute_metrics(wl_nm: np.ndarray, y_db: np.ndarray, window_nm: Tuple[float, float],
                    smooth: int = 11) -> PeakMetrics:
    """Find the strongest peak inside `window_nm`, compute Q via 3-dB FWHM, and
    compute local ER as (mean of flanking-passband levels) − peak."""
    w_lo, w_hi = window_nm
    in_win = (wl_nm >= w_lo) & (wl_nm <= w_hi)
    if not np.any(in_win):
        return PeakMetrics(None, None, None)

    y_smooth = _movmean(y_db, smooth)
    idxs = np.where(in_win)[0]
    local = y_smooth[idxs]
    pk_local = int(np.argmax(local))
    pk_idx   = idxs[pk_local]
    lam_pk   = float(wl_nm[pk_idx])
    pk_db    = float(y_smooth[pk_idx])

    fwhm = _fwhm_3db(wl_nm, y_smooth, pk_idx)
    q    = (lam_pk / fwhm) if (fwhm and fwhm > 0) else None

    # Local ER: max value in the 4 nm flanking passbands either side of the window.
    pad = 4.0
    left_band  = (wl_nm >= w_lo - pad) & (wl_nm <  w_lo)
    right_band = (wl_nm >  w_hi)       & (wl_nm <= w_hi + pad)
    flank = []
    if np.any(left_band):  flank.append(float(np.max(y_smooth[left_band])))
    if np.any(right_band): flank.append(float(np.max(y_smooth[right_band])))
    er_db = (float(np.mean(flank)) - pk_db) if flank else None

    return PeakMetrics(lambda_nm=lam_pk, q=q, er_db=er_db)


# ─────────────────────────────────────────────────────────────────────────────
# Plot one device
# ─────────────────────────────────────────────────────────────────────────────

def plot_overlay(out_path: str,
                 record:  dict,
                 sim:     Optional[Tuple[np.ndarray, np.ndarray]],
                 exp:     Optional[Tuple[np.ndarray, np.ndarray]],
                 win_nm:  Tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    title = (f"{record['folder']}/{record['device_id']}  "
             f"({record['subname']})   pitch={record['pitch_nm']} nm")

    # Crop to a sensible plot window: pitch window ± 4 nm.
    pad = 4.0
    x_lo, x_hi = win_nm[0] - pad, win_nm[1] + pad

    if exp is not None:
        wl_e, db_e = exp
        sel = (wl_e >= x_lo) & (wl_e <= x_hi)
        ax.plot(wl_e[sel], db_e[sel], color="black", lw=1.0, label="experiment")
    if sim is not None:
        wl_s, db_s = sim
        sel = (wl_s >= x_lo) & (wl_s <= x_hi)
        ax.plot(wl_s[sel], db_s[sel], color="tab:red", lw=1.4, label="simulation")

    ax.axvspan(win_nm[0], win_nm[1], color="0.9", alpha=0.5, zorder=-10,
               label=f"{int(record['pitch_nm'])} nm pitch window")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("loss / S21 (dB)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="Top-level local results dir, e.g. results_from_athena/it11_devices_500")
    p.add_argument("--manifest", default=None,
                   help="Cards manifest JSON (default: <results-dir>/../_cards_manifest.json)")
    p.add_argument("--out-dir", default=None,
                   help="Output dir for PNGs and summary.csv (default: <results-dir>/comparison)")
    p.add_argument("--pitch-only", type=int, default=None, choices=[500, 516],
                   help="Process only one pitch (500 or 516); default = both, 500 first")
    args = p.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    out_dir     = os.path.abspath(args.out_dir or os.path.join(results_dir, "comparison"))
    manifest    = os.path.abspath(args.manifest
                                  or os.path.join(os.path.dirname(results_dir),
                                                  "_cards_manifest.json"))

    if not os.path.exists(manifest):
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        sys.exit(1)

    with open(manifest) as f:
        records = json.load(f)["records"]

    if args.pitch_only is not None:
        records = [r for r in records if int(r["pitch_nm"]) == args.pitch_only]

    # 500 nm first, then 516; secondary sort by folder / device id for stability.
    records.sort(key=lambda r: (int(r["pitch_nm"]), r["folder"], r["device_id"]))

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.csv")
    cols = ["device_id", "folder", "pitch_nm", "subname",
            "sim_lambda_nm", "exp_lambda_nm",
            "sim_Q", "exp_Q",
            "sim_ER_dB", "exp_ER_dB",
            "sim_mat", "exp_mat"]

    n_ok = 0
    with open(summary_path, "w", newline="") as f_sum:
        w = csv.DictWriter(f_sum, fieldnames=cols)
        w.writeheader()

        for rec in records:
            pitch = int(rec["pitch_nm"])
            win   = PITCH_WINDOW_NM[pitch]
            label = rec["label"]

            sim_path = _find_sim_mat(results_dir, label)
            exp_path = _find_exp_mat(rec["mat_dir"], rec["device_id"])

            sim_data = None
            exp_data = None
            if sim_path is not None:
                try:    sim_data = _load_sim_spectrum(sim_path)
                except Exception as e: print(f"  [{label}] sim load failed: {e}")
            else:
                print(f"  [{label}] sim .mat not found under {results_dir}")
            if exp_path is not None:
                try:    exp_data = _load_exp_spectrum(exp_path)
                except Exception as e: print(f"  [{label}] exp load failed: {e}")

            sim_m = compute_metrics(*sim_data, win) if sim_data else PeakMetrics(None, None, None)
            exp_m = compute_metrics(*exp_data, win) if exp_data else PeakMetrics(None, None, None)

            png = os.path.join(out_dir, f"{label}.png")
            try:
                plot_overlay(png, rec, sim_data, exp_data, win)
            except Exception as e:
                print(f"  [{label}] plot failed: {e}")

            w.writerow({
                "device_id":      rec["device_id"],
                "folder":         rec["folder"],
                "pitch_nm":       pitch,
                "subname":        rec["subname"],
                "sim_lambda_nm":  f"{sim_m.lambda_nm:.4f}" if sim_m.lambda_nm else "",
                "exp_lambda_nm":  f"{exp_m.lambda_nm:.4f}" if exp_m.lambda_nm else "",
                "sim_Q":          f"{sim_m.q:.0f}" if sim_m.q else "",
                "exp_Q":          f"{exp_m.q:.0f}" if exp_m.q else "",
                "sim_ER_dB":      f"{sim_m.er_db:.2f}" if sim_m.er_db else "",
                "exp_ER_dB":      f"{exp_m.er_db:.2f}" if exp_m.er_db else "",
                "sim_mat":        sim_path or "",
                "exp_mat":        exp_path or "",
            })
            if sim_data is not None or exp_data is not None:
                n_ok += 1
            print(f"  [{label}] sim_λ={sim_m.lambda_nm}  exp_λ={exp_m.lambda_nm}")

    print(f"\nWrote {n_ok}/{len(records)} comparisons to: {out_dir}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
