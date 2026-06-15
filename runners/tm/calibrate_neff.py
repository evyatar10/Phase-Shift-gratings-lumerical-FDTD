"""
Local FDE (MODE) calibration of the effective index n_eff(λ) for the pi-shift
Bragg grating — for BOTH TE and TM — used to compute the TM pitch that re-centers
the TM resonance on the TE wavelength.

WHY THIS EXISTS
  The FDTD example holds the MATERIAL index constant (n_core=1.9963, n_clad=1.444),
  but the EFFECTIVE index n_eff is still wavelength-dependent through WAVEGUIDE
  (geometric) dispersion. So the naive "scale the pitch by the wavelength ratio"
  under-predicts the pitch. FDE gives n_eff(λ) directly, which fixes that.

METHOD
  - FDE the two grating cross-sections — wide tooth (950 nm) and narrow tooth
    (650 nm), both 350 nm tall, constant n_core / n_clad — and average them
    (50/50 duty):  n̄_eff(λ) = ½·[n_eff_wide(λ) + n_eff_narrow(λ)].
  - Bragg / cavity condition:  λ_res = 2·n̄_eff(λ_res)·Λ.
  - Anchor the FDE curve to the FDTD-measured n_eff (cancels FDE↔FDTD and
    uniform-vs-grating offsets), then read n̄_eff,TM at the TARGET λ_TE and solve:
        Λ'_TM = λ_TE / (2·n̄_eff,TM(λ_TE)).
  - One FDTD run at Λ'_TM is still the gold-standard confirmation.

Runs LOCALLY — MODE/FDE is CPU and fast. FDE is a SEPARATE Lumerical product from
FDTD, so this needs a MODE license on the machine you run it on.

Usage:
  python -m runners.tm.calibrate_neff
      [--out <dir>]            default: results_from_athena/run_tm_vs_te/results
      [--lam-te 1570.74] [--lam-tm 1523.57] [--pitch-nm 500]
          (defaults auto-read from result_*_te.mat / _tm.mat in --out if present)
      [--wl-start 1.50e-6] [--wl-stop 1.62e-6] [--n-points 200]

Writes <out>/neff_calibration.mat: wl_nm, neff_te_avg, neff_tm_avg (FDE), the
FDTD anchors, and the recommended TM pitch (anchored + pure-FDE).
"""

import argparse
import glob
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config  # noqa: E402

IS_HELPER = True  # local tool, not a cluster-dispatchable runner

# ── Cross-section geometry (matches GeometryConfig defaults / build_base_cfg) ──
CORE_HEIGHT_M = 350e-9
WIDE_WIDTH_M = 950e-9     # avg 800 + depth/2 (300/2)
NARROW_WIDTH_M = 650e-9   # avg 800 − depth/2
N_CORE = 1.9963
N_CLAD = 1.444

# Default FDTD anchors (overridden by --out result_*.mat if present, or CLI)
DEFAULT_LAM_TE_NM = 1570.74
DEFAULT_LAM_TM_NM = 1523.57
DEFAULT_PITCH_NM = 500.0


def _import_lumapi():
    """Import lumapi lazily (only when actually running FDE)."""
    try:
        import lumapi  # noqa
        return lumapi
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("lumapi", config.LUMAPI_PATH)
        lumapi = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lumapi)
        return lumapi


def _build_section(mode, width_m, wl_mid):
    """Build a constant-index waveguide cross-section (2D Z-normal FDE)."""
    mode.switchtolayout()
    mode.deleteall()

    mode.addfde()
    mode.set("solver type", "2D Z normal")
    mode.set("x", 0); mode.set("x span", 4e-6)   # thickness direction
    mode.set("y", 0); mode.set("y span", 6e-6)   # width direction
    mode.set("z", 0)
    mode.set("wavelength", wl_mid)

    mode.addrect()                               # cladding (constant index)
    mode.set("name", "clad")
    mode.set("material", "<Object defined dielectric>")
    mode.set("index", N_CLAD)
    mode.set("x", 0); mode.set("x span", 10e-6)
    mode.set("y", 0); mode.set("y span", 10e-6)
    mode.set("z", 0); mode.set("z span", 10e-6)

    mode.addrect()                               # core (constant index)
    mode.set("name", "core")
    mode.set("material", "<Object defined dielectric>")
    mode.set("index", N_CORE)
    mode.set("x", 0); mode.set("x span", CORE_HEIGHT_M)
    mode.set("y", 0); mode.set("y span", width_m)
    mode.set("z", 0); mode.set("z span", 10e-6)

    mode.addmesh()
    mode.set("name", "core_mesh")
    mode.set("x", 0); mode.set("x span", CORE_HEIGHT_M + 0.5e-6)
    mode.set("y", 0); mode.set("y span", width_m + 0.5e-6)
    mode.set("z", 0); mode.set("z span", 10e-6)
    mode.set("dx", 10e-9); mode.set("dy", 10e-9); mode.set("dz", 10e-9)


def _pick_mode(mode, polarization, n_trial=6):
    """Find the fundamental mode of the requested polarization.
    TE-like => TE polarization fraction > 0.5; TM-like => < 0.5. Returns the
    matching mode with the largest n_eff (the fundamental of that polarization)."""
    try:
        mode.setanalysis("number of trial modes", n_trial)
    except Exception:
        try:
            mode.set("number of trial modes", n_trial)
        except Exception:
            pass
    mode.findmodes()
    best_idx, best_neff = None, -1.0
    for i in range(1, n_trial + 1):
        name = f"FDE::data::mode{i}"
        try:
            neff = float(np.real(np.asarray(mode.getdata(name, "neff")).ravel()[0]))
            te_frac = float(np.asarray(
                mode.getdata(name, "TE polarization fraction")).ravel()[0])
        except Exception:
            break  # no more modes
        is_te = te_frac > 0.5
        want_te = (polarization.upper() == "TE")
        if is_te == want_te and neff > best_neff:
            best_idx, best_neff = i, neff
    if best_idx is None:
        raise RuntimeError(f"no {polarization} mode found among {n_trial} trial modes")
    return best_idx


def neff_vs_wl(mode, width_m, polarization, wl_start, wl_stop, n_points):
    """FDE frequency sweep → (wl_nm ascending, neff_real) for one cross-section."""
    _build_section(mode, width_m, 0.5 * (wl_start + wl_stop))
    mode.run()
    mode.setanalysis("wavelength", wl_start)
    k = _pick_mode(mode, polarization)
    mode.selectmode(k)
    mode.setanalysis("track selected mode", 1)
    mode.setanalysis("detailed dispersion calculation", 0)
    mode.setanalysis("stop wavelength", wl_stop)
    mode.setanalysis("number of points", int(n_points))
    mode.frequencysweep()
    neff = np.squeeze(np.real(mode.getdata("frequencysweep", "neff")))
    f = np.squeeze(mode.getdata("frequencysweep", "f"))
    wl_nm = (299792458.0 / f) * 1e9
    if wl_nm[0] > wl_nm[-1]:
        wl_nm = np.flip(wl_nm); neff = np.flip(neff)
    return wl_nm, neff


def _read_anchor(out_dir, suffix, default_lam_nm):
    """(lam_res_nm, pitch_m) from result_*<suffix>.mat in out_dir, else defaults."""
    hits = [p for p in glob.glob(os.path.join(out_dir, "result_*.mat"))
            if os.path.basename(p).endswith(suffix + ".mat")
            and "summary" not in os.path.basename(p)]
    if not hits:
        return default_lam_nm, DEFAULT_PITCH_NM * 1e-9
    m = sio.loadmat(sorted(hits)[0])
    lam = float(np.asarray(m["resonance_wavelength_nm"]).ravel()[0])
    pitch = float(np.asarray(m["pitch_m"]).ravel()[0])
    return lam, pitch


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console is cp1252
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        config.BASE_SAVE_DIR if os.path.isabs(getattr(config, "BASE_SAVE_DIR", "")) else ".",
        ""))
    ap.add_argument("--lam-te", type=float, default=None)
    ap.add_argument("--lam-tm", type=float, default=None)
    ap.add_argument("--pitch-nm", type=float, default=None)
    ap.add_argument("--wl-start", type=float, default=1.50e-6)
    ap.add_argument("--wl-stop", type=float, default=1.62e-6)
    ap.add_argument("--n-points", type=int, default=200)
    # Reuse an EXISTING calibration (no MODE rerun): given a target wavelength,
    # print the pitch for that polarization from the saved neff_calibration.mat.
    ap.add_argument("--predict-target", type=float, default=None,
                    help="target wavelength in nm; reuse saved calibration, no FDE rerun")
    ap.add_argument("--predict-pol", default="TM", choices=["TE", "TM"])
    ap.add_argument("--cal-file", default=None,
                    help="path to neff_calibration.mat (default: <out>/neff_calibration.mat)")
    args = ap.parse_args()

    # Default output dir: the downloaded run_tm_vs_te results next to this repo.
    out_dir = args.out
    if not out_dir or out_dir == ".":
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "results_from_athena", "run_tm_vs_te", "results")
    os.makedirs(out_dir, exist_ok=True)

    # ── Reuse mode: predict a pitch for a new target from a saved calibration ──
    if args.predict_target is not None:
        cal_path = args.cal_file or os.path.join(out_dir, "neff_calibration.mat")
        cal = sio.loadmat(cal_path)
        wl = np.asarray(cal["wl_nm"]).ravel()
        pol = args.predict_pol.upper()
        avg_key = "neff_tm_avg" if pol == "TM" else "neff_te_avg"
        off_key = "fde_anchor_offset_tm" if pol == "TM" else "fde_anchor_offset_te"
        neff_avg = np.asarray(cal[avg_key]).ravel()
        offset = float(np.asarray(cal[off_key]).ravel()[0])
        lam = args.predict_target
        neff_cal = float(np.interp(lam, wl, neff_avg)) + offset
        pitch_pred = lam * 1e-9 / (2.0 * neff_cal) * 1e9
        print(f"reuse: predict {pol} pitch for target {lam:.2f} nm  (from {cal_path})")
        print(f"  n̄_eff,{pol}({lam:.2f}) calibrated = {neff_cal:.4f}")
        print(f"  predicted pitch = {pitch_pred:.2f} nm")
        print(f"  → FDE estimate; confirm/secant-refine with one FDTD run "
              f"(TM_PITCH_NM={pitch_pred:.2f}).")
        return {"target_nm": lam, "pitch_pred_nm": pitch_pred, "pol": pol}

    lam_te_nm, _ = (_read_anchor(out_dir, "_te", DEFAULT_LAM_TE_NM)
                    if args.lam_te is None else (args.lam_te, None))
    lam_tm_nm, pitch_m = (_read_anchor(out_dir, "_tm", DEFAULT_LAM_TM_NM)
                          if args.lam_tm is None else (args.lam_tm, None))
    if args.pitch_nm is not None:
        pitch_m = args.pitch_nm * 1e-9
    if pitch_m is None:
        pitch_m = DEFAULT_PITCH_NM * 1e-9
    print(f"anchors: λ_TE={lam_te_nm:.3f} nm, λ_TM={lam_tm_nm:.3f} nm, "
          f"pitch={pitch_m*1e9:.1f} nm")

    lumapi = _import_lumapi()
    mode = lumapi.MODE(hide=True)
    try:
        wl, te_wide = neff_vs_wl(mode, WIDE_WIDTH_M, "TE", args.wl_start, args.wl_stop, args.n_points)
        _,  te_narr = neff_vs_wl(mode, NARROW_WIDTH_M, "TE", args.wl_start, args.wl_stop, args.n_points)
        _,  tm_wide = neff_vs_wl(mode, WIDE_WIDTH_M, "TM", args.wl_start, args.wl_stop, args.n_points)
        _,  tm_narr = neff_vs_wl(mode, NARROW_WIDTH_M, "TM", args.wl_start, args.wl_stop, args.n_points)
    finally:
        mode.close()

    neff_te_avg = 0.5 * (te_wide + te_narr)
    neff_tm_avg = 0.5 * (tm_wide + tm_narr)

    # Anchor each polarization's FDE curve to its FDTD-measured n_eff, then read
    # the TM index at the TARGET λ_TE and solve the Bragg condition for the pitch.
    neff_tm_fdtd = lam_tm_nm / (2.0 * pitch_m * 1e9)         # = λ_tm / (2Λ)
    neff_te_fdtd = lam_te_nm / (2.0 * pitch_m * 1e9)
    off_tm = neff_tm_fdtd - float(np.interp(lam_tm_nm, wl, neff_tm_avg))
    off_te = neff_te_fdtd - float(np.interp(lam_te_nm, wl, neff_te_avg))

    neff_tm_at_te = float(np.interp(lam_te_nm, wl, neff_tm_avg)) + off_tm
    pitch_rec_m = lam_te_nm * 1e-9 / (2.0 * neff_tm_at_te)
    pitch_pure_fde_m = lam_te_nm * 1e-9 / (
        2.0 * float(np.interp(lam_te_nm, wl, neff_tm_avg)))

    out = {
        "wl_nm": wl,
        "neff_te_avg": neff_te_avg, "neff_tm_avg": neff_tm_avg,
        "neff_te_wide": te_wide, "neff_te_narrow": te_narr,
        "neff_tm_wide": tm_wide, "neff_tm_narrow": tm_narr,
        "n_core_const": N_CORE, "n_clad_const": N_CLAD,
        "core_height_m": CORE_HEIGHT_M,
        "wide_width_m": WIDE_WIDTH_M, "narrow_width_m": NARROW_WIDTH_M,
        "lam_te_nm": lam_te_nm, "lam_tm_nm": lam_tm_nm, "pitch_nm": pitch_m * 1e9,
        "neff_tm_fdtd_anchor": neff_tm_fdtd, "neff_te_fdtd_anchor": neff_te_fdtd,
        "fde_anchor_offset_tm": off_tm, "fde_anchor_offset_te": off_te,
        "neff_tm_at_lam_te": neff_tm_at_te,
        "pitch_tm_recommended_nm": pitch_rec_m * 1e9,
        "pitch_tm_pure_fde_nm": pitch_pure_fde_m * 1e9,
    }
    out_path = os.path.join(out_dir, "neff_calibration.mat")
    sio.savemat(out_path, out)

    print(f"\n══ FDE n_eff calibration ══")
    print(f"  n̄_eff,TM(λ_TM={lam_tm_nm:.2f}) FDTD anchor = {neff_tm_fdtd:.4f}")
    print(f"  n̄_eff,TM(λ_TE={lam_te_nm:.2f}) calibrated  = {neff_tm_at_te:.4f}")
    print(f"  recommended TM pitch (anchored) = {pitch_rec_m*1e9:.2f} nm")
    print(f"  recommended TM pitch (pure FDE) = {pitch_pure_fde_m*1e9:.2f} nm")
    print(f"  → confirm with one FDTD run at the anchored pitch.")
    print(f"  saved: {out_path}")
    return out


if __name__ == "__main__":
    main()
