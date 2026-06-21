"""
Shared logic for the TM polarization runners in runners/tm/:

  run_te.py        — single wide TE scan (one predefined window).
  run_tm.py        — single wide TM scan (one predefined window).
  run_tm_vs_te.py  — TE-vs-TM comparison on identical geometry: one single wide
                     scan per polarization (2 sims), run sequentially in one job
                     or — with deploy --pol-array — as a 2-task GPU array. Writes a
                     comparison summary .mat (both resonances, n_eff, Δλ).

All three use the same single-scan step (run_one_scan): one predefined wide
window per polarization, no scout/refine two-step. Every output lands in the
study's standard layouts/ + results/ folders (config.LAYOUTS_DIR /
config.RESULTS_DIR) — no separate hand-off folder, no plots (viewed in MATLAB).

The pitch that re-centers TM on the TE wavelength is computed separately, from an
FDE n_eff(λ) calibration — see runners/tm/calibrate_neff.py. NOTE: n_eff is still
wavelength-dependent here via WAVEGUIDE (geometric) dispersion even though the
MATERIAL index is held constant, so a naive λ-ratio under-predicts the pitch.

Physics notes:
  - TM (E along z) couples much more weakly to sidewall corrugation than TE —
    the TM field concentrates at the top/bottom interfaces (Chen et al.,
    Opt. Express 23, 25295, 2015). Expect a much narrower TM stopband.
"""

import copy
import glob
import os

import numpy as np
import scipy.io as sio

import config
from runners.single.run_simulation import run_single_sim
from simulation_config import SimulationConfig

IS_HELPER = True  # not a dispatchable runner (athena_run.py auto-discovery)

# Shared results folder for the whole TM/TE study. All three runners (run_te,
# run_tm, run_tm_vs_te) write here instead of into a per-runner folder named after
# the script. athena_run.py reads STUDY_DIR_NAME off the dispatched runner module
# and sets RUN_NAME, so every TM/TE run lands in results/<STUDY_DIR_NAME>/
# (config.RESULTS_DIR). Result filenames already encode polarization / pitch /
# far-field / fields, so the two polarizations never collide in one folder.
STUDY_DIR_NAME = "tm_te"

# Single wide scan window (all runners). One window per polarization, wide enough
# to contain both resonances for this device (TE ≈ 1571, TM ≈ 1524 nm).
COMPARE_CENTER_M = 1.571e-6
COMPARE_WIDTH_NM = 10
COMPARE_N_POINTS = 2001       # ~25 pm spacing across the 150 nm window


def build_base_cfg(cfg: SimulationConfig) -> SimulationConfig:
    """Baseline for this example — pinned explicitly so the Athena
    dispatcher's global tweaks (detuning, mode) and run_simulation's IT11
    calibration are NOT inherited."""
    # Pitch defaults to 500 nm; override with TM_PITCH_NM for the pitch-correction
    # rerun (e.g. TM_PITCH_NM=518 to re-center TM on the TE wavelength). The pitch
    # is recorded in the result filename (run_one_scan) so runs don't collide.
    cfg.grating.pitch_m = float(os.environ.get("TM_PITCH_NM", "500")) * 1e-9
    cfg.grating.n_periods_each_side = 80
    cfg.grating.cavity_neg_detuning_nm = 0.0     # true default device
    cfg.apodization.enabled = False
    # Constant indices follow the grating-coupler project's values
    # (library values at 1.55 µm: Si3N4 "Luke", SiO2 "Palik") instead of the
    # IT11-calibrated n_core used elsewhere in this repo.
    cfg.material.use_constant_materials = True
    cfg.material.n_core_const = 1.9963
    cfg.material.n_clad_const = 1.444
    # Constant-index backend A/B: TM_CONST_MODE=sampled|object overrides the
    # repo default. Unset -> inherit the MaterialConfig default (so a later
    # default-flip to "object" is honored automatically).
    cfg.material.const_material_mode = os.environ.get(
        "TM_CONST_MODE", cfg.material.const_material_mode)
    # Mesh: "optimization" (dx=50 nm) by default; TM_MESH=accurate -> dx~35 nm
    # (for convergence checks). Recorded in the result filename (run_one_scan).
    cfg.mesh.simulation_mode = os.environ.get("TM_MESH", "optimization")
    cfg.farfield.enabled = False                 # refined runs may re-enable via TM_FARFIELD
    return cfg


def _farfield_enabled() -> bool:
    return os.environ.get("TM_FARFIELD", "0") == "1"


# ── Sub-run helpers ──────────────────────────────────────────────────────────

def _spectral(cfg, center_m, width_nm, n_points):
    cfg.spectral.center_wavelength_m = center_m
    cfg.spectral.scan_width_nm = width_nm
    cfg.spectral.n_wl_points = int(n_points)


def stopband_nm(res):
    """Width of the contiguous T<0.5 region around the resonance (from a scan)."""
    wl = np.asarray(res["wl_nm"]).ravel()
    T = np.asarray(res["T"]).ravel()
    idx = int(np.argmin(np.abs(wl - float(res["resonance_wavelength_nm"]))))
    below = T < 0.5
    if not below.any():
        return 0.0
    lo = idx
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = idx
    while hi < len(T) - 1 and below[hi + 1]:
        hi += 1
    return float(wl[hi] - wl[lo])


# ── Single wide scan (run_tm_vs_te) ──────────────────────────────────────────

def run_one_scan(base_cfg, polarization,
                 center_m=COMPARE_CENTER_M, width_nm=COMPARE_WIDTH_NM,
                 n_points=COMPARE_N_POINTS) -> dict:
    """One single wide FDTD scan for `polarization` — no scout/refine two-step.
    The window is wide enough to contain the resonance outright. Returns the same
    summary-dict shape that run_stitch consumes (T/R spectra live in the .mat)."""
    cfg = copy.deepcopy(base_cfg)
    cfg.source.polarization = polarization
    cfg.material.n_eff_guess = center_m / (2.0 * cfg.grating.pitch_m)
    # Field-profile capture: spectra-only by default (comparison is about the
    # spectra). TM_RECORD_2D=1 turns on the XY/YZ/XZ 2D profile monitors so the
    # cavity cross-section (+ top/side views) can be viewed in MATLAB
    # (matlab_plotting/plot_field_poynting.m). The wide scan band places a 2D
    # monitor frequency point within ~1 nm of the resonance, enough to render
    # the (dB-normalized) mode pattern.
    record_2d = os.environ.get("TM_RECORD_2D", "0") == "1"
    cfg.monitors.record_2d_fields = record_2d
    cfg.farfield.enabled = _farfield_enabled()
    # Manual domain-size override for one-off checks: SPAN_MULT (× λ_center).
    # Inert unless set; the default box is unchanged (1.8, or 5.0 with far-field).
    span_mult = os.environ.get("SPAN_MULT")
    if span_mult:
        cfg.span_multiplier_override = float(span_mult)
    _spectral(cfg, center_m, width_nm, n_points)
    # Tag with the polarization, plus the pitch when it differs from the 500 nm
    # default — so a pitch-corrected rerun doesn't overwrite the original.
    suffix = f"_{polarization.lower()}"
    pitch_nm = base_cfg.grating.pitch_m * 1e9
    if abs(pitch_nm - 500.0) > 0.5:
        suffix += f"_P{pitch_nm:.1f}".replace(".", "p")   # 0.1 nm res, e.g. _P518p3
    if base_cfg.mesh.simulation_mode == "accurate":
        suffix += "_acc"                                  # don't clobber optimization results
    if cfg.span_multiplier_override is not None:
        suffix += f"_M{cfg.span_multiplier_override:.1f}".replace(".", "p")  # domain check, e.g. _M1p5
    if record_2d:
        suffix += "_fields"                               # spectra+field run; distinct .mat
    # Constant-index backend A/B: only when TM_CONST_MODE is explicitly set, tag the
    # result with the backend (_smp / _obj) so the two runs get distinct filenames and
    # never overwrite the existing un-tagged TM-study results. Normal runs are untouched.
    if os.environ.get("TM_CONST_MODE"):
        suffix += "_obj" if base_cfg.material.const_material_mode == "object" else "_smp"
    res = run_single_sim(cfg, show_plots=False, tag_suffix=suffix, save_figs=False)
    lam_nm = float(res["resonance_wavelength_nm"])
    pitch_m = base_cfg.grating.pitch_m
    return {
        "polarization": polarization,
        "pitch_nm": pitch_m * 1e9,
        "n_periods_each_side": base_cfg.grating.n_periods_each_side,
        "n_core_const": base_cfg.material.n_core_const,
        "n_clad_const": base_cfg.material.n_clad_const,
        "lambda_res_nm": lam_nm,
        "n_eff": lam_nm * 1e-9 / (2.0 * pitch_m),   # Bragg condition λ = 2·n_eff·Λ
        "stopband_nm": stopband_nm(res),
        "spectral_fwhm_nm": float(res["spectral_fwhm_nm"]),
        "resonance_T": float(res["resonance_transmission"]),
        "scan_mat": res["results_path"],
    }


# ── Stitch ───────────────────────────────────────────────────────────────────

def _build_summary(te, tm) -> dict:
    """Comparison summary from the TE and TM scan dicts — pure arithmetic, no I/O.
    Each dict carries pitch_nm / n_periods_each_side / materials / lambda_res_nm /
    n_eff / stopband_nm / spectral_fwhm_nm / resonance_T (from run_one_scan or
    reconstructed from a result_*.mat by stitch_dir). The pitch that re-centers TM
    on λ_TE is computed separately by calibrate_neff.py (FDE), not here."""
    lam_te_nm, lam_tm_nm = te["lambda_res_nm"], tm["lambda_res_nm"]
    return {
        "pitch_nm": te["pitch_nm"],
        "n_periods_each_side": te["n_periods_each_side"],
        "n_core_const": te.get("n_core_const", float("nan")),
        "n_clad_const": te.get("n_clad_const", float("nan")),
        "lambda_res_te_nm": lam_te_nm,
        "lambda_res_tm_nm": lam_tm_nm,
        "delta_lambda_nm": lam_te_nm - lam_tm_nm,
        "n_eff_te": te["n_eff"],
        "n_eff_tm": tm["n_eff"],
        "stopband_te_nm": te["stopband_nm"],
        "stopband_tm_nm": tm["stopband_nm"],
        "resonance_T_te": te["resonance_T"],
        "resonance_T_tm": tm["resonance_T"],
        "spectral_fwhm_te_nm": te["spectral_fwhm_nm"],
        "spectral_fwhm_tm_nm": tm["spectral_fwhm_nm"],
    }


def run_stitch(te, tm) -> dict:
    """Write the comparison summary .mat into config.RESULTS_DIR. Used by the
    LOCAL/sequential run_tm_vs_te (both scans in one process). The parallel GPU
    array path runs one polarization per task and stitches afterwards with
    stitch_dir() on the downloaded folder."""
    summary = _build_summary(te, tm)
    n_side = int(summary["n_periods_each_side"])
    mat_path = os.path.join(config.RESULTS_DIR, f"result_tm_vs_te_summary_N{n_side}.mat")
    sio.savemat(mat_path, summary)
    print("\n══ TM vs TE summary ══")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  summary: {mat_path}")
    return summary


def _chain_from_mat(path, n_core_const, n_clad_const) -> dict:
    """Reconstruct a scan summary-dict from a per-polarization result_*.mat."""
    m = sio.loadmat(path)
    pitch_m = float(np.asarray(m["pitch_m"]).ravel()[0])
    lam_nm = float(np.asarray(m["resonance_wavelength_nm"]).ravel()[0])
    res = {"wl_nm": m["wl_nm"], "T": m["T"], "resonance_wavelength_nm": lam_nm}
    return {
        "pitch_nm": pitch_m * 1e9,
        "n_periods_each_side": int(np.asarray(m["n_periods_each_side"]).ravel()[0]),
        "n_core_const": n_core_const,
        "n_clad_const": n_clad_const,
        "lambda_res_nm": lam_nm,
        "n_eff": lam_nm * 1e-9 / (2.0 * pitch_m),
        "stopband_nm": stopband_nm(res),
        "spectral_fwhm_nm": float(np.asarray(m["spectral_fwhm_nm"]).ravel()[0]),
        "resonance_T": float(np.asarray(m["resonance_transmission"]).ravel()[0]),
    }


def stitch_dir(results_dir, n_core_const=1.9963, n_clad_const=1.444) -> str:
    """Build result_tm_vs_te_summary_N<n>.mat from the two per-polarization
    result_*.mat in `results_dir`. This is the parallel-array stitch step: each
    GPU task wrote one polarization (…_te.mat / …_tm.mat), so the comparison is
    assembled here, locally, after download. Returns the summary .mat path.

    (materials default to build_base_cfg's constants — they aren't stored in the
    result .mat and are informational only in the summary.)"""
    def _find(suffix):
        hits = [p for p in glob.glob(os.path.join(results_dir, "result_*.mat"))
                if os.path.basename(p).endswith(suffix + ".mat")
                and "summary" not in os.path.basename(p)]
        if not hits:
            raise FileNotFoundError(
                f"no result_*{suffix}.mat in {results_dir} — did both array "
                f"tasks finish and download?")
        return sorted(hits)[0]

    te = _chain_from_mat(_find("_te"), n_core_const, n_clad_const)
    tm = _chain_from_mat(_find("_tm"), n_core_const, n_clad_const)
    summary = _build_summary(te, tm)
    n_side = int(summary["n_periods_each_side"])
    out = os.path.join(results_dir, f"result_tm_vs_te_summary_N{n_side}.mat")
    sio.savemat(out, summary)
    print(f"[stitch] wrote {out}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return out
