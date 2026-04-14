"""
Post-simulation analysis pipeline for Pi-Shift Bragg Grating FDTD.

After the FDTD solver completes, this module handles all data extraction
and analysis. The top-level entry point is analyze_simulation(), which
runs the full pipeline in a clear, sequential order.

Pipeline stages:
  1. S-parameter extraction and phase correction  (extract_s_parameters)
  2. Resonance detection                          (find_resonance)
  3. Field profile extraction and FWHM            (extract_field_profile)
  4. 2D field monitor extraction                  (extract_2d_fields)
  5. 3D field monitor extraction                  (extract_3d_fields)
  6. Far-field and near-field extraction           (extract_farfield_data)
  7. Results assembly                             (assemble_results)
  8. Save to .mat                                 (save_results)
  9. INTERCONNECT export                          (export_interconnect)
 10. Plotting                                     (plot_results)

Each stage is a standalone function and can be called independently — for
example, to re-run just the far-field analysis on an already-loaded session,
or to re-plot from a saved results dict.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import analysis
import config
from sim_helpers import (
    extract_and_process_field_profile,
    extract_farfield,
    extract_monitor_nearfield,
    find_bragg_resonance,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Data containers
#
# Simple dataclasses to carry the output of each analysis stage.
# Using named fields instead of raw dicts makes the data flow between
# stages explicit and easy to follow.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SParameters:
    """Broadband S-parameters after phase correction."""
    wl:   np.ndarray   # Wavelength array [m]
    R:    np.ndarray   # Reflectance |S11|²
    T:    np.ndarray   # Transmittance |S21|²
    Loss: np.ndarray   # Radiation loss = 1 - R - T
    T_mat: np.ndarray  # Transfer matrix [2×2×N_wl]
    S11:  np.ndarray   # Complex S11
    S21:  np.ndarray   # Complex S21


@dataclass
class ResonanceResult:
    """Location of the cavity resonance peak."""
    idx:             int    # Index into S-parameter wavelength array
    wavelength_m:    float  # Resonance wavelength [m]
    transmission:    float  # Transmission at resonance (peak value inside stopband)
    spectral_fwhm_m: float  # Half-power (3 dB) spectral bandwidth [m]


@dataclass
class FieldProfileResult:
    """1D longitudinal field profile along the grating axis."""
    x:                  np.ndarray  # Position array [m], cropped to grating extent
    intensity_1d:       np.ndarray  # |E|² integrated over transverse Y
    envelope_1d:        np.ndarray  # Peak envelope of the standing-wave pattern
    fwhm_m:             float       # FWHM of the envelope [m]
    monitor_wavelength_m: float     # Actual monitor wavelength closest to resonance [m]


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — S-parameter extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_s_parameters(sim, cfg) -> SParameters:
    """
    Extract broadband S-parameters from the live FDTD session and apply
    phase corrections (feed-line de-embedding, Bragg slope removal,
    resonance phase alignment).

    Requires: sim.fdtd session open and simulation complete.
    """
    wl, R, T, Loss, T_mat, S11, S21 = sim.get_s_and_t_matrix(
        neff_mat_file=config.NEFF_DATA_PATH,
        correct_length=cfg.phase_correction.do_length_correction,
        correct_envelope_and_t_phase=cfg.phase_correction.do_envelope_correction,
    )
    return SParameters(wl=wl, R=R, T=T, Loss=Loss, T_mat=T_mat, S11=S11, S21=S21)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Resonance detection
# ═══════════════════════════════════════════════════════════════════════════════

def find_resonance(s_params: SParameters) -> ResonanceResult:
    """
    Locate the cavity resonance peak within the Bragg stopband.

    Uses stopband-based detection (looks for the transmission peak inside
    the low-T region). Falls back to the global maximum if no stopband is found.
    Also computes the half-power (3 dB) spectral bandwidth via peak_widths.
    """
    from scipy.signal import peak_widths

    idx = find_bragg_resonance(s_params.wl, s_params.T)
    widths, _, _, _ = peak_widths(s_params.T, [idx], rel_height=0.5)
    dw = float(s_params.wl[1] - s_params.wl[0])
    return ResonanceResult(
        idx=idx,
        wavelength_m=float(s_params.wl[idx]),
        transmission=float(s_params.T[idx]),
        spectral_fwhm_m=float(widths[0]) * dw,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Field profile
# ═══════════════════════════════════════════════════════════════════════════════

def extract_field_profile(sim, resonance: ResonanceResult) -> FieldProfileResult:
    """
    Extract the 1D field intensity profile along the grating axis at the
    resonance wavelength. Integrates |E|² over the transverse Y axis,
    crops to the grating extent, and computes the peak envelope and FWHM.

    Requires: sim.fdtd session open and simulation complete.
    """
    f_x, I_x_1D, I_x_envelope, fwhm_val, actual_mon_wl = extract_and_process_field_profile(
        sim, resonance.wavelength_m
    )
    return FieldProfileResult(
        x=f_x,
        intensity_1d=I_x_1D,
        envelope_1d=I_x_envelope,
        fwhm_m=fwhm_val,
        monitor_wavelength_m=float(actual_mon_wl),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4 — 2D field monitors
# ═══════════════════════════════════════════════════════════════════════════════

def extract_2d_fields(sim) -> dict:
    """
    Extract broadband E-field data from the 2D profile monitors:
      - field_profile_2D_XY   : top view (XY plane)
      - field_profile_2D_YZ_cross : cross-section (YZ plane)

    Returns a dict with keys 'xy' and 'yz_cross', each containing
    coordinate arrays and complex E-field data. Returns {} if the
    monitors were not enabled.

    Requires: sim.fdtd session open and simulation complete.
    """
    if not sim.record_2d_fields_top_and_cross:
        return {}

    print("Extracting 2D XY (Top View) data...")
    res_xy = sim.fdtd.getresult("field_profile_2D_XY", "E")
    lam_xy = np.squeeze(res_xy['lambda'])
    xy_data = {
        'x':        np.squeeze(res_xy['x']),
        'y':        np.squeeze(res_xy['y']),
        'z':        np.squeeze(res_xy['z']),
        'E_res':    res_xy['E'],
        'lambda_3d': lam_xy,
    }
    del res_xy

    try:
        print("  Extracting Poynting vector for XY monitor...")
        P_xy = sim.fdtd.getresult("field_profile_2D_XY", "P")
        xy_data['P_res'] = P_xy['P']
        del P_xy
    except Exception as e:
        print(f"  Warning: Could not extract Poynting vector for XY: {e}")

    print("Extracting 2D YZ (Cross Section View) data...")
    res_yz = sim.fdtd.getresult("field_profile_2D_YZ_cross", "E")
    yz_data = {
        'x':        np.squeeze(res_yz['x']),
        'y':        np.squeeze(res_yz['y']),
        'z':        np.squeeze(res_yz['z']),
        'E_res':    res_yz['E'],
        'lambda_3d': np.squeeze(res_yz['lambda']),
    }
    del res_yz

    try:
        print("  Extracting Poynting vector for YZ monitor...")
        P_yz = sim.fdtd.getresult("field_profile_2D_YZ_cross", "P")
        yz_data['P_res'] = P_yz['P']
        del P_yz
    except Exception as e:
        print(f"  Warning: Could not extract Poynting vector for YZ: {e}")

    print("Extracting 2D XZ (Side View) data...")
    res_xz = sim.fdtd.getresult("field_profile_2D_XZ_side", "E")
    xz_data = {
        'x':        np.squeeze(res_xz['x']),
        'y':        np.squeeze(res_xz['y']),
        'z':        np.squeeze(res_xz['z']),
        'E_res':    res_xz['E'],
        'lambda_3d': np.squeeze(res_xz['lambda']),
    }
    del res_xz

    try:
        print("  Extracting Poynting vector for XZ monitor...")
        P_xz = sim.fdtd.getresult("field_profile_2D_XZ_side", "P")
        xz_data['P_res'] = P_xz['P']
        del P_xz
    except Exception as e:
        print(f"  Warning: Could not extract Poynting vector for XZ: {e}")

    return {'xy': xy_data, 'yz_cross': yz_data, 'xz_side': xz_data}


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5 — 3D field monitor
# ═══════════════════════════════════════════════════════════════════════════════

def extract_3d_fields(sim) -> dict:
    """
    Extract E-field data from the full 3D volume monitor.

    Returns a dict with coordinate arrays and complex E-field data,
    or {} if the 3D monitor was not enabled.

    Note: 3D monitors produce large arrays. Use sparingly.

    Requires: sim.fdtd session open and simulation complete.
    """
    if not getattr(sim, 'record_3d_fields', False):
        return {}

    print("Extracting full 3D field data...")
    res_3d = sim.fdtd.getresult("field_profile_3D", "E")
    data = {
        'x':        np.squeeze(res_3d['x']),
        'y':        np.squeeze(res_3d['y']),
        'z':        np.squeeze(res_3d['z']),
        'E_res':    res_3d['E'],
        'lambda_3d': np.squeeze(res_3d['lambda']),
    }
    del res_3d
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 6 — Far-field and near-field
# ═══════════════════════════════════════════════════════════════════════════════

def extract_farfield_data(sim, cfg) -> dict:
    """
    Extract far-field radiation patterns and near-field surface data from
    the side and top planar monitors.

    Returns a flat dict with keys:
      farfield_side, farfield_top     — E², ux, uy direction cosine grids
      nearfield_side, nearfield_top   — complex E on monitor surface
      farfield_config                 — recording parameters

    Missing monitors return empty dicts for their key. Returns {} entirely
    if far-field recording was not enabled.

    Requires: sim.fdtd session open and simulation complete.
    """
    if not sim.record_farfield:
        return {}

    ff_res = cfg.farfield.ff_resolution
    result = {'farfield_config': {'ff_res': ff_res}}

    print("\nExtracting far-field and near-field data...")
    for monitor_name, key in [("side_monitor", "side"), ("top_monitor", "top")]:
        ff = extract_farfield(sim.fdtd, monitor_name, ff_res=ff_res)
        nf = extract_monitor_nearfield(sim.fdtd, monitor_name)
        result[f"farfield_{key}"]  = ff if ff is not None else {}
        result[f"nearfield_{key}"] = nf if nf is not None else {}

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 7 — Assemble results
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_results(
    sim,
    s_params:      SParameters,
    resonance:     ResonanceResult,
    field_profile: FieldProfileResult,
    fields_2d:     dict,
    fields_3d:     dict,
    farfield:      dict,
) -> dict:
    """
    Combine all analysis stage outputs into a single flat dict suitable
    for saving to a .mat file or passing to plotting functions.

    The dict layout matches the format expected by the MATLAB analysis scripts.
    """
    results = {
        # Spectral response
        'wl_m':          s_params.wl,
        'wl_nm':         s_params.wl * 1e9,
        'T':             s_params.T,
        'R':             s_params.R,
        'loss':          s_params.Loss,
        'T_matrix':      s_params.T_mat,
        'S11_complex':   s_params.S11,
        'S21_complex':   s_params.S21,
        # Resonance
        'resonance_wavelength_nm':  resonance.wavelength_m * 1e9,
        'resonance_transmission':   resonance.transmission,
        'spectral_fwhm_nm':         resonance.spectral_fwhm_m * 1e9,
        # Device geometry
        'L_device':               2.0 * sim.x_grating_end,
        'pitch_m':                sim.pitch,
        'n_periods_each_side':    sim.n_periods_each_side,
        'core_height_m':          sim.core_height,
        'avg_corrugation_width_m': 0.5 * (sim.width_narrow + sim.width_wide),
        'corrugation_depth_m':    sim.width_wide - sim.width_narrow,
        # 1D field profile
        'field_x':                  field_profile.x,
        'field_energy_density_1D':  field_profile.intensity_1d,
        'field_envelope_1D':        field_profile.envelope_1d,
        'fwhm_m':                   field_profile.fwhm_m,
    }

    # 2D field monitors (only include if data was recorded)
    if fields_2d:
        results['field_xy']       = fields_2d.get('xy', np.array([]))
        results['field_yz_cross'] = fields_2d.get('yz_cross', np.array([]))
        results['field_xz_side']  = fields_2d.get('xz_side', np.array([]))

    # 3D field monitor
    if fields_3d:
        results['field_3d'] = fields_3d

    # Far-field and near-field (only include non-empty entries)
    for key, value in farfield.items():
        if value:  # skip empty dicts from failed extractions
            results[key] = value

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 8 — Save
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(results: dict, results_path: str) -> None:
    """Save the assembled results dict to a MATLAB .mat file."""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    sio.savemat(results_path, results)
    print(f"Results saved to: {results_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 9 — INTERCONNECT export
# ═══════════════════════════════════════════════════════════════════════════════

def export_interconnect(s_params: SParameters, results_dir: str, tag: str) -> None:
    """
    Export S-parameters to Lumerical INTERCONNECT format (.txt).

    The output file is placed alongside the .mat results file:
      <results_dir>/interconnect_symmetric_<tag>.txt
    """
    path = os.path.join(results_dir, f"interconnect_symmetric_{tag}.txt")
    analysis.export_for_interconnect_symmetric(path, s_params.wl, s_params.S11, s_params.S21)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 10 — Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(
    s_params:      SParameters,
    resonance:     ResonanceResult,
    field_profile: FieldProfileResult,
    tag:           str = "",
) -> None:
    """
    Generate two standard diagnostic figures:
      1. Spectral response (T, R) with resonance marker
      2. Longitudinal field profile with envelope and FWHM annotation
    """
    # Figure 1: Spectral response
    plt.figure(num="Spectrum", figsize=(10, 6))
    plt.plot(s_params.wl * 1e9, s_params.T, label="T (Modal)")
    plt.plot(s_params.wl * 1e9, s_params.R, label="R (Modal)")
    plt.plot(resonance.wavelength_m * 1e9, resonance.transmission,
             'ro', label="Detected Resonance")
    plt.title(f"Spectral Response\nScan: {tag}")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized power")
    plt.legend()
    plt.grid(True)

    # Figure 2: Field profile
    fwhm_um = field_profile.fwhm_m * 1e6
    y_max   = np.max(field_profile.envelope_1d)
    y_min   = np.min(field_profile.envelope_1d)
    target_y = y_min + 0.5 * (y_max - y_min)

    plt.figure(num="Field Profile", figsize=(10, 6))
    plt.plot(field_profile.x * 1e6, field_profile.envelope_1d,
             'r-', linewidth=2.5, label="Field Envelope (Peak Tracing)")
    plt.plot(field_profile.x * 1e6, field_profile.intensity_1d,
             'b-', alpha=0.3, linewidth=0.5, label="Raw |E|²")
    plt.hlines(target_y, -fwhm_um / 2, fwhm_um / 2, colors='k', linestyles='dashed')
    plt.text(0, target_y * 1.05, f"FWHM = {fwhm_um:.2f} µm",
             ha='center', color='black', fontweight='bold')
    plt.title(f"Mode Profile (Cropped)\nResonance at {field_profile.monitor_wavelength_m * 1e9:.2f} nm")
    plt.xlabel("Position x [µm]")
    plt.ylabel("Integrated Energy Density (a.u.)")
    plt.legend()
    plt.grid(True)


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_simulation(sim, cfg, results_path: str, tag: str = "") -> dict:
    """
    Run the full post-simulation analysis pipeline.

    Call this immediately after sim.fdtd.run() completes. It extracts all
    data from the live FDTD session, processes it, saves results to disk,
    and prepares plots.

    Parameters
    ----------
    sim          : PiShiftBraggFDTD instance (session must still be open)
    cfg          : SimulationConfig
    results_path : Full path to the output .mat file
    tag          : Short label used in plot titles and the INTERCONNECT filename

    Returns
    -------
    dict  — assembled results (same data that is saved to results_path)
    """
    results_dir = os.path.dirname(results_path)

    # 1. S-parameters
    print("Extracting S-parameters...")
    s_params = extract_s_parameters(sim, cfg)

    # 2. Resonance
    print("Finding resonance peak...")
    resonance = find_resonance(s_params)
    print(f"  → {resonance.wavelength_m * 1e9:.3f} nm  (T = {resonance.transmission:.3f})")

    # 3. Field profile
    print("Extracting field profile...")
    field_profile = extract_field_profile(sim, resonance)
    print(f"  → FWHM = {field_profile.fwhm_m * 1e6:.4f} µm")

    # 4. 2D fields
    fields_2d = extract_2d_fields(sim)

    # 5. 3D fields
    fields_3d = extract_3d_fields(sim)

    # 6. Far-field + near-field
    farfield = extract_farfield_data(sim, cfg)

    # 7. Assemble
    results = assemble_results(sim, s_params, resonance, field_profile,
                               fields_2d, fields_3d, farfield)

    # 8. Save
    save_results(results, results_path)

    # 9. INTERCONNECT export (optional)
    if cfg.run.export_interconnect:
        print("Exporting for INTERCONNECT...")
        export_interconnect(s_params, results_dir, tag)

    # 10. Plot (caller is responsible for plt.show())
    plot_results(s_params, resonance, field_profile, tag)

    return results
