"""
Far-field monitor distance convergence test.

Places multiple top (Z-normal) and side (Y-normal) monitors at different
distances from the waveguide, runs a single FDTD simulation, and saves
all far-field + near-field data to one .mat file for analysis.
"""

import sys
import os
import time
import numpy as np
import scipy.io as sio

# Add parent directory so we can import from the main project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bragg_device import PiShiftBraggFDTD
from sim_helpers import extract_farfield, extract_monitor_nearfield
from simulation_config import SimulationConfig
import config

# ── Config ────────────────────────────────────────────────────────────────
N_STEPS = 6         # number of monitor distances to test


CONVERGENCE_SPAN_MULT = 6  # Extra-large domain: more room for far monitors than standard (5×)


# ── Polarization ──────────────────────────────────────────────────────────
# CONV_POL selects the polarization, mirroring convergence_testing/
# run_mesh_convergence.py. The default "TE" reproduces the original far-field
# convergence byte-for-byte — nothing in `_apply_polarization`'s TM branch runs
# unless CONV_POL=TM (so the existing TE layout/.mat filenames are untouched).
#
# CONV_POL=TM switches to the TM-study device (calibrated pitch + constant
# indices, see runners/tm/ and project_tm_convergence_study) and re-centers the
# source on the TM cavity resonance, so the single-frequency far-field monitors
# ("use source limits" + 1 frequency point) sample the radiating TM mode. TM
# artifacts get a "_tm" filename tag so the TE results are never overwritten.
POLARIZATION = (os.environ.get("CONV_POL") or "TE").upper()   # unset OR empty → TE
_POL_TAG     = "" if POLARIZATION == "TE" else f"_{POLARIZATION.lower()}"

# TM-study device parameters (used ONLY when POLARIZATION == "TM").
# Mirrors convergence_testing/run_mesh_convergence.py — see that file and
# runners/tm/PITCH_ALIGNMENT.md for the pitch-calibration rationale.
TM_PITCH_NM      = 516.14     # calibrated pitch (n_core 1.97) re-centers TM near the TE wavelength
TM_N_CORE        = 1.97       # Si3N4 (user 2026-06-28; was 1.9963 "Luke")
TM_N_CLAD        = 1.444      # SiO2 "Palik" @ 1.55 µm
# Source center → the single-point far-field is sampled here. 1571.4 nm is the
# TM cavity resonance on the production mesh (cells=5 / dy=dz=50 nm, which this
# runner uses) from the mesh study's Phase-X table — so the monitors capture the
# on-resonance radiating cavity mode (NOT the mesh study's 1567 nm scan center,
# which is a wide-window center, not the peak).
TM_CENTER_M      = 1.5587e-6   # new TM resonance at n 1.97 / pitch 516.14 (was 1.5714 @ 1.9963/518.3)
TM_SCAN_WIDTH_NM = 40.0
TM_N_POINTS      = 2001


def _apply_polarization(cfg: SimulationConfig) -> None:
    """Mutate `cfg` in place for the selected polarization.

    For TE (default) this is a no-op — the caller's cfg (incl. the Athena
    dispatcher's detuning/mesh tweaks) is left untouched. For TM we switch to the
    canonical TM-study device (calibrated pitch, constant indices, zero detuning)
    and re-center the scan on the TM resonance, matching the TM mesh-convergence
    study so the recommended far-field monitor distance applies to the same TM
    device being analyzed.
    """
    if POLARIZATION != "TM":
        return
    cfg.source.polarization            = "TM"
    cfg.grating.pitch_m                = TM_PITCH_NM * 1e-9
    cfg.grating.cavity_neg_detuning_nm = 0.0   # canonical TM device — drop the TE-tuned detuning
    cfg.material.use_constant_materials = True
    cfg.material.n_core_const          = TM_N_CORE
    cfg.material.n_clad_const          = TM_N_CLAD
    cfg.spectral.center_wavelength_m   = TM_CENTER_M
    cfg.spectral.scan_width_nm         = TM_SCAN_WIDTH_NM
    cfg.spectral.n_wl_points           = TM_N_POINTS
    # Seed the mode solver near the Bragg n_eff (λ = 2·n_eff·Λ).
    cfg.material.n_eff_guess = (
        cfg.spectral.center_wavelength_m / (2.0 * cfg.grating.pitch_m))


def run_convergence(cfg: SimulationConfig = None):
    if cfg is None:
        cfg = SimulationConfig()

    _apply_polarization(cfg)
    print(f"\n{'=' * 62}")
    print(f"  FAR-FIELD MONITOR-DISTANCE CONVERGENCE  [polarization = {POLARIZATION}]")
    if POLARIZATION == "TM":
        print(f"  TM device: pitch={TM_PITCH_NM} nm, n_core={TM_N_CORE}, "
              f"n_clad={TM_N_CLAD}, source center={TM_CENTER_M * 1e9:.1f} nm")
    print(f"{'=' * 62}")

    lam = cfg.spectral.center_wavelength_m
    core_h = cfg.geometry.core_height_m
    w_wide = cfg.geometry.width_wide_m
    calc_y_span = w_wide + CONVERGENCE_SPAN_MULT * lam
    calc_z_span = core_h + CONVERGENCE_SPAN_MULT * lam

    farfield_x_span_m = cfg.farfield.farfield_x_span_m
    monitor_y_span_m = calc_y_span - 0.5 * lam
    monitor_z_span_m = calc_z_span - 0.5 * lam

    # ── Distance arrays ───────────────────────────────────────────────────
    # Top monitors: Z from (core surface + 0.5*lambda) to (PML - 0.5*lambda)
    z_min = core_h / 2 + 0.5 * lam
    z_max = calc_z_span / 2 - 0.5 * lam
    top_z_positions = np.linspace(z_min, z_max, N_STEPS)

    # Side monitors: Y from (waveguide edge + 0.5*lambda) to (PML - 0.5*lambda)
    y_min = w_wide / 2 + 0.5 * lam
    y_max = calc_y_span / 2 - 0.5 * lam
    side_y_positions = np.linspace(y_min, y_max, N_STEPS)

    print("Top monitor Z positions (um):", top_z_positions * 1e6)
    print("  Distance from core surface (um):", (top_z_positions - core_h / 2) * 1e6)
    print("  Distance from core surface (lambda):", (top_z_positions - core_h / 2) / lam)
    print()
    print("Side monitor Y positions (um):", side_y_positions * 1e6)
    print("  Distance from waveguide edge (um):", (side_y_positions - w_wide / 2) * 1e6)
    print("  Distance from waveguide edge (lambda):", (side_y_positions - w_wide / 2) / lam)

    # ── Build simulation (disable built-in optional monitors) ─────────────
    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled = False

    device_kwargs = cfg.to_device_kwargs()
    # Use the 6× domain instead of the config-derived spans
    device_kwargs['y_span'] = calc_y_span
    device_kwargs['z_span'] = calc_z_span
    device_kwargs['monitor_y_span_m'] = monitor_y_span_m
    device_kwargs['monitor_z_span_m'] = monitor_z_span_m

    sim = PiShiftBraggFDTD(**device_kwargs)

    sim.build()
    sim.update_scan(
        center_lambda_m=cfg.spectral.center_wavelength_m,
        width_nm=cfg.spectral.scan_width_nm,
        n_points=cfg.spectral.n_wl_points,
    )

    # ── Add convergence test monitors ─────────────────────────────────────
    cavity_x = sim.cavity_length / 2.0
    ff_res = cfg.farfield.ff_resolution

    print(f"\nAdding {N_STEPS} top monitors (Z-normal)...")
    for i, z_pos in enumerate(top_z_positions):
        name = f"top_monitor_{i+1}"
        sim.fdtd.addprofile()
        sim.fdtd.set("name", name)
        sim.fdtd.set("monitor type", "2D Z-normal")
        sim.fdtd.set("x", cavity_x)
        sim.fdtd.set("x span", farfield_x_span_m)
        sim.fdtd.set("y", 0)
        sim.fdtd.set("y span", monitor_y_span_m)
        sim.fdtd.set("z", z_pos)
        sim.fdtd.set("override global monitor settings", 1)
        sim.fdtd.set("use source limits", 1)
        sim.fdtd.set("frequency points", 1)
        print(f"  {name}: z = {z_pos*1e6:.3f} um  ({(z_pos - core_h/2)/lam:.2f} lambda from core)")

    print(f"\nAdding {N_STEPS} side monitors (Y-normal)...")
    for i, y_pos in enumerate(side_y_positions):
        name = f"side_monitor_{i+1}"
        sim.fdtd.addprofile()
        sim.fdtd.set("name", name)
        sim.fdtd.set("monitor type", "2D Y-normal")
        sim.fdtd.set("x", cavity_x)
        sim.fdtd.set("x span", farfield_x_span_m)
        sim.fdtd.set("y", y_pos)
        sim.fdtd.set("z", 0)
        sim.fdtd.set("z span", monitor_z_span_m)
        sim.fdtd.set("override global monitor settings", 1)
        sim.fdtd.set("use source limits", 1)
        sim.fdtd.set("frequency points", 1)
        print(f"  {name}: y = {y_pos*1e6:.3f} um  ({(y_pos - w_wide/2)/lam:.2f} lambda from edge)")

    # ── Save layout and run ───────────────────────────────────────────────
    layout_path = os.path.join(config.LAYOUTS_DIR, f"layout_convergence_farfield{_POL_TAG}.fsp")
    sim.fdtd.save(layout_path)
    print(f"\nLayout saved: {layout_path}")

    start = time.perf_counter()
    sim.fdtd.run()
    print(f"Simulation time: {time.perf_counter() - start:.1f} s")

    # ── Extract far-field and near-field from all monitors ────────────────
    print("\n--- Extracting top monitor data ---")
    top_ff_list = []
    top_nf_E2_list = []
    top_nf_x = None
    top_nf_y = None

    for i in range(N_STEPS):
        name = f"top_monitor_{i+1}"
        ff = extract_farfield(sim.fdtd, name, ff_res=ff_res)
        if ff is None:
            raise RuntimeError(f"Failed to extract far-field from {name}")
        top_ff_list.append(ff)

        nf = extract_monitor_nearfield(sim.fdtd, name)
        if nf is None:
            raise RuntimeError(f"Failed to extract near-field from {name}")

        E = nf["E_res"]
        E2 = np.sum(np.abs(E) ** 2, axis=-1)
        E2 = np.squeeze(E2)
        if E2.ndim > 2:
            E2 = E2[..., -1]
        E2 = np.squeeze(E2)
        top_nf_E2_list.append(E2)

        if top_nf_x is None:
            top_nf_x = np.squeeze(nf["x"])
            top_nf_y = np.squeeze(nf["y"])

    print("\n--- Extracting side monitor data ---")
    side_ff_list = []
    side_nf_E2_list = []
    side_nf_x = None
    side_nf_z = None

    for i in range(N_STEPS):
        name = f"side_monitor_{i+1}"
        ff = extract_farfield(sim.fdtd, name, ff_res=ff_res)
        if ff is None:
            raise RuntimeError(f"Failed to extract far-field from {name}")
        side_ff_list.append(ff)

        nf = extract_monitor_nearfield(sim.fdtd, name)
        if nf is None:
            raise RuntimeError(f"Failed to extract near-field from {name}")

        E = nf["E_res"]
        E2 = np.sum(np.abs(E) ** 2, axis=-1)
        E2 = np.squeeze(E2)
        if E2.ndim > 2:
            E2 = E2[..., -1]
        E2 = np.squeeze(E2)
        side_nf_E2_list.append(E2)

        if side_nf_x is None:
            side_nf_x = np.squeeze(nf["x"])
            side_nf_z = np.squeeze(nf["z"])

    # ── Save to .mat ──────────────────────────────────────────────────────
    results_path = os.path.join(config.RESULTS_DIR, f"convergence_farfield{_POL_TAG}.mat")

    mat_data = {
        "n_steps": N_STEPS,
        "polarization": POLARIZATION,
        "pitch_m": cfg.grating.pitch_m,
        "n_core": cfg.material.n_core_const,
        "n_clad": cfg.material.n_clad_const,
        "lambda_res": lam,
        "ff_res": ff_res,
        "core_h": core_h,
        "w_wide": w_wide,
        "calc_y_span": calc_y_span,
        "calc_z_span": calc_z_span,
        # Distance arrays
        "top_z_positions": top_z_positions,
        "side_y_positions": side_y_positions,
        # Top monitor far-field
        "top_E2": np.stack([d["E2"] for d in top_ff_list]),
        "top_ux": top_ff_list[0]["ux"],
        "top_uy": top_ff_list[0]["uy"],
        # Side monitor far-field
        "side_E2": np.stack([d["E2"] for d in side_ff_list]),
        "side_ux": side_ff_list[0]["ux"],
        "side_uy": side_ff_list[0]["uy"],
        # Top monitor near-field
        "top_nf_E2": np.stack(top_nf_E2_list),
        "top_nf_x": top_nf_x,
        "top_nf_y": top_nf_y,
        # Side monitor near-field
        "side_nf_E2": np.stack(side_nf_E2_list),
        "side_nf_x": side_nf_x,
        "side_nf_z": side_nf_z,
    }

    sio.savemat(results_path, mat_data, do_compression=True)
    print(f"\nResults saved: {results_path}")

    sim.close()
    print("Done.")


# Auto-discovery contract: athena_run.py looks for `run` on each runner module.
run = run_convergence


if __name__ == "__main__":
    run_convergence()
