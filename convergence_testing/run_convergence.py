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
from run_simulation import extract_farfield, extract_monitor_nearfield
import config

# ── Config ────────────────────────────────────────────────────────────────
FF_RES = 201        # ux/uy grid resolution (must match plot scripts)
IDX_F = 1           # 1-based frequency index
N_STEPS = 6         # number of monitor distances to test


def run_convergence():
    # ── Simulation parameters (same as run_simulation.py) ─────────────────
    lambda_res_est = 1.5625e-6
    scan_width_nm = 16.0
    n_points_global = 3001

    avg_corr = 800e-9
    corr_depth = 200e-9
    w_wide = avg_corr + corr_depth / 2
    w_narrow = avg_corr - corr_depth / 2
    core_h = 350e-9

    span_multiplier = 6
    calc_y_span = w_wide + span_multiplier * lambda_res_est
    calc_z_span = core_h + span_multiplier * lambda_res_est

    pitch = 500e-9
    N_periods = 100
    cav_len = pitch / 2.0

    farfield_x_span_m = 30e-6
    monitor_y_span_m = calc_y_span - 0.5 * lambda_res_est
    monitor_z_span_m = calc_z_span - 0.5 * lambda_res_est

    # ── Distance arrays ───────────────────────────────────────────────────
    # Top monitors: Z from (core surface + 0.5*lambda) to (PML - 0.5*lambda)
    z_min = core_h / 2 + 0.5 * lambda_res_est
    z_max = calc_z_span / 2 - 0.5 * lambda_res_est
    top_z_positions = np.linspace(z_min, z_max, N_STEPS)

    # Side monitors: Y from (waveguide edge + 0.5*lambda) to (PML - 0.5*lambda)
    y_min = w_wide / 2 + 0.5 * lambda_res_est
    y_max = calc_y_span / 2 - 0.5 * lambda_res_est
    side_y_positions = np.linspace(y_min, y_max, N_STEPS)

    print("Top monitor Z positions (um):", top_z_positions * 1e6)
    print("  Distance from core surface (um):", (top_z_positions - core_h / 2) * 1e6)
    print("  Distance from core surface (lambda):", (top_z_positions - core_h / 2) / lambda_res_est)
    print()
    print("Side monitor Y positions (um):", side_y_positions * 1e6)
    print("  Distance from waveguide edge (um):", (side_y_positions - w_wide / 2) * 1e6)
    print("  Distance from waveguide edge (lambda):", (side_y_positions - w_wide / 2) / lambda_res_est)

    # ── Build simulation (no built-in far-field or 2D monitors) ───────────
    sim = PiShiftBraggFDTD(
        pitch=pitch,
        n_periods_each_side=N_periods,
        n_apod_periods_each_side=10,
        width_narrow=w_narrow,
        width_wide=w_wide,
        width_port=1000e-9,
        core_height=core_h,
        substrate_thickness=10e-6,
        override_cavity_length_nm=False,
        y_span=calc_y_span,
        z_span=calc_z_span,
        material_db_path=config.MATERIAL_DB_PATH,
        n_periods_dist_to_port=20,
        n_wls_dist_port_to_pml=5.0,
        n_eff_guess=1.55,
        n_wl_points=n_points_global,
        use_apodization=True,
        center_mod_depth_nm=10.0,
        use_cavity_mesh_override=True,
        use_symmetry=True,
        use_z_symmetry=True,
        use_constant_materials=True,
        n_core_const=1.977,
        # Disable all optional monitors
        record_2d_fields_top_and_cross=False,
        record_3d_fields=False,
        record_farfield=False,
        monitor_y_span_m=monitor_y_span_m,
        monitor_z_span_m=monitor_z_span_m,
    )

    sim.build()
    sim.update_scan(center_lambda_m=lambda_res_est, width_nm=scan_width_nm, n_points=n_points_global)

    # ── Add convergence test monitors ─────────────────────────────────────
    cavity_x = sim.cavity_length / 2.0

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
        print(f"  {name}: z = {z_pos*1e6:.3f} um  ({(z_pos - core_h/2)/lambda_res_est:.2f} lambda from core)")

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
        print(f"  {name}: y = {y_pos*1e6:.3f} um  ({(y_pos - w_wide/2)/lambda_res_est:.2f} lambda from edge)")

    # ── Save layout and run ───────────────────────────────────────────────
    layout_path = os.path.join(config.LAYOUTS_DIR, "layout_convergence_farfield.fsp")
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
        ff = extract_farfield(sim.fdtd, name)
        if ff is None:
            raise RuntimeError(f"Failed to extract far-field from {name}")
        top_ff_list.append(ff)

        nf = extract_monitor_nearfield(sim.fdtd, name)
        if nf is None:
            raise RuntimeError(f"Failed to extract near-field from {name}")

        # Pre-compute |E|^2 image (sum over components, squeeze to 2D)
        E = nf["E_res"]  # complex array
        E2 = np.sum(np.abs(E) ** 2, axis=-1)  # sum over Ex,Ey,Ez
        E2 = np.squeeze(E2)
        # For Z-normal: remaining dims should be (Nx, Ny)
        # Take last freq index if multiple
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
        ff = extract_farfield(sim.fdtd, name)
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
    results_path = os.path.join(config.RESULTS_DIR, "convergence_farfield.mat")

    mat_data = {
        "n_steps": N_STEPS,
        "lambda_res": lambda_res_est,
        "ff_res": FF_RES,
        "core_h": core_h,
        "w_wide": w_wide,
        "calc_y_span": calc_y_span,
        "calc_z_span": calc_z_span,
        # Distance arrays
        "top_z_positions": top_z_positions,
        "side_y_positions": side_y_positions,
        # Top monitor far-field: stack to (n_steps, FF_RES, FF_RES)
        "top_E2": np.stack([d["E2"] for d in top_ff_list]),
        "top_ux": top_ff_list[0]["ux"],
        "top_uy": top_ff_list[0]["uy"],
        # Side monitor far-field
        "side_E2": np.stack([d["E2"] for d in side_ff_list]),
        "side_ux": side_ff_list[0]["ux"],
        "side_uy": side_ff_list[0]["uy"],
        # Top monitor near-field (pre-computed |E|^2)
        "top_nf_E2": np.stack(top_nf_E2_list),
        "top_nf_x": top_nf_x,
        "top_nf_y": top_nf_y,
        # Side monitor near-field
        "side_nf_E2": np.stack(side_nf_E2_list),
        "side_nf_x": side_nf_x,
        "side_nf_z": side_nf_z,
    }

    sio.savemat(results_path, mat_data)
    print(f"\nResults saved: {results_path}")

    sim.close()
    print("Done.")


if __name__ == "__main__":
    run_convergence()
