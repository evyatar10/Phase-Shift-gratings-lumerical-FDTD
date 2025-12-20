import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt

# Try to import lumapi normally
try:
    import lumapi
except ImportError:
    import importlib.util

    # Adjust path if necessary to match your installation
    LUMAPI_PATH = r"C:\\Program Files\\Lumerical\\v252\\api\\python\\lumapi.py"
    spec = importlib.util.spec_from_file_location("lumapi", LUMAPI_PATH)
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)


def run_fde_sweep_builtin(base_dir):
    # ------------------------------------------------------------------
    # 1. PARAMETERS
    # ------------------------------------------------------------------
    # Core dimensions
    width_nm = 700e-9
    height_nm = 350e-9

    # Sweep settings
    center_lambda = 1.573e-6
    scan_width_nm = 40.0
    n_points = 10

    # --- DYNAMIC SPAN CALCULATION ---
    # Formula: Core Dimension + 1.8 * Center Wavelength
    y_span = width_nm + (1.8 * center_lambda)
    z_span = height_nm + (1.8 * center_lambda)

    half_w = 0.5 * scan_width_nm * 1e-9
    lam_min = center_lambda - half_w
    lam_max = center_lambda + half_w

    # ------------------------------------------------------------------
    # 2. SETUP & BUILD
    # ------------------------------------------------------------------
    layouts_dir = os.path.join(base_dir, "layouts")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(layouts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    layout_path = os.path.join(layouts_dir, "FDE_sweep_builtin.lms")
    result_path = os.path.join(results_dir, "FDE_sweep_results.mat")

    mode = lumapi.MODE()

    # --- Materials Setup ---
    custom_sin = "SiN_custom"
    custom_sio2 = "SiO2_custom"
    core_mat_base = "Si3N4 (Silicon Nitride) - Luke"
    clad_mat_base = "SiO2 (Glass) - Palik"

    script = f'''
    if (haveresult("{custom_sin}", "material")) {{ deletematerial("{custom_sin}"); }}
    if (haveresult("{custom_sio2}", "material")) {{ deletematerial("{custom_sio2}"); }}

    m1 = copymaterial("{core_mat_base}");
    setmaterial(m1, "name", "{custom_sin}");
    m2 = copymaterial("{clad_mat_base}");
    setmaterial(m2, "name", "{custom_sio2}");

    setmaterial("{custom_sin}",  "specify fit range", 1);
    setmaterial("{custom_sio2}", "specify fit range", 1);
    setmaterial("{custom_sin}",  "wavelength min", {lam_min});
    setmaterial("{custom_sin}",  "wavelength max", {lam_max});
    setmaterial("{custom_sio2}", "wavelength min", {lam_min});
    setmaterial("{custom_sio2}", "wavelength max", {lam_max});
    setmaterial("{custom_sin}",  "make fit passive", 1);
    setmaterial("{custom_sio2}", "make fit passive", 1);
    '''
    mode.eval(script)

    # --- Geometry ---
    mode.addrect()
    mode.set("name", "waveguide")
    mode.set("x", 0);
    mode.set("x span", 1e-6)
    mode.set("y", 0);
    mode.set("y span", width_nm)
    mode.set("z", 0);
    mode.set("z span", height_nm)
    mode.set("material", custom_sin)

    # --- FDE Region ---
    mode.addfde()
    mode.set("solver type", "2D X normal")
    mode.set("x", 0)
    mode.set("y", 0);
    mode.set("y span", y_span)
    mode.set("z", 0);
    mode.set("z span", z_span)
    mode.set("background material", custom_sio2)

    # Boundaries
    for bc in ["y min bc", "y max bc", "z min bc", "z max bc"]:
        mode.set(bc, "PML")

    # --- Mesh Settings ---
    # Using 'maximum mesh step' (0.05 um) for consistent resolution
    mode.set("define y mesh by", "maximum mesh step")
    mode.set("define z mesh by", "maximum mesh step")
    mode.set("dy", 0.05e-6)
    mode.set("dz", 0.05e-6)

    # ------------------------------------------------------------------
    # 3. RUN FREQUENCY SWEEP
    # ------------------------------------------------------------------
    print(f"1. Setting calculation wavelength to start: {lam_min * 1e6:.4f} um")
    mode.setanalysis("wavelength", lam_min)

    print("2. Calculating modes at start wavelength...")
    mode.findmodes()

    print("3. Selecting Fundamental Mode (Mode 1)...")
    mode.selectmode(1)

    print("4. Configuring Frequency Sweep...")
    mode.setanalysis("track selected mode", 1)
    mode.setanalysis("detailed dispersion calculation", 1)
    mode.setanalysis("stop wavelength", lam_max)
    mode.setanalysis("number of points", n_points)

    print(f"5. Running Frequency Sweep ({n_points} points)...")
    mode.frequencysweep()

    # ------------------------------------------------------------------
    # 4. EXTRACT RESULTS
    # ------------------------------------------------------------------
    print("6. Extracting data...")

    # Get Raw Data
    # Lumerical frequency sweep stores 'f' (Hz), not 'wavelength'
    freq_Hz = np.squeeze(mode.getdata("frequencysweep", "f"))
    neff_complex = np.squeeze(mode.getdata("frequencysweep", "neff"))

    # Get Loss (Default Lumerical unit is dB/m)
    loss_db_m = np.squeeze(mode.getdata("frequencysweep", "loss"))

    # --- UNIT CONVERSIONS ---
    c0 = 299792458.0
    wavelengths_out = c0 / freq_Hz

    # Convert dB/m to dB/cm (Divide by 100)
    loss_db_cm = loss_db_m / 100.0

    # Save Layout
    mode.save(layout_path)
    print(f"Layout saved: {layout_path}")

    # Save .mat
    mat_data = {
        "wavelengths": wavelengths_out,
        "frequency": freq_Hz,
        "neff_complex": neff_complex,
        "loss_dB_cm": loss_db_cm,
        "loss_dB_m": loss_db_m,
        "neff_real": np.real(neff_complex),
        "neff_imag": np.imag(neff_complex)
    }

    sio.savemat(result_path, mat_data)
    print(f"Results saved: {result_path}")

    mode.close()

    # ------------------------------------------------------------------
    # 5. PLOTTING WITH DUAL AXIS & CORRECT UNITS
    # ------------------------------------------------------------------
    print("7. Plotting results...")

    wl_um = wavelengths_out * 1e6
    neff_real = np.real(neff_complex)
    neff_imag = np.imag(neff_complex)

    plt.figure(figsize=(10, 8))

    # --- SUBPLOT 1: Complex Effective Index (Dual Axis) ---
    ax1 = plt.subplot(2, 1, 1)

    # Left Axis: Real Part
    color_real = 'tab:blue'
    lns1 = ax1.plot(wl_um, neff_real, color=color_real, marker='.', label=r"Re($n_{eff}$)")
    ax1.set_xlabel(r"Wavelength ($\mu m$)")
    ax1.set_ylabel("Real Part", color=color_real, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_real)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Right Axis: Imaginary Part
    ax2 = ax1.twinx()
    color_imag = 'tab:orange'
    lns2 = ax2.plot(wl_um, neff_imag, color=color_imag, marker='x', linestyle='--', label=r"Im($n_{eff}$)")
    ax2.set_ylabel("Imaginary Part", color=color_imag, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_imag)

    # Combined Legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')

    plt.title("Complex Effective Index vs Wavelength")

    # --- SUBPLOT 2: Loss (dB/cm) ---
    plt.subplot(2, 1, 2)
    plt.plot(wl_um, loss_db_cm, 'r.-', linewidth=2)
    plt.title("Propagation Loss vs Wavelength")
    plt.xlabel(r"Wavelength ($\mu m$)")
    plt.ylabel("Loss (dB/cm)")  # Now matches the converted data
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Update this path to your folder
    user_base_dir = r"C:\Users\evyat\Lumerical\pi_shifts_FDTD_results\neff_vs_wl"
    run_fde_sweep_builtin(user_base_dir)