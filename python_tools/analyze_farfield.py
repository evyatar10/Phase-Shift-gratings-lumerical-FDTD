import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import config

# --- LUMAPI INITIALIZATION ---
try:
    import lumapi
except ImportError:
    LUMAPI_PATH = r"C:\\Program Files\\Lumerical\\v252\\api\\python\\lumapi.py"
    spec = importlib.util.spec_from_file_location("lumapi", LUMAPI_PATH)
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)

def analyze_farfield(fsp_path, monitor_name, target_wl):
    """
    Connects to Lumerical, extracts far-field data for a given monitor,
    and performs 3D far-field analysis in Python.
    """
    print(f"\n--- Analyzing {monitor_name} in {os.path.basename(fsp_path)} ---")
    
    # Initialize Lumerical session and load the saved FSP file
    fdtd = lumapi.FDTD()
    try:
        fdtd.load(fsp_path)
    except Exception as e:
        print(f"Failed to load file {fsp_path}. Error: {e}")
        return

    # 1. Safely get Wavelength Index using getresult (more robust than getdata)
    try:
        res = fdtd.getresult(monitor_name, "E")
        wl_array = np.squeeze(res['lambda'])
        
        # Find closest wavelength
        idx_f = np.argmin(np.abs(wl_array - target_wl))
        idx_f_1_based = int(idx_f + 1) # Lumerical eval scripts are 1-based
        actual_wl = wl_array[idx_f]
        print(f"Target WL: {target_wl*1e9:.2f} nm, Closest Monitor WL: {actual_wl*1e9:.2f} nm")
    except Exception as e:
        print(f"Error: Monitor '{monitor_name}' not found or lacks data. Did you run the updated simulation first?\nDetails: {e}")
        fdtd.close()
        return

    # --- LUMERICAL FAR-FIELD COMMANDS VIA EVAL ---
    # We send a script to Lumerical to do the heavy lifting of NTFF transformations
    # Removed transmission() to prevent eval crashes on profile monitors
    script = f"""
    mname = '{monitor_name}';
    idx = {idx_f_1_based};
    res = 201;
    
    # Calculate |E|^2
    E2 = farfield3d(mname, idx, res, res);
    ux = farfieldux(mname, idx, res, res);
    uy = farfielduy(mname, idx, res, res);
    
    # Calculate Complex Vectors
    E_vec = farfieldvector3d(mname, idx, res, res);
    Ex = pinch(E_vec, 3, 1);
    Ey = pinch(E_vec, 3, 2);
    Ez = pinch(E_vec, 3, 3);
    
    # Calculate Polar Vectors
    E_pol = farfieldpolar3d(mname, idx, res, res);
    Er = pinch(E_pol, 3, 1);
    Etheta = pinch(E_pol, 3, 2);
    Ephi = pinch(E_pol, 3, 3);
    
    # Integrate Power (Method 1: 30-degree cone)
    half_angle = 30;
    temp_cone = farfield3dintegrate(E2, ux, uy, half_angle, 0, 0);
    temp_far = farfield3dintegrate(E2, ux, uy);
    """
    
    try:
        fdtd.eval(script)
    except Exception as e:
        print(f"Lumerical evaluation failed for {monitor_name}. Check if the monitor has field data. Error: {e}")
        fdtd.close()
        return

    # --- PULL DATA BACK TO PYTHON ---
    E2 = np.squeeze(fdtd.getv("E2"))
    ux = np.squeeze(fdtd.getv("ux"))
    uy = np.squeeze(fdtd.getv("uy"))
    
    Ex = np.squeeze(fdtd.getv("Ex"))
    Ey = np.squeeze(fdtd.getv("Ey"))
    Ez = np.squeeze(fdtd.getv("Ez"))
    
    Er = np.squeeze(fdtd.getv("Er"))
    Etheta = np.squeeze(fdtd.getv("Etheta"))
    Ephi = np.squeeze(fdtd.getv("Ephi"))
    
    temp_cone = float(np.squeeze(fdtd.getv("temp_cone")))
    temp_far = float(np.squeeze(fdtd.getv("temp_far")))
    
    # Calculate fraction of radiation inside the 30-degree cone
    if temp_far > 0:
        cone_fraction = (temp_cone / temp_far) * 100
        print(f"Fraction of total radiated power within 30-deg cone: {cone_fraction:.2f}%")
    else:
        print("Warning: Total radiated power is 0. Monitor might be completely dark.")

    fdtd.close()

    # --- PLOTTING ---
    # Filter for valid hemisphere. Use indexing='ij' to match Lumerical's raw array shapes
    UX, UY = np.meshgrid(ux, uy, indexing='ij')
    valid_mask = (UX**2 + UY**2) <= 1.0
    
    E2_plot = np.where(valid_mask, E2, np.nan)
    
    # Convert to dB safely
    with np.errstate(divide='ignore'):
        E2_plot_dB = 10 * np.log10(E2_plot / np.nanmax(E2_plot))
    E2_plot_dB[E2_plot_dB < -60] = -60

    # Figure 1: |E|^2
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    c1 = ax1.contourf(ux, uy, E2_plot_dB, levels=50, cmap='jet')
    ax1.set_title(f"{monitor_name}: |E|^2 [dB]\nWL: {actual_wl*1e9:.2f} nm")
    ax1.set_xlabel("ux")
    ax1.set_ylabel("uy")
    ax1.axis('equal')
    fig1.colorbar(c1, ax=ax1, label='Normalized Intensity [dB]')

    # Figure 2: Vector Components (Ex, Ey, Ez)
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    comps = [Ex, Ey, Ez]
    titles = ["|Ex|", "|Ey|", "|Ez|"]
    for i, ax in enumerate(axes):
        comp_plot = np.abs(comps[i])
        comp_plot = np.where(valid_mask, comp_plot, np.nan)
        c = ax.contourf(ux, uy, comp_plot, levels=50, cmap='viridis')
        ax.set_title(f"{monitor_name}: {titles[i]}")
        ax.set_xlabel("ux")
        ax.set_ylabel("uy")
        ax.axis('equal')
        fig2.colorbar(c, ax=ax)

    # Figure 3: Polar Components (Er, Etheta, Ephi)
    fig3, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    pol_comps = [Er, Etheta, Ephi]
    pol_titles = ["|Er|", "|Etheta|", "|Ephi|"]
    for i, ax in enumerate(axes2):
        p_plot = np.abs(pol_comps[i])
        p_plot = np.where(valid_mask, p_plot, np.nan)
        c = ax.contourf(ux, uy, p_plot, levels=50, cmap='magma')
        ax.set_title(f"{monitor_name}: {pol_titles[i]}")
        ax.set_xlabel("ux")
        ax.set_ylabel("uy")
        ax.axis('equal')
        fig3.colorbar(c, ax=ax)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure this is pointing to a file generated AFTER the recent script updates
    layout_file = os.path.join(config.LAYOUTS_DIR, r"C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles_5_long\layouts\layout_100_periods_CONST.fsp")
    
    # Target resonance wavelength
    target_resonance = 1.562e-6 
    
    # Analyze Top Monitor
    analyze_farfield(layout_file, "top_monitor", target_resonance)
    
    # Analyze Side Monitor
    analyze_farfield(layout_file, "side_monitor", target_resonance)