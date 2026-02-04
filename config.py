import os

# Base paths
BASE_SAVE_DIR = r"C:\Users\evyat\Lumerical\long_bragg_grating_interconnect\bragg_fdtd_elements_v9_3d_profiles_min_80"
NEFF_DATA_PATH = r"C:\Users\evyat\Lumerical\pi_shifts_FDTD_results\neff_vs_wl_new\FDE_sweep_results.mat"
MATERIAL_DB_PATH = None  # or r'C:\...\lgt_materials.mdf'

# Create directories automatically
LAYOUTS_DIR = os.path.join(BASE_SAVE_DIR, "layouts")
RESULTS_DIR = os.path.join(BASE_SAVE_DIR, "results")

os.makedirs(LAYOUTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)