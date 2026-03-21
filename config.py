import os

# Base paths
BASE_SAVE_DIR = r"C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles_8"
NEFF_DATA_PATH = r"C:\Users\evyat\Lumerical\pi_shifts_FDTD_results\neff_vs_wl_new\FDE_sweep_results.mat"
MATERIAL_DB_PATH = None  # or r'C:\...\lgt_materials.mdf'
LUMAPI_PATH = r"C:\Program Files\Lumerical\v252\api\python\lumapi.py"

# Create directories automatically
LAYOUTS_DIR = os.path.join(BASE_SAVE_DIR, "layouts")
RESULTS_DIR = os.path.join(BASE_SAVE_DIR, "results")

os.makedirs(LAYOUTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)