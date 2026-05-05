import os

USE_GPU = True  # Set False on HPC/server (overridden by server_run.py)

# Base paths
BASE_SAVE_DIR = r"C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\cavity_width_comparison_v2"
NEFF_DATA_PATH = r"C:\Users\evyat\Lumerical\pi_shifts_FDTD_results\neff_vs_wl_new\FDE_sweep_results.mat"
MATERIAL_DB_PATH = None  # or r'C:\...\lgt_materials.mdf'
LUMAPI_PATH = r"C:\Program Files\Lumerical\v261\api\python\lumapi.py"


def __getattr__(name: str) -> str:
    """Compute LAYOUTS_DIR and RESULTS_DIR dynamically from BASE_SAVE_DIR.

    This ensures that if BASE_SAVE_DIR is changed after import (e.g.
    ``config.BASE_SAVE_DIR = "new_path"``), the derived paths always
    reflect the current value instead of the value at import time.
    """
    if name in ("LAYOUTS_DIR", "RESULTS_DIR"):
        sub = "layouts" if name == "LAYOUTS_DIR" else "results"
        run_name = os.environ.get("RUN_NAME", "")
        if run_name:
            # Per-run folder with the same layouts/ + results/ split inside,
            # mirroring run_mesh_convergence.py's CONV_DIR/{layouts,results}.
            path = os.path.join(BASE_SAVE_DIR, run_name, sub)
        else:
            path = os.path.join(BASE_SAVE_DIR, sub)
        os.makedirs(path, exist_ok=True)
        return path
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")