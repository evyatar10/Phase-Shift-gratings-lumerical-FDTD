"""
Athena (GPU) analog of hpc/scripts/server_run.py.

Patches config.py paths for Athena at runtime, enables GPU on the FDTD
resource, and dispatches to the same simulation/sweep modules as the Zeus
pipeline — without touching Zeus-specific code.

This file runs INSIDE the Lumerical container on Athena compute nodes.
It is invoked by hpc_gpu/jobs/run_python_gpu.sh via:
    xvfb-run -a python /work/scripts/athena_run.py
"""
import glob
import os
import sys
import traceback

# ── 1. Locate the project root and add it to the Python path ─────────────────
# Inside the container, the project is mounted at /work/project/
# This script is at /work/scripts/athena_run.py
PROJECT_DIR = "/work/project"
if not os.path.isdir(PROJECT_DIR):
    # Fallback: derive from script location when testing outside container
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_DIR)

# ── 2. Patch config BEFORE any other project imports ─────────────────────────
#    Patching BASE_SAVE_DIR is sufficient — LAYOUTS_DIR and RESULTS_DIR are
#    derived dynamically via config.__getattr__ (see config.py).
import config

config.BASE_SAVE_DIR  = '/work/results'
config.NEFF_DATA_PATH = '/work/data/FDE_sweep_results.mat'
config.USE_GPU        = True   # Athena has GPUs — opposite of Zeus override
config.LUMAPI_PATH    = '/opt/lumerical/v261/api/python/lumapi.py'

if not os.path.exists(config.LUMAPI_PATH):
    print("\n" + "=" * 60)
    print("STARTUP ERROR — lumapi.py not found in the container")
    print("=" * 60)
    print(f"Expected at: {config.LUMAPI_PATH}")
    print("\nThe Lumerical container may not have installed correctly.")
    print("Rebuild with:  bash hpc_gpu/container/build.sh")
    sys.exit(1)

print("=" * 60)
print("[athena_run] Configuration")
print(f"  LUMAPI_PATH    = {config.LUMAPI_PATH}")
print(f"  BASE_SAVE_DIR  = {config.BASE_SAVE_DIR}")
print(f"  NEFF_DATA_PATH = {config.NEFF_DATA_PATH}")
print(f"  USE_GPU        = {config.USE_GPU}")
print(f"  REQUIRE_GPU    = {os.environ.get('REQUIRE_GPU', '0')}")
print("=" * 60)

# ── 3. Import lumapi and configure GPU resource ───────────────────────────────
#    Unlike Zeus (which requires an LD_PRELOAD ompt stub and Xvfb namespace
#    workarounds), inside the Lumerical container the environment is clean —
#    no glibc mismatch, no lib path conflicts. lumapi loads directly.
sys.path.insert(0, os.path.dirname(config.LUMAPI_PATH))
import lumapi as _lumapi

# ── 4. Import simulation modules (will use the patched config) ────────────────
from simulation_config import SimulationConfig

# ── 5. Configure simulation parameters ───────────────────────────────────────
cfg = SimulationConfig()
cfg.mesh.simulation_mode = "optimization"

# Override any parameter here without touching the original files:
# cfg.grating.n_periods_each_side = 120
# cfg.apodization.enabled = True

# ── 6. Dispatch to selected script ───────────────────────────────────────────
_SCRIPTS = {
    "single_sim":       ("run_simulation",                          "run_single_sim"),
    "sweep_shift":      ("ToothShift.run_sweep_innermost_shift",    "run_sweep_innermost_shift"),
    "sweep_inner_size": ("ToothShift.run_sweep_inner_tooth_size",   "run_sweep_inner_tooth_size"),
}
_run_script = os.environ.get("RUN_SCRIPT", "single_sim")
if _run_script not in _SCRIPTS:
    print(f"ERROR: Unknown RUN_SCRIPT='{_run_script}'. Valid options: {list(_SCRIPTS)}")
    sys.exit(1)

_module_name, _func_name = _SCRIPTS[_run_script]
import importlib
_module = importlib.import_module(_module_name)
_run_func = getattr(_module, _func_name)
print(f"[athena_run] Running: {_run_script} ({_module_name}.{_func_name})")

# ── 7. Enable GPU on the FDTD resource before running ────────────────────────
#    Patch lumapi.FDTD.__init__ (the method), NOT the class itself.
#    Replacing the class object breaks lumapi's own super(FDTD, self) call
#    because "FDTD" is looked up by name in lumapi's module globals at runtime —
#    swapping the class (or replacing it with a function) makes super() see the
#    wrong type and raises TypeError. Patching only __init__ keeps the class
#    identity intact so lumapi's super() continues to resolve correctly.

_original_FDTD_init = _lumapi.FDTD.__init__
_REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "0") == "1"

def _patched_FDTD_init(self, *args, **kwargs):
    _original_FDTD_init(self, *args, **kwargs)
    try:
        self.setresource("FDTD", 1, "device type", "GPU")
        print("[athena_run] GPU enabled on FDTD resource 1 (device type='GPU').")
    except Exception as _e:
        msg = f"could not enable GPU via setresource: {_e}"
        if _REQUIRE_GPU:
            print(f"[athena_run] FATAL: {msg}")
            print("[athena_run] REQUIRE_GPU=1 — refusing to run on CPU.")
            sys.exit(2)
        print(f"[athena_run] WARNING: {msg}")
        print("[athena_run] Simulation will proceed on CPU as fallback.")
        return

    # Confirm what the engine will actually use.
    try:
        device_type = self.getresource("FDTD", 1, "device type")
        print(f"[athena_run] getresource('FDTD',1,'device type') = {device_type!r}")
        if _REQUIRE_GPU and str(device_type).strip().upper() != "GPU":
            print(f"[athena_run] FATAL: device type read back as {device_type!r}, expected 'GPU'.")
            sys.exit(2)
    except Exception as _e:
        print(f"[athena_run] WARNING: getresource readback failed: {_e}")

_lumapi.FDTD.__init__ = _patched_FDTD_init

# ── 8. Run — with fallback notification on failure ────────────────────────────
try:
    _run_func(cfg)
    print("\n[athena_run] Pipeline completed successfully.")

except Exception as e:
    print("\n" + "=" * 60)
    print("SIMULATION FAILED — ACTION REQUIRED")
    print("=" * 60)
    print(f"Error type : {type(e).__name__}")
    print(f"Message    : {e}")
    print("\nFull traceback:")
    traceback.print_exc()

    # Check whether any .fsp layout was saved before the failure
    layouts_dir = config.LAYOUTS_DIR
    fsp_files = glob.glob(os.path.join(layouts_dir, "*.fsp"))

    if fsp_files:
        latest_fsp = max(fsp_files, key=os.path.getmtime)
        fsp_name   = os.path.basename(latest_fsp)
        print("\n" + "-" * 60)
        print("FALLBACK OPTION — Run the engine job directly:")
        print("-" * 60)
        print(f"  Layout saved at: {latest_fsp}")
        print()
        print("  Submit engine-only job from your local machine:")
        print(f"    bash hpc_gpu/deploy_athena.sh --option1 --fsp {fsp_name}")
        print()
        print("  Or manually on Athena:")
        print(f"    sbatch --export=FSP_FILE=\"{fsp_name}\" "
              f"/home/evyatarrubin/bragg_sim_gpu/jobs/run_fsp_gpu.sh")
    else:
        print("\n" + "-" * 60)
        print("NO .fsp layout file was found in:")
        print(f"  {layouts_dir}")
        print()
        print("The failure likely occurred before the layout was saved.")
        print("Check the SLURM output log (.out file) for details.")

    print("=" * 60)
    sys.exit(1)
