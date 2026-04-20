"""
Server-side wrapper for Pi-Shift Bragg Grating FDTD pipeline.

Patches config.py paths for Zeus at runtime — the original file is never modified.
If the simulation fails mid-run, prints a clear fallback message and exits non-zero.
"""
import glob
import os
import sys
import traceback

# ── 1. Add project directory to Python path ─────────────────────────────────
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_DIR)

# ── 2. Patch config BEFORE any other project imports ─────────────────────────
#    config.LAYOUTS_DIR and config.RESULTS_DIR are derived dynamically from
#    BASE_SAVE_DIR via __getattr__, so patching BASE_SAVE_DIR is sufficient.
import config

config.BASE_SAVE_DIR  = '/home/evyatarrubin/bragg_sim/results'
config.NEFF_DATA_PATH = '/home/evyatarrubin/bragg_sim/data/FDE_sweep_results.mat'

# Locate lumapi.py — check both known Lumerical installation paths
_api_candidates = [
    '/usr/local/lumerical-2021R2.5/api/python/lumapi.py',
    '/usr/local/lumerical-2021R1/api/python/lumapi.py',
    '/usr/local/lumerical/api/python/lumapi.py',
]
for _path in _api_candidates:
    if os.path.exists(_path):
        config.LUMAPI_PATH = _path
        break
else:
    print("\n" + "=" * 60)
    print("STARTUP ERROR — lumapi.py not found")
    print("=" * 60)
    print("Searched locations:")
    for p in _api_candidates:
        print(f"  {p}")
    print("\nFix: run this on Zeus to find the correct path:")
    print("  find /usr/local -name 'lumapi.py' 2>/dev/null")
    print("Then update _api_candidates in hpc/scripts/server_run.py")
    sys.exit(1)

print("=" * 60)
print("[server_run] Configuration")
print(f"  LUMAPI_PATH    = {config.LUMAPI_PATH}")
print(f"  BASE_SAVE_DIR  = {config.BASE_SAVE_DIR}")
print(f"  NEFF_DATA_PATH = {config.NEFF_DATA_PATH}")
print("=" * 60)

# ── 2b. Patch lumapi.ENVIRONPATH to use our fdtd-solutions wrapper ────────────
#   libinterop-api uses Qt QProcess with setProcessEnvironment(), which builds
#   a custom env that may not carry FDTD_LD_LIBRARY_PATH from the parent shell.
#   Our wrapper (jobs/bin/fdtd-solutions) hardcodes the library paths so the
#   fix works on every compute node regardless of QProcess env handling.
_WRAPPER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'jobs', 'bin')
)
sys.path.insert(0, os.path.dirname(config.LUMAPI_PATH))
import lumapi as _lumapi
_lumapi.ENVIRONPATH = _WRAPPER_DIR + ':' + _lumapi.ENVIRONPATH
print(f"[server_run] lumapi wrapper dir: {_WRAPPER_DIR}")

# ── 3. Import simulation modules (will use the patched config) ────────────────
from simulation_config import SimulationConfig

# ── 4. Configure ─────────────────────────────────────────────────────────────
cfg = SimulationConfig()
cfg.mesh.simulation_mode = "optimization"

# Override any parameter here without touching the original files:
# cfg.grating.n_periods_each_side = 120
# cfg.apodization.enabled = True
# cfg.farfield.enabled = True

# ── 5. Dispatch to selected script ───────────────────────────────────────────
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
print(f"[server_run] Running: {_run_script} ({_module_name}.{_func_name})")

# ── 6. Run — with fallback notification on failure ───────────────────────────
try:
    _run_func(cfg)
    print("\n[server_run] Pipeline completed successfully.")

except Exception as e:
    print("\n" + "=" * 60)
    print("SIMULATION FAILED — ACTION REQUIRED")
    print("=" * 60)
    print(f"Error type : {type(e).__name__}")
    print(f"Message    : {e}")
    print("\nFull traceback:")
    traceback.print_exc()

    # Check whether the .fsp layout was saved before the failure
    layouts_dir = config.LAYOUTS_DIR
    fsp_files = glob.glob(os.path.join(layouts_dir, "*.fsp"))

    if fsp_files:
        latest_fsp = max(fsp_files, key=os.path.getmtime)
        fsp_name   = os.path.basename(latest_fsp)
        print("\n" + "-" * 60)
        print("FALLBACK OPTION — Run the engine job manually:")
        print("-" * 60)
        print(f"  Layout saved at: {latest_fsp}")
        print()
        print("  Submit the fallback PBS job with:")
        print(f"    qsub -v FSP_FILE=\"{fsp_name}\" /home/evyatarrubin/bragg_sim/jobs/run_fsp_job.sh")
        print()
        print("  NOTE: Post-processing will NOT run automatically in fallback mode.")
        print("        Re-open the result .fsp manually in Lumerical to extract data,")
        print("        or contact your supervisor for a post-processing-only script.")
    else:
        print("\n" + "-" * 60)
        print("NO .fsp layout file was found in:")
        print(f"  {layouts_dir}")
        print()
        print("The failure likely occurred before the layout was saved.")
        print("Check the PBS output log (.out file) for details.")

    print("=" * 60)
    sys.exit(1)   # non-zero exit so PBS marks the job as failed
