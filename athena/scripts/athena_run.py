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
_run_name = os.environ.get("RUN_SCRIPT", "")
if _run_name:
    os.environ["RUN_NAME"] = _run_name
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
cfg.grating.cavity_neg_detuning_nm = 5.76   # phase-matched detuning (= pitch/2 − 244.24 nm)

# Server policy: delete Lumerical .h5 scratch after each iteration so NFS quota
# does not balloon over a sweep (each scratch dir is comparable to the .fsp).
# The .mat is the deliverable; the .fsp is kept for re-runs. Override per-job
# with KEEP_H5=1 (sbatch --export=...,KEEP_H5=1) when you need raw monitor data.
cfg.run.cleanup_lumerical_data = os.environ.get("KEEP_H5", "0") != "1"
print(f"  cleanup_lumerical_data = {cfg.run.cleanup_lumerical_data} "
      f"(KEEP_H5={os.environ.get('KEEP_H5', '0')})")

# Override any parameter here without touching the original files:
# cfg.grating.n_periods_each_side = 120
# cfg.apodization.enabled = True

# ── 6. Dispatch to selected script (auto-discovery) ──────────────────────────
# RUN_SCRIPT can be:
#   • a bare module name discovered under runners/single/ or convergence_testing/
#     (e.g. "run_simulation", "run_mesh_convergence"). Module must define a
#     top-level `run(cfg=None)` callable. Files with IS_HELPER=True or starting
#     with "_" are skipped.
#   • an entry from _ALIAS_SCRIPTS below — used by chained workflows whose
#     entry function isn't `run` (e.g. compare_3d_field_prelim).
import importlib

_AUTO_DIRS = [
    ("runners.single",       os.path.join(PROJECT_DIR, "runners", "single")),
    ("runners.tm",           os.path.join(PROJECT_DIR, "runners", "tm")),
    ("convergence_testing",  os.path.join(PROJECT_DIR, "convergence_testing")),
]

# Chain-trigger aliases: entry functions that aren't the conventional `run`.
# Keep this small — only for programmatic invocation by SPEC modules with
# PRELIM_RUN_SCRIPT. Auto-discovered names take priority over aliases.
_ALIAS_SCRIPTS = {
    "compare_3d_field_prelim": ("runners.single.compare_3d_field_default_vs_shift",
                                "run_compare_3d_field_prelim"),
    # Run the full mesh-convergence sweep (Phase X then YZ, in-process true
    # coordinate descent) as ONE sequential Option-2 GPU job. The module sets
    # IS_HELPER=True (so it's hidden from the Single picker) and is therefore not
    # auto-discovered; this alias makes `--option2 --run=run_mesh_convergence`
    # dispatch its top-level run(). Polarization is selected via CONV_POL env.
    "run_mesh_convergence": ("convergence_testing.run_mesh_convergence", "run"),
    # Back-compat for old picker labels — will be removed once docs/scripts
    # are updated to use bare module names.
    "single_sim":   ("runners.single.run_simulation",   "run_single_sim"),
    "simple_bragg": ("runners.single.run_simple_bragg", "run_simple_sim"),
}


def _discover_runners() -> dict:
    """Scan auto-discovery dirs; return {bare_module_name: dotted_module_path}.
    A module is included iff it defines a top-level `run` callable AND does
    not set IS_HELPER=True. Files starting with '_' are skipped."""
    found = {}
    for pkg, dir_path in _AUTO_DIRS:
        if not os.path.isdir(dir_path):
            continue
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            mod_short = fname[:-3]
            dotted = f"{pkg}.{mod_short}"
            try:
                mod = importlib.import_module(dotted)
            except Exception as _e:
                print(f"[athena_run]   skip {dotted}: import error ({_e!r})")
                continue
            if getattr(mod, "IS_HELPER", False):
                continue
            if not callable(getattr(mod, "run", None)):
                continue
            found[mod_short] = dotted
    return found


_DISCOVERED = _discover_runners()
print(f"[athena_run] discovered runners: {sorted(_DISCOVERED)}")

# Back-compat: existing job scripts default to RUN_SCRIPT=single_sim. Map the
# old hardcoded registry names to the auto-discovered module names so old
# submissions still work after the auto-discovery refactor.
_LEGACY_RUN_SCRIPT = {
    "single_sim":     "run_simulation",
    "simple_bragg":   "run_simple_bragg",
    "run_experiment": "run_experiment",
}

_run_script = os.environ.get("RUN_SCRIPT", "run_simulation")
_run_script = _LEGACY_RUN_SCRIPT.get(_run_script, _run_script)

if _run_script in _DISCOVERED:
    _module_name = _DISCOVERED[_run_script]
    _func_name   = "run"
elif _run_script in _ALIAS_SCRIPTS:
    _module_name, _func_name = _ALIAS_SCRIPTS[_run_script]
else:
    valid = sorted(set(_DISCOVERED) | set(_ALIAS_SCRIPTS))
    print(f"ERROR: Unknown RUN_SCRIPT='{_run_script}'. Valid options: {valid}")
    sys.exit(1)

_module = importlib.import_module(_module_name)
_run_func = getattr(_module, _func_name)
print(f"[athena_run] Running: {_run_script} ({_module_name}.{_func_name})")

# Optional per-runner results-folder override. By default the output folder is the
# runner's own name (RUN_NAME=RUN_SCRIPT, set near the top). A runner can instead
# declare a shared STUDY_DIR_NAME so a family of runners writes into ONE study
# folder — e.g. run_te / run_tm / run_tm_vs_te all land in results/tm_te/ rather
# than three separate per-runner folders. config.RESULTS_DIR reads RUN_NAME lazily,
# so updating it here (before run()) is honored.
_study_dir = getattr(_module, "STUDY_DIR_NAME", None)
if _study_dir:
    os.environ["RUN_NAME"] = _study_dir
    print(f"[athena_run] {_run_script} declares STUDY_DIR_NAME={_study_dir!r} -> results/{_study_dir}/")

# Optional per-runner override hook. If the runner module defines
# `build_cfg(cfg) -> cfg`, apply it here so local (__main__) and cluster
# runs share one source of truth for parameter overrides. Without this hook
# any setup inside a runner's `if __name__ == "__main__":` is silently
# ignored on the cluster (the dispatcher imports the module and calls run()
# directly, never executing __main__).
_build_cfg = getattr(_module, "build_cfg", None)
if callable(_build_cfg):
    print(f"[athena_run] Applying {_run_script}.build_cfg(cfg) overrides")
    cfg = _build_cfg(cfg)

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
    # All results are written and sim.close() has been called by the simulation
    # module's finally block. Lumapi's interpreter-shutdown cleanup of the
    # already-closed fdtd-solutions session returns non-zero (no traceback,
    # marks the SLURM job FAILED despite success). Skip that chain by exiting
    # immediately with a clean status — flush stdio first so logs aren't lost.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

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
        print(f"    bash athena/deploy_athena.sh --option1 --fsp {fsp_name}")
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
