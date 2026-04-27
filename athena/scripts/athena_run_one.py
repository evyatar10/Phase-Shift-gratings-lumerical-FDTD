"""
Single-iteration entry point for Athena SLURM array tasks (--option3).

Reads:
  SWEEP_KIND      shift | inner_size | generic | mesh_conv_a | mesh_conv_b | spec
  SWEEP_INDEX     line index (0-based) into SWEEP_LIST   (= SLURM_ARRAY_TASK_ID)
  SWEEP_LIST      path to sweep_list.txt (default: /work/data/sweep_list.txt)
  SWEEP_PARAM     for kind=generic: dot-path of the swept config field
  SWEEP_FIXED_DZ  for kind=mesh_conv_a: dz_divisor value to fix (default DEFAULT_DZ_DIV)
  SWEEP_FIXED_CELLS  for kind=mesh_conv_b: cells_per_half_period to fix
                      (default: best_cells from Phase A checkpoint, else DEFAULT_CELLS)
  SWEEP_SPEC_MODULE  for kind=spec: dotted module path of a study file that
                      exposes a top-level SPEC: SweepSpec
                      (e.g. runners.sweeps.apod_and_shift)

Same container plumbing as athena_run.py — only the dispatch differs.
"""
import glob
import os
import sys
import traceback

PROJECT_DIR = "/work/project"
if not os.path.isdir(PROJECT_DIR):
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_DIR)

# ── 1. Patch config BEFORE any other project imports ─────────────────────────
import config

config.BASE_SAVE_DIR  = '/work/results'
config.NEFF_DATA_PATH = '/work/data/FDE_sweep_results.mat'
config.USE_GPU        = True
config.LUMAPI_PATH    = '/opt/lumerical/v261/api/python/lumapi.py'

if not os.path.exists(config.LUMAPI_PATH):
    print(f"STARTUP ERROR — lumapi.py not found at: {config.LUMAPI_PATH}")
    sys.exit(1)

# ── 2. Read sweep dispatch env ───────────────────────────────────────────────
SWEEP_KIND  = os.environ.get("SWEEP_KIND", "")
SWEEP_LIST  = os.environ.get("SWEEP_LIST", "/work/data/sweep_list.txt")
SWEEP_INDEX = os.environ.get("SWEEP_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID", ""))

VALID_KINDS = {"shift", "inner_size", "generic", "mesh_conv_a", "mesh_conv_b", "spec"}
if SWEEP_KIND not in VALID_KINDS:
    print(f"ERROR: invalid SWEEP_KIND={SWEEP_KIND!r}; expected one of {sorted(VALID_KINDS)}")
    sys.exit(1)
if not SWEEP_INDEX:
    print("ERROR: SWEEP_INDEX (or SLURM_ARRAY_TASK_ID) not set")
    sys.exit(1)
try:
    idx = int(SWEEP_INDEX)
except ValueError:
    print(f"ERROR: SWEEP_INDEX={SWEEP_INDEX!r} is not an integer")
    sys.exit(1)

if not os.path.exists(SWEEP_LIST):
    print(f"ERROR: sweep list not found at {SWEEP_LIST}")
    sys.exit(1)

with open(SWEEP_LIST) as f:
    lines = [ln.strip() for ln in f if ln.strip()]

if idx < 0 or idx >= len(lines):
    print(f"ERROR: SWEEP_INDEX={idx} out of range (file has {len(lines)} lines)")
    sys.exit(1)

line = lines[idx]
print("=" * 60)
print(f"[athena_run_one] kind={SWEEP_KIND}  index={idx}/{len(lines) - 1}  line={line!r}")
print(f"  USE_GPU={config.USE_GPU}  REQUIRE_GPU={os.environ.get('REQUIRE_GPU', '0')}")
print("=" * 60)

# ── 3. Import lumapi and patch FDTD __init__ to enable GPU resource ──────────
sys.path.insert(0, os.path.dirname(config.LUMAPI_PATH))
import lumapi as _lumapi

_original_FDTD_init = _lumapi.FDTD.__init__
_REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "0") == "1"


def _patched_FDTD_init(self, *args, **kwargs):
    _original_FDTD_init(self, *args, **kwargs)
    try:
        self.setresource("FDTD", 1, "device type", "GPU")
        print("[athena_run_one] GPU enabled on FDTD resource 1.")
    except Exception as _e:
        msg = f"could not enable GPU via setresource: {_e}"
        if _REQUIRE_GPU:
            print(f"[athena_run_one] FATAL: {msg}")
            sys.exit(2)
        print(f"[athena_run_one] WARNING: {msg}; will run on CPU.")
        return
    try:
        device_type = self.getresource("FDTD", 1, "device type")
        print(f"[athena_run_one] device type readback = {device_type!r}")
        if _REQUIRE_GPU and str(device_type).strip().upper() != "GPU":
            print(f"[athena_run_one] FATAL: readback was {device_type!r}, expected 'GPU'.")
            sys.exit(2)
    except Exception as _e:
        print(f"[athena_run_one] WARNING: getresource readback failed: {_e}")


_lumapi.FDTD.__init__ = _patched_FDTD_init

# ── 4. Build a SimulationConfig with the same defaults athena_run.py uses ────
from simulation_config import SimulationConfig

cfg = SimulationConfig()
cfg.mesh.simulation_mode = "optimization"
cfg.grating.cavity_neg_detuning_nm = 5.76
cfg.run.cleanup_lumerical_data = os.environ.get("KEEP_H5", "0") != "1"


# ── 5. Dispatch ───────────────────────────────────────────────────────────────
def _run_kind_mesh_conv_a(line: str) -> None:
    """Convergence Phase A: one cells_per_half_period value."""
    from convergence_testing.run_mesh_convergence import (
        _run_one, _make_cfg, DEFAULT_DY_DIV, DEFAULT_DZ_DIV,
    )
    cells = int(line)
    dz_div = float(os.environ.get("SWEEP_FIXED_DZ", DEFAULT_DZ_DIV))
    mc_cfg = _make_cfg()
    rec = _run_one(mc_cfg, cells=cells, dy_div=DEFAULT_DY_DIV, dz_div=dz_div,
                   tag=f"phA_array_cells_{cells}")
    _save_array_part("phA", idx, rec)


def _run_kind_mesh_conv_b(line: str) -> None:
    """Convergence Phase B: one dz_divisor value (cells fixed from Phase A)."""
    from convergence_testing.run_mesh_convergence import (
        _run_one, _make_cfg, DEFAULT_DY_DIV, DEFAULT_CELLS,
    )
    dz_div = float(line)
    cells = int(os.environ.get("SWEEP_FIXED_CELLS", DEFAULT_CELLS))
    mc_cfg = _make_cfg()
    rec = _run_one(mc_cfg, cells=cells, dy_div=DEFAULT_DY_DIV, dz_div=dz_div,
                   tag=f"phB_array_dzdiv_{dz_div}")
    _save_array_part("phB", idx, rec)


def _save_array_part(phase_tag: str, task_idx: int, record: dict) -> None:
    """Per-task JSON for the convergence sweep — avoids concurrent writes to
    the shared checkpoint. Aggregate locally after the array completes."""
    import json
    from convergence_testing.run_mesh_convergence import CONV_DIR
    os.makedirs(CONV_DIR, exist_ok=True)
    out = os.path.join(CONV_DIR, f"array_part_{phase_tag}_{task_idx:04d}.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[athena_run_one] wrote per-task result: {out}")


def _run_kind_spec(line: str) -> None:
    """One sim from a SweepSpec module's expand() at index SWEEP_INDEX.

    The line content is unused — the array task index is the source of truth.

    Optional module-level hooks (all backward-compatible — no hook == old behavior):
      module.BASE             - SimulationConfig used as the base for SPEC.expand()
                                Lets a sweep encode all non-swept settings (e.g.
                                record_3d_fields, farfield.enabled) in one place
                                instead of per-task code.
      LOCKED_LAMBDA_FILE env  - Path to a JSON sidecar written by a prelim job
                                ({'lambda_res_m': float, ...}). When present, the
                                file is read and cfg.spectral.center_wavelength_m
                                is overwritten with lambda_res_m before running.
                                Used for the prelim → array chain.
    """
    import importlib
    import json
    from runners.single.run_simulation import run_single_sim
    module_name = os.environ.get("SWEEP_SPEC_MODULE", "")
    if not module_name:
        raise RuntimeError("SWEEP_SPEC_MODULE env var required for kind=spec")
    module = importlib.import_module(module_name)
    if not hasattr(module, "SPEC"):
        raise RuntimeError(f"Module {module_name!r} has no top-level SPEC attribute")

    # Optional per-module BASE config. Modules without BASE keep their existing
    # behavior (expand() with no base ⇒ default SimulationConfig per combo).
    spec_base = getattr(module, "BASE", None)
    configs = module.SPEC.expand(base=spec_base)

    if idx < 0 or idx >= len(configs):
        raise IndexError(f"SWEEP_INDEX={idx} out of range (spec produces {len(configs)} configs)")
    cfg_for_task = configs[idx]

    # Optional per-task center-wavelength lock from a prerequisite (prelim) job.
    locked_path = os.environ.get("LOCKED_LAMBDA_FILE", "").strip()
    if locked_path:
        if os.path.exists(locked_path):
            with open(locked_path) as f:
                locked = json.load(f)
            lam_m = float(locked["lambda_res_m"])
            cfg_for_task.spectral.center_wavelength_m = lam_m
            print(f"[athena_run_one] LOCKED_LAMBDA_FILE applied: "
                  f"center_wavelength_m = {lam_m*1e9:.4f} nm  ({locked_path})")
        else:
            print(f"[athena_run_one] WARNING: LOCKED_LAMBDA_FILE={locked_path} "
                  f"does not exist; using cfg's center_wavelength = "
                  f"{cfg_for_task.spectral.center_wavelength_m*1e9:.4f} nm.")

    # Apply common Athena overrides on top of the spec-built config (the spec
    # encodes the science; these encode the runtime environment policy).
    cfg_for_task.run.cleanup_lumerical_data = cfg.run.cleanup_lumerical_data
    run_single_sim(cfg_for_task)


_DISPATCH = {
    "mesh_conv_a": _run_kind_mesh_conv_a,
    "mesh_conv_b": _run_kind_mesh_conv_b,
    "spec":        _run_kind_spec,
}

try:
    _DISPATCH[SWEEP_KIND](line)
    print(f"\n[athena_run_one] task {idx} completed successfully.")
    # Same trick as athena_run.py:144 — bypass lumapi's interpreter-shutdown
    # cleanup that returns non-zero and would flip SLURM to FAILED.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

except Exception as e:
    print("\n" + "=" * 60)
    print(f"TASK {idx} FAILED — {type(e).__name__}: {e}")
    print("=" * 60)
    traceback.print_exc()
    sys.exit(1)
