"""
Single-iteration entry point for Athena SLURM array tasks (--option3).

Reads:
  SWEEP_KIND      mesh_conv_x | mesh_conv_yz | shutoff_conv_shutoff | spec
  SWEEP_INDEX     line index (0-based) into SWEEP_LIST   (= SLURM_ARRAY_TASK_ID)
  SWEEP_LIST      path to sweep_list.txt (default: /work/data/sweep_list.txt)
  SWEEP_FIXED_DYZ_NM  for kind=mesh_conv_x: global+box dyz to fix (default DEFAULT_DYZ_NM)
  SWEEP_FIXED_CELLS   for kind=mesh_conv_yz: cells_per_half_period to fix
                       (default: best_cells from Phase X checkpoint, else DEFAULT_CELLS)
  SWEEP_SPEC_MODULE   for kind=spec: dotted module path exposing SPEC: SweepSpec
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
_run_name = (os.environ.get("SWEEP_SPEC_MODULE", "").rsplit(".", 1)[-1]
             or os.environ.get("RUN_SCRIPT", ""))
if _run_name:
    os.environ["RUN_NAME"] = _run_name
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

VALID_KINDS = {"shift", "inner_size", "generic", "mesh_conv_x", "mesh_conv_yz",
               "shutoff_conv_shutoff", "spec", "cards",
               "inverse_design", "gradient_free_design"}
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
def _run_kind_mesh_conv_x(line: str) -> None:
    """Mesh-convergence X-axis (dx): sweeps one cells_per_half_period value."""
    from convergence_testing.run_mesh_convergence import (
        _run_one, _make_cfg, DEFAULT_DYZ_NM, CONV_DIR,
    )
    cells = int(line)
    dyz_nm = float(os.environ.get("SWEEP_FIXED_DYZ_NM", "") or DEFAULT_DYZ_NM)
    mc_cfg = _make_cfg()
    rec = _run_one(mc_cfg, cells=cells, dyz_nm=dyz_nm,
                   tag=f"phX_array_cells_{cells}")
    _save_array_part(CONV_DIR, "phX", idx, rec)


def _run_kind_mesh_conv_yz(line: str) -> None:
    """Mesh-convergence YZ-axis (dy=dz): sweeps one dyz_max_step_nm value
    (cells fixed from the X-phase via SWEEP_FIXED_CELLS env)."""
    from convergence_testing.run_mesh_convergence import (
        _run_one, _make_cfg, DEFAULT_CELLS, CONV_DIR,
    )
    dyz_nm = float(line)
    cells = int(os.environ.get("SWEEP_FIXED_CELLS", "") or DEFAULT_CELLS)
    mc_cfg = _make_cfg()
    rec = _run_one(mc_cfg, cells=cells, dyz_nm=dyz_nm,
                   tag=f"phYZ_array_dyz_{dyz_nm}")
    _save_array_part(CONV_DIR, "phYZ", idx, rec)


def _run_kind_shutoff_conv(line: str) -> None:
    """Auto-shutoff convergence: sweeps one auto_shutoff_min value."""
    from convergence_testing.run_auto_shutoff_convergence import (
        _run_one, _make_cfg, CONV_DIR,
    )
    shutoff_min = float(line)
    sc_cfg = _make_cfg()
    rec = _run_one(sc_cfg, shutoff_min=shutoff_min,
                   tag=f"phSHUTOFF_array_{shutoff_min:.0e}")
    _save_array_part(CONV_DIR, "phSHUTOFF", idx, rec)


def _save_array_part(conv_dir: str, phase_tag: str, task_idx: int, record: dict) -> None:
    """Per-task JSON for a convergence sweep — avoids concurrent writes to
    the shared checkpoint. Aggregate locally after the array completes."""
    import json
    os.makedirs(conv_dir, exist_ok=True)
    out = os.path.join(conv_dir, f"array_part_{phase_tag}_{task_idx:04d}.json")
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

    # When SWEEP_IS_PRELIM=1 the task runs PRELIM_SPEC/PRELIM_BASE from the
    # same module instead of SPEC/BASE. IS_PRELIM flag triggers sidecar write.
    is_prelim = os.environ.get("SWEEP_IS_PRELIM", "0") == "1" or \
                bool(getattr(module, "IS_PRELIM", False))
    spec_attr = "PRELIM_SPEC" if is_prelim else "SPEC"
    base_attr = "PRELIM_BASE" if is_prelim else "BASE"
    if not hasattr(module, spec_attr):
        raise RuntimeError(f"Module {module_name!r} has no {spec_attr} attribute")
    spec_base = getattr(module, base_attr, None)
    configs = getattr(module, spec_attr).expand(base=spec_base)

    if idx < 0 or idx >= len(configs):
        raise IndexError(f"SWEEP_INDEX={idx} out of range (spec produces {len(configs)} configs)")
    cfg_for_task = configs[idx]

    # Per-task locked-lambda path. {idx} expands to SLURM_ARRAY_TASK_ID so a
    # prelim ARRAY can write per-task sidecars and the main array can read
    # the matching one (task k ↔ task k).
    locked_path = os.environ.get("LOCKED_LAMBDA_FILE", "").strip()
    if locked_path:
        locked_path = locked_path.replace("{idx}", str(idx))

    # Read the sidecar only for non-prelim tasks. Prelim tasks WRITE it after
    # the sim completes (below), so the file shouldn't exist yet at this point.
    if locked_path and not is_prelim:
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
    result = run_single_sim(cfg_for_task)

    # Prelim sweep tasks write a per-task λ_res sidecar for the downstream main
    # array to consume. The main array's LOCKED_LAMBDA_FILE template uses the
    # same {idx} so task k ↔ task k.
    if is_prelim and locked_path:
        lam_nm = float(result["resonance_wavelength_nm"])
        payload = {
            "lambda_res_nm": lam_nm,
            "lambda_res_m":  lam_nm * 1e-9,
            "scan_width_nm": float(getattr(cfg_for_task.spectral, "scan_width_nm", 0.0)),
            "source":        module_name,
            "task_idx":      idx,
        }
        os.makedirs(os.path.dirname(locked_path) or ".", exist_ok=True)
        with open(locked_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[athena_run_one] prelim wrote λ_res = {lam_nm:.4f} nm → {locked_path}")


def _run_kind_cards(line: str) -> None:
    """One sim from a module's CARDS list (explicit per-device card list).

    Required env: SWEEP_SPEC_MODULE (dotted path to module exposing
    CARDS: list[ExperimentCard], optionally with BASE: SimulationConfig
    and RECORDS: list[DeviceRecord] parallel to CARDS).

    Each task gets its own RUN_NAME = "<module_short>/<card.label>" so that
    layouts/results from different devices land in separate folders and never
    collide on the auto-generated tag.
    """
    import importlib
    from runners.single.run_simulation import run_single_sim
    module_name = os.environ.get("SWEEP_SPEC_MODULE", "")
    if not module_name:
        raise RuntimeError("SWEEP_SPEC_MODULE env var required for kind=cards")
    module = importlib.import_module(module_name)
    if not hasattr(module, "CARDS"):
        raise RuntimeError(f"Module {module_name!r} has no top-level CARDS attribute")
    cards = module.CARDS
    if idx < 0 or idx >= len(cards):
        raise IndexError(f"SWEEP_INDEX={idx} out of range (CARDS has {len(cards)} entries)")

    card = cards[idx]
    base = getattr(module, "BASE", None)
    cfg_for_task = card.to_config(base=base)

    # Per-task RUN_NAME nests this device's output under the sweep label.
    # config.RESULTS_DIR / LAYOUTS_DIR are computed dynamically each access,
    # so changing RUN_NAME here propagates to run_single_sim's path resolution.
    module_short = module_name.rsplit(".", 1)[-1]
    label = (card.label or f"task_{idx:04d}").replace("/", "_").replace("\\", "_")
    os.environ["RUN_NAME"] = f"{module_short}/{label}"

    # Apply runtime override (license / disk-cleanup) on top of the card's cfg.
    # Do NOT override cavity_neg_detuning_nm — the card encodes the geometry
    # exactly as fabricated.
    cfg_for_task.run.cleanup_lumerical_data = cfg.run.cleanup_lumerical_data
    print(f"[athena_run_one] cards: idx={idx} label={label!r} "
          f"RUN_NAME={os.environ['RUN_NAME']!r}")
    run_single_sim(cfg_for_task)


def _run_kind_inverse_design(line: str) -> None:
    """One lumopt-driver run from an InverseDesignSpec module.

    The line content is unused; the array task index is the start_idx.

    Required env: SWEEP_SPEC_MODULE (dotted path to module exposing SPEC: InverseDesignSpec).
    Optional env: SWEEP_BASELINE_LAMBDA_NM (skip baseline run when supplied).
    """
    import importlib
    from runners.inverse_design.inverse_design import run_inverse_design

    module_name = os.environ.get("SWEEP_SPEC_MODULE", "")
    if not module_name:
        raise RuntimeError("SWEEP_SPEC_MODULE env var required for kind=inverse_design")
    module = importlib.import_module(module_name)
    if not hasattr(module, "SPEC"):
        raise RuntimeError(f"Module {module_name!r} has no top-level SPEC attribute")
    spec = module.SPEC
    base = getattr(module, "BASE", None)
    if base is None:
        from simulation_config import SimulationConfig
        base = SimulationConfig()
    base.run.cleanup_lumerical_data = cfg.run.cleanup_lumerical_data

    baseline_nm = os.environ.get("SWEEP_BASELINE_LAMBDA_NM", "").strip()
    baseline_m = float(baseline_nm) * 1e-9 if baseline_nm else None

    run_inverse_design(
        cfg=base,
        spec=spec,
        start_idx=idx,
        baseline_lambda_m=baseline_m,
    )


def _run_kind_gradient_free_design(line: str) -> None:
    """One Lumerical-PSO run from a GradientFreeDesignSpec module."""
    import importlib
    from runners.gradient_free_design.gradient_free_design import run_gradient_free_design

    module_name = os.environ.get("SWEEP_SPEC_MODULE", "")
    if not module_name:
        raise RuntimeError("SWEEP_SPEC_MODULE env var required for kind=gradient_free_design")
    module = importlib.import_module(module_name)
    if not hasattr(module, "SPEC"):
        raise RuntimeError(f"Module {module_name!r} has no top-level SPEC attribute")
    spec = module.SPEC
    base = getattr(module, "BASE", None)
    if base is None:
        from simulation_config import SimulationConfig
        base = SimulationConfig()
    base.run.cleanup_lumerical_data = cfg.run.cleanup_lumerical_data

    baseline_nm = os.environ.get("SWEEP_BASELINE_LAMBDA_NM", "").strip()
    baseline_m = float(baseline_nm) * 1e-9 if baseline_nm else None

    run_gradient_free_design(
        cfg=base,
        spec=spec,
        start_idx=idx,
        baseline_lambda_m=baseline_m,
    )


_DISPATCH = {
    "mesh_conv_x":            _run_kind_mesh_conv_x,
    "mesh_conv_yz":           _run_kind_mesh_conv_yz,
    "shutoff_conv_shutoff":   _run_kind_shutoff_conv,
    "spec":                   _run_kind_spec,
    "cards":                  _run_kind_cards,
    "inverse_design":         _run_kind_inverse_design,
    "gradient_free_design":   _run_kind_gradient_free_design,
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
