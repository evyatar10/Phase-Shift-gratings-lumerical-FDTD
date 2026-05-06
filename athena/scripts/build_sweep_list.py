"""
Build sweep_list.txt locally for an Athena --option3 SLURM array submission.

Each line in the output file becomes one SLURM array task. The line format
depends on SWEEP_KIND (interpreted on the cluster by athena_run_one.py):

  mesh_conv_a            <cells_per_half_period>
  mesh_conv_b            <dyz_max_step_nm>          (absolute max-mesh-step in nm,
                                                     applied to global+box dy=dz)
  shutoff_conv_shutoff   <auto_shutoff_min>         (FDTD field-energy threshold,
                                                     formatted like 1e-06)
  spec                   <task_index>               (line content unused;
                                                     remote imports SWEEP_SPEC_MODULE
                                                     and picks expand()[idx])

Sources of value lists:
  mesh_conv_a / b         - convergence_testing/run_mesh_convergence.py module constants
  shutoff_conv_shutoff    - convergence_testing/run_auto_shutoff_convergence.py SHUTOFF_VALUES
  spec                    - any module that defines a top-level SPEC: SweepSpec.
                            Your per-study file: runners/sweeps/<name>.py

Usage (called by deploy_athena.sh; can also run standalone):
  python athena/scripts/build_sweep_list.py --kind spec --module runners.sweeps.innermost_shift --output sweep_list.txt

The script also prints a SWEEP_META: line with metadata that deploy_athena.sh
captures and forwards as sbatch --export env vars (SWEEP_FIXED_CELLS for
mesh_conv_b, SWEEP_SPEC_MODULE for spec).
"""

import argparse
import importlib
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_DIR)


def _values_mesh_conv_x():
    from convergence_testing.run_mesh_convergence import PHASE_X_VALUES
    return [f"{v}" for v in PHASE_X_VALUES]


def _values_mesh_conv_yz():
    from convergence_testing.run_mesh_convergence import (
        PHASE_YZ_VALUES, DEFAULT_CELLS, CHECKPOINT,
    )
    # YZ-phase fixes cells_per_half_period at the X-phase converged value.
    # Prefer the checkpoint's recorded "best_cells_phase_x" if it exists, else
    # fall back to DEFAULT_CELLS and warn — usually means the user submitted
    # the YZ array before the X-phase aggregator job completed on Athena.
    import json
    fixed_cells = DEFAULT_CELLS
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            data = json.load(f)
        if "best_cells_phase_x" in data:
            fixed_cells = int(data["best_cells_phase_x"])
        else:
            print(f"WARNING: checkpoint {CHECKPOINT} has no best_cells_phase_x; "
                  f"falling back to DEFAULT_CELLS={DEFAULT_CELLS}. "
                  f"Did you `--results` after the X-phase aggregator finished?",
                  file=sys.stderr)
    else:
        print(f"WARNING: no checkpoint at {CHECKPOINT}; "
              f"falling back to DEFAULT_CELLS={DEFAULT_CELLS}.",
              file=sys.stderr)
    return [f"{v}" for v in PHASE_YZ_VALUES], fixed_cells


def _values_shutoff_conv():
    from convergence_testing.run_auto_shutoff_convergence import SHUTOFF_VALUES
    # Format as scientific so athena_run_one can parse with float() unambiguously.
    return [f"{v:.0e}" for v in SHUTOFF_VALUES]


def _values_spec(module_name: str, prelim: bool = False):
    """Import a study module exposing SPEC (or PRELIM_SPEC); return one line per config."""
    module = importlib.import_module(module_name)
    attr = "PRELIM_SPEC" if prelim else "SPEC"
    if not hasattr(module, attr):
        raise RuntimeError(f"Module {module_name!r} has no top-level {attr} attribute")
    spec = getattr(module, attr)
    base = getattr(module, "PRELIM_BASE" if prelim else "BASE", None)
    configs = spec.expand(base=base)
    return [f"{i}" for i in range(len(configs))]


def _values_inverse_design(module_name: str):
    """Import an inverse-design study (exposes SPEC: InverseDesignSpec); one line per multi-start."""
    module = importlib.import_module(module_name)
    if not hasattr(module, "SPEC"):
        raise RuntimeError(f"Module {module_name!r} has no top-level SPEC attribute")
    spec = module.SPEC
    starts = spec.get_starts()
    return [f"{i}" for i in range(len(starts))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True,
                        choices=["mesh_conv_x", "mesh_conv_yz",
                                 "shutoff_conv_shutoff", "spec", "inverse_design"])
    parser.add_argument("--output", required=True,
                        help="Path to sweep_list.txt (one line per array task)")
    parser.add_argument("--module", default=None,
                        help="(kind=spec only) dotted module path of a study file "
                             "exposing top-level SPEC: SweepSpec")
    parser.add_argument("--prelim", action="store_true",
                        help="(kind=spec only) build list from PRELIM_SPEC instead of SPEC")
    args = parser.parse_args()

    meta = {"kind": args.kind}

    if args.kind == "mesh_conv_x":
        lines = _values_mesh_conv_x()
    elif args.kind == "mesh_conv_yz":
        lines, fixed_cells = _values_mesh_conv_yz()
        meta["fixed_cells"] = str(fixed_cells)
    elif args.kind == "shutoff_conv_shutoff":
        lines = _values_shutoff_conv()
    elif args.kind == "inverse_design":
        if not args.module:
            print("ERROR: --module is required for --kind inverse_design")
            sys.exit(1)
        lines = _values_inverse_design(args.module)
        meta["spec_module"] = args.module
    elif args.kind == "spec":
        if not args.module:
            print("ERROR: --module is required for --kind spec")
            sys.exit(1)
        lines = _values_spec(args.module, prelim=args.prelim)
        meta["spec_module"] = args.module
        # Optional spec-module hooks for chained workflows. deploy_athena.sh
        # picks these up: PRELIM_RUN_SCRIPT triggers a prerequisite single-sim
        # job with that RUN_SCRIPT, and LOCKED_LAMBDA_FILE is forwarded as an
        # env var to both jobs (the prelim writes it; the array reads it).
        # Modules without these constants keep current behavior (no chaining).
        _study_module = importlib.import_module(args.module)
        if hasattr(_study_module, "PRELIM_RUN_SCRIPT"):
            meta["prelim_run_script"] = str(_study_module.PRELIM_RUN_SCRIPT)
        if hasattr(_study_module, "PRELIM_SPEC"):
            # Prelim spec lives in the same module. Emit the prelim task count
            # so deploy_athena.sh can build a separate sweep_list for it using
            # --prelim on a second call to this script.
            prelim_base = getattr(_study_module, "PRELIM_BASE", None)
            prelim_n = len(_study_module.PRELIM_SPEC.expand(base=prelim_base))
            meta["has_prelim_spec"] = "true"
            meta["prelim_n_tasks"]  = str(prelim_n)
        if hasattr(_study_module, "LOCKED_LAMBDA_FILE"):
            meta["locked_lambda_file"] = str(_study_module.LOCKED_LAMBDA_FILE)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} task lines to: {args.output}")
    # deploy_athena.sh greps for this prefix to extract metadata for sbatch --export
    for k, v in meta.items():
        print(f"SWEEP_META: {k}={v}")


if __name__ == "__main__":
    main()
