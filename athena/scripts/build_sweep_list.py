"""
Build sweep_list.txt locally for an Athena --option3 SLURM array submission.

Each line in the output file becomes one SLURM array task. The line format
depends on SWEEP_KIND (interpreted on the cluster by athena_run_one.py):

  mesh_conv_a    <cells_per_half_period>
  mesh_conv_b    <dz_divisor>
  spec           <task_index>                         (line content unused;
                                                       remote imports SWEEP_SPEC_MODULE
                                                       and picks expand()[idx])

Sources of value lists:
  mesh_conv_a / b     - convergence_testing/run_mesh_convergence.py module constants
  spec                - any module that defines a top-level SPEC: SweepSpec.
                        Your per-study file: runners/sweeps/<name>.py

Usage (called by deploy_athena.sh; can also run standalone):
  python athena/scripts/build_sweep_list.py --kind spec --module runners.sweeps.innermost_shift --output sweep_list.txt

The script also prints a SWEEP_META: line with metadata that deploy_athena.sh
captures and forwards as sbatch --export env vars (SWEEP_FIXED_DZ for
mesh_conv_b, SWEEP_SPEC_MODULE for spec).
"""

import argparse
import importlib
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_DIR)


def _values_mesh_conv_a():
    from convergence_testing.run_mesh_convergence import PHASE_A_VALUES
    return [f"{v}" for v in PHASE_A_VALUES]


def _values_mesh_conv_b():
    from convergence_testing.run_mesh_convergence import (
        PHASE_B_VALUES, DEFAULT_CELLS, CHECKPOINT,
    )
    # Phase B fixes cells_per_half_period at the converged Phase A value.
    # Prefer the checkpoint's recorded "best_cells" if it exists, else fall back
    # to DEFAULT_CELLS (matches run_mesh_convergence.py's _run_phase fallback).
    import json
    fixed_cells = DEFAULT_CELLS
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            data = json.load(f)
        if "best_cells_phase_a" in data:
            fixed_cells = int(data["best_cells_phase_a"])
    return [f"{v}" for v in PHASE_B_VALUES], fixed_cells


def _values_spec(module_name: str):
    """Import a study module exposing SPEC: SweepSpec; return one line per config."""
    module = importlib.import_module(module_name)
    if not hasattr(module, "SPEC"):
        raise RuntimeError(f"Module {module_name!r} has no top-level SPEC attribute")
    spec = module.SPEC
    configs = spec.expand()
    # Line content is unused on the remote (athena_run_one.py imports the same
    # module and picks expand()[SLURM_ARRAY_TASK_ID]). Use the index for
    # human readability when inspecting sweep_list.txt.
    return [f"{i}" for i in range(len(configs))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True,
                        choices=["mesh_conv_a", "mesh_conv_b", "spec"])
    parser.add_argument("--output", required=True,
                        help="Path to sweep_list.txt (one line per array task)")
    parser.add_argument("--module", default=None,
                        help="(kind=spec only) dotted module path of a study file "
                             "exposing top-level SPEC: SweepSpec")
    args = parser.parse_args()

    meta = {"kind": args.kind}

    if args.kind == "mesh_conv_a":
        lines = _values_mesh_conv_a()
    elif args.kind == "mesh_conv_b":
        lines, fixed_cells = _values_mesh_conv_b()
        meta["fixed_cells"] = str(fixed_cells)
    elif args.kind == "spec":
        if not args.module:
            print("ERROR: --module is required for --kind spec")
            sys.exit(1)
        lines = _values_spec(args.module)
        meta["spec_module"] = args.module

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} task lines to: {args.output}")
    # deploy_athena.sh greps for this prefix to extract metadata for sbatch --export
    for k, v in meta.items():
        print(f"SWEEP_META: {k}={v}")


if __name__ == "__main__":
    main()
