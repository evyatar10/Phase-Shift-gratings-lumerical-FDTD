"""
Multi-parameter sweep specification — project-level abstraction.

Specify only the parameters that vary between simulations as lists of values.
Everything else uses SimulationConfig defaults (or your provided base config).
The same SweepSpec runs locally (sequential), on Zeus (single PBS job), or on
Athena (SLURM job array, parallel) — only the runner backend changes.

Per-study usage pattern:

    # runners/sweeps/apod_and_shift.py
    from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec

    SPEC = SweepSpec(
        n_apod_periods_each_side  = [0, 3, 5],
        innermost_tooth_shift_nm  = [0, 50, 100, 150],
        cavity_neg_detuning_nm    = [5.76],
        label = "apod_and_shift",
    )

    if __name__ == "__main__":
        run_sweep_spec(SPEC, target="local")     # or "athena", "zeus"

The set of swept fields is the same one ExperimentCard exposes. Field-name →
config-path mapping comes from experiment_card._CARD_FIELD_MAP (single source
of truth) — to add a new sweepable parameter, add it there once.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field, fields
from typing import List, Literal, Optional

from experiment_card import _CARD_FIELD_MAP
from simulation_config import SimulationConfig, set_nested_attr


# ═══════════════════════════════════════════════════════════════════════════════
# Spec
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SweepSpec:
    """
    Multi-parameter sweep over any subset of the fields in _CARD_FIELD_MAP.

    Each field is a list of values to sweep over (or None to leave at default).
    The cartesian product of all populated fields is run, unless mode='zipped'
    (in which case all populated fields must have the same length and are
    iterated in lockstep).
    """

    # ── Geometry / grating ───────────────────────────────────────────────────
    n_periods_each_side:        Optional[List[int]]   = None
    n_apod_periods_each_side:   Optional[List[int]]   = None
    center_mod_depth_nm:        Optional[List[float]] = None
    corrugation_depth_nm:       Optional[List[float]] = None
    pitch_nm:                   Optional[List[float]] = None

    # ── Cavity / phase-shift ────────────────────────────────────────────────
    innermost_tooth_shift_nm:   Optional[List[float]] = None
    cavity_neg_detuning_nm:     Optional[List[float]] = None
    cavity_length_nm:           Optional[List[float]] = None
    cavity_width_nm:            Optional[List[float]] = None
    lengthen_cavity:            Optional[List[bool]]  = None

    # ── Apodization ─────────────────────────────────────────────────────────
    apod_method:                Optional[List[str]]   = None     # 'none' | 'linear' | 'tanh'
    tanh_steepness:             Optional[List[float]] = None

    # ── Spectral / monitors ─────────────────────────────────────────────────
    center_wavelength_nm:       Optional[List[float]] = None
    scan_width_nm:              Optional[List[float]] = None
    farfield:                   Optional[List[bool]]  = None

    # ── Source / polarization ───────────────────────────────────────────────
    polarization:               Optional[List[str]]   = None     # ["TE"], ["TM"], or ["TE","TM"]

    # ── Two side-by-side coupled devices (radiative-coupling study) ───────────
    n_devices:                  Optional[List[int]]   = None     # [1] | [2]
    device_gap_nm:              Optional[List[float]] = None     # lateral edge-to-edge gap (nm)
    device_stagger_nm:          Optional[List[float]] = None     # longitudinal Δx offset of device 2 (nm)
    corrugation_depth_2_nm:     Optional[List[float]] = None     # device-2 corrugation depth (nm)

    # ── Behavior ────────────────────────────────────────────────────────────
    mode: Literal["cartesian", "zipped"] = "cartesian"
    label: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    def _populated(self) -> List[tuple]:
        """List of (field_name, values) for fields the user actually set."""
        out = []
        for f in fields(self):
            if f.name in ("mode", "label"):
                continue
            v = getattr(self, f.name)
            if v is None:
                continue
            if not isinstance(v, (list, tuple)) or len(v) == 0:
                raise ValueError(
                    f"SweepSpec field {f.name!r} must be a non-empty list, got {v!r}"
                )
            if f.name not in _CARD_FIELD_MAP:
                raise ValueError(
                    f"SweepSpec field {f.name!r} is not in _CARD_FIELD_MAP. "
                    f"Add it in experiment_card.py to make it sweepable."
                )
            out.append((f.name, list(v)))
        return out

    def expand(self, base: Optional[SimulationConfig] = None) -> List[SimulationConfig]:
        """Materialize all sweep combinations as a list of SimulationConfig."""
        populated = self._populated()
        if not populated:
            raise ValueError("SweepSpec is empty — at least one field must be set.")

        if self.mode == "zipped":
            lens = {len(vals) for _, vals in populated}
            if len(lens) != 1:
                raise ValueError(
                    f"mode='zipped' requires all fields to have the same length; got {lens}"
                )
            n = lens.pop()
            combos = [tuple(vals[i] for _, vals in populated) for i in range(n)]
        else:  # cartesian
            combos = list(itertools.product(*[vals for _, vals in populated]))

        names = [name for name, _ in populated]
        configs: List[SimulationConfig] = []
        for combo in combos:
            cfg = copy.deepcopy(base) if base is not None else SimulationConfig()
            for name, raw_value in zip(names, combo):
                dot_path, transform = _CARD_FIELD_MAP[name]
                value = transform(raw_value) if transform is not None else raw_value
                set_nested_attr(cfg, dot_path, value)
                # Mirror ExperimentCard semantics for apodization on/off.
                # apod_method=='none' disables; any other value enables.
                if name == "apod_method":
                    cfg.apodization.enabled = (raw_value != "none")
            # Final invariant: n_apod_periods_each_side=0 always disables
            # apodization, regardless of which order the fields were applied.
            if cfg.apodization.n_apod_periods_each_side == 0:
                cfg.apodization.enabled = False
            configs.append(cfg)
        return configs

    def describe(self) -> str:
        """Human-readable summary of the populated fields and total task count."""
        populated = self._populated()
        lines = [f"SweepSpec(label={self.label!r}, mode={self.mode!r})"]
        for name, vals in populated:
            lines.append(f"  {name:30s} = {vals}")
        n = (
            sum(1 for _ in itertools.product(*[v for _, v in populated]))
            if self.mode == "cartesian"
            else (len(populated[0][1]) if populated else 0)
        )
        lines.append(f"  -> {n} simulation(s)")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep_spec(
    spec: SweepSpec,
    target: str = "local",
    base: Optional[SimulationConfig] = None,
) -> Optional[List[dict]]:
    """
    Execute a SweepSpec on the chosen target.

    target = "local"   → sequential in the current Python process
    target = "zeus"    → ssh + qsub a single PBS job that loops sequentially
    target = "athena"  → ssh + sbatch a SLURM array (parallel; one task per config)
    """
    print(spec.describe())
    print()

    if target == "local":
        return _run_local(spec, base)
    elif target == "zeus":
        raise NotImplementedError(
            "target='zeus' is planned but not yet implemented. "
            "For now, run with target='local' on the Zeus head node, or use "
            "the existing Zeus deploy.sh path."
        )
    elif target == "athena":
        raise NotImplementedError(
            "target='athena' is invoked via the deploy script, not Python. "
            "Run: bash dgx/deploy_dgx.sh --option2  (DGX) or bash dgx/deploy_dgx.sh --option2  (Athena)"
        )
    else:
        raise ValueError(f"Unknown target {target!r}. Choose 'local'|'zeus'|'athena'.")


def _run_local(spec: SweepSpec, base: Optional[SimulationConfig]) -> List[dict]:
    """Sequential execution in the current Python process."""
    import gc
    import matplotlib.pyplot as plt
    from runners.single.run_simulation import run_single_sim

    configs = spec.expand(base)
    results: List[dict] = []
    n = len(configs)
    print("=" * 60)
    print(f"SWEEP_SPEC LOCAL — {n} simulation(s)  [label={spec.label!r}]")
    print("=" * 60)

    for i, cfg in enumerate(configs, 1):
        print(f"\n>>> RUN {i}/{n} <<<")
        try:
            r = run_single_sim(cfg)
            results.append(r)
        except Exception as e:
            print(f"ERROR in run {i}: {e}")
            raise
        finally:
            plt.close("all")
            gc.collect()

    print("\n" + "=" * 60)
    print(f"SWEEP_SPEC COMPLETE — {len(results)}/{n} succeeded.")
    print("=" * 60)
    return results
