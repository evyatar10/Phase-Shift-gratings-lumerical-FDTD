"""
InverseDesignSpec — the on-disk description of an inverse-design study.

Mirrors `runners.sweeps.sweep_spec.SweepSpec` in spirit: one Python file per
study under `runners/inverse_design/<study_name>.py` declares an
`InverseDesignSpec`, the deploy scripts pick it up and submit a SLURM array
job where each task index runs one independent lumopt driver from its own
starting point.

Parameter vector layout (mirror-symmetric, applied identically to both sides):
    p = [dw_d1, dw_d2, ..., dw_dN,        # full corrugation depth (W_wide - W_narrow), nm
         shift_d1, shift_d2, ..., shift_dN, # gap shift, nm
         cavity_width]                      # cavity-segment width, nm

Length = 2 * n_free_inner_teeth + 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class InverseDesignSpec:
    """Configuration for one inverse-design study."""

    # ── Free parameters and bounds ──────────────────────────────────────────
    n_free_inner_teeth: int = 2
    # Per-tooth bounds on full corrugation depth (DW = W_wide - W_narrow), in nm.
    # Length must equal n_free_inner_teeth.
    dw_bounds_nm: List[Tuple[float, float]] = field(
        default_factory=lambda: [(60.0, 240.0), (60.0, 240.0)]
    )
    # Per-tooth bounds on gap shift, in nm. Length must equal n_free_inner_teeth.
    shift_bounds_nm: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 200.0), (0.0, 200.0)]
    )
    # Cavity width bounds in nm.
    cavity_width_bounds_nm: Tuple[float, float] = (600.0, 1100.0)

    # ── Initial points (one per multi-start). Each is a parameter vector
    # of length 2*n_free_inner_teeth + 1 in the order documented above.
    # If None, n_starts random Latin-hypercube samples are drawn from bounds.
    initial_points: Optional[List[List[float]]] = None
    n_starts: int = 4
    seed: int = 0

    # ── Optimizer ───────────────────────────────────────────────────────────
    max_iter: int = 30
    optimizer_method: str = "L-BFGS-B"   # passed to scipy.optimize.minimize via lumopt
    optimizer_pgtol: float = 1e-6
    optimizer_ftol: float = 1e-6
    use_concurrent_adjoint_solves: bool = True   # Forward + adjoint run simultaneously

    # ── Geometry constraints ────────────────────────────────────────────────
    enforce_mirror_symmetry: bool = True   # always True at bragg_device level for now
    lengthen_cavity: bool = True            # cavity absorbs total shift to keep x_grating_end constant

    # ── Study metadata ──────────────────────────────────────────────────────
    label: str = ""

    # ─────────────────────────────────────────────────────────────────────────

    def n_params(self) -> int:
        return 2 * self.n_free_inner_teeth + 1

    def all_bounds_nm(self) -> List[Tuple[float, float]]:
        """Flat list of bounds in parameter-vector order: [dw..., shift..., cavity]."""
        return list(self.dw_bounds_nm) + list(self.shift_bounds_nm) + [self.cavity_width_bounds_nm]

    def validate(self) -> None:
        n = self.n_free_inner_teeth
        if n < 1:
            raise ValueError(f"n_free_inner_teeth must be >= 1, got {n}.")
        if len(self.dw_bounds_nm) != n:
            raise ValueError(
                f"dw_bounds_nm length ({len(self.dw_bounds_nm)}) "
                f"must equal n_free_inner_teeth ({n})."
            )
        if len(self.shift_bounds_nm) != n:
            raise ValueError(
                f"shift_bounds_nm length ({len(self.shift_bounds_nm)}) "
                f"must equal n_free_inner_teeth ({n})."
            )
        if self.initial_points is not None:
            for i, p in enumerate(self.initial_points):
                if len(p) != self.n_params():
                    raise ValueError(
                        f"initial_points[{i}] has length {len(p)}, "
                        f"expected {self.n_params()} (= 2*N + 1 with N={n})."
                    )

    def latin_hypercube_starts(self) -> List[List[float]]:
        """Generate n_starts Latin-hypercube samples in the bound box."""
        import numpy as np
        rng = np.random.default_rng(self.seed)
        bounds = self.all_bounds_nm()
        d = len(bounds)
        # LHS: divide each axis into n_starts bins, randomize a permutation per axis.
        u = (rng.random((self.n_starts, d)) + np.array([rng.permutation(self.n_starts) for _ in range(d)]).T) / self.n_starts
        starts = []
        for i in range(self.n_starts):
            p = [bounds[j][0] + u[i, j] * (bounds[j][1] - bounds[j][0]) for j in range(d)]
            starts.append(p)
        return starts

    def get_starts(self) -> List[List[float]]:
        """Return the list of starting points (explicit or generated)."""
        self.validate()
        if self.initial_points is not None:
            return [list(p) for p in self.initial_points]
        return self.latin_hypercube_starts()


def regular_grating_start(
    cfg, n_free_inner_teeth: int, cavity_width_nm: float = 800.0,
) -> List[float]:
    """Build the parameter vector corresponding to the unmodified regular grating.

    All N freed teeth use the same full corrugation depth as the rest of the
    grating (cfg.geometry.corrugation_depth_m), zero inner-tooth shift, and
    the supplied cavity width (default 800 nm = avg_corrugation_width_m).

    This is the recommended baseline starting point: peak T at the optimum
    must exceed peak T at this start, otherwise the optimizer hasn't found
    an improvement.
    """
    full_depth_nm = cfg.geometry.corrugation_depth_m * 1e9
    return (
        [float(full_depth_nm)] * n_free_inner_teeth      # dw_d for d=1..N
        + [0.0] * n_free_inner_teeth                      # shift_d for d=1..N
        + [float(cavity_width_nm)]                        # cavity_width_nm
    )

    def describe(self) -> str:
        lines = [f"InverseDesignSpec(label={self.label!r})"]
        lines.append(f"  n_free_inner_teeth = {self.n_free_inner_teeth}")
        lines.append(f"  parameters         = {self.n_params()}")
        lines.append(f"  dw_bounds_nm       = {self.dw_bounds_nm}")
        lines.append(f"  shift_bounds_nm    = {self.shift_bounds_nm}")
        lines.append(f"  cavity_bounds_nm   = {self.cavity_width_bounds_nm}")
        lines.append(f"  n_starts           = {self.n_starts}")
        lines.append(f"  max_iter           = {self.max_iter}")
        lines.append(f"  optimizer          = {self.optimizer_method}")
        return "\n".join(lines)


def params_to_kwargs(p, n_free_inner_teeth: int) -> dict:
    """
    Decompose a flat parameter vector into bragg_device kwargs.

    p layout: [dw_1, ..., dw_N, shift_1, ..., shift_N, cavity_width]
    Returns: {"inner_dw_nm": [...], "inner_shift_nm": [...], "cavity_width_m": ...}
    """
    n = n_free_inner_teeth
    expected = 2 * n + 1
    if len(p) != expected:
        raise ValueError(
            f"params_to_kwargs: expected vector of length {expected} "
            f"(2*N+1 with N={n}), got {len(p)}."
        )
    inner_dw_nm = list(p[0:n])
    inner_shift_nm = list(p[n:2*n])
    cavity_width_nm = float(p[2*n])
    return {
        "inner_dw_nm": inner_dw_nm,
        "inner_shift_nm": inner_shift_nm,
        "cavity_width_m": cavity_width_nm * 1e-9,
    }
