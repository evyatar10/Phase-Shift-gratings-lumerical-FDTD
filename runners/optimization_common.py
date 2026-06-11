"""
Shared base SimulationConfig for the optimization runner families
(inverse_design, gradient_free_design, fd_gradient_design,
lumerical_native_optimization).

Every optimize_transmission.py / smoke_test.py across those families uses the
same base config and differs ONLY in device size (n_periods_each_side: 20 for
smoke tests, 80 for production). Before this module existed the 8-line block
was copy-pasted into all 8 spec files; consolidating it here keeps the
smoke/production configs from silently drifting apart.

Deploy-safety note (Athena/DGX menu contract):
  This module lives at runners/ root deliberately. The deploy scripts'
  study menus grep runners/<family>/*.py for the literal text "SPEC ="
  (and runners/single/*.py for a top-level `run` callable) — the runners/
  root directory is never scanned, so nothing here can leak into a menu.
  Conversely: never add a file containing top-level "SPEC =" inside a
  family directory unless it SHOULD appear as a selectable study.
  See runners/README.md for the full contract.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_config import SimulationConfig


def make_optimization_base(n_periods: int) -> SimulationConfig:
    """Base config shared by all optimization studies.

    n_periods: periods each side of the cavity — 20 for smoke tests
    (~10x faster FDTD), 80 for production runs.
    """
    base = SimulationConfig()
    base.grating.n_periods_each_side = n_periods
    base.grating.lengthen_cavity     = True             # NOT optimizing cavity length;
                                                        # absorbs Σ(shifts) so total
                                                        # device length stays constant.
    base.mesh.simulation_mode        = "optimization"   # device-wide dx=50 nm; specs
                                                        # add their own freed-region
                                                        # override via mesh_override_dxyz_nm.
    base.spectral.scan_width_nm      = 10.0             # full bandgap window; the FOM
                                                        # weight/window restricts further.
    # Optimization only needs T(λ) at Port_2; field-profile and far-field
    # monitors are pure overhead over the 60+ FDTD runs per study. Specs can
    # flip these back on for post-mortem field visualization.
    base.monitors.record_2d_fields = False   # XY top, YZ cross, XZ side profiles
    base.monitors.record_3d_fields = False   # full 3D field volume
    base.farfield.enabled          = False   # side + top far-field monitors
    return base
