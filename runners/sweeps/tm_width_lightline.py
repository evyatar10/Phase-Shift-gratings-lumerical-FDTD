"""
Light-line-margin study — does widening the guide cut the TM radiation loss?

Physics (docs/loss_reduction_research_2026-07-03.md, option #2): the TM mode's
n_eff (~1.508) sits barely above n_clad (1.444), which is WHY the radiated lobe
hugs the axis and the loss is large. Widening the corrugated core raises n_eff
and pulls the mode away from the cladding light line; the leaky fraction should
drop roughly exponentially in the margin (Quan & Loncar 2011, design step iv).
Constraint tracked per row: the SPATIAL mode width fwhm_m must stay ~unchanged
(user's hard requirement) — width alone weakens the relative corrugation and
widens the mode, so each width is swept BOTH at fixed corrugation (400 nm) and
at proportionally scaled corrugation (corr/width = 0.5, the anchored ratio),
letting the analysis separate margin gain from kappa loss.

Grid (zipped), all on the anchored TM device (pitch 516.83, h 350, n 1.97/1.444):
  control 800/400 + width-only {900,1000,1100,1200}/400
                  + scaled     {900/450, 1000/500, 1100/550, 1200/600}  => 9 tasks
Window: WIDE (center 1590 nm / 200 nm / 7001 pts) — n_eff growth red-shifts the
resonance by up to ~70 nm across the sweep; every row shares the same window so
the control is at identical numerics. Converged box y=6.8 um / z-mult 5.42;
optimization mesh (survey — winners get an accurate-mesh pass).

Expected resonances: 1558.6 nm (control) drifting red toward ~1630 nm at 1200 nm
width. Judge: loss(row) at its own resonance vs control, at comparable fwhm_m.

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_width_lightline
Output -> results/tm_width_lightline/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BOX_Y_UM = 6.8       # converged y (tm_span_convergence, job 116854)
BOX_Z_MULT = 5.42    # converged z (round 2, job 116870): z = 8.8 um

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON (r=150 default)!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
# Wide shared window — the resonance red-shifts with width. 4001 pts (50 pm):
# 7001 pts at this box OOM'd (port monitors store fields x freq points; the
# 6.8x8.8 um cross-section doubles the per-point cost vs the old 4.8 um box —
# job 116974 task 2 died OUT_OF_MEMORY at 7001).
BASE.spectral.center_wavelength_m = 1.590e-6
BASE.spectral.scan_width_nm = 200.0
BASE.spectral.n_wl_points = 4001

_widths = [800.0, 900.0, 1000.0, 1100.0, 1200.0, 900.0, 1000.0, 1100.0, 1200.0]
_corrs  = [400.0, 400.0, 400.0,  400.0,  400.0,  450.0, 500.0,  550.0,  600.0]

assert len(_widths) == len(_corrs) == 9
assert len(set(zip(_widths, _corrs))) == 9, "duplicate (width, corr) row -> tag collision"

SPEC = SweepSpec(
    avg_width_nm         = _widths,
    corrugation_depth_nm = _corrs,
    mode  = "zipped",
    label = "tm_width_lightline",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
