"""PRE-DISPATCH BOUNDS CHECK — run this before EVERY validate_c325 dispatch.

Three tasks died in seconds tonight on the same class of bug: a seed or a
detune point that lies outside its own spec's bounds, which lumopt2 rejects
outright (`_check_params`, parametrization.py:674). Each cost a dispatch cycle
and, twice, a queue slot behind a slow forward.

This reproduces EXACTLY what the runner does to the parameter vector — the
seed, the run_adjoint_only detune perturbation, the BEST_T9636 seeding — and
checks it against `param_bounds(spec)` locally, with zero GPU.

Usage:  python predispatch_check.py
"""
import os
import sys

sys.path.insert(0, r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes")

import dataclasses
import numpy as np

from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_v2_proj import SPEC as PSPEC
from runners.lumopt2_design.best_designs import BEST_T9636

VAL = eng.CampaignSpec(label="lumopt2_val_c325")


def detune1(p):
    """The exact perturbation run_adjoint_only / run_validate_gradient apply
    at detune=1 (lumopt2_design.py) — including the COMB detune that the
    frozen-comb specs reject."""
    p = np.asarray(p, dtype=float).copy()
    p[eng.SL_SHIFT] = 20.0
    p[eng.SL_R.start + eng.COMB_N_HALF] = 100.0
    p[eng.SL_X.start + eng.COMB_N_HALF] += 50.0
    p[eng.I_DCOMB] = 1750.0
    return p


def check(name, spec, p, clamp=False):
    b = np.asarray(eng.param_bounds(spec), dtype=float)
    if clamp:
        p = np.clip(p, b[:, 0], b[:, 1])
    bad = np.where((p < b[:, 0] - 1e-9) | (p > b[:, 1] + 1e-9))[0]
    ok = len(bad) == 0
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: {len(bad)} out of bounds"
          + ("" if ok else
             f"  first: idx {bad[0]} val {p[bad[0]]:.4f} "
             f"bounds [{b[bad[0],0]:.3f}, {b[bad[0],1]:.3f}]"))
    return ok


print("PRE-DISPATCH BOUNDS CHECK\n")
allok = True

# tasks 42/43 — C re-fit at production numerics, detune=1 (comb IS detuned)
for quad in (False, True):
    spec = dataclasses.replace(
        PSPEC, n_wl_points=151, wg_pure=True, wg_project=False, free_comb=True,
        label="cfit", adj_fix_field_re=(0.0 if quad else 1.0),
        adj_fix_field_im=(1.0 if quad else 0.0))
    allok &= check(f"task {43 if quad else 42} (C re-fit, detune1)",
                   spec, detune1(eng.seed_params(spec)))

# task 38 — P2, uniform seed, no detune
s38 = dataclasses.replace(PSPEC, n_wl_points=251, scan_width_nm=5.0,
                          scan_center_nm=1564.614, label="p2")
allok &= check("task 38 (P2 uniform seed)", s38, eng.seed_params(s38), clamp=True)

# task 39 — P3, BEST_T9636 with inner-8 lowered, clamped
s39 = dataclasses.replace(PSPEC, n_wl_points=251, scan_width_nm=5.0,
                          scan_center_nm=1566.409, label="p3")
p39 = np.asarray(BEST_T9636, dtype=float).copy()
i0 = eng.SL_CORR.start
p39[i0:i0 + 8] -= 30.0
allok &= check("task 39 (P3 see-saw seed)", s39, p39, clamp=True)

# task 41 / the campaign — uniform seed under the projection spec
allok &= check("task 41 + campaign (uniform seed)", PSPEC,
               eng.seed_params(PSPEC))

print("\n" + ("ALL SEEDS IN BOUNDS — safe to dispatch"
               if allok else "*** DO NOT DISPATCH — fix the bounds first ***"))
sys.exit(0 if allok else 1)
