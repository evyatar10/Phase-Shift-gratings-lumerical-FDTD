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

# tasks 50/51 + the d1 campaign — BEST_T9636 seed under the ns2 spec
# (param_bounds derives the frozen-comb slivers and the shift trust box from
# the seed, so this checks the exact vector the runner will submit)
from runners.lumopt2_design.campaign_v2_proj_d1 import SPEC as DSPEC
allok &= check("task 51 + d1 campaign (BEST seed, ns2 spec)", DSPEC,
               np.asarray(BEST_T9636, dtype=float))
s50 = dataclasses.replace(DSPEC, n_periods_side=60, two_kl_floor=0.0,
                          fwhm0_um=None, label="ns2smoke")
allok &= check("task 50 (ns2 smoke, N=60)", s50,
               np.asarray(BEST_T9636, dtype=float))

# d1u — uniform-lane continuation: c1's best in-band row (the row the copied
# log will resume from) must sit inside the bounds the runtime derives after
# recentring on it (run_campaign overwrites seed_override per attempt).
import json as _j
_c1_log = (r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes"
           r"\results_from_athena\lumopt2_v2_proj_c1"
           r"\lumopt2_v2_proj_c1_evals.jsonl")
try:
    from runners.lumopt2_design.campaign_v2_proj_d1u import SPEC as USPEC
    _rows = [_j.loads(l) for l in open(_c1_log, encoding="utf-8")]
    _fw0 = float(USPEC.fwhm0_um)
    _ok_rows = [r for r in _rows if r.get("fwhm_env_um")
                and 0.98 <= r["fwhm_env_um"] / _fw0 <= 1.02]
    _best = max(_ok_rows, key=lambda r: r["fom"])
    _p = np.asarray(_best["params"], dtype=float)
    _spec_rt = dataclasses.replace(USPEC, seed_override=tuple(_p))
    allok &= check(f"d1u resume row (c1 eval {_best['eval']}, fom "
                   f"{_best['fom']:.5f})", _spec_rt, _p)
except (OSError, ValueError) as e:
    print(f"  FAIL d1u resume row: {e}")
    allok = False

# task 49 — fast-then-rescale: s5 best, shifts scaled, replay under VAL spec
from runners.lumopt2_design.best_designs import UNIFORM_S5_FAST_BEST
_f = 1564.614 / 1565.8347
s49 = dataclasses.replace(VAL, label="lumopt2_val_c325_rescale_s5",
                          scan_width_nm=10.0, n_wl_points=501,
                          scan_center_nm=1564.614, free_comb=False)
p49 = np.asarray(UNIFORM_S5_FAST_BEST, dtype=float).copy()
p49[eng.SL_SHIFT] *= _f
try:
    p49 = eng.replay_params(s49, p49)      # asserts bounds itself
    allok &= check("task 49 (rescale_s5 replay)", s49, p49)
except AssertionError as e:
    print(f"  FAIL task 49 (rescale_s5 replay): {e}")
    allok = False

print("\n" + ("ALL SEEDS IN BOUNDS — safe to dispatch"
               if allok else "*** DO NOT DISPATCH — fix the bounds first ***"))
# ★INDEX-REACHABILITY (2026-08-28, after TWO collisions with _GFR_RUNGS'
# membership branch ate tasks 27 and 34): our named tasks must not be
# captured by any membership set and must appear exactly once as a literal.
import re as _re
from runners.lumopt2_design import validate_c325 as _v
_src = open(_v.__file__, encoding="utf-8").read()
_bad = False
for _idx, _name in ((41, "lam-chain toy"), (46, "control twin"),
                    (47, "pipeline smoke"), (48, "ride toy"),
                    (49, "rescale s5"), (50, "ns2 smoke"),
                    (51, "ns2 ride toy"), (52, "reuse smoke"),
                    (53, "gW angle probe")):
    _hits = len(_re.findall(rf"task_idx == {_idx}:", _src))
    _in_gfr = _idx in _v._GFR_RUNGS
    _ok = (_hits == 1) and not _in_gfr and _idx < _v.N_TASKS
    print(f"  {'OK  ' if _ok else 'FAIL'} task {_idx} ({_name}): "
          f"literal x{_hits}, in _GFR_RUNGS={_in_gfr}, < N_TASKS={_idx < _v.N_TASKS}")
    _bad |= not _ok
if _bad:
    print("*** INDEX GATE FAILED ***")
    allok = False
print("INDEXES REACHABLE")
sys.exit(0 if allok else 1)
