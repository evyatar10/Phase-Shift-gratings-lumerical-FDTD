"""TM identity gate: the engine's session-free outputs must not change.

Study dir: runners/lumopt2_design/gates/ | Created 2026-10-04 | zero GPU, no Lumerical.
Purpose: snapshot every spec-dependent, session-free output of lumopt2_design.py
for the live TM specs, so the TE device-profile refactor can prove it left every
TM spec bit-identical (floats compared by repr, i.e. exactly).

Usage (from the repo root, PYTHONIOENCODING=utf-8):
  python runners/lumopt2_design/gates/gate_tm_identity.py --write   # new reference
  python runners/lumopt2_design/gates/gate_tm_identity.py           # compare, exit 1 on any diff

Covered: seed_params, param_bounds, replay_params, tooth_names, scatterer/comb
counts, build_base_cfg (every attribute, nested), make_func at the seed and at a
perturbed grating, two_kappa_L, kappa/elong penalties + grads, the penalty that
attach_penalty wires for each spec (fake project), sigma_hat_of,
boundary_weights, _line_from_res on a synthetic field_profile result, softW +
its gradient, the softW boxcar width nb and smoothing matrix, the import-source
injection guard, recenter_nm.
NOT covered (need a live FDTD/lumopt2 session): build_base_fsp, make_project's
region/containment asserts, the log callback rows, run_validate_gradient /
run_adjoint_only detune vectors (checked separately when that code changes).
Calls into functions whose signature gains a TE argument pass the spec's value
only when the parameter exists, so the same file runs on old and new engine.
"""
import dataclasses
import hashlib
import inspect
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np

from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_v2_proj import SPEC as C1
from runners.lumopt2_design.campaign_v2_proj_d1 import SPEC as D1
from runners.lumopt2_design.campaign_v2_proj_d1u import SPEC as D1U
from runners.lumopt2_design.campaign_v2_uniform import SPEC as UNI

SNAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots", "tm_identity.json")
SPECS = {
    "C1": C1, "D1": D1, "D1U": D1U, "UNI": UNI,
    "smoke50": dataclasses.replace(D1, n_periods_side=60, two_kl_floor=0.0, fwhm0_um=None),
    "C1_bare": dataclasses.replace(C1, bare=True),
    "C1_freecomb": dataclasses.replace(C1, free_comb=True),
}
N = 25            # every spec above is the 25-free-tooth TM layout


def r(v):
    """JSON-safe exact record: floats by repr, arrays/tuples elementwise."""
    if isinstance(v, dict):
        return {str(k): r(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [r(x) for x in v]
    if isinstance(v, np.ndarray):
        return [r(x) for x in v.tolist()]
    while hasattr(v, "_value"):          # autograd box
        v = v._value
    if isinstance(v, (bool, str)) or v is None:
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return repr(v)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes()).hexdigest()


def call(fn, *args, **maybe):
    """Pass the keyword args the function actually accepts (old vs new engine)."""
    params = inspect.signature(fn).parameters
    return fn(*args, **{k: v for k, v in maybe.items() if k in params})


def flatten(obj, prefix=""):
    """Every attribute reachable through nested dataclasses (incl. unknown
    attributes set on them — the config-override trap); callables skipped."""
    out = {}
    for k, v in sorted(vars(obj).items()):
        key = f"{prefix}{k}"
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            out.update(flatten(v, key + "."))
        elif not callable(v):
            out[key] = r(v)
    return out


def captured(fn):
    try:
        return r(fn())
    except Exception as e:
        return f"RAISES {type(e).__name__}"


class FakeProject:
    """Just enough for attach_penalty: raw FOM 0 and raw gradient 0, so the
    wrapped handles return exactly -pen(p) and -pen_grad(p)."""
    def __init__(self, p0):
        self.parametrization = type("P", (), {"get_initial_params": lambda s: p0})()
        self.compute_fom = lambda params=None: 0.0
        self.compute_gradient = lambda params=None: np.zeros(len(p0))


def synthetic_res(n_side, pitch_nm):
    """field_profile-shaped result dict (x, y, z=1, lambda, comp) with a
    standing-wave envelope, extending past the grating crop."""
    rng = np.random.default_rng(1)
    x = np.linspace(-1.2, 1.2, 961) * n_side * pitch_nm * 1e-9
    y = np.linspace(-0.6e-6, 0.6e-6, 13)
    lam = np.array([1563.0, 1564.21, 1565.0]) * 1e-9
    env = np.exp(-(x / 9e-6) ** 2) * np.cos(2 * np.pi * x / (pitch_nm * 1e-9)) ** 2 + 0.01
    E = (rng.normal(size=(len(x), len(y), 1, len(lam), 3))
         + 1j * rng.normal(size=(len(x), len(y), 1, len(lam), 3)))
    E = E * np.sqrt(env)[:, None, None, None, None]
    return {"x": x, "y": y, "lambda": lam, "E": E}


def record(spec):
    rec = {}
    pitch = getattr(spec, "pitch_nm", eng.PITCH_NM)
    pol = getattr(spec, "polarization", "TM")
    seed = eng.seed_params(spec)
    rec["seed_params"] = r(seed)
    rec["param_bounds"] = r(eng.param_bounds(spec))
    rec["replay_seed"] = captured(lambda: eng.replay_params(spec, seed))
    rec["tooth_names"] = r(call(eng.tooth_names, spec.n_periods_side,
                                n_free=getattr(spec, "n_free", N)))
    rec["comb_count"] = eng.comb_count(spec)
    rec["recenter_nm"] = r(getattr(spec, "recenter_nm", eng.RECENTER_NM))
    rec["cfg"] = flatten(eng.build_base_cfg(spec))

    # perturbed grating: corr/avg/shift + cavity width; comb slots untouched
    p_pert = seed.copy()
    dv = np.random.default_rng(0).normal(0.0, 1.0, size=3 * N + 1)
    p_pert[:3 * N] += dv[:3 * N]
    p_pert[eng.I_CAV] += dv[-1]
    func = eng.make_func(spec)
    rec["func_seed"] = r(func(seed))
    rec["func_pert"] = r(func(p_pert))

    p_rho = seed.copy()
    p_rho[:N] *= 1.05                       # rho 1.05 (out of the +2% band)
    p_rho[2 * N:3 * N] = 10.0               # elong 500 nm (past the 120 nm deadband)
    for name, p in (("seed", seed), ("pert", p_pert), ("rho", p_rho)):
        if "spec" in inspect.signature(eng.two_kappa_L).parameters:
            rec[f"two_kL_{name}"] = r(eng.two_kappa_L(p, spec))
        else:
            rec[f"two_kL_{name}"] = r(eng.two_kappa_L(p, spec.n_periods_side))
        rec[f"kappa_pen_{name}"] = r(eng.kappa_penalty(p))
        rec[f"elong_pen_{name}"] = r(eng.elong_penalty(p))
        rec[f"sigma_hat_{name}"] = r(eng.sigma_hat_of(spec, p))
    rec["kappa_grad_rho"] = r(eng._kappa_penalty_grad(p_rho))
    rec["elong_grad_rho"] = r(eng._elong_penalty_grad(p_rho))

    def attached():
        proj = FakeProject(seed)
        eng.attach_penalty(proj, dataclasses.replace(spec))
        return {"fom_rho": proj.compute_fom(p_rho),
                "grad_rho": proj.compute_gradient(p_rho),
                "fom_pert": proj.compute_fom(p_pert)}
    rec["attach_penalty"] = captured(attached)
    rec["boundary_weights"] = r(eng.boundary_weights(spec))

    # profile line + softW on a synthetic result (no session)
    res = synthetic_res(spec.n_periods_side, pitch)
    x, I, aux = call(eng._line_from_res, res, 1564.21, spec.n_periods_side, pitch_nm=pitch)
    rec["line"] = {"x": sha(x), "I": sha(I), "keep": sha(aux["keep"]), "wy": sha(aux["wy"]),
                   "i_lam": int(aux["i_lam"]), "n": int(len(x))}
    fringe = pitch / 2000.0 if hasattr(spec, "pitch_nm") else eng.WG_FRINGE_UM
    sw, gw = call(eng.softw_and_weight, x, I, fringe_um=fringe)
    rec["softw"] = {"value": r(sw), "grad": sha(gw)}
    for dx_nm in sorted({50.0, spec.region_dx_nm, eng.DX_PITCHLOCK_NM}):
        dx_um = dx_nm / 1000.0
        xs = np.arange(-300, 301) * dx_um
        Is = np.exp(-(xs / 9.0) ** 2) * np.cos(2 * np.pi * xs / (pitch / 1000.0)) ** 2 + 0.01
        rec[f"wsmooth_dx{dx_nm!r}"] = {
            "nb": max(1, int(round(fringe / dx_um))),
            "matrix": sha(call(eng._wsmooth_matrix, len(xs), dx_um, fringe_um=fringe)),
            "softw": r(call(eng.soft_width_of_line, xs, Is, fringe_um=fringe))}

    # import-source injection guard on synthetic planes (x, y, comp)
    tan_plane = np.zeros((40, 7, 3)); tan_plane[..., 0] = 1.0; tan_plane[..., 2] = 0.5
    ez_plane = np.zeros((40, 7, 3)); ez_plane[..., 2] = 1.0
    rec["src_guard_tan"] = captured(lambda: call(eng.check_import_src_injects, tan_plane,
                                                 polarization=pol))
    rec["src_guard_ez"] = captured(lambda: call(eng.check_import_src_injects, ez_plane,
                                                polarization=pol))
    return rec


def module_constants():
    names = ["PITCH_NM", "CORR_NM", "AVG_W_NM", "KAPPA_PER_UM", "N_FREE", "SL_CORR",
             "SL_AVG", "SL_SHIFT", "SL_R", "SL_X", "I_DCOMB", "I_CAV", "N_PARAMS",
             "DX_PITCHLOCK_NM", "RECENTER_NM", "WG_FRINGE_UM", "CELLS_PER_PITCH"]
    return {n: r(getattr(eng, n)) for n in names}


def main():
    snap = {"_module": module_constants()}
    snap.update({name: record(spec) for name, spec in SPECS.items()})
    snap = json.loads(json.dumps(snap))     # normalise tuples -> lists
    if "--write" in sys.argv:
        os.makedirs(os.path.dirname(SNAP), exist_ok=True)
        with open(SNAP, "w") as f:
            json.dump(snap, f, indent=0, sort_keys=True)
        print(f"wrote {SNAP} ({len(SPECS)} specs)")
        return 0
    with open(SNAP) as f:
        ref = json.load(f)
    diffs = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a or k not in b:
                    # A NEW builder-config key whose default is inert (False / None /
                    # 0 / empty) is another study adding an opt-in knob to the shared
                    # SimulationConfig (2026-10-05: farfield.save_surface_eh from the
                    # far-field study tripped this gate). It cannot change a TM scene,
                    # so report it without failing; any other new/missing key fails.
                    if (k in b and "/cfg" in path
                            and b[k] in (False, None, 0, 0.0, "", [], "False", "None", "0", "0.0", "[]")):
                        print(f"INFO new inert config key {path}/{k} = {b[k]!r} (not a difference)")
                        continue
                    diffs.append(f"{path}/{k}: {'missing now' if k not in b else 'new key'}")
                else:
                    walk(a[k], b[k], f"{path}/{k}")
        elif a != b:
            diffs.append(f"{path}: ref {str(a)[:120]} | now {str(b)[:120]}")
    walk(ref, snap, "")
    for d in diffs:
        print("DIFF", d)
    if diffs:
        print(f"TM IDENTITY: {len(diffs)} DIFFERENCES — refactor changed TM behaviour")
        return 1
    print(f"TM IDENTITY: ALL IDENTICAL ({len(SPECS)} specs + module constants)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
