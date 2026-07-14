"""Offline engine of the scatterer response-matrix (Green's) program — NO FDTD.

Study dir: runners/scatterers/   |   Created 2026-07-12   |   Job(s): consumes A/B/C/E
Purpose: everything after the simulations — gate checks, the linear inverse solve
that picks the scatterer COMBINATION, and the predicted-vs-measured validation.
numpy/scipy only (no lumapi, no license): runs locally or on the Athena login node.

Subcommands (run from the project root):
  gates      stage A (+B when present): numerics floor, jitter floor, superposition
             (Born) verdict -> MIN_SPACING_NM. The go/no-go before stage C.
  solve      stage C: builds b (control) and A (responses), prints the continuous
             LS ceiling FIRST (go/no-go), then the greedy binary combination and
             the best periodic comb; writes greens_report.json.
  solve2     stages C + C2 (2D grid): superposition gates FIRST (cross-row kill
             criterion), then LS + bounded-weight ceilings (2-row vs row-1-only),
             exhaustive 1/2/3-site over BOTH rows, greedy; greens2_report.json.
  validate   stage E: measured far-field reduction + dT/dloss per combination vs
             control, side by side with stage-D predictions.
  selftest   synthetic planted-optimum recovery (no data needed) — run after edits.

Objective throughout: total far-field POWER, side+top monitors concatenated with
direction-cosine solid-angle weights:  P = sum |w*E|^2,  w = sqrt(dux*duy/uz).
Minimizing ||b + A x||^2 = the radiated power of the device WITH the combination
(first Born). b must be in the objective: without it the optimum is trivially x=0.
"""

from __future__ import annotations   # keep signature hints inert on older login-node pythons

import argparse
import glob
import json
import os
import sys

import numpy as np
import scipy.io as sio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results_from_athena")
STAGE_DIR = {
    "a": "scat_a_baseline",
    "b": "scat_b_gates",
    "c": "scat_c_response",
    "c2": "scat_c2_row2",
    "e": "scat_e_validate",
}

# ── Gate thresholds ────────────────────────────────────────────────────────────
PEAK_T_FLOOR = 0.05          # dead parametric device shows T ~ 0.0008 (§2 low floor)
RES_SHIFT_FRAC_MAX = 0.10    # per-run |lambda_res - control| must be < 0.1*|FWHM|
PAIR_ERR_MAX = 0.05          # superposition-error gate. 5% = the published
                             # "independent scatterers" convention (dependent-vs-
                             # independent boundary defined at 5% error, JQSRT 246,
                             # 106924) and the one-shot-design verdict of the
                             # 2026-07-13 literature review: residual-power floor
                             # scales as e^2 (Vellekoop fidelity / Strehl / null-
                             # depth laws), so 5% keeps >=99.75% of the linear
                             # optimum; 1% buys nothing measurable and costs
                             # column strength. Iterate+validate designs may
                             # relax to 0.15-0.20.
UX2_MASK = 0.98              # hemisphere mask ux^2+uy^2 <= 0.98 (clip the horizon)
UZ_MIN = 0.02                # uz clip inside the solid-angle weight


# ═══════════════════════════════════════════════════════════════════════════════
# Loading + vectorization
# ═══════════════════════════════════════════════════════════════════════════════

def _stage_dir(stage: str, override: str | None) -> str:
    if override:
        d = override
    else:
        d = os.path.join(RESULTS_ROOT, STAGE_DIR[stage])
    for cand in (os.path.join(d, "results"), d):
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "result_*.mat")):
            return cand
    raise FileNotFoundError(
        f"no result_*.mat found under {d} (looked in ./ and ./results/) — "
        f"download the stage first (bash athena/deploy_athena.sh --results)")


def _monitor_vector(ff: dict) -> np.ndarray:
    """Solid-angle-weighted complex field vector of one far-field monitor."""
    for k in ("Ex_c", "Ey_c", "Ez_c", "ux", "uy"):
        if k not in ff:
            raise KeyError(f"far-field dict lacks {k!r} — was farfield.save_complex on?")
    ux = np.asarray(ff["ux"], dtype=float).ravel()
    uy = np.asarray(ff["uy"], dtype=float).ravel()
    UX, UY = np.meshgrid(ux, uy, indexing="ij")
    u2 = UX ** 2 + UY ** 2
    mask = u2 <= UX2_MASK
    uz = np.sqrt(np.clip(1.0 - u2, UZ_MIN ** 2, None))
    dux = float(np.mean(np.diff(ux)))
    duy = float(np.mean(np.diff(uy)))
    w = np.sqrt(dux * duy / uz)
    parts = [(w * np.asarray(ff[k]))[mask] for k in ("Ex_c", "Ey_c", "Ez_c")]
    return np.concatenate(parts).astype(complex)


def _check_e2_consistency(ff: dict) -> float:
    """Max relative |E_c|^2 vs stored E2 deviation on the strong part of the mask."""
    E2 = np.asarray(ff["E2"], dtype=float)
    Ec2 = sum(np.abs(np.asarray(ff[k])) ** 2 for k in ("Ex_c", "Ey_c", "Ez_c"))
    ux = np.asarray(ff["ux"], dtype=float).ravel()
    uy = np.asarray(ff["uy"], dtype=float).ravel()
    UX, UY = np.meshgrid(ux, uy, indexing="ij")
    sel = ((UX ** 2 + UY ** 2) <= UX2_MASK) & (E2 > 1e-6 * float(E2.max()))
    if not np.any(sel):
        return np.nan
    return float(np.max(np.abs(Ec2[sel] - E2[sel]) / E2[sel]))


def load_run(path: str, need_complex: bool = True) -> dict | None:
    """Load one result_*.mat into {r_nm, x_nm(list), y_nm, v, power, lam_res, ...}."""
    m = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
    run = {"file": path, "name": os.path.basename(path)}
    run["r_nm"] = float(m.get("scatterer_r_m", 0.0)) * 1e9
    n_sites = int(m.get("scatterer_n_sites", 0) or 0)
    if n_sites > 1 or "scatterer_x_list_m" in m:
        xl = np.atleast_1d(np.asarray(m["scatterer_x_list_m"], dtype=float))
        run["x_nm"] = [round(float(x) * 1e9, 1) for x in xl]
    else:
        run["x_nm"] = [round(float(m.get("scatterer_x_m", 0.0)) * 1e9, 1)]
    run["y_nm"] = round(float(m.get("scatterer_y_m", 0.0)) * 1e9, 1)
    # Per-site y (2D rows): stored scatterer_y_list_m when the run has an x list
    # (post_processing broadcasts the scalar when no explicit y list was given).
    if "scatterer_y_list_m" in m:
        yl = np.atleast_1d(np.asarray(m["scatterer_y_list_m"], dtype=float))
        run["y_site_nm"] = [round(float(y) * 1e9, 1) for y in yl]
    else:
        run["y_site_nm"] = [run["y_nm"]] * len(run["x_nm"])
    run["sites"] = list(zip(run["x_nm"], run["y_site_nm"]))
    run["lam_res_nm"] = float(m.get("resonance_wavelength_nm", np.nan))
    run["fwhm_nm"] = abs(float(m.get("spectral_fwhm_nm", np.nan)))
    run["peak_T"] = float(m.get("resonance_transmission", np.nan))
    wl = np.asarray(m.get("wl_nm", []), dtype=float).ravel()
    run["window"] = (float(wl.min()), float(wl.max())) if wl.size else (np.nan, np.nan)
    # Loss at the resonance index (full budget, not one far-field window).
    if wl.size and np.isfinite(run["lam_res_nm"]):
        i = int(np.argmin(np.abs(wl - run["lam_res_nm"])))
        for key in ("T", "R", "loss"):
            arr = np.asarray(m.get(key, []), dtype=float).ravel()
            run[f"{key}_res"] = float(arr[i]) if arr.size == wl.size else np.nan
    # §2 sanity gate.
    problems = []
    if not np.isfinite(run["lam_res_nm"]):
        problems.append("no resonance found")
    elif not (run["window"][0] <= run["lam_res_nm"] <= run["window"][1]):
        problems.append(f"resonance {run['lam_res_nm']:.2f} nm outside window {run['window']}")
    if not (run["peak_T"] > PEAK_T_FLOOR):
        problems.append(f"peak T {run['peak_T']:.4f} below floor {PEAK_T_FLOOR} (dead device?)")
    run["problems"] = problems
    # Complex far-field vector (side + top concatenated).
    run["v"] = None
    if need_complex:
        try:
            vs = [_monitor_vector(m[k]) for k in ("farfield_side", "farfield_top")
                  if isinstance(m.get(k), dict) and m[k]]
            if not vs:
                raise KeyError("no farfield_side/farfield_top structs in file")
            run["v"] = np.concatenate(vs)
            run["e2_dev"] = max(_check_e2_consistency(m[k])
                                for k in ("farfield_side", "farfield_top")
                                if isinstance(m.get(k), dict) and m[k])
            run["ff_lam_nm"] = float(m["farfield_side"]["lam"]) * 1e9 \
                if isinstance(m.get("farfield_side"), dict) and "lam" in m["farfield_side"] else np.nan
        except KeyError as e:
            run["problems"].append(str(e))
    return run


def load_stage(stage: str, override: str | None, need_complex: bool = True) -> list[dict]:
    d = _stage_dir(stage, override)
    runs = []
    for f in sorted(glob.glob(os.path.join(d, "result_*.mat"))):
        try:
            runs.append(load_run(f, need_complex=need_complex))
        except Exception as e:                        # unreadable file: report, keep going
            print(f"  SKIP {os.path.basename(f)}: {e}")
    print(f"Loaded {len(runs)} runs from {d}")
    return runs


def _controls_and_rest(runs):
    ctrl = [r for r in runs if r["r_nm"] == 0.0 and r["v"] is not None]
    rest = [r for r in runs if r["r_nm"] > 0.0]
    return ctrl, rest


def _shift_gate(runs, ctrl):
    """Per-run resonance-pull check vs the control (dispersive contamination)."""
    bad = []
    for r in runs:
        if not np.isfinite(r["lam_res_nm"]) or not np.isfinite(ctrl["lam_res_nm"]):
            continue
        dl = abs(r["lam_res_nm"] - ctrl["lam_res_nm"])
        r["dlam_nm"] = dl
        if ctrl["fwhm_nm"] > 0 and dl > RES_SHIFT_FRAC_MAX * ctrl["fwhm_nm"]:
            bad.append((r["name"], dl))
    return bad


# ═══════════════════════════════════════════════════════════════════════════════
# gates — noise floor + superposition verdict (stages A + B)
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_gates(args):
    a_runs = load_stage("a", args.dir_a)
    # The stage-A prelim .mat (ports-only, no "_ff" in the tag) has no far-field
    # keys by design — don't report it as a failure.
    a_runs = [r for r in a_runs if "_ff" in r["name"] or r["v"] is not None]
    a_ok = [r for r in a_runs if r["v"] is not None and not r["problems"]]
    for r in a_runs:
        for p in r["problems"]:
            print(f"  GATE FAIL {r['name']}: {p}")
    if not a_ok:
        print("STOP: no usable stage-A baseline (see failures above).")
        return 1
    a_ctrl = a_ok[0]
    print(f"\nStage A baseline: {a_ctrl['name']}")
    print(f"  lambda_res = {a_ctrl['lam_res_nm']:.3f} nm  |FWHM| = {a_ctrl['fwhm_nm']:.3f} nm"
          f"  peak T = {a_ctrl['peak_T']:.4f}  loss@res = {a_ctrl.get('loss_res', np.nan):.4f}")
    print(f"  far-field recorded at {a_ctrl.get('ff_lam_nm', np.nan):.3f} nm; "
          f"E2 vs |E_c|^2 max deviation = {a_ctrl.get('e2_dev', np.nan):.2e}")

    try:
        b_runs = load_stage("b", args.dir_b)
    except FileNotFoundError:
        print("\nStage B not downloaded yet — stage-A checks only (rerun gates after B).")
        return 0

    ctrls, rest = _controls_and_rest(b_runs)
    if not ctrls:
        print("STOP: stage B has no usable r=0 control.")
        return 1
    b_ctrl = ctrls[0]
    for name, dl in _shift_gate(rest, b_ctrl):
        print(f"  RESONANCE-PULL WARNING {name}: dlam = {dl*1e3:.1f} pm "
              f"(> {RES_SHIFT_FRAC_MAX:.0%} of FWHM — dispersive contamination)")

    nb = float(np.linalg.norm(b_ctrl["v"]))
    floor_cross = float(np.linalg.norm(b_ctrl["v"] - a_ctrl["v"]))
    print(f"\nCross-job control repeat (pure numerics floor): "
          f"||dv|| = {floor_cross:.3e}  ({floor_cross/nb:.2%} of baseline norm)")

    def find(r_nm, xs, y):
        for r in rest:
            if (abs(r["r_nm"] - r_nm) < 0.5 and len(r["x_nm"]) == len(xs)
                    and r["v"] is not None and abs(r["y_nm"] - y) < 0.5
                    and all(abs(a - b) < 0.5 for a, b in zip(r["x_nm"], xs))):
                return r
        return None

    from runners.scatterers import _common
    X0, Y0, J, R0 = (_common.X_REF_NM, _common.Y0_NM, _common.JITTER_NM,
                     _common.RADIUS_NM)
    DP = _common.PAIR_GATE_SPACING_NM
    ref = find(R0, [X0], Y0)
    if ref is None:
        print("STOP: reference single (RADIUS_NM, X_REF, Y0) missing from stage B.")
        return 1
    resp_ref = ref["v"] - b_ctrl["v"]
    nr = float(np.linalg.norm(resp_ref))
    print(f"Reference response |r(X_REF={X0:.0f})| = {nr:.3e}"
          f"  -> response / numerics-floor ratio = {nr/max(floor_cross,1e-300):.1f}x")
    for lbl, xs, y in (("x+J", [X0 + J], Y0), ("y+J", [X0], Y0 + J), ("xy+J", [X0 + J], Y0 + J)):
        r = find(R0, xs, y)
        if r is not None:
            d = float(np.linalg.norm((r["v"] - b_ctrl["v"]) - resp_ref))
            print(f"  half-cell jitter {lbl:6s}: ||r_jit - r_ref|| = {d:.3e} ({d/nr:.1%} of |r|)"
                  f"  [contains the real {J:.0f} nm phase advance, upper bound on noise]")

    print("\nSuperposition (Born) test — pair vs sum of singles:")
    min_spacing = None
    passing = []
    for s in _common.SEPARATIONS_NM:
        pair = find(R0, sorted([X0, X0 + s]), Y0)
        single_b = find(R0, [X0 + s], Y0)
        if pair is None or single_b is None:
            print(f"  s = {s:6.0f} nm: MISSING runs — skipped")
            continue
        ra = resp_ref
        rb = single_b["v"] - b_ctrl["v"]
        rp = pair["v"] - b_ctrl["v"]
        e = float(np.linalg.norm(rp - ra - rb)) / max(np.linalg.norm(ra), np.linalg.norm(rb))
        ok = e < PAIR_ERR_MAX
        passing.append((s, ok))
        print(f"  s = {s:6.0f} nm: error = {e:.1%}  {'PASS' if ok else 'FAIL'}")

    # (r, y) size/distance gate — biggest radius at the closest y that stays linear.
    pair_x = sorted(_common.GATE_PAIR_X_NM)
    print(f"\n(r, y) gate — cavity-straddling pair {pair_x} (spacing {DP:.0f} nm, worst case):")
    grid = []
    for rg in _common.R_GATE_LIST_NM:
        for yg in _common.Y_GATE_LIST_NM:
            if yg < _common.y_min_nm(rg):
                continue
            sa = find(rg, [pair_x[0]], yg)
            sb = find(rg, [pair_x[1]], yg)
            pr = find(rg, pair_x, yg)
            if sa is None or sb is None or pr is None:
                print(f"  r={rg:4.0f} y={yg:5.0f}: MISSING runs — skipped")
                continue
            ra, rb = sa["v"] - b_ctrl["v"], sb["v"] - b_ctrl["v"]
            e = float(np.linalg.norm((pr["v"] - b_ctrl["v"]) - ra - rb)) \
                / max(np.linalg.norm(ra), np.linalg.norm(rb))
            pull = abs(pr["lam_res_nm"] - b_ctrl["lam_res_nm"])
            pull_ok = b_ctrl["fwhm_nm"] > 0 and pull < RES_SHIFT_FRAC_MAX * b_ctrl["fwhm_nm"]
            snr = float(min(np.linalg.norm(ra), np.linalg.norm(rb))) / max(floor_cross, 1e-300)
            ok = e < PAIR_ERR_MAX and pull_ok and snr > 3.0
            grid.append({"r_nm": rg, "y_nm": yg, "resp_norm": float(np.linalg.norm(ra)),
                         "pair_err": e, "pull_nm": pull, "snr": snr, "ok": bool(ok)})
            why = "" if ok else (" [noise-limited]" if snr <= 3.0 else
                                 (" [nonlinear]" if e >= PAIR_ERR_MAX else " [resonance pull]"))
            print(f"  r={rg:4.0f} y={yg:5.0f}: |resp|={np.linalg.norm(ra):.3e}  "
                  f"snr={snr:5.1f}x  err={e:.1%}  pull={pull*1e3:.0f}pm  "
                  f"{'PASS' if ok else 'FAIL' + why}")
    rec = None
    for rg in sorted(_common.R_GATE_LIST_NM, reverse=True):
        cands = sorted([g for g in grid if g["r_nm"] == rg and g["ok"]],
                       key=lambda g: g["y_nm"])
        if cands:
            rec = cands[0]
            break
    if rec:
        print(f"\nRECOMMENDED (biggest linear scatterer, closest clean y): "
              f"RADIUS_NM = {rec['r_nm']:.0f}, Y0_NM = {rec['y_nm']:.0f} "
              f"-> paste into runners/scatterers/_common.py before stage C.")
    for s, ok in sorted(passing):
        if ok and all(o for t, o in passing if t >= s):
            min_spacing = s
            break
    gates_out = os.path.join(_stage_dir("b", args.dir_b), "gates_report.json")
    with open(gates_out, "w") as f:
        json.dump({"floor_cross_norm": floor_cross, "ref_response_norm": nr,
                   "baseline_norm": nb, "min_spacing_nm": min_spacing,
                   "pair_errors": [[s, bool(ok)] for s, ok in passing],
                   "ry_grid": grid,
                   "recommended": rec}, f, indent=2)
    print(f"\nGates report written: {os.path.abspath(gates_out)}")
    if min_spacing is None:
        print("\nVERDICT: FAIL — superposition does not hold at any tested spacing "
              "(or runs missing). STOP: do not dispatch stage C; this is a valid negative.")
        return 1
    print(f"\nVERDICT: PASS — use MIN_SPACING_NM = {min_spacing:.0f} in scat_e_validate.py / solve.")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# solve — the inverse problem (stage C)
# ═══════════════════════════════════════════════════════════════════════════════

def _power(v):
    return float(np.vdot(v, v).real)


def _greedy(b, cols, x_pos, min_spacing, allowed_fn=None):
    """Greedy binary forward selection + swap refinement. Returns index list.
    allowed_fn(j, sel) overrides the default same-row |dx| spacing rule (2D solve)."""
    chosen: list[int] = []
    F = b.copy()

    def allowed(j, sel):
        if allowed_fn is not None:
            return allowed_fn(j, sel)
        return all(abs(x_pos[j] - x_pos[k]) >= min_spacing - 1e-6 for k in sel)

    improved = True
    while improved:
        improved = False
        base_p = _power(F)
        best_j, best_p = None, base_p
        for j in range(len(cols)):
            if j in chosen or not allowed(j, chosen):
                continue
            p = _power(F + cols[j])
            if p < best_p - 1e-15:
                best_j, best_p = j, p
        if best_j is not None:
            chosen.append(best_j)
            F = F + cols[best_j]
            improved = True
    # Swap pass: replace any member with any allowed alternative if it helps.
    for _ in range(5):
        changed = False
        for i, j in enumerate(list(chosen)):
            others = [k for k in chosen if k != j]
            F_wo = b + sum((cols[k] for k in others), np.zeros_like(b))
            best_k, best_p = j, _power(F_wo + cols[j])
            for k in range(len(cols)):
                if k in others or not allowed(k, others):
                    continue
                p = _power(F_wo + cols[k])
                if p < best_p - 1e-15:
                    best_k, best_p = k, p
            if best_k != j:
                chosen[i] = best_k
                changed = True
        if not changed:
            break
    return sorted(chosen, key=lambda j: x_pos[j])


def _bounded_ceiling(b, A):
    """Best leak-power reduction with real per-column weights w in [0,1] — the
    'radius-tunable pillars, phase fixed by position' ceiling. Solved as a small
    box-constrained QP on the NORMALIZED Gram form (raw scale ~1e-14 stalls
    lsq_linear; L-BFGS-B on the normalized form converges — session-verified
    against the exact single-site optimum). Returns (reduction_frac, w)."""
    from scipy.optimize import minimize
    pb = _power(b)
    f = np.real(np.conj(A.T) @ b) / pb
    H = np.real(np.conj(A.T) @ A) / pb
    n = A.shape[1]
    res = minimize(lambda w: 1.0 + 2 * f @ w + w @ H @ w,
                   np.zeros(n), jac=lambda w: 2 * f + 2 * H @ w,
                   method="L-BFGS-B", bounds=[(0.0, 1.0)] * n,
                   options={"maxiter": 2000, "ftol": 1e-15})
    return 1.0 - float(res.fun), res.x


def _periodic_scan(b, cols, x_pos, min_spacing):
    """Best constant-period + phase comb on the measured grid."""
    xs = np.asarray(x_pos)
    step = float(np.min(np.diff(np.unique(xs)))) if len(xs) > 1 else 0.0
    best = None
    max_m = max(2, int(round((xs.max() - xs.min()) / max(step, 1e-9))) )
    for m in range(2, min(max_m, 30) + 1):
        period = m * step
        if period < min_spacing - 1e-6:
            continue
        tol = 0.01 * step
        for k in range(m):
            rem = np.mod(xs - xs.min() - k * step, period)
            sel = [j for j in range(len(xs))
                   if min(rem[j], period - rem[j]) < tol]
            if len(sel) < 2:
                continue
            p = _power(b + sum((cols[j] for j in sel), np.zeros_like(b)))
            if best is None or p < best["power"]:
                best = {"power": p, "period_nm": period,
                        "offset_nm": float(xs.min() + k * step),
                        "indices": sel}
    return best


def cmd_solve(args):
    runs = load_stage("c", args.dir_c)
    ctrls, rest = _controls_and_rest(runs)
    if not ctrls:
        print("STOP: stage C has no usable r=0 control (the b vector).")
        return 1
    ctrl = ctrls[0]
    singles = sorted([r for r in rest if len(r["x_nm"]) == 1 and r["v"] is not None],
                     key=lambda r: r["x_nm"][0])
    dropped = [r["name"] for r in rest if r["v"] is None or r["problems"]]
    for r in rest:
        for p in r["problems"]:
            print(f"  GATE FAIL {r['name']}: {p}")
    for name, dl in _shift_gate(singles, ctrl):
        print(f"  RESONANCE-PULL WARNING {name}: dlam = {dl*1e3:.1f} pm")
    singles = [r for r in singles if not r["problems"]]
    if dropped:
        print(f"NOTE: {len(dropped)} position(s) dropped (gate failures above) — "
              f"the combination is chosen from the surviving {len(singles)}.")
    if len(singles) < 3:
        print("STOP: fewer than 3 usable response columns.")
        return 1

    b = ctrl["v"]
    x_pos = [r["x_nm"][0] for r in singles]
    cols = [r["v"] - b for r in singles]
    P0 = _power(b)
    print(f"\nBaseline (control) far-field power P0 = {P0:.4e}   "
          f"loss@res = {ctrl.get('loss_res', np.nan):.4f}  T@res = {ctrl.get('T_res', np.nan):.4f}")

    # 1) Continuous LS ceiling — the fundamental limit of this response basis.
    A = np.stack(cols, axis=1)
    x_star, *_ = np.linalg.lstsq(A, -b, rcond=None)
    ceiling = 1.0 - _power(b + A @ x_star) / P0
    print(f"\n[1] CONTINUOUS LS CEILING: {ceiling:.1%} of the monitored far-field power "
          f"is cancellable with IDEAL per-site weights.")
    print(f"    DERIVED upper bound on dT (= ceiling x baseline resonant loss "
          f"{ctrl.get('loss_res', np.nan):.3f}): {ceiling * ctrl.get('loss_res', np.nan):+.4f}"
          f"  — assumes all cancelled radiation recycles into T; stage E measures the truth.")
    if ceiling < args.ceiling_stop:
        print(f"    Ceiling below --ceiling-stop={args.ceiling_stop:.0%}: the route is "
              f"measured-limited. Consider stopping here (valid negative result).")

    # 2) Greedy binary combination (the actual scatterer sequence).
    chosen = _greedy(b, cols, x_pos, args.min_spacing)
    Pg = _power(b + sum((cols[j] for j in chosen), np.zeros_like(b)))
    red_g = 1.0 - Pg / P0
    print(f"\n[2] GREEDY BINARY COMBINATION ({len(chosen)} scatterer pairs, "
          f"min spacing {args.min_spacing:.0f} nm): power reduction {red_g:.1%}")
    print(f"    positions (nm): {[x_pos[j] for j in chosen]}")

    # 3) Best constant-period + phase comb from the SAME data.
    per = _periodic_scan(b, cols, x_pos, args.min_spacing)
    if per:
        red_p = 1.0 - per["power"] / P0
        print(f"\n[3] BEST PERIODIC COMB: period {per['period_nm']:.0f} nm, first site "
              f"{per['offset_nm']:.0f} nm, {len(per['indices'])} sites: reduction {red_p:.1%}")
        print(f"    -> free positions vs constant period: "
              f"{'free WINS' if red_g > red_p + 0.005 else 'comparable — period+phase suffices'}")

    report = {
        "created": "stage-D solve",
        "baseline_file": ctrl["name"],
        "lambda_res_nm": ctrl["lam_res_nm"],
        "baseline_T_res": ctrl.get("T_res"), "baseline_loss_res": ctrl.get("loss_res"),
        "positions_nm": x_pos,
        "response_norm": [float(np.linalg.norm(c)) for c in cols],
        "response_phase_rad": [float(np.angle(np.vdot(b, c))) for c in cols],
        "ls_ceiling_frac": ceiling,
        "greedy": {"positions_nm": [x_pos[j] for j in chosen],
                   "power_reduction_frac": red_g},
        "periodic_best": ({"period_nm": per["period_nm"], "offset_nm": per["offset_nm"],
                           "positions_nm": [x_pos[j] for j in per["indices"]],
                           "power_reduction_frac": 1.0 - per["power"] / P0} if per else None),
        "min_spacing_nm": args.min_spacing,
        "dropped_runs": dropped,
    }
    out = os.path.join(_stage_dir("c", args.dir_c), "greens_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written: {os.path.abspath(out)}")
    print("\nPaste into scat_e_validate.py:")
    print(f"CANDIDATE_ARRAYS = [\n    {report['greedy']['positions_nm']},")
    if report["periodic_best"]:
        print(f"    {report['periodic_best']['positions_nm']},")
    print("]")
    print(f"MIN_SPACING_NM = {args.min_spacing}")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# solve2 — 2-row inverse solve (stage C row 1 + stage C2 row 2)
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_solve2(args):
    runs1 = load_stage("c", args.dir_c)
    runs2 = load_stage("c2", args.dir_c2)
    ctrls1, rest1 = _controls_and_rest(runs1)
    ctrls2, rest2 = _controls_and_rest(runs2)
    if not ctrls2:
        print("STOP: stage C2 has no usable r=0 control (the fresh b vector).")
        return 1
    ctrl = ctrls2[0]
    b = ctrl["v"]
    P0 = _power(b)
    if ctrls1:
        dc = float(np.linalg.norm(ctrls1[0]["v"] - b) / np.linalg.norm(b))
        print(f"cross-job control consistency |b_C - b_C2| / |b| = {dc:.2e} "
              f"(row-1 columns reused across jobs)")
    for runs, tag in ((rest1, "C"), (rest2, "C2")):
        for r in runs:
            for p in r["problems"]:
                print(f"  GATE FAIL [{tag}] {r['name']}: {p}")
    singles = [r for r in rest1 + rest2
               if len(r["x_nm"]) == 1 and r["v"] is not None and not r["problems"]]
    pairs2 = [r for r in rest2
              if len(r["x_nm"]) == 2 and r["v"] is not None and not r["problems"]]
    for name, dl in _shift_gate(singles + pairs2, ctrl):
        print(f"  RESONANCE-PULL WARNING {name}: dlam = {dl*1e3:.1f} pm")

    by_site = {}
    for r in singles:                       # C loaded first; C2 (same-job) wins ties
        by_site[r["sites"][0]] = r
    sites = sorted(by_site)                 # list of (x_nm, y_nm)
    cols = [by_site[s]["v"] - b for s in sites]
    rows = sorted({s[1] for s in sites})
    print(f"\nBaseline (C2 control): P0 = {P0:.4e}   T@res = {ctrl.get('T_res', np.nan):.4f}"
          f"   loss@res = {ctrl.get('loss_res', np.nan):.4f}")
    print("columns: " + ", ".join(
        f"{sum(1 for s in sites if s[1] == y)} @ y={y:.0f}" for y in rows))

    # [0] Superposition gates — the kill criterion for any cross-row prediction.
    print(f"\n[0] SUPERPOSITION GATES (pair vs sum of singles, {PAIR_ERR_MAX:.0%} gate):")
    gate_fail, gate_out = False, []
    for p in pairs2:
        ss = p["sites"]
        missing = [s for s in ss if s not in by_site]
        if missing:
            print(f"  SKIP {p['name']}: singles missing for {missing}")
            continue
        r_pair = p["v"] - b
        r_sum = sum(by_site[s]["v"] - b for s in ss)
        den = max(float(np.linalg.norm(by_site[s]["v"] - b)) for s in ss)
        err = float(np.linalg.norm(r_pair - r_sum)) / den
        kind = "cross-row" if abs(ss[0][1] - ss[1][1]) > 0.5 else "in-row-2"
        ok = err <= PAIR_ERR_MAX
        gate_fail |= not ok
        gate_out.append({"sites": [list(s) for s in ss], "kind": kind, "err": err, "ok": ok})
        print(f"  {kind:9s} {ss}: err = {err:.1%}  {'PASS' if ok else 'FAIL'}")
    if gate_fail:
        print("\nVERDICT: FAIL — superposition beyond the 5% gate. STOP: 2-row "
              "predictions are not trustworthy; the single-row results stand (valid negative).")
        return 1

    # [1] Ceilings: LS (any complex weights) + bounded w in [0,1], 2-row vs row-1-only.
    A = np.stack(cols, axis=1)
    x_star, *_ = np.linalg.lstsq(A, -b, rcond=None)
    ls_ceiling = 1.0 - _power(b + A @ x_star) / P0
    bq2, _w = _bounded_ceiling(b, A)
    idx1 = [i for i, s in enumerate(sites) if abs(s[1] - rows[0]) < 0.5]
    bq1, _ = _bounded_ceiling(b, A[:, idx1])
    print(f"\n[1] CEILINGS: LS (ideal complex weights) {ls_ceiling:.1%}")
    print(f"    bounded w in [0,1]:  2-row {bq2:.1%}   row-1-only {bq1:.1%}"
          f"   ->  row 2 adds {bq2 - bq1:+.1%} at the weight-tuned limit")

    # [2] Exhaustive binary 1/2/3-site over the union (Gram identity).
    g = 2 * np.real(np.conj(b) @ A)
    s_ = np.real(np.sum(np.conj(A) * A, axis=0))
    G = 2 * np.real(A.conj().T @ A)
    xs_a = np.array([s[0] for s in sites])
    ys_a = np.array([s[1] for s in sites])

    def ok2(i, j):
        return (abs(ys_a[i] - ys_a[j]) > 0.5) or \
               (abs(xs_a[i] - xs_a[j]) >= args.min_spacing - 1e-6)

    p1 = P0 + g + s_
    n = len(sites)
    j1 = int(np.argmin(p1))
    best = {1: (float(p1[j1]), [j1])}
    ii, jj = np.triu_indices(n, 1)
    okp = np.array([ok2(a, c) for a, c in zip(ii, jj)])
    P2 = np.where(okp, p1[ii] + p1[jj] - P0 + G[ii, jj], np.inf)
    k2 = int(np.argmin(P2))
    best[2] = (float(P2[k2]), [int(ii[k2]), int(jj[k2])])
    b3 = (np.inf, None)
    for a in range(n):
        for c in range(a + 1, n):
            if not ok2(a, c):
                continue
            base = p1[a] + p1[c] - P0 + G[a, c]
            ks = np.array([k for k in range(c + 1, n) if ok2(a, k) and ok2(c, k)])
            if ks.size == 0:
                continue
            P3 = base + (p1[ks] - P0) + G[a, ks] + G[c, ks]
            kk = int(np.argmin(P3))
            if P3[kk] < b3[0]:
                b3 = (float(P3[kk]), [a, c, int(ks[kk])])
    best[3] = b3
    print(f"\n[2] EXHAUSTIVE BINARY (within-row spacing >= {args.min_spacing:.0f} nm; "
          f"cross-row unconstrained — gate-verified):")
    for k in (1, 2, 3):
        pk, sel = best[k]
        if sel is None:
            continue
        print(f"  best {k}-site: {[sites[j] for j in sel]}   reduction {1 - pk/P0:+.1%}")

    # [3] Greedy + swap over the union (k free).
    chosen = _greedy(b, cols, [s[0] for s in sites], args.min_spacing,
                     allowed_fn=lambda j, sel: all(ok2(j, k) for k in sel))
    red_g = 1.0 - _power(b + sum((cols[j] for j in chosen), np.zeros_like(b))) / P0
    print(f"\n[3] GREEDY+SWAP: {len(chosen)} sites {[sites[j] for j in chosen]}"
          f"   reduction {red_g:+.1%}")

    report = {
        "created": "stage-D solve2 (2-row)",
        "baseline_file": ctrl["name"],
        "baseline_T_res": ctrl.get("T_res"), "baseline_loss_res": ctrl.get("loss_res"),
        "rows_y_nm": [float(y) for y in rows],
        "sites": [list(s) for s in sites],
        "gates": gate_out,
        "ls_ceiling_frac": ls_ceiling,
        "bounded_ceiling_2row": bq2, "bounded_ceiling_row1": bq1,
        "exhaustive": {str(k): {"sites": [list(sites[j]) for j in best[k][1]],
                                "power_reduction_frac": 1.0 - best[k][0] / P0}
                       for k in (1, 2, 3) if best[k][1] is not None},
        "greedy": {"sites": [list(sites[j]) for j in chosen],
                   "power_reduction_frac": red_g},
        "min_spacing_nm": args.min_spacing,
    }
    out = os.path.join(_stage_dir("c2", args.dir_c2), "greens2_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written: {os.path.abspath(out)}")
    print("\nPaste into scat_e_validate.py (entries are [x, y] site pairs):")
    print("CANDIDATE_ARRAYS = [")
    for k in (2, 3):
        pk, sel = best[k]
        if sel is not None:
            print(f"    {[[float(sites[j][0]), float(sites[j][1])] for j in sel]},")
    print("]")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# validate — predicted vs measured (stage E)
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_validate(args):
    runs = load_stage("e", args.dir_e)
    ctrls, rest = _controls_and_rest(runs)
    if not ctrls:
        print("STOP: stage E has no usable r=0 control.")
        return 1
    ctrl = ctrls[0]
    P0 = _power(ctrl["v"])
    report = None
    try:
        with open(os.path.join(_stage_dir("c", args.dir_c), "greens_report.json")) as f:
            report = json.load(f)
    except Exception:
        print("NOTE: greens_report.json not found — measured numbers only.")

    print(f"Control: T@res = {ctrl.get('T_res', np.nan):.4f}  "
          f"loss@res = {ctrl.get('loss_res', np.nan):.4f}  lambda = {ctrl['lam_res_nm']:.3f} nm")
    for r in sorted(rest, key=lambda r: len(r["x_nm"])):
        for p in r["problems"]:
            print(f"  GATE FAIL {r['name']}: {p}")
        if r["v"] is None:
            continue
        red = 1.0 - _power(r["v"]) / P0
        dT = r.get("T_res", np.nan) - ctrl.get("T_res", np.nan)
        dL = r.get("loss_res", np.nan) - ctrl.get("loss_res", np.nan)
        pred = ""
        if report:
            for key in ("greedy", "periodic_best"):
                cand = report.get(key)
                if cand and sorted(cand["positions_nm"]) == sorted(r["x_nm"]):
                    pred = (f"  [predicted reduction {cand['power_reduction_frac']:.1%} ({key})]")
        ys = set(r["y_site_nm"])
        lbl = (f"{r['x_nm']} @ y={r['y_site_nm'][0]:.0f}" if len(ys) == 1
               else f"{r['sites']}")
        print(f"\n{len(r['x_nm'])} sites {lbl}:")
        print(f"  MEASURED far-field power reduction = {red:+.1%}{pred}")
        print(f"  MEASURED dT = {dT:+.4f}   dloss = {dL:+.4f}   "
              f"dlam = {(r['lam_res_nm'] - ctrl['lam_res_nm'])*1e3:+.0f} pm")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# selftest — synthetic planted-optimum recovery (no data)
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_selftest(args):
    rng = np.random.default_rng(7)
    M, N, step = 600, 49, 135.0
    x_pos = [round((j - N // 2) * step, 1) for j in range(N)]

    def smooth_col(seed_phase):
        z = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        return np.exp(1j * seed_phase) * z / np.linalg.norm(z)

    cols = [smooth_col(2 * np.pi * j / N) for j in range(N)]
    planted = [4, 14, 24, 34, 44]                       # spaced 10 steps = 1350 nm
    b = -0.9 * sum(cols[j] for j in planted) \
        + 0.005 * (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(M)

    A = np.stack(cols, axis=1)
    x_star, *_ = np.linalg.lstsq(A, -b, rcond=None)
    ceiling = 1.0 - _power(b + A @ x_star) / _power(b)
    assert ceiling > 0.95, f"LS ceiling too low on synthetic problem: {ceiling:.2%}"

    chosen = _greedy(b, cols, x_pos, min_spacing=270.0)
    assert set(chosen) == set(planted), f"greedy missed the planted set: {chosen} vs {planted}"

    # Spacing constraint: a decoy adjacent (1 step) to a planted site must be excluded.
    cols2 = cols + [cols[4] * 1.001]
    x2 = x_pos + [x_pos[4] + step]                      # 135 nm from planted site 4
    chosen2 = _greedy(b, cols2, x2, min_spacing=270.0)
    assert not (4 in chosen2 and len(cols) in chosen2), "min-spacing constraint violated"

    # Periodic scan finds a planted comb.
    comb = [j for j in range(N) if j % 7 == 3]
    b3 = -0.9 * sum(cols[j] for j in comb)
    per = _periodic_scan(b3, cols, x_pos, min_spacing=270.0)
    assert per and abs(per["period_nm"] - 7 * step) < 1e-6, f"periodic scan failed: {per}"
    assert sorted(per["indices"]) == comb, "periodic scan picked the wrong comb"

    # 2-row machinery: bounded-weight ceiling (planted amplitude 0.9 lies in [0,1],
    # so near-complete cancellation must be reachable with bounded weights) ...
    bq, w = _bounded_ceiling(b, A)
    assert bq > 0.95, f"bounded ceiling too low on synthetic problem: {bq:.2%}"
    assert np.all(w >= -1e-9) and np.all(w <= 1.0 + 1e-9), "bounded weights out of box"
    # ... and the allowed_fn hook (cross-row spacing rule plumbing): restricting the
    # greedy to the planted set must still recover exactly that set.
    chosen3 = _greedy(b, cols, x_pos, 270.0, allowed_fn=lambda j, sel: j in planted)
    assert set(chosen3) == set(planted), f"allowed_fn greedy failed: {chosen3}"

    print(f"selftest PASS  (ceiling {ceiling:.1%}, bounded {bq:.1%}, greedy recovered "
          f"{len(planted)} planted sites, spacing + allowed_fn held, periodic comb "
          f"period {per['period_nm']:.0f} nm found)")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gates", help="stage A/B noise + superposition verdict")
    g.add_argument("--dir-a", default=None); g.add_argument("--dir-b", default=None)
    s = sub.add_parser("solve", help="stage C inverse solve -> greens_report.json")
    s.add_argument("--dir-c", default=None)
    s.add_argument("--min-spacing", type=float, default=270.0,
                   help="minimum |dx| between chosen scatterers (nm), from the gates verdict")
    s.add_argument("--ceiling-stop", type=float, default=0.02,
                   help="print a stop suggestion if the LS ceiling is below this fraction")
    s2 = sub.add_parser("solve2", help="2-row inverse solve (stage C + C2) -> greens2_report.json")
    s2.add_argument("--dir-c", default=None); s2.add_argument("--dir-c2", default=None)
    s2.add_argument("--min-spacing", type=float, default=270.0,
                    help="minimum |dx| between chosen scatterers WITHIN a row (nm)")
    v = sub.add_parser("validate", help="stage E predicted vs measured")
    v.add_argument("--dir-e", default=None); v.add_argument("--dir-c", default=None)
    sub.add_parser("selftest", help="synthetic planted-optimum recovery")
    args = ap.parse_args()
    sys.exit({"gates": cmd_gates, "solve": cmd_solve, "solve2": cmd_solve2,
              "validate": cmd_validate, "selftest": cmd_selftest}[args.cmd](args))


if __name__ == "__main__":
    sys.path.insert(0, PROJECT_ROOT)
    main()
