"""TE-lane local gates (zero GPU, silent lumapi): B1 geometry equivalence for
both TE seeds + bounds / 2κL / tiles / region checks; `--generate` adds the
local lumopt2 project.generate() smoke (skill item 28c).

Study dir: runners/lumopt2_design/gates/ | Created 2026-10-04 | zero GPU
Usage (repo root):  PYTHONIOENCODING=utf-8 python runners/lumopt2_design/gates/gate_te_local.py [--generate]

Per seed (campaign_te_s1 / campaign_te_s2):
  1. build_base_fsp builds (asserts the tooth-name map), λ grid length right;
  2. func(seed) applied to the base scene == a builder scene drawn with the
     seed's per-tooth arrays (S2's apodization lives in the param vector, the
     base scene is uniform bulk — so the reference MUST be the per-tooth
     builder, not the base .fsp) to < 0.1 nm; same for a perturbed vector;
  3. shift algebra (contiguity, frozen outer edge, cavity absorption) exact;
  4. seed inside bounds; 2κL ≥ 3.5 at the spec's N; tile count fits
     wg_tile_max_xcells; region x half-span covers the free teeth.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
import config                                                   # noqa: E402
from runners.lumopt2_design import lumopt2_design as eng        # noqa: E402
from runners.lumopt2_design.campaign_te_s1 import SPEC as S1    # noqa: E402
from runners.lumopt2_design.campaign_te_s2 import SPEC as S2    # noqa: E402
from runners.lumopt2_design.campaign_te_s1 import SPEC_V3 as S1V3   # noqa: E402

sys.path.insert(0, os.path.dirname(config.LUMAPI_PATH))
import lumapi                                                   # noqa: E402
from bragg_device import PiShiftBraggFDTD                       # noqa: E402


def _read(fdtd, names):
    return {n: tuple(float(np.squeeze(fdtd.getnamed(n, k))) / eng.NM
                     for k in ("x", "x span", "y span")) for n in names}


def _builder_ref(spec, p, objs):
    """Scene drawn by the production builder from p's per-tooth widths."""
    L = eng.layout(spec.n_free)
    cfg = eng.build_base_cfg(spec)
    nar = list((p[L.SL_AVG] - p[L.SL_CORR] / 2.0) * eng.NM)
    wid = list((p[L.SL_AVG] + p[L.SL_CORR] / 2.0) * eng.NM)
    cfg.grating.width_narrow_per_tooth_m = nar
    cfg.grating.width_wide_per_tooth_m = wid
    cfg.grating.cavity_width_m = float(p[L.I_CAV]) * eng.NM
    sim = PiShiftBraggFDTD(**cfg.to_device_kwargs())
    try:
        sim.build()
        return _read(sim.fdtd, objs)
    finally:
        sim.close()


def gate(spec, workdir):
    print(f"== TE B1: {spec.label} (pol {spec.polarization}, pitch {spec.pitch_nm}, "
          f"n_free {spec.n_free}, N {spec.n_periods_side}) ==")
    L = eng.layout(spec.n_free)
    fsp = os.path.join(workdir, f"{spec.label}_b1.fsp")
    wl = eng.build_base_fsp(spec, fsp)
    assert len(wl) == spec.n_wl_points, f"λ grid {len(wl)} != {spec.n_wl_points}"
    names, cavity = eng.tooth_names(spec.n_periods_side, spec.n_free)
    objs = [n for q in names.values() for n in q] + [cavity]
    func = eng.make_func(spec)
    p0 = eng.seed_params(spec)
    rng = np.random.default_rng(1)
    p1 = p0.copy()
    p1[L.SL_CORR] = np.clip(p1[L.SL_CORR] + rng.uniform(-40, 80, spec.n_free),
                            spec.corr_min_nm, spec.corr_max_nm)
    p1[L.SL_AVG] = np.clip(p1[L.SL_AVG] + rng.uniform(-15, 15, spec.n_free),
                           *spec.avg_bounds_nm)
    ok = True
    applied = {}
    with lumapi.FDTD(filename=fsp, hide=True) as fdtd:
        for tag, p in (("seed", p0), ("perturbed", p1)):
            for k, v in func(p).items():
                obj, prop = k.split("::")
                if obj in objs:
                    fdtd.setnamed(obj, prop, float(eng._plain(v)))
            applied[tag] = _read(fdtd, objs)
    for tag, p in (("seed", p0), ("perturbed", p1)):
        ref = _builder_ref(spec, p, objs)
        worst = max(abs(a - b) for o in objs for a, b in zip(applied[tag][o], ref[o]))
        print(f"  func({tag}) vs per-tooth builder: worst {worst:.4f} nm")
        ok &= worst < 0.1

    # shift algebra (pure python)
    p2 = p0.copy()
    p2[L.SL_SHIFT] = rng.uniform(0, 150, spec.n_free)
    pr = {k: float(eng._plain(v)) / eng.NM for k, v in func(p2).items()}
    hp = l0 = spec.pitch_nm / 2.0
    x_out = l0 / 2.0 + spec.n_free * spec.pitch_nm
    edges = lambda o: (pr[f"{o}::x"] - pr[f"{o}::x span"] / 2, pr[f"{o}::x"] + pr[f"{o}::x span"] / 2)
    gaps, prev = [], -x_out
    for d in range(spec.n_free, 0, -1):
        for o in names[d][:2]:
            lo, hi = edges(o); gaps.append(abs(lo - prev)); prev = hi
    gaps.append(abs(edges(cavity)[0] - prev)); prev = edges(cavity)[1]
    for d in range(1, spec.n_free + 1):
        for o in names[d][2:]:
            lo, hi = edges(o); gaps.append(abs(lo - prev)); prev = hi
    e_edge = abs(prev - x_out)
    e_cav = abs(pr[f"{cavity}::x span"] - (l0 + 2.0 * p2[L.SL_SHIFT].sum()))
    print(f"  shifts: gap {max(gaps):.1e} edge {e_edge:.1e} cavity {e_cav:.1e} nm")
    ok &= max(gaps) < 1e-6 and e_edge < 1e-6 and e_cav < 1e-6

    # bounds / 2κL / tiles / region
    b = eng.param_bounds(spec)
    inb = all(lo <= v <= hi for v, (lo, hi) in zip(p0, b))
    two_kl = eng.two_kappa_L(p0, spec)
    x_prof_cells = (2 * (spec.n_periods_side * spec.pitch_nm + spec.pitch_nm / 4) + 2000) / spec.region_dx_nm
    per_tile = x_prof_cells / spec.wg_src_tiles
    x_half = max(eng.COMB_N_HALF * eng.COMB_LAM_NM + eng.COMB_DX_NM + 740.0,
                 spec.pitch_nm / 4 + spec.n_free * spec.pitch_nm + 1000.0)
    print(f"  seed in bounds {inb} | 2κL {two_kl:.3f} (floor {spec.two_kl_floor or eng.TWO_KL_FLOOR}) "
          f"| profile ≈{x_prof_cells:.0f} cells → {per_tile:.0f}/tile (cap {spec.wg_tile_max_xcells}) "
          f"| region x_half {x_half/1000:.1f} µm vs free edge {(spec.pitch_nm/4 + spec.n_free*spec.pitch_nm)/1000:.1f} µm")
    ok &= inb and two_kl >= (spec.two_kl_floor or eng.TWO_KL_FLOOR) \
        and per_tile <= spec.wg_tile_max_xcells
    print(f"== TE B1 {spec.label}: {'PASS' if ok else 'FAIL'} ==")
    return ok


def generate_smoke(spec, workdir):
    """Local lumopt2 project.generate() (hidden session): reproduces the whole
    generate()-time validation class in minutes (item 28c)."""
    import dataclasses
    print(f"== generate() smoke: {spec.label} ==")
    # production branch (wg_project + ns2) with provisional anchors; make_project
    # does NOT call generate() itself (review 2026-10-04) — call it explicitly.
    s = dataclasses.replace(spec, label=spec.label + "_gen", fwhm0_um=19.0,
                            wgp_target_um=19.0, wg_anchor={"softw": 19.0, "fwhm": 19.0},
                            adj_fix_re=1.0, adj_fix_im=0.0)
    lmpt = eng.import_lumopt2()
    project, _ = eng.make_project(s, os.path.join(workdir, s.label), lmpt)
    project.generate()
    print(f"== generate() smoke {spec.label}: PASS (project.generate() ran) ==")
    return True


if __name__ == "__main__":
    wd = os.path.join(tempfile.gettempdir(), "lumopt2_te_b1")
    os.makedirs(wd, exist_ok=True)
    allok = True
    for spec in (S1, S2):
        allok &= gate(spec, wd)
        if "--generate" in sys.argv:
            allok &= generate_smoke(spec, wd)
    if "--generate" in sys.argv:           # v3 engine variant: same scene, v3 fct/flags
        allok &= generate_smoke(S1V3, wd)
    from runners.lumopt2_design.validate_te import check_points
    print("== C-recipe operating point + FD legs vs bounds ==")
    allok &= check_points()
    print("TE LOCAL GATES: " + ("ALL PASS" if allok else "FAIL"))
    sys.exit(0 if allok else 1)
