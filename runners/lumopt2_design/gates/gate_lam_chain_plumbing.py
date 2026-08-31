"""PLUMBING GATE for defect #19's selector passes (2026-08-26).

WHY THIS EXISTS: job 137267 died after 2:03 of GPU time on
    IndexError: invalid index to scalar variable
because the selector was written `anp.abs(x[0])[i]`, assuming x was a LIST of
FOM entry results. It is not: autograd takes the jacobian w.r.t. the FLAT
vector [T(lam_0) .. T(lam_{n_wl-1}), softW]. The math gates all passed --
they tested the FORMULA, never the CALL PATH. This gate drives the selectors
through autograd exactly as lumopt2's base_fom.get_jacobian does, so the same
failure is caught locally in under a second with zero GPU.

Run: python gate_lam_chain_plumbing.py
"""
import sys

sys.path.insert(0, r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes")

import numpy as np
import autograd.numpy as anp
from autograd import jacobian

from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_v2_proj import SPEC

ok = True

# Build the REAL fct over the REAL flat layout the engine uses.
wl = np.linspace(SPEC.scan_center_nm - SPEC.scan_width_nm / 2,
                 SPEC.scan_center_nm + SPEC.scan_width_nm / 2,
                 SPEC.n_wl_points)
n_wl = len(wl)
fct_top = eng.make_fct_v2(wl, SPEC)

# A plausible flat x: a Lorentzian T grid plus the softW entry at the end.
g = 0.405
T = 0.9012 * g**2 / ((wl - SPEC.scan_center_nm) ** 2 + g**2)
x = np.concatenate([T, [18.35]])
print(f"flat x: {x.shape[0]} entries = {n_wl} wavelengths + 1 softW")

# 1. the layout assumption itself
scalar_ok = np.isscalar(x[0]) or np.ndim(x[0]) == 0
print(f"  {'OK  ' if scalar_ok else 'FAIL'} x[0] is a SCALAR (not the T array) "
      f"-- the assumption that killed 137267")
ok &= scalar_ok

# 2. the top-level fct must run on this layout (it is what wg_project uses)
try:
    v = float(fct_top(x))
    print(f"  OK   make_fct_v2 evaluates on the flat x -> {v:.6f}")
except Exception as e:
    print(f"  FAIL make_fct_v2 raised {e!r}")
    ok = False

# 3. THE ACTUAL FAILING CALL PATH: autograd jacobian of each selector.
i_pk = int(np.argmax(T))
k = max(1, min(int(round(0.5 * 0.81 / (wl[1] - wl[0]))), i_pk - 1, n_wl - 2 - i_pk))
i_lo, i_hi = i_pk - k, i_pk + k
print(f"  peak idx {i_pk}, k {k} -> i_lo {i_lo}, i_hi {i_hi}")

for name, sel, idx in (("width  x[-1]", lambda z: z[-1], n_wl),
                       ("Tlo    x[i_lo]", lambda z: anp.abs(z[i_lo]), i_lo),
                       ("Thi    x[i_hi]", lambda z: anp.abs(z[i_hi]), i_hi)):
    try:
        J = jacobian(sel)(x)
        onehot = (J.shape == x.shape and abs(J[idx] - 1.0) < 1e-9
                  and abs(J.sum() - 1.0) < 1e-9)
        print(f"  {'OK  ' if onehot else 'FAIL'} {name}: jacobian is one-hot at "
              f"{idx} (shape {J.shape}, J[{idx}]={J[idx]:+.3g}, "
              f"sum={J.sum():+.3g})")
        ok &= onehot
    except Exception as e:
        print(f"  FAIL {name}: jacobian raised {type(e).__name__}: {e}")
        ok = False

# 4. the OLD broken form must still fail -- proves this gate has teeth
try:
    jacobian(lambda z: anp.abs(z[0])[i_lo])(x)
    print("  FAIL the old x[0][i] form did NOT raise -- gate has no teeth")
    ok = False
except IndexError:
    print("  OK   the old x[0][i] form still raises IndexError (gate has teeth)")

# 5. SESSION-STATE hazard (job 137845_41, 2026-08-27): dEps conversion
# (compute_gradient_from_fields -> update_structure) inside
# calculate_gradient_fields runs while the CAD is in ANALYSIS mode and dies
# with "use switchtolayout first". The fix stashes the summed field arrays
# (gfields_Tlo/Thi) and converts at the DRIVER. Assert, from source, that the
# conversion cannot creep back inside — and that the driver does convert.
import inspect
mod_src = inspect.getsource(eng)
i0 = mod_src.index("def calculate_gradient_fields")
cgf_src = mod_src[i0:mod_src.index("return gT", i0)]
# strip comments — the hazard is a CALL, not a mention in a comment
cgf_src = "\n".join(l.split("#", 1)[0] for l in cgf_src.splitlines())
inside_clean = ("compute_gradient_from_fields" not in cgf_src
                and "gfields_Tlo" in cgf_src and "gfields_Thi" in cgf_src)
print(f"  {'OK  ' if inside_clean else 'FAIL'} calculate_gradient_fields "
      f"stashes gfields_Tlo/Thi and does NOT convert in-place (analysis-mode "
      f"dEps hazard, 137845_41)")
ok &= inside_clean
drv_src = inspect.getsource(eng.run_projected)
drv_ok = ("gfields_Tlo" in drv_src
          and "compute_gradient_from_fields" in drv_src)
print(f"  {'OK  ' if drv_ok else 'FAIL'} run_projected converts the stashed "
      f"selector fields at the driver (layout-mode site)")
ok &= drv_ok

# 6. d1 (2026-08-30): the ns2 spec must build the same flat-x fct — the two-
# constraint law changes the STEP, not the FOM plumbing — and the driver must
# route gLam as its own constraint row (never fold the chain into gW under
# ns2), read from source like check 5.
from runners.lumopt2_design.campaign_v2_proj_d1 import SPEC as DSPEC
try:
    fct_d1 = eng.make_fct_v2(wl, DSPEC)
    v_d1 = float(fct_d1(x))
    jac_w = float(jacobian(fct_d1)(x)[n_wl])
    d1_ok = np.isfinite(v_d1) and jac_w == 0.0
    print(f"  {'OK  ' if d1_ok else 'FAIL'} d1 spec fct on flat x -> "
          f"{v_d1:.6f}, width jac {jac_w:+.3g} (pure-T under wg_project)")
    ok &= d1_ok
except Exception as e:
    print(f"  FAIL d1 spec fct raised {e!r}")
    ok = False
drv2 = inspect.getsource(eng.run_projected)
ns2_ok = ("gLam_vec = gLam" in drv2 and "_ns2_step" in inspect.getsource(eng)
          and "if ns2:" in drv2 and "gW = gW + chain" in drv2)
print(f"  {'OK  ' if ns2_ok else 'FAIL'} run_projected keeps gLam as its own "
      f"row under ns2 (chain not folded into gW)")
ok &= ns2_ok

# 7. wgp_reuse_k (2026-08-31): the width-row reuse must (a) default OFF,
# (b) guard BOTH the width-adjoint read and the gfields_W pass on
# spec._wg_reuse, (c) substitute the STORED gW in the driver, and (d) mark
# any reject as dirty so the next iterate refreshes.
r_ok = eng.CampaignSpec().wgp_reuse_k == 0
i0 = mod_src.index("def _compute_adjoint_fields_phased")
adj_src = mod_src[i0:i0 + 1500]
r_ok &= '_wg_reuse' in adj_src and "[0.0] * len(" in adj_src
i0 = mod_src.index("def calculate_gradient_fields")
cgf2 = mod_src[i0:mod_src.index("return gT", i0)]
r_ok &= '_wg_reuse' in cgf2 and "gfields_W = None" in cgf2
r_ok &= ('acc.get("gW_raw")' in drv2 and "reuse_dirty = True" in drv2
         and "_queue_adjoint_jobs_filtered" in drv2
         and drv2.count("reuse_dirty = True") >= 2)
print(f"  {'OK  ' if r_ok else 'FAIL'} wgp_reuse_k: default off, zero-read + "
      f"pass-skip guarded, stored-gW substitution, rejects mark dirty")
ok &= r_ok

# 8. BEHAVIORAL test of the filtered adjoint queue (audit #6: string greps
# cannot fail on the modes that matter). Fake project/fom/runner; assert the
# width job is queued when _wg_reuse is False and absent when True.
from types import SimpleNamespace


def _fake(width_flag_reuse):
    jobs = []
    adjs = [{"sim_result": SimpleNamespace(monitor_name="Port_2"),
             "adj_filename": "a_port.fsp"},
            {"sim_result": SimpleNamespace(monitor_name="field_profile_adj"),
             "adj_filename": "a_width.fsp"}]
    fom = SimpleNamespace(
        config_map=SimpleNamespace(config_map={None: {"adjoints": adjs}}),
        _is_width=lambda sr: sr.monitor_name == "field_profile_adj")
    proj = SimpleNamespace(
        fom=fom,
        _verify_simulation_files_exist=lambda fns: None,
        runner=SimpleNamespace(
            add_job=lambda task, label: jobs.append(label)))
    sp = SimpleNamespace(_wg_reuse=width_flag_reuse)
    eng._queue_adjoint_jobs_filtered(proj, sp)
    return jobs


j_fresh, j_reuse = _fake(False), _fake(True)
beh_ok = (any("field_profile_adj" in j for j in j_fresh)
          and not any("field_profile_adj" in j for j in j_reuse)
          and any("Port_2" in j for j in j_reuse))
print(f"  {'OK  ' if beh_ok else 'FAIL'} filtered queue BEHAVIOR: width job "
      f"queued fresh ({len(j_fresh)}) / skipped on reuse ({len(j_reuse)}), "
      f"port always queued")
ok &= beh_ok
# near-null guard: rho_T below 0.02 must shrink the step proportionally
gT9 = np.random.default_rng(3).standard_normal(191)
gW9 = np.random.default_rng(4).standard_normal(191)
gL9 = np.random.default_rng(5).standard_normal(191)
D9 = np.ones(191)
# build gT nearly inside span{gW,gL}: tiny orthogonal residue
A9 = np.stack([gW9, gL9], axis=1)
M9 = A9.T @ A9
gT_in = A9 @ np.linalg.solve(M9, A9.T @ gT9)
resid = gT9 - gT_in
gT_null = gT_in + 1e-3 * resid          # rho_T ~ 1e-3 * |resid|/|gT| << 0.02
s_null, _, d_null = eng._ns2_step(gT_null, gW9, gL9, D9, 18.35, 18.35, 0.1,
                                  1566.0, 1566.0, 0.05, 10.0)
nn_ok = (d_null["rho_T"] < 0.02
         and float(np.max(np.abs(s_null))) < 10.0 * (d_null["rho_T"] / 0.02)
         * 1.01 + 1e-9)
print(f"  {'OK  ' if nn_ok else 'FAIL'} near-null guard: rho_T "
      f"{d_null['rho_T']:.2e} -> step inf-norm "
      f"{float(np.max(np.abs(s_null))):.3g} nm (< cap*rho/0.02)")
ok &= nn_ok

print("\n" + ("ALL PASS" if ok else "*** GATE FAILED ***"))
raise SystemExit(0 if ok else 1)
