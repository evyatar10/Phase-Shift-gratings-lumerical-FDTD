"""Gate: per-tooth shift width row and the cw noise floor (opt-in knobs, 2026-10-11).

Study dir: runners/lumopt2_design/ | Created 2026-10-11 | Fable review of the shift machinery
  K1  defaults: no knob set -> gW untouched, curved flag still degrades (legacy behaviour)
  K2  scalar knob reproduces the old inline override on live shifts only
  K3  profile knob writes per-tooth values, NaN keeps the adjoint entry, frozen teeth untouched
  K4  profile wins over the scalar
  K5  cw remainder: curved cleared only if slope spread x dlam < remainder; a flat slope with
      strong curvature (GPT counterexample, built through cw_stats) stays bad; no stats -> bad
"""
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from runners.lumopt2_design import lumopt2_design as eng  # noqa: E402
from runners.lumopt2_design import campaign_te_s2 as s2  # noqa: E402

base = s2.SPEC_V3
L = eng.layout(base.n_free)
n = L.SL_SHIFT.stop - L.SL_SHIFT.start
rng = np.random.default_rng(3)
gW = rng.normal(size=L.N_PARAMS)
lo, hi = np.zeros(L.N_PARAMS), np.ones(L.N_PARAMS)
hi[L.SL_SHIFT.start + 7] = lo[L.SL_SHIFT.start + 7]          # tooth 8 frozen

g, tag = eng.shift_width_override(base, gW, lo, hi, L)
assert tag is None and np.array_equal(g, gW), "K1"
row_curved = {"cw_curved": True, "cw_um_per_nm": 1e-4}
assert eng.cw_is_bad(base, row_curved, True, None, 0.1) is True, "K1 curved"

sc = dataclasses.replace(base, wgp_gw_shift_um_per_nm=-0.0004)
g, tag = eng.shift_width_override(sc, gW, lo, hi, L)
exp = gW.copy()
exp[L.SL_SHIFT] = np.where((hi - lo)[L.SL_SHIFT] > 0.01, -0.0004, gW[L.SL_SHIFT])
assert tag == -0.0004 and np.array_equal(g, exp), "K2"

prof = np.full(n, np.nan)
prof[1:6] = -0.0005
prof[6:10] = 0.0008
pr = dataclasses.replace(base, wgp_gw_shift_profile=tuple(prof))
g, tag = eng.shift_width_override(pr, gW, lo, hi, L)
s0 = L.SL_SHIFT.start
assert tag == "profile", "K3 tag"
assert np.allclose(g[s0 + 1:s0 + 6], -0.0005) and np.allclose(g[s0 + 8:s0 + 10], 0.0008), "K3 values"
assert g[s0 + 7] == gW[s0 + 7], "K3 frozen tooth untouched"
assert g[s0] == gW[s0] and np.array_equal(g[s0 + 10:L.SL_SHIFT.stop], gW[s0 + 10:L.SL_SHIFT.stop]), "K3 NaN keeps adjoint"
assert np.array_equal(np.delete(g, np.r_[L.SL_SHIFT]), np.delete(gW, np.r_[L.SL_SHIFT])), "K3 non-shift untouched"

both = dataclasses.replace(pr, wgp_gw_shift_um_per_nm=-0.0004)
assert np.array_equal(eng.shift_width_override(both, gW, lo, hi, L)[0], g), "K4"

fl = dataclasses.replace(base, wgp_cw_remainder_um=0.02)
wl = np.linspace(1559.464, 1561.464, 501)
i_pk, m = 240, 12


def stats(f):
    st = eng.cw_stats(lambda j: f(wl[j] - wl[i_pk]), wl, i_pk, m)
    st.pop("spans")
    return st


# GPT counterexample: W = W0 + 20 dl^2 -> every central slope is 0, yet 0.2 um at 0.1 nm
quad = stats(lambda d: 19.6 + 20.0 * d * d)
assert abs(quad["cw_um_per_nm"]) < 1e-9 and quad["cw_spread_um_per_nm"] < 1e-9, "K5 quad slopes vanish"
quad["cw_curved"] = True                       # force the flag: the remainder must still refuse
assert eng.cw_is_bad(fl, quad, True, None, 0.1) is True, "K5 symmetric quadratic stays bad"
# nearly linear, small curvature: remainder 0.5*0.2*0.01 = 0.001 um -> cleared
lin = stats(lambda d: 19.6 + 0.3 * d + 0.2 * d * d)
lin["cw_curved"] = True
assert eng.cw_is_bad(fl, lin, True, None, 0.1) is False, "K5 small remainder -> ok"
assert eng.cw_is_bad(fl, {"cw_curved": True, "cw_um_per_nm": 0.0}, True, None, 0.1) is True, "K5 no stats -> bad"
assert eng.cw_is_bad(fl, {"cw_curved": False, "twin_lam_nm": 1560.0}, True, 1560.5, 0.1) is True, "K5 twin"
assert eng.cw_is_bad(fl, lin, False, None, None) is True, "K5 non-v3 (no dlam) stays as flagged"
assert eng.cw_is_bad(base, lin, True, None, 0.1) is True, "K5 knob off -> legacy (flag kept)"
print("GATE width_row_knobs PASS  K1-K5")
