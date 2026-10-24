"""Compact tooth-group BUMP knobs on the 20-um overshoot: measured radiation response basis for a linear cancellation solve.

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Why (MEASURED 2026-10-20, Athena 265078-80 vs 182131): a 20-nm mean-width ripple of period 4.7 um over teeth 4-40
radiates 3.3-4x the whole leak but overlaps it by only eta^2 = 2.4 % -- its fringes are narrow (90 % at grazing)
while the leak is broad; the band-limited inverse FT of the leak puts 71 % of its source within |x| <= 5 um.
So the knobs are COMPACT and CENTRAL: mean-width bumps (same dw on the wide and narrow tooth, both arms) on
period-index groups {0,1} {2,3} {4,5} {6,7} {8,9} {10,11} (index 0 = the first period beside the cavity, centre
at (j+0.75) pitch), +5 nm one-sided (cards 0-5), plus {2,3} at -5 nm (card 6, +-pair linearity) and +2.5 nm
(card 7, half-step slope check, GPT review 2026-10-20).
Dispatched as labelled ExperimentCards (kind=cards): each task's outputs land under <module>/<label>, so the
identical auto tags (per-tooth lists are not hashed) never collide and no tag-only geometry change is needed.
Numerics = round C exactly: BASE is the fully expanded round-C seed config (farfield_sph_20um with FF_SPH_ROUND=C:
conformal, 150 ps, shutoff off, box 6.8/6.81, window 1559.77 +-2 nm, tube E/H); cards override only the tooth
lists. Control = 182131 seed row (results_from_athena/farfield_sph_20um_c).
Analysis: scratch bump_solve.py: b_j = c_j - c (per stored energy, Lorentzian-at-lam_ff), G alpha = -h with
lambda/W held (null-space), eta^2 = removable share; then ONE validation run at the solved profile.
Dispatch (Athena):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --cards=runners.sweeps.farfield_sph_20um_bumps --max-concurrent=4
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF_SPH_ROUND"] = "C"
from runners.sweeps.farfield_sph_20um import SPEC as SPEC_C, BASE as BASE_C, N_SIDE  # noqa: E402  (round-C numerics)
from runners.sweeps import itai_hh_nt60w20 as hh  # noqa: E402
from experiment_card import ExperimentCard  # noqa: E402

BUMP_NM = 5.0
GROUPS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
KNOBS = [(g, +BUMP_NM) for g in GROUPS] + [((2, 3), -BUMP_NM), ((2, 3), +BUMP_NM / 2)]


def bump_teeth(teeth_idx, dw_nm):
    nar, wid = (list(v) for v in hh.teeth(N_SIDE))
    for d in teeth_idx:
        nar[d] += dw_nm
        wid[d] += dw_nm
    return nar, wid


BASE = next(iter(SPEC_C.expand(base=BASE_C)))          # the seed row, fully expanded (pitch, widths, box, window, far field)
CARDS = [ExperimentCard(width_narrow_per_tooth_nm=bump_teeth(g, dw)[0], width_wide_per_tooth_nm=bump_teeth(g, dw)[1],
                        label=f"g{g[0]:02d}{g[1]:02d}_{'p' if dw > 0 else 'm'}{abs(dw):g}nm")
         for g, dw in KNOBS]

if __name__ == "__main__":
    nar0, wid0 = hh.teeth(N_SIDE)
    for i, ((g, dw), card) in enumerate(zip(KNOBS, CARDS)):
        cfg = card.to_config(base=BASE)
        nar = [v * 1e9 for v in cfg.grating.width_narrow_per_tooth_m]; wid = [v * 1e9 for v in cfg.grating.width_wide_per_tooth_m]
        diff = [d for d in range(N_SIDE) if abs(nar[d] - nar0[d]) > 1e-6 or abs(wid[d] - wid0[d]) > 1e-6]
        assert diff == list(g), diff
        assert all(abs((nar[d] - nar0[d]) - (wid[d] - wid0[d])) < 1e-6 for d in range(N_SIDE))
        print(f"card {i} {card.label:12s}: teeth {g} dw {dw:+.1f} | pitch {cfg.grating.pitch_m*1e9:.2f} N {cfg.grating.n_periods_each_side} "
              f"| win {cfg.spectral.center_wavelength_m*1e9:.2f}+-{cfg.spectral.scan_width_nm/2:.0f} x{cfg.spectral.n_wl_points} | box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} "
              f"| shutoff {cfg.mesh.auto_shutoff_min} eh {cfg.farfield.save_surface_eh} ffpts {cfg.farfield.farfield_freq_points} | sim ps {os.environ.get('TM_SIM_TIME_PS')} | pol {cfg.source.polarization}")
