"""Mean-width RIPPLE probe on the 20-um overshoot: the radiation response to one complex knob.

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Purpose (user 2026-10-20: cancel the overshoot's radiation with an ADDED structure): the leak of
the overshoot device is a line source along x; 43 % of it sits at grazing |u_x| 0.8-0.9 (DERIVED
from the round-C tube far field, Athena 182131). A slow MEAN-width ripple, the same dw added to the
wide and the narrow tooth of each period, is co-located with that source (same y, z -> the same
azimuthal pattern), leaves the wide-narrow contrast untouched, and scatters the +-beta guided
waves into the cone at u_x = (beta -+ 2*pi/period)/k_clad: period 4.7 um -> u_x = +-0.85.
One ripple = one complex knob q = A*exp(i*phi). Three rows measure its response b and linearity:
phi = 0 and 180 deg (+h / -h: changes must be equal and opposite), phi = 90 deg (imaginary part).
Control = the round-C seed row (results_from_athena/farfield_sph_20um_c, T 0.98327, Q_L 7736,
radiated 0.70 % of input) at IDENTICAL numerics: this module imports round C's ROWS/BASE and only
replaces the per-tooth width lists. Verdict per knob: q* = -Re<b,c>/|b|^2, eta^2 = removable share
of the leak (whole sphere, per stored energy), plus the measured dW/dq and dlambda/dq.
Tag trap: generate_file_tag hashes only the first/last tooth, which the ripple does not touch
(window = teeth 4-40, raised-cosine ramps of 4 teeth) -> one phase per MODULE (results dir =
module name): farfield_sph_20um_ripple (0 deg), _ripple180, _ripple90.
Amplitude 20 nm: zero-DC, so the first-order resonance shift averages out (local dn_eff 3e-3,
phase error over a half period 0.03 rad); the lumopt2 TE campaign had these average widths free
inside a +-25 nm box and gained +0.0006 T in 8 evals -> this probe measures the RADIATION
response directly, it does not assume the optimizer missed it.
Dispatch (Athena, 1 task per module, same memory as round C):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_20um_ripple --max-concurrent=1
Output -> results/farfield_sph_20um_ripple*/results/ ; analyse with python_tools/farfield_multipole.py --lmax 150.
"""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF_SPH_ROUND"] = "C"
from runners.sweeps.farfield_sph_20um import ROWS, BASE, N_SIDE, zmult  # noqa: E402  (round-C numerics)
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402
from runners.sweeps import itai_hh_nt60w20 as hh  # noqa: E402

RIPPLE_NM  = 20.0
PERIOD_UM  = 4.7                                   # -> u_x = +-0.85, the grazing lobes
PHASE_DEG  = float(os.environ.get("FF_RIPPLE_DEG", "0"))
TOOTH_LO, TOOTH_HI, RAMP = 4, 40, 4                # window in tooth index (0 = cavity segment)


def window(d):
    """Raised-cosine window, 1 on teeth LO+RAMP..HI-RAMP, 0 outside LO..HI."""
    if d < TOOTH_LO or d > TOOTH_HI:
        return 0.0
    if d < TOOTH_LO + RAMP:
        return 0.5 * (1 - math.cos(math.pi * (d - TOOTH_LO) / RAMP))
    if d > TOOTH_HI - RAMP:
        return 0.5 * (1 - math.cos(math.pi * (TOOTH_HI - d) / RAMP))
    return 1.0


def ripple_teeth(n_side):
    """Itai's per-tooth (narrow, wide) nm with the mean-width ripple added to BOTH."""
    nar, wid = (list(v) for v in hh.teeth(n_side))
    for d in range(n_side):
        x_um = d * hh.PITCH_NM * 1e-3                 # tooth d sits ~d pitches from the cavity
        dw = RIPPLE_NM * window(d) * math.cos(2 * math.pi * x_um / PERIOD_UM + math.radians(PHASE_DEG))
        nar[d] += dw
        wid[d] += dw
    return nar, wid


seed = ROWS[0]                                         # round-C overshoot seed row
ROW = seed[:5] + (ripple_teeth(N_SIDE),) + seed[6:]
SPEC = SweepSpec(
    polarization              = [ROW[0]],
    pitch_nm                  = [ROW[1]],
    corrugation_depth_nm      = [ROW[2]],
    avg_width_nm              = [ROW[3]],
    cavity_width_nm           = [ROW[4]],
    width_narrow_per_tooth_nm = [ROW[5][0]],
    width_wide_per_tooth_nm   = [ROW[5][1]],
    n_periods_each_side       = [N_SIDE],
    center_wavelength_nm      = [ROW[6]],
    scan_width_nm             = [ROW[7]],
    y_span_um                 = [ROW[8]],
    span_mult                 = [zmult(ROW[9], ROW[6])],
    mode  = "zipped",
    label = "farfield_sph_20um_ripple",
)

if __name__ == "__main__":
    nar0, wid0 = hh.teeth(N_SIDE)
    nar, wid = ROW[5]
    print(f"phase {PHASE_DEG:.0f} deg, period {PERIOD_UM} um, A {RIPPLE_NM} nm, pitch {hh.PITCH_NM}")
    print(" d   x_um   dw_nm   narrow   wide")
    for d in range(0, 44):
        print(f"{d:2d} {d*hh.PITCH_NM*1e-3:6.2f} {wid[d]-wid0[d]:+7.2f} {nar[d]:8.1f} {wid[d]:8.1f}")
    assert all(abs((a - a0) - (b - b0)) < 1e-9 for a, a0, b, b0 in zip(nar, nar0, wid, wid0)), "ripple must be equal on both teeth"
    assert all(a == a0 and b == b0 for a, a0, b, b0 in zip(nar[41:], nar0[41:], wid[41:], wid0[41:]))
    dc = sum(wid[d] - wid0[d] for d in range(N_SIDE)); print("DC sum of ripple (nm):", round(dc, 2))
    print("max wide tooth (nm):", max(wid), " min narrow tooth (nm):", min(nar))
    for i, cfg in enumerate(SPEC.expand(base=BASE)):
        print(f"task {i}: pitch {cfg.grating.pitch_m*1e9:.2f} N {cfg.grating.n_periods_each_side} window "
              f"{cfg.spectral.center_wavelength_m*1e9:.2f} +- {cfg.spectral.scan_width_nm/2:.1f} nm x {cfg.spectral.n_wl_points} "
              f"| box y {cfg.y_span*1e6:.2f} z {cfg.z_span*1e6:.2f} | shutoff {cfg.mesh.auto_shutoff_min} "
              f"| surface_eh {cfg.farfield.save_surface_eh} | sim_time_env {os.environ.get('TM_SIM_TIME_PS')}")
