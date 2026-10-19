"""Far field of the ~15.5-um-mode pi-shift gratings at the LAB length N=96/side.

Study dir: runners/sweeps/   |   Created 2026-10-19   |   Job(s): see memory project_farfield_sph_15um.md
Purpose (user 2026-10-19): the 15.5-um family next to farfield_sph_20um; then harmonic cancellation.
PHASE A (this SPEC): the two LOW-Q plain devices, standard port-driven steady state (same pipeline,
ports, feeds and normalization as farfield_sph_20um round C -- verified 2026-10-19):
  0 TE plain corr 316 / pitch 500 / W800. corr from predict_q3db (te_q3db_c250 moved by the corr knob,
              width 15.5 um AT N=96 -- EXPECTED-grade for TE). EXPECTED: T 0.77, Q_L 2.9e3, W 15.5,
              lam ~1558.8 (-1.2 nm vs c250, FDE).
  1 TM plain corr 400 / pitch 516.83 / W800 (the anchored co-resonant TM default; knob says 402 for
              15.5 um at N=96). EXPECTED: T 0.77, Q_L 2.6e3, W ~15.5, lam ~1558.1 (N80/corr400
              Athena 128918 measured 1558.066).
Q_L ~3e3 -> amplitude decay 2Q/w ~5 ps -> 150 ps = ~30 decay times (complete). The regular pipeline
shows no port leak (round C R 5e-5 at resonance), so no straight-guide reference is needed.
The Itai overshoot tables below are for PHASE C (high-Q decay runs, needs the monitor-window engine knob);
they are NOT in this SPEC. Geometry: his bragg_lib chain, index correction ONLY, his target_lambda 1600,
pitch 514 (lab), N=96 -> bulk 750/1250, cavity 950.3 nm; gate + independent re-derivation 0.000 nm.
EXPECTED resonance at pitch 514 in our model ~1618 +-5 nm. Rows with identical scalar params get the
SAME file tag (no per-tooth hash) -> each overshoot needs its own module.

Numerics = round C of farfield_sph_20um (conformal, dx 50, fixed time, shutoff 1e-12, tube E/H):
TE box 6.8/6.81 (converged), TM 8.0/8.8 (stored TM numerics).

Dispatch:  SBATCH_MEM=200G bash athena/deploy_athena.sh --option3                --spec=runners.sweeps.farfield_sph_15um --max-concurrent=2
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from simulation_config import SimulationConfig

N_SIDE = 96
PITCH_ITAI_NM = 514.0       # Itai's lab pitch (user 2026-10-19), not our 1560 retune
BULK_NARROW_NM, BULK_WIDE_NM, CAVITY_W_NM = 750.0, 1250.0, 950.3   # phase C

# drawn widths, cavity-first d=1.., his chain with index correction ONLY at his target_lambda 1600;
# outer teeth = bulk. Gate + independent re-derivation from bragg_lib: 0.000 nm.
HH60_NARROW_NM = [
    950.3, 912.5, 894.5, 868.1, 843.0, 820.0, 798.5, 778.2, 759.0, 741.1,
    724.3, 708.6, 694.1, 680.6, 668.2, 656.6, 646.0, 636.2, 627.3, 619.1,
    611.9, 605.9, 601.8, 600.1, 600.1, 600.1, 601.9, 605.1, 614.7, 626.4,
    638.9, 652.4, 667.1, 682.9, 697.0, 708.3, 716.3, 720.3, 720.3, 716.0,
    707.8, 696.2, 682.3, 667.3, 653.3, 640.1, 627.7, 615.9, 604.5, 602.6,
    602.2, 602.1, 602.2, 602.6, 613.5, 634.6, 657.8, 684.1, 713.1, 736.2,
    750.0,
]
HH60_WIDE_NM = [
    950.3, 990.5, 1011.6, 1045.1, 1080.3, 1115.7, 1152.3, 1190.4, 1229.9, 1270.8,
    1312.9, 1356.3, 1400.5, 1445.2, 1490.3, 1535.5, 1580.2, 1623.8, 1665.7, 1705.1,
    1740.9, 1770.7, 1791.5, 1800.1, 1800.1, 1800.1, 1791.0, 1774.9, 1726.8, 1669.9,
    1611.5, 1552.7, 1494.4, 1437.3, 1391.1, 1357.1, 1334.6, 1323.5, 1323.7, 1335.2,
    1358.5, 1393.7, 1439.3, 1493.4, 1549.2, 1606.0, 1663.3, 1720.8, 1777.9, 1787.6,
    1789.4, 1789.9, 1789.3, 1787.6, 1732.6, 1631.0, 1530.8, 1433.1, 1343.3, 1282.7,
    1250.0,
]
NT80X_NARROW_NM = [
    950.3, 919.5, 898.2, 873.3, 850.9, 829.5, 809.2, 789.9, 771.6, 754.4,
    738.1, 722.9, 708.6, 695.3, 682.8, 671.2, 660.4, 650.4, 641.2, 632.7,
    625.0, 618.0, 611.9, 606.6, 602.4, 600.1, 600.1, 600.1, 601.6, 605.8,
    611.7, 619.7, 628.7, 638.5, 648.0, 657.3, 666.1, 674.1, 681.3, 684.8,
    682.5, 677.4, 671.9, 666.3, 658.3, 646.8, 635.9, 625.8, 616.3, 609.7,
    606.3, 605.5, 605.2, 605.3, 605.7, 608.4, 616.3, 633.3, 665.9, 719.4,
    789.1, 845.1, 856.4, 857.6, 858.7, 866.0, 873.5, 866.8, 840.5, 803.1,
    765.9, 733.0, 704.1, 678.7, 656.2, 633.7, 613.0, 612.6, 651.4, 714.4,
    760.1,
]
NT80X_WIDE_NM = [
    950.3, 982.6, 1007.1, 1038.3, 1068.8, 1100.7, 1133.8, 1168.1, 1203.6, 1240.2,
    1277.9, 1316.6, 1356.3, 1396.7, 1437.6, 1478.9, 1520.2, 1561.1, 1601.3, 1640.0,
    1676.6, 1710.5, 1740.9, 1767.4, 1788.5, 1800.1, 1800.1, 1800.1, 1792.3, 1771.5,
    1741.7, 1702.4, 1658.9, 1613.5, 1571.6, 1532.7, 1498.1, 1468.4, 1442.8, 1430.8,
    1438.8, 1456.6, 1476.4, 1497.5, 1528.8, 1576.9, 1625.1, 1672.5, 1718.7, 1751.5,
    1768.7, 1772.8, 1774.3, 1773.6, 1771.6, 1758.2, 1718.8, 1637.4, 1498.9, 1325.9,
    1169.5, 1077.2, 1061.1, 1059.5, 1057.9, 1047.9, 1038.0, 1046.9, 1084.0, 1144.3,
    1215.4, 1290.5, 1369.5, 1452.0, 1537.3, 1635.3, 1735.0, 1737.2, 1556.9, 1339.7,
    1227.7,
]


def pad(nar, wid, n_side=N_SIDE):
    k = n_side - len(nar)
    return list(nar) + [BULK_NARROW_NM] * k, list(wid) + [BULK_WIDE_NM] * k


# (pol, pitch, corr, avg, cavity, teeth, centre nm, window nm, box y, box z)
ROWS = [
    ("TE", 500.0, 316.0, 800.0, None, None, 1558.8, 10.0, 6.8, 6.81),
    ("TM", 516.83, 400.0, 800.0, None, None, 1558.1, 10.0, 8.0, 8.80),
]
os.environ["TM_SIM_TIME_PS"] = "150"   # fixed run length, read by bragg_device at build

# PHASE B (module farfield_sph_15um_b sets FF15_ROUND=B): GATE of the decay (ring-down) mode on the
# STORED 20-um overshoot = farfield_sph_20um round C seed row (Athena 182131_0: steady state at 150 ps,
# Q_L 7736, T 0.98327, tube 0.70 % of input, Q_rad 2.2e6, M(1,0) 3.5 %, <l> 15.0). Identical geometry,
# box, window, tube; only the run length (75 ps) and the monitor start window (t0 = 30 / 45 ps, one per
# task -> shifted-window stability) differ. PASS (python_tools/ringdown_budget.py): Q_L from the
# time_cavity decay 7736 +-5 %; decay branching P_tube/P_total = Q_L/Q_rad = 0.35 % (+-20 %);
# harmonic fractions vs the stored table (top 10 within 0.5 pt); t0=30 vs 45 agree to 5 %.
# RESULT 265002: Q_L decay PASS (7736), but the GPU solver DROPPED the window (t0 30/45 fields identical);
# the window knob was then REMOVED from the engine (2026-10-20). Re-running B now = a plain 75-ps short run.
ROUND = os.environ.get("FF15_ROUND", "A")
# PHASE B2 (module farfield_sph_15um_b2): 265002 showed the GPU solver ignores monitor apodization
# (identical fields at t0 30/45 ps; Q_L from the time trace = 7736 exactly). Same gate device, NO window,
# two short run lengths T1 19 ps / T2 26 ps (~1.5 / 2 amplitude decay times, like a 250-ps run of a
# Q~1e5 device). Tests: (i) Ansys's high-Q amplitude correction gamma = 1/(1-exp(-w T/2Q)) applied to
# the short-run harmonics / tube power / T vs the stored 150-ps steady state; (ii) the DIFFERENCE of
# the two runs' DFTs = an exact hard window [T1, T2] (FDTD is deterministic, identical up to T1).
if ROUND == "B2":
    SIM_PS = [19.0, 26.0]
# PHASE C (modules farfield_sph_15um_c_nt80 / _c_hh60: one device per module, because both draw the
# same scalar tag). B2 PASSED (265023): a plain run of ~2 amplitude decay times gives Q_L (decay fit),
# the harmonic table (0.01-0.03 pt) and gamma-corrected radiated power (0.4 %); T ~0.006 low.
# STEP "LOC" (60 ps, 1618 +-10 nm): locate the resonance (port peak + time-trace frequency).
# STEP "PROD" (~2 decay times, window +-0.5 nm around the located lambda so the 81 far-field points sit
# within half a linewidth): Q_L, harmonics, radiated power. Tube 100 um > device (+-49.5 um), so the
# end caps sit in the feed guide (+-50 um) and their flux is the guided output (ports at +-59.75 um).
# LOC DONE 2026-10-20 (Athena 265031 / 265032, 60 ps): time-trace fit from 30 ps vs port finder ->
#   Nt80 lam 1619.452 / 1619.460 nm, Q_L 9.91e4 (lab 120k); HH60 1619.396 / 1619.399 nm, Q_L 7.65e4.
# PROD: run = ~2 amplitude decay times 2Q/w (Nt80 170 ps, HH60 131 ps); source 4 nm (short pulse as in
# the B2 gate); far-field monitors + caps in their own +-0.25 nm window (use_2d_window) -> 81 far-field
# points every 6.25 pm, projection <= 0.2 linewidth from resonance (B2 gate passed at 1.5-2 decay times).
C_STEP = os.environ.get("FF15_C_STEP", "PROD")
C_LAM_NM = ({"C_NT80": 1618.0, "C_HH60": 1618.0} if C_STEP == "LOC"
            else {"C_NT80": 1619.455, "C_HH60": 1619.398})
C_SIM_PS = {"LOC": 60.0, "PROD": {"C_NT80": 340.0, "C_HH60": 265.0}.get(ROUND)}
if ROUND in ("C_NT80", "C_HH60"):
    nar, wid = (NT80X_NARROW_NM, NT80X_WIDE_NM) if ROUND == "C_NT80" else (HH60_NARROW_NM, HH60_WIDE_NM)
    ROWS = [("TE", PITCH_ITAI_NM, BULK_WIDE_NM - BULK_NARROW_NM, (BULK_WIDE_NM + BULK_NARROW_NM) / 2,
             CAVITY_W_NM, pad(nar, wid), C_LAM_NM[ROUND], 20.0 if C_STEP == "LOC" else 4.0, 6.8, 6.81)]
    os.environ["TM_SIM_TIME_PS"] = str(C_SIM_PS[C_STEP])
if ROUND in ("B", "B2"):
    from runners.sweeps import itai_hh_nt60w20 as hh
    _gate = ("TE", hh.PITCH_NM, round(hh.BULK_WIDE_NM - hh.BULK_NARROW_NM, 1), hh.AVG_WIDTH_NM,
             hh.CAVITY_W_NM, hh.teeth(98), 1559.77, 4.0, 6.8, 6.81)
    ROWS = [_gate, _gate]
    N_SIDE = 98
    _TASK = int(os.environ.get("SWEEP_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID") or 0)
    os.environ["TM_SIM_TIME_PS"] = str(SIM_PS[_TASK]) if ROUND == "B2" else "75"


def zmult(z_um, lam_nm):
    """span_multiplier giving z_span = core 0.35 um + mult * lambda_centre."""
    return round((z_um - 0.35) / (lam_nm * 1e-3), 2)


BASE = SimulationConfig()
BASE.spectral.n_wl_points = 2001
BASE.mesh.auto_shutoff_min = 1e-12          # fixed time, shutoff off (TE lane standing default)
BASE.monitors.record_2d_fields = False
BASE.farfield.enabled = True
BASE.farfield.save_complex = True
BASE.farfield.save_nearfield = False
BASE.farfield.save_surface_eh = True
BASE.farfield.farfield_x_span_m = 80e-6
BASE.farfield.ff_resolution = 401
BASE.farfield.farfield_freq_points = 81
if ROUND in ("C_NT80", "C_HH60"):
    BASE.farfield.short_run = True                 # end caps + time traces (skill highq-short-run)
    BASE.farfield.farfield_x_span_m = 100e-6
    if C_STEP == "PROD":     # source/ports keep the gate's short pulse (4 nm); far field + caps own +-0.25 nm
        BASE.farfield.use_2d_window = True
        BASE.monitors.monitor_2d_center_nm = C_LAM_NM[ROUND]
        BASE.monitors.monitor_2d_span_nm = 0.5
if ROUND in ("B", "B2"):
    BASE.farfield.short_run = True

SPEC = SweepSpec(
    polarization              = [r[0] for r in ROWS],
    pitch_nm                  = [r[1] for r in ROWS],
    corrugation_depth_nm      = [r[2] for r in ROWS],
    avg_width_nm              = [r[3] for r in ROWS],
    cavity_width_nm           = [r[4] for r in ROWS],
    width_narrow_per_tooth_nm = [None if r[5] is None else r[5][0] for r in ROWS],
    width_wide_per_tooth_nm   = [None if r[5] is None else r[5][1] for r in ROWS],
    n_periods_each_side       = [N_SIDE] * len(ROWS),
    center_wavelength_nm      = [r[6] for r in ROWS],
    scan_width_nm             = [r[7] for r in ROWS],
    y_span_um                 = [r[8] for r in ROWS],
    span_mult                 = [zmult(r[9], r[6]) for r in ROWS],
    mode  = "zipped",
    label = "farfield_sph_15um",
)

if __name__ == "__main__":
    for i, cfg in enumerate(SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        nar = kw.get("width_narrow_per_tooth_m")
        print(f"task {i}: {cfg.source.polarization} pitch {cfg.grating.pitch_m*1e9:.2f} "
              f"corr {cfg.geometry.corrugation_depth_m*1e9:.1f} N {cfg.grating.n_periods_each_side} "
              f"| {cfg.spectral.center_wavelength_m*1e9:.2f} +- {cfg.spectral.scan_width_nm/2:.1f} nm "
              f"x {cfg.spectral.n_wl_points} | box y {cfg.y_span*1e6:.2f} z {cfg.z_span*1e6:.2f} um "
              f"| ff {kw['farfield_freq_points']} pts | teeth "
              + (f"per-tooth d1 {nar[0]*1e9:.1f} d96 {nar[-1]*1e9:.1f}" if nar else "uniform")
              )
