"""SECOND-TEETH ANTENNA on the 15.5-um Nt80x: a small sub-corrugation superimposed on the sidewalls near the pi shift.

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Idea (user 2026-10-20): ADD geometry, not re-shape it -- a second set of small teeth on top of the existing teeth, periodic or
not, acting as an antenna that radiates against the leak. Physics: a sub-corrugation of period L2 scatters the +-beta guided waves
into u_x = -+(n_eff - lam/L2)/n_clad (first order). L2 = 578 nm aims at the grazing lobes |u_x| ~ 0.85 that carry ~50 % of the
Nt80x leak (n_eff 1.575, lam 1619.5, n_clad 1.444); L2 = 1000 nm aims broadside. Depth sets the amplitude, the offset dx the
phase, the extent the angular width; L2 has no 2*beta component, so the Bragg coupling is untouched at first order. Co-located
with the leak (same sidewalls), so whole-sphere cancellation is possible in principle -- unlike cladding teeth.
Realisation: rect SiN sites (index n_core) 120 nm along x, SNAPPED to the centre of the nearest WIDE half-period (257 nm flat
wall, so the footprint never straddles a tooth step; the model's positions move by <= 130 nm), 60 nm tall in y of which only the PROTRUSION d sticks out of the
local sidewall (the rest sits inside the core: y_centre = half_width(x) + d - 30 nm), mirrored +-y, both arms.
REVISED 2026-10-20 after two reviews (GPT 01a158c6 + the first-order antenna model, scratch subteeth_model.py): 55-nm sub-teeth
would radiate 2-4x the leak and shift lambda by ~1.3 nm (outside the +-0.25 nm far-field window); a half-period shift is not a
negative depth and a quarter shift is not quadrature; L2 = 1000 nm is near-commensurate with 2*beta (+53 % kappa). Rows now:
  A  periodic L2 578 nm, 7/side at |x| = 1.0 + k*L2, protrusion 5 nm      (EXPECTED |b|^2/|c|^2 ~0.03, dlam ~ +0.13 nm)
  B  same sites, protrusion 2.5 nm                                        (depth linearity: b_A vs 2 b_B)
  C  APERIODIC model design, 10 pairs (x_k nm, d_k nm) mirrored +-x +-y: (557,7) (2870,1.5) (3770,3) (4027,6.5) (4155,2)
     (4412,7.5) (4926,3.5) (5440,3.5) (5568,3) (5697,4)   -- first-order prediction P/Pc 0.948, dlam +0.14, W-neutral
  D  the same design at half depth                                        (direction + linearity test of C)
  E  DESIGN 2 (model, arms independent, 12 sites, d <= 10 nm; predicted P/Pc 0.860, dlam +0.15 nm, W-neutral)
  F  design 2 at half depth                                                (predicted 0.921)
Responses b = row - control (265046) per stored energy (cavity-trace reference); acceptance = total P/U, Q_i, T, lambda, W.
Dispatch (Athena, 7 seats/row on A100; submit held if the pool is above 33 in use):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_subteeth --max-concurrent=2 --qos=4d_1g --gpu=a100
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "C_NT80"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um import ROWS, BASE as BASE_C, N_SIDE, zmult, PITCH_ITAI_NM  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

P_UM = PITCH_ITAI_NM * 1e-3
LCAV_UM = P_UM / 2                                    # cavity segment length = pitch/2 (bragg_device default)
NAR0, WID0 = ROWS[0][5]                               # per-period narrow/wide widths (nm), index 0 = first period
SITE_LEN_NM, SITE_Y_SPAN_NM = 120.0, 60.0                 # rect size; the protrusion d_k is what sticks out of the wall
X0_UM, N_SITES, L2_NM = 1.0, 7, 578.0
APERIODIC = [(557, 7.0), (2870, 1.5), (3770, 3.0), (4027, 6.5), (4155, 2.0), (4412, 7.5), (4926, 3.5), (5440, 3.5), (5568, 3.0), (5697, 4.0)]
DESIGN2 = [(+514, 10.0), (+1028, 10.0), (+2056, 2.0), (+3598, 10.0), (+4112, 10.0), (+6682, 10.0),
           (-257, 10.0), (-771, 5.5), (-3341, 10.0), (-3855, 10.0), (-4369, 10.0), (-4883, 10.0)]   # signed x (arms differ), <=1-nm sites dropped
VARIANTS = [("A_L578_d5", 1.0), ("B_L578_d2p5", 0.5), ("C_aperiodic_x1", 1.0), ("D_aperiodic_x0p5", 0.5),
            ("E_design2_x1", 1.0), ("F_design2_x0p5", 0.5)]


def half_width_nm(x_um):
    """Local core half-width at |x|: right arm draws narrow then wide within each period, left arm wide then narrow."""
    s = abs(x_um) - LCAV_UM / 2
    if s < 0:
        return 950.3 / 2
    j = int(s // P_UM); frac = (s - j * P_UM) / P_UM
    narrow_first = x_um > 0
    w = (NAR0[j] if frac < 0.5 else WID0[j]) if narrow_first else (WID0[j] if frac < 0.5 else NAR0[j])
    return w / 2


def snap_to_wide_half(x_um):
    """Centre of the WIDE half-period nearest to |x| (keeps the 120-nm footprint on one flat wall; GPT plan item 2).
    Right arm (x>0): narrow then wide within each period -> wide-half centre at s = (j + 0.75) P; left arm: wide first -> (j + 0.25) P."""
    sgn = 1 if x_um >= 0 else -1
    s = abs(x_um) - LCAV_UM / 2
    j = max(0, int(round((s - (0.75 if sgn > 0 else 0.25) * P_UM) / P_UM)))
    return sgn * (LCAV_UM / 2 + (j + (0.75 if sgn > 0 else 0.25)) * P_UM)


def sites(name, scale):
    """(x list nm, y list nm): sites snapped to wide-half centres; y_centre = local half width + protrusion - y_span/2."""
    if name.startswith(("A", "B")):
        elems = [(X0_UM * 1e3 + k * L2_NM, 5.0) for k in range(N_SITES)]
    elif name.startswith(("E", "F")):
        elems = list(DESIGN2)                             # signed x: each entry is ONE site on its own arm
    else:
        elems = list(APERIODIC)
    xs, ys = [], []
    arms = ((+1,),) if name.startswith(("E", "F")) else ((+1,), (-1,))
    for (sgn,) in arms:
        used = set()
        for x_nm, d_nm in elems:
            x_um = snap_to_wide_half(sgn * x_nm * 1e-3)
            sg = 1 if x_um >= 0 else -1
            while round(x_um, 4) in used:                 # distinct wide halves: step one period outward (2026-10-20 fix)
                x_um = snap_to_wide_half(x_um + sg * P_UM)
            used.add(round(x_um, 4))
            xs.append(x_um * 1e3)
            ys.append(half_width_nm(x_um) + scale * d_nm - SITE_Y_SPAN_NM / 2)
    assert len(set(round(v, 1) for v in xs)) == len(xs), "duplicate sites"
    return xs, ys


BASE = copy.deepcopy(BASE_C)
BASE.scatterer.enabled = True
BASE.scatterer.shape = "rect"
BASE.scatterer.mirrored_y = True
seed = ROWS[0]
XL = [sites(name, sc) for name, sc in VARIANTS]
SPEC = SweepSpec(
    polarization              = [seed[0]] * len(VARIANTS),
    pitch_nm                  = [seed[1]] * len(VARIANTS),
    corrugation_depth_nm      = [seed[2]] * len(VARIANTS),
    avg_width_nm              = [seed[3]] * len(VARIANTS),
    cavity_width_nm           = [seed[4]] * len(VARIANTS),
    width_narrow_per_tooth_nm = [seed[5][0]] * len(VARIANTS),
    width_wide_per_tooth_nm   = [seed[5][1]] * len(VARIANTS),
    n_periods_each_side       = [N_SIDE] * len(VARIANTS),
    center_wavelength_nm      = [seed[6]] * len(VARIANTS),
    scan_width_nm             = [seed[7]] * len(VARIANTS),
    y_span_um                 = [seed[8]] * len(VARIANTS),
    span_mult                 = [zmult(seed[9], seed[6])] * len(VARIANTS),
    scatterer_shape           = ["rect"] * len(VARIANTS),
    scatterer_x_span_um       = [SITE_LEN_NM * 1e-3] * len(VARIANTS),
    scatterer_y_span_nm       = [SITE_Y_SPAN_NM] * len(VARIANTS),
    scatterer_index           = [1.97] * len(VARIANTS),
    scatterer_mirrored_y      = [True] * len(VARIANTS),
    scatterer_x_list_nm       = [xl[0] for xl in XL],
    scatterer_y_list_nm       = [xl[1] for xl in XL],
    mode  = "zipped",
    label = "farfield_sph_15um_subteeth",
)

if __name__ == "__main__":
    for (name, sc), cfg in zip(VARIANTS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        xs = [v * 1e6 for v in kw["scatterer_x_list_m"]]; ys = [v * 1e9 for v in kw["scatterer_y_list_m"]]
        prot = [y + SITE_Y_SPAN_NM / 2 - half_width_nm(x) for x, y in zip(xs, ys)]
        edge_prot = [max(y + SITE_Y_SPAN_NM / 2 - half_width_nm(x + e) for e in (-0.06, 0.06)) for x, y in zip(xs, ys)]
        assert max(edge_prot) - max(prot) < 1e-6, ("footprint straddles a tooth step", max(edge_prot))
        print(f"{name}: {len(xs)} sites, x {xs[0]:.3f}..{xs[len(xs)//2-1]:.3f} / {xs[len(xs)//2]:.3f}.. um, protrusion {min(prot):.1f}-{max(prot):.1f} nm (edges {max(edge_prot):.1f}), "
              f"rect {kw['scatterer_x_span_m']*1e9:.0f}x{kw['scatterer_y_span_m']*1e9:.0f} nm n {kw.get('scatterer_index')} mirrored {kw['scatterer_mirrored_y']} "
              f"| short {cfg.farfield.short_run} sim ps {os.environ.get('TM_SIM_TIME_PS')} box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f}")
