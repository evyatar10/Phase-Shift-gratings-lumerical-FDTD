"""TE PERIODIC pillar row (photonic-crystal-like comb) — aimed phase circle.

Study dir: runners/scatterers/   |   Created 2026-08-10   |   Job(s): TBD
Purpose (user, explicit): simulate the TE periodic tens-of-pillars device on
IGUM. Design from the measured TE map (IGUM 51469): TE radiates a BROAD double
hump peaking at |ux| ~ 0.75-0.8 (no grazing needle). The comb's first backward
order is AIMED at the hump center: Lambda = lambda_res/(n_eff + n_clad*0.75)
= 1559.4/(1.5594 + 1.083) = 590 nm (n_eff 1.5594 = lambda/2/pitch, MEASURED
lambda). No TE transit-phase calibration exists -> run the FULL 4-point phase
circle; a constant offset only rotates the circle.

Rows (zipped, 4): dx = 0 / 148 / 295 / 443  (0/90/180/270 deg of 590).
Posts: 41/row (tens, +-11.8 um), r=110, h350, d=1.5 um (TE gamma ~2.37/um ->
e^-gamma*d at 1.5 um matches the TM winner's drive attenuation at 1.8 um;
perpendicular-polarizability factor ~0.7 puts amplitude in the TM-proven class).
NO control row (CLAUDE.md section-6 no-rerun rule): anchor = TE ctrl T 0.8756 /
lambda 1559.389 / Q 1411, MEASURED TODAY at identical numerics on THIS cluster
(scat_z_teffmap row 0).

REGISTERED: the hump is broad (half-max spans ux -0.88..+0.92) — the aimed
beam (width ~0.045) can only carve a notch; a FLAT circle = no coherent
addressability -> TE periodic axis closed by direct measurement. GATE: any row
T > 0.8774 (ctrl + floor 0.0018) = lever exists -> refine amplitude/length.

Physics line (CLAUDE.md section 4): TE h350, pitch 500, corr 300, W800,
N=80/side; resonance ~1559.4, window 1549.0-1569.0 (20 nm / 1501), box y=16.

Dispatch (IGUM, user-named; license peak: 3 Athena + 2 here <= 6):
    SBATCH_MEM=160G ARRAY_TIME=03:00:00 bash igum/deploy_igum.sh \
        --option3 --spec=runners.scatterers.scat_te_comb --max-concurrent=2
Output -> results/scat_te_comb/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

TE_PITCH_NM    = 500.0
TE_CORR_NM     = 300.0
TE_TOOTH_EDGE  = 550.0            # corr-300 wide-tooth half-width (nm)
LAM_NM         = 590.0            # aims backward order at measured hump ux ~ -0.75
R_NM           = 110.0
D_NM           = 1500.0
H_NM           = 350.0
N_HALF         = 20               # 41 posts, +-11.8 um
DX_LIST        = [0.0, 148.0, 295.0, 443.0]   # 0/90/180/270 deg of 590

BOX_Y_UM       = 16.0
SCAN_CENTER_NM = 1559.0
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 1501

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

def comb_x(dx_nm):
    return [round(k * LAM_NM + dx_nm, 1) for k in range(-N_HALF, N_HALF + 1)]

assert D_NM - R_NM >= TE_TOOTH_EDGE + _common.GAP_MIN_NM
assert D_NM + R_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0
assert N_HALF * LAM_NM + max(DX_LIST) + R_NM + 2000.0 <= 80 * TE_PITCH_NM
assert len(set(DX_LIST)) == len(DX_LIST)

SPEC = SweepSpec(
    polarization         = ["TE"] * len(DX_LIST),
    pitch_nm             = [TE_PITCH_NM] * len(DX_LIST),
    corrugation_depth_nm = [TE_CORR_NM] * len(DX_LIST),
    center_wavelength_nm = [SCAN_CENTER_NM] * len(DX_LIST),
    scatterer_radius_nm  = [R_NM] * len(DX_LIST),
    scatterer_x_list_nm  = [comb_x(dx) for dx in DX_LIST],
    scatterer_y_list_nm  = [[D_NM] * (2 * N_HALF + 1)] * len(DX_LIST),
    scatterer_height_nm  = [H_NM] * len(DX_LIST),
    mode  = "zipped",
    label = "scat_te_comb",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print("anchor (51469 row 0, SAME cluster+numerics): TE ctrl T 0.8756 | gate 0.8774")
