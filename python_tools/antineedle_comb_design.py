# antineedle_comb_design.py
# Study: anti-needle comb design (zero-GPU), 2026-08-09.
# Data: scat_h_retrocomb ctrl (job 123563) + scat_o_comb1800 full-depth comb
#       (job 125285, "green": Lambda=551nm d=1.8um H12000) + scat_c ctrl (120976).
# Purpose: predict how much of the grazing-needle radiation a cladding SiN comb
#   can cancel when its period aims the carrier-out-coupled beam AT the needle
#   (Friedrich-Wintgen-style two-channel destructive interference), vs period,
#   comb length, and comb x-shift. Calibrated on the measured green run.
#   Saves curves to docs/antineedle_comb_design.mat for the MATLAB figure
#   (matlab_plotting/plot_antineedle_design.m) and prints the design table.

import numpy as np
import scipy.io as sio

ROOT = "results_from_athena"
LAM_UM = 1.558972          # green resonance [um]; ctrl 1558.612 (+0.36nm loading)
N_CLAD = 1.444
KAPPA = 0.0446             # measured envelope decay [1/um] (theory gate 2026-07-08)
GREEN_LAM_NM = 551.0
UY_BAND = 0.35             # needle uy extent used everywhere
WIN_UX = (-1.0, -0.90)     # needle window, negative side (positive side symmetric)

load = lambda f: sio.loadmat(f, squeeze_me=True, struct_as_record=False)
ctrl = load(f"{ROOT}/scat_h_retrocomb/results/result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat")
grn = load(f"{ROOT}/scat_o_comb1800/results/"
           "result_N80_TM_avg_Ybox16p0_Zbox8p8_scR110_arr151_"
           "X-41325to41325_Y1800to1800_C400_pair_H12000_ff.mat")

ux = ctrl["farfield_side"].ux
uy = ctrl["farfield_side"].uy
Ec = ctrl["farfield_side"].Ez_c            # rows = ux (95% of E2 is Ez)
dE = grn["farfield_side"].Ez_c - Ec        # comb-radiated field (1st order)
k0 = 2 * np.pi / LAM_UM

# empirical n_eff from the green beam angle: n_eff = lam/Lambda - n_clad*|ux_beam|
band = np.abs(uy) <= UY_BAND
prof = lambda E, m: np.sqrt((np.abs(E[m, :][:, band]) ** 2).sum(1))
m_beam = (ux >= -0.97) & (ux <= -0.85)
p = prof(dE, m_beam) ** 2
ux_beam = (ux[m_beam] * p).sum() / p.sum()
n_eff = LAM_UM / (GREEN_LAM_NM / 1000) - N_CLAD * abs(ux_beam)

win_x = (ux >= WIN_UX[0]) & (ux <= WIN_UX[1])
win_y = np.abs(uy) <= UY_BAND
E_n = Ec[np.ix_(win_x, win_y)]
P_n = (np.abs(E_n) ** 2).sum()

m_g = (ux >= -0.95) & (ux <= -0.90)        # comb-beam uy envelope, measured
g_uy = np.sqrt((np.abs(dE[m_g, :]) ** 2).mean(0))
g_uy /= g_uy.max()

def comb_field(lam_nm, L_um, apod=False):
    """Model comb-radiated field over the needle window (shape only)."""
    lam_g = lam_nm / 1000
    n = max(3, int(round(L_um / lam_g)))
    xs = (np.arange(n) - (n - 1) / 2) * lam_g
    w = np.exp(-KAPPA * np.abs(xs)) if apod else np.ones(n)
    kx = k0 * n_eff - k0 * N_CLAD * ux[win_x] - 2 * np.pi / lam_g
    af = (w[None, :] * np.exp(1j * np.outer(kx, xs))).sum(1)
    return af[:, None] * g_uy[None, win_y]

def cancel(lam_nm, L_um, apod=False):
    """Needle-power fraction removable at optimal complex amplitude."""
    B = comb_field(lam_nm, L_um, apod)
    return abs((np.conj(B) * E_n).sum()) ** 2 / ((np.abs(B) ** 2).sum() * P_n)

# validation: green geometry beam center must reproduce the measurement
p = (np.abs(comb_field(GREEN_LAM_NM, 83.2)) ** 2).sum(1)
ux_model = (ux[win_x] * p).sum() / p.sum()
print(f"n_eff (empirical) {n_eff:.4f}; green beam model {ux_model:.3f} vs measured {ux_beam:.3f}")

table = [(GREEN_LAM_NM, 83.2, False), (539, 83.2, False), (539, 17, False),
         (539, 30, False), (539, 17, True)]
for lam_nm, L, ap in table:
    print(f"Lambda={lam_nm:.0f}nm L={L:4.1f}um apod={ap} -> cancel {100*cancel(lam_nm, L, ap):5.1f}%")

lam_scan = np.arange(528, 553, 1.0)
c17 = np.array([cancel(l, 17) for l in lam_scan])
c83 = np.array([cancel(l, 83.2) for l in lam_scan])
best = lam_scan[np.argmax(c17)]

# x-shift (phase) curve at the scan optimum, optimal |w|
B = comb_field(best, 17)
w_opt = -(np.conj(B) * E_n).sum() / (np.abs(B) ** 2).sum()
dx = np.linspace(0, best / 1000, 121)
P_dx = [(np.abs(E_n + w_opt * np.exp(2j * np.pi * d / (best / 1000)) * B) ** 2).sum() / P_n
        for d in dx]
print(f"best Lambda {best:.0f}nm: needle power x{min(P_dx):.2f} (best dx) .. x{max(P_dx):.2f} (worst dx)")

# model beam profile at design point for the figure
p_design = (np.abs(comb_field(best, 17)) ** 2).sum(1)
sio.savemat("docs/antineedle_comb_design.mat", {
    "ux_win": ux[win_x], "ux": ux,
    "prof_needle": prof(Ec, np.ones_like(ux, bool)),
    "prof_green": prof(dE, np.ones_like(ux, bool)),
    "prof_design": np.sqrt(p_design),
    "lam_scan": lam_scan, "cancel_L17": c17, "cancel_L83": c83,
    "dx_nm": dx * 1000, "P_dx": np.array(P_dx),
    "best_lambda_nm": best, "n_eff": n_eff, "ux_beam_green": ux_beam,
})
print("saved docs/antineedle_comb_design.mat")
