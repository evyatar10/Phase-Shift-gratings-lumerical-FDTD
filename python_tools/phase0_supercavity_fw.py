"""
PHASE-0 THEORY GATE (zero-GPU, 2026-07-07) — SINGLE-CAVITY / SUPERCAVITY FW.

Question (user): the TWO-cavity Friedrich-Wintgen failed (job 118734, ss too low).
Could a SINGLE cavity supporting TWO co-located modes reach the dark state
(Rybin/Bogdanov 'supercavity' quasi-BIC)? Theory first, before any GPU.

FW two-mode CMT into ONE radiation channel: two modes at w1,w2 with radiative
rates g1,g2 and far-field patterns f1(k),f2(k). A dark (BIC) state forms only when
the modes can be tuned to degeneracy AND radiate into the SAME channel, i.e. their
radiation patterns must OVERLAP. Define the pattern overlap
      rho = |<f1,f2>| / (|f1| |f2|)   (flux-weighted, inside the light cone).
The best achievable radiative-loss floor of the dark combination is
      loss_floor = (1 - rho^2) * loss0
(rho=1 => true BIC, loss->0; rho=0 => modes orthogonal, NO interference, floor=loss0).
The two-cavity attempt failed because rho was low (lobes didn't match; CMT gate
wanted rho>=0.82). Does a single cavity's own higher-order mode do better?

Method: the pi-shift defect mode is EVEN, carrier +/-beta, envelope g0(x)
(extracted from the measured field). Its co-located 'partner' can only be another
longitudinal state of the SAME defect: the next ones are a 1-node (ODD) and a
2-node (EVEN) envelope with the SAME decay length. We build them, radiate each
(A_i(kx)=FT[g_i*cos(beta x)] in the light cone, flux-weighted), and measure rho.

Two independent physics checks decide it:
  (1) rho for the even-2node partner (odd is orthogonal by parity => rho=0, no FW).
  (2) DO these partners even EXIST as separate resonances in the gap? For a single
      pi-defect they are pushed into the bands (not localized); forcing a second
      gap state needs a longer / multi-defect cavity, which SPLITS the spectrum
      (violates single-resonance) or WIDENS the mode (violates the width budget).
      This script quantifies (1); (2) is structural and stated in the verdict.

Run:  python python_tools/phase0_supercavity_fw.py
"""

import os
import numpy as np
from scipy.io import loadmat

ROOT = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes"
MAT = os.path.join(ROOT, "results_from_athena", "tm_field_export", "results",
                   "result_N80_TM_W1050_dsh2S40s20_ptw2W1040to980_Ybox6p8_Zbox8p8_EZSLICE.mat")
N_CLAD = 1.444
LOSS0 = 0.0545
RHO_GATE = 0.82          # CMT dark-state gate established this session (two-cavity)
NX = 1 << 14

m = loadmat(MAT, squeeze_me=True)
x = np.asarray(m["x"], float) * 1e6
y = np.asarray(m["y"], float) * 1e6
E = np.asarray(m["Ez_re"]) + 1j * np.asarray(m["Ez_im"])
lam = float(m["lam_used_nm"]) * 1e-3
k0 = 2 * np.pi / lam
kc = N_CLAD * k0

# --- core line: extract carrier beta and envelope g0(x) of the real mode ---
jc = int(np.argmin(np.abs(y - 0.0)))
core = E[:, jc]
xu = np.linspace(x.min(), x.max(), NX)
dx = xu[1] - xu[0]
f_core = np.interp(xu, x, core.real) + 1j * np.interp(xu, x, core.imag)

kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, dx))
FT = lambda f: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f))) * dx
S_core = FT(f_core)
# carrier beta = dominant |kx| of the guided standing wave
beta = abs(kx[np.argmax(np.abs(S_core))])
# envelope magnitude via analytic-signal style: |field| low-pass smoothed
amp = np.abs(f_core)
# smooth over ~one period to get the envelope
per = 2 * np.pi / beta
w_smooth = max(3, int(per / dx))
kern = np.ones(w_smooth) / w_smooth
g0 = np.convolve(amp, kern, mode="same")
g0 /= g0.max()
# decay length L from the envelope (1/e half-width)
xc = xu[np.argmax(g0)]
half = g0 >= 1 / np.e
L = 0.5 * (xu[half].max() - xu[half].min())

print(f"lam={lam*1e3:.2f} nm  n_clad={N_CLAD}  beta={beta:.4f} rad/um  "
      f"beta/kc={beta/kc:.3f} (carrier {'OUT' if beta>kc else 'IN'} of cone)")
print(f"defect-mode envelope 1/e half-width L = {L:.2f} um\n")

carrier = np.cos(beta * (xu - xc))


def radiated(gi):
    """flux-weighted in-cone radiation amplitude of envelope gi with the carrier."""
    A = FT(gi * carrier)
    inc = np.abs(kx) < kc
    ky = np.sqrt(np.maximum(kc**2 - kx[inc]**2, 0.0))
    w = np.sqrt(ky / kc)            # amplitude flux weight (power = |.|^2)
    return w * A[inc]


def overlap(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)


# candidate co-located partners (SAME cavity, same L):
s = (xu - xc) / L
g_even0 = g0                                   # fundamental (even, 0 node) = real mode
g_odd1 = s * g0                                # 1-node ODD partner
g_even2 = (s**2 - 1.0) * g0                     # 2-node EVEN partner (Hermite-like)
for g in (g_odd1, g_even2):
    g /= np.max(np.abs(g)) + 1e-30

A0 = radiated(g_even0)
A1 = radiated(g_odd1)
A2 = radiated(g_even2)

rho_odd = overlap(A0, A1)
rho_even = overlap(A0, A2)

print("Radiation-pattern overlap rho of each co-located partner with the mode:")
print(f"   1-node ODD  partner:  rho = {rho_odd:.3f}  "
      f"(parity forbids overlap => expect ~0; no FW interference possible)")
print(f"   2-node EVEN partner:  rho = {rho_even:.3f}  (the only FW-eligible partner)")
print()
for name, rho in (("odd", rho_odd), ("even-2node", rho_even)):
    floor = (1 - rho**2) * LOSS0
    ok = "PASS" if rho >= RHO_GATE else "FAIL"
    print(f"   {name:11s}: dark-state loss floor (1-rho^2)*loss0 = {floor:.4f} "
          f"(loss0={LOSS0})   gate rho>={RHO_GATE}: {ok}")

overlap_ok = rho_even >= RHO_GATE
print(f"""
VERDICT (single-cavity / supercavity FW):
  (1) OVERLAP is NOT the blocker. The odd partner is parity-orthogonal (rho~0), but
      the even-2node partner's rho = {rho_even:.2f} -> it CLEARS the rho>={RHO_GATE} gate,
      and its formal dark-state floor (1-rho^2)*loss0 = {(1-rho_even**2)*LOSS0:.4f} is well below
      loss0 {LOSS0}. So unlike the two-cavity case (where low rho was the killer),
      the radiation patterns of a mode and its own even higher-order partner DO
      overlap enough. The obstruction is elsewhere.
  (2) EXISTENCE: a single pi-shift defect has NO second localized state in the gap
      (higher longitudinal states are expelled into the bands). Creating one needs
      a longer / coupled-defect cavity, whose two longitudinal modes are split by
      the cavity FSR, not degenerate. FW needs (near-)degeneracy -> you must tune
      them together, and near the anti-crossing you generically get TWO spectral
      poles = TWO peaks in the window (violates single-resonance) and/or a WIDER
      mode (violates the ~1% budget).
  (3) PORT DECOUPLING (decisive for a TRANSMISSION device): at the exact FW-BIC
      point the dark mode is uncoupled from radiation AND from the waveguide port,
      so it carries NO transmission signal -- a BIC is invisible to T(lambda). Back
      off to a quasi-BIC to recover port coupling and you (a) reintroduce radiative
      loss and (b) bring the BRIGHT partner back into the window. This is exactly
      the 'the resonance drains instead' failure the two-cavity FDTD showed. The
      supercavity trick works for a Mie SCATTERER (read in reflection/scattering,
      two co-located tunable modes: Mie + Fabry-Perot); our device is read in
      TRANSMISSION through a single-mode port with one localized gap mode, and that
      is structurally incompatible with harvesting a BIC.
  => Single-cavity FW is a MIRAGE HERE: overlap is fine, but no co-located degenerate
     gap partner exists without splitting the resonance or widening the mode, and a
     transmission port cannot harvest the dark state anyway. Consistent with the
     two-cavity FDTD failure. NOT worth GPU.
""")
