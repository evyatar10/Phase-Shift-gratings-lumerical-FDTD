"""
PHASE-0 (4.1) — Single-resonance BIC: symmetry table + Friedrich-Wintgen
2-mode CMT design map. Zero GPU.

Part A  Symmetry-protected BIC (4.1a): classify the operating defect mode,
        the input port mode, and the measured radiation continuum coupling
        under every mirror of the device. Data anchor: the y-parity and
        x-parity of the in-cone radiating amplitude measured by
        phase0_greens_overlap.py from the accurate-mesh stack field export.
        GATE: a realizable, single-resonance defect whose mode is odd under a
        mirror the continuum coupling is even under — WITHOUT killing the
        (even) port coupling.

Part B  Friedrich-Wintgen at one location (4.1b): 2x2 non-Hermitian CMT,
        H = [[0, k],[k, D]] - i*G,  G = [[g1, r*sqrt(g1*g2)],[r*sqrt(g1*g2), g2]]
        mode 1 = the operating defect resonance (radiative width g1 from the
        measured stack numbers), mode 2 = a co-located DIFFERENT-family
        partner with radiative width g2, near-field coupling k, detuning D,
        channel-pattern overlap r (r=1: same radiation channel; the FW zero
        exists only on the shared part -> max suppression = 1/(1-r^2)).
        All rates/detunings in nm of wavelength (internally consistent).
        GATE: a (k, D, g2) region with radiative suppression >= 3x, partner
        pushed >= 15 nm out of the window, and mode-2 admixture <= 2%
        (spatial-width guard), at physically achievable k.

Part C  Vertical-channel anti-phase out-coupler (bonus from the 4.4 framing):
        a weak 2*Lambda superperiod component turns the guided envelope into
        a vertical secondary source that can destructively interfere with the
        existing vertical leak (38% of the loss). Order-of-magnitude sizing.

Measured anchors (stack, accurate mesh):
  lam = 1556.6 nm, Q_tot = 1404, T = 0.9449, loss = 0.0545
  -> symmetric-cavity CMT: T = (1-x)^2, loss = 2x(1-x), x = g_rad/g_tot
     x = 0.0277 reproduces BOTH T and loss (internal consistency check below)
  in-cone radiation y-parity: 100.0% EVEN / 0.0% odd (phase0_greens_overlap)
  in-cone radiation x-parity: 49% even / 51% odd (stack; W800: 82/18)

Run:  python python_tools/phase0_bic_cmt.py
"""

import numpy as np

LAM = 1556.6          # nm
Q_TOT = 1404.0
T_MEAS = 0.9449
LOSS_MEAS = 0.0545

# ------------------------------------------------------------ rate extraction
print("=" * 74)
print("Rate extraction (symmetric 2-port + radiation CMT)")
print("=" * 74)
x = 1 - np.sqrt(T_MEAS)                    # g_rad / g_tot
print(f"x = g_rad/g_tot from T=(1-x)^2: {x:.4f}")
print(f"  consistency: predicted loss 2x(1-x) = {2*x*(1-x):.4f} vs measured "
      f"{LOSS_MEAS:.4f}  (R = x^2 = {x**2:.4f} ~ 0)")
G_TOT = LAM / Q_TOT                        # total width, nm
G1 = x * G_TOT                             # radiative width of mode 1, nm
G_PORT = G_TOT - G1
Q_RAD = LAM / G1
print(f"g_tot = {G_TOT:.3f} nm  -> radiative g1 = {G1:.4f} nm (Q_rad = {Q_RAD:.0f}), "
      f"port width = {G_PORT:.3f} nm")

# ============================================================== A. symmetry
print()
print("=" * 74)
print("A. SYMMETRY TABLE — is a symmetry-protected single-resonance BIC open?")
print("=" * 74)
print("""
mirror          device sym?  operating mode   port mode   continuum coupling
--------------  -----------  ---------------  ----------  -------------------------
sigma_y (y->-y)   YES          EVEN (fund.)     EVEN        EVEN: measured 100.0% of
                                                            in-cone power is y-even
sigma_x (about    ~ (arms      ~EVEN            n/a (prop-  BOTH parities: measured
 defect)          differ)                       agating)    49%/51% even/odd (stack)
sigma_z (z->-z)   YES          EVEN (TM fund.)  EVEN        both parities available
                                                            (up/down continuum)

Protection logic per mirror:
 * sigma_y: the continuum coupling is PURE EVEN (measured) -> a y-ODD defect
   mode would be exactly decoupled from the in-plane near-axial channel.
   BUT the input port mode is y-EVEN: a y-odd cavity mode has ZERO port
   coupling by the same integral -> T = 0. Protection and transmission are
   killed by the SAME symmetry. An asymmetric launch (broken-mirror taper)
   re-admits both equally -> no net protection, just insertion loss.
 * sigma_x: continuum coupling carries BOTH parities (49/51) -> no x-parity
   assignment can forbid the coupling. (Also the physical arms are not
   x-mirror images: left arm is wide-adjacent, right narrow-adjacent.)
 * sigma_z: with uniform height + symmetric cladding, up/down radiation is a
   two-channel continuum containing both z-parities -> nothing to forbid.
 * k-space (Gamma-point) protection does not apply: the resonance sits at the
   band edge beta = pi/pitch = 1.508*k0; the periodic Bloch part diffracts to
   |beta - 2pi/pitch| = 1.508*k0 > n_clad*k0 -> the PERIODIC grating does not
   radiate at all (arms are near-lossless); ALL loss is envelope broadening
   around +-beta. Gamma-point symmetry-protected BIC machinery needs the
   resonance at kx=0 -> not our operating point.
""")
print("GATE A VERDICT: FAIL — no mirror exists under which the continuum is")
print("even and a PORT-COUPLED single-resonance mode can be odd. The only")
print("protecting mirror (sigma_y) kills transmission identically. DROP 4.1a.")

# ============================================================== B. FW-CMT map
print()
print("=" * 74)
print("B. FRIEDRICH-WINTGEN 2-mode CMT design map (all widths/detunings in nm)")
print("=" * 74)
print("""FW condition (r=1): k*(g1-g2) = sqrt(g1*g2)*(w1-w2). With a LOSSY
partner g2 >> g1 the dark state is automatically mode-1-dominated:
  admixture |c2|^2 -> g1/(g1+g2)   (small)
  required detuning D = w2-w1 ~ +- k*sqrt(g2/g1)   (large -> partner far away)
Max suppression at channel overlap r: S_max = 1/(1-r^2)  (r=1 -> exact BIC).
""")

sep_min, admix_max, sup_goal = 15.0, 0.02, 3.0
kappas = np.geomspace(0.05, 15.0, 160)
dets = np.linspace(-40.0, 40.0, 801)

print(f"requirements: suppression >= {sup_goal}x, partner separation >= "
      f"{sep_min} nm, admixture <= {admix_max*100:.0f}%")
print(f"\n{'g2 (nm)':>8} {'rho':>5} | {'best sup':>9} {'@k':>6} {'@D':>7} "
      f"{'sep':>6} {'admix':>7} | feasible k-range (nm)")
summary = {}
for g2 in (0.5, 2.0, 5.0, 15.0):
    for rho in (1.0, 0.95, 0.9, 0.8, 0.6):
        best = None
        feas_k = []
        for kap in kappas:
            ok_here = False
            for D in dets:
                H = np.array([[0.0, kap], [kap, D]], complex)
                Gm = np.array([[G1, rho * np.sqrt(G1 * g2)],
                               [rho * np.sqrt(G1 * g2), g2]], complex)
                ev, vec = np.linalg.eig(H - 1j * Gm)
                i_op = int(np.argmax(np.abs(vec[0, :])))
                i_pa = 1 - i_op
                im_op = -ev[i_op].imag
                sup = G1 / max(im_op, 1e-12)
                sep = abs(ev[i_op].real - ev[i_pa].real)
                admix = abs(vec[1, i_op])**2 / (np.abs(vec[:, i_op])**2).sum()
                if sep >= sep_min and admix <= admix_max:
                    if best is None or sup > best[0]:
                        best = (sup, kap, D, sep, admix)
                    if sup >= sup_goal:
                        ok_here = True
            if ok_here:
                feas_k.append(kap)
        tag = (f"{min(feas_k):.2f} - {max(feas_k):.2f}" if feas_k else "none")
        if best:
            print(f"{g2:8.1f} {rho:5.2f} | {min(best[0], 9999):9.1f} "
                  f"{best[1]:6.2f} {best[2]:+7.1f} {best[3]:6.1f} "
                  f"{best[4]*100:6.2f}% | {tag}")
            summary[(g2, rho)] = (best, feas_k)

# representative design point -> device-level prediction
g2, rho = 5.0, 0.9
(best, feas_k) = summary[(g2, rho)]
sup, kap, D, sep, admix = best
im_new = G1 / sup
x_new = im_new / (im_new + G_PORT)
print(f"\nREPRESENTATIVE POINT (g2={g2} nm, rho={rho}): k={kap:.2f} nm, "
      f"D={D:+.1f} nm")
print(f"  suppression {sup:.1f}x -> new loss = {2*x_new*(1-x_new):.4f} "
      f"(from {LOSS_MEAS}); T -> {(1-x_new)**2:.4f}; partner at "
      f"{sep:.1f} nm with width ~{g2 + G1:.1f} nm (broad, shallow, out of the "
      f"+-20 nm window)")

print("""
Physical partner shortlist (k is the design knob; all planar, 350 nm height):
 1. SHORT side-coupled parallel strip cavity (few-period Bragg ridge along y,
    gap 0.3-0.8 um): k set by evanescent overlap. Guided-mode lateral decay
    gamma_y = 1.75 /um -> tail ratio e^{-1.75*gap}; supermode splitting at
    gap ~0.4-0.7 um lands k ~ 0.5-3 nm (the side-by-side study machinery
    already builds exactly this geometry). Partner Q ~ 100-800 (small N)
    -> g2 ~ 2-15 nm. Detune partner by pitch/width -> D ~ -k*sqrt(g2/g1).
 2. A few-period SECOND-order-corrugated patch superimposed near the defect
    (every-other-tooth width alternation): creates a co-located leaky
    resonance of a different Bloch family; k via direct spatial overlap.
 3. NOT a cladding Mie rod: SiN/oxide contrast gives rod Q ~ 1-3
    (g2 ~ hundreds of nm) -> no usable resonance to interfere with.

What CMT CANNOT predict: rho (the radiation-pattern overlap of the two
modes). S_max = 1/(1-rho^2): rho >= 0.82 is REQUIRED for 3x. rho is set by
how similar the partner's radiation lobe is to the defect's near-axial lobe
— plausibly high for partner #1 (same family of leakage, same plane), and it
is exactly what the FDTD scan measures. This is the registered risk.
""")
print("GATE B VERDICT: PASS (conditional). A feasible (k, D, g2) region exists")
print("with suppression >= 3x, partner >= 15 nm away, admixture <= 2%, at")
print("k ~ 0.5-3 nm achievable by evanescent side coupling. The unknown is")
print("rho >= 0.82 — measured by the scan itself, not assumable in advance.")

# ============================================== C. vertical anti-phase coupler
print()
print("=" * 74)
print("C. Vertical-channel anti-phase out-coupler (2*Lambda superperiod)")
print("=" * 74)
P_vert = 0.38 * LOSS_MEAS
print(f"vertical leak to cancel: 0.38 * {LOSS_MEAS} = {P_vert:.4f} of input power")
print("""A weak every-other-tooth width alternation dW2 (superperiod 2*Lambda)
adds a first-order out-coupling k_oc ~ (dW2/W) * kappa_gr acting on the
resonant envelope -> vertical secondary amplitude a_oc ~ k_oc * L_env * E_env.
Matching |a_oc|^2 to the existing vertical leak fraction needs only
  P_oc ~ P_vert = 0.0207  ->  amplitude ratio ~ sqrt(P_vert) = 0.14
i.e. a PERTURBATIVE alternation (nm-scale dW2), phased by the alternation's
spatial offset (which tooth is wide) and x-position of the patch. Two knobs
(amplitude, phase) against one complex target amplitude -> a 2D scan can in
principle null the vertical channel IF the existing vertical leak is
spatially coherent (near-axial spikes suggest it is). Risks: (i) the
alternation also back-couples in-plane (detunes the resonance -> re-trim),
(ii) leak coherence unmeasured. This is the cheapest planar handle that
touches the 38% vertical share — worth rows in the BIC array (same builder
knob: width_wide_per_tooth alternation).""")
