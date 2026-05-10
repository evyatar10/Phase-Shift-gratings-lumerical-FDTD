"""
Decisive test: take the BASELINE .fsp (rectangles, no polygon) and modify it
to use a SINGLE-WAVELENGTH source at 1560.254 nm. Run, read T at that λ.

If T ≈ 0.866 (matches the original baseline broadband-pulse result):
    → Single-λ source is fine; the polygon is the issue in lumopt.
If T ≈ 0.75 (matches the lumopt single-λ result):
    → Single-λ source is the issue; the polygon is innocent.
    → Need to switch lumopt to broadband-with-Gaussian-FOM-weights.
"""

import sys
import numpy as np

sys.path.insert(0, '/opt/lumerical/v261/api/python')
import lumapi

BASELINE = '/work/baseline.fsp'
TARGET_LAMBDA = 1.560253992166315e-6   # exactly what lumopt uses


def main():
    fdtd = lumapi.FDTD()
    fdtd.load(BASELINE)
    fdtd.switchtolayout()

    # Reconfigure the source to single-wavelength matching lumopt's setup.
    fdtd.setglobalsource("set wavelength", 1)
    fdtd.setglobalsource("wavelength start", TARGET_LAMBDA)
    fdtd.setglobalsource("wavelength stop", TARGET_LAMBDA)
    fdtd.setglobalmonitor("frequency points", 1)
    fdtd.setnamed("FDTD::ports", "monitor frequency points", 1)

    print(f"Reconfigured source to single λ = {TARGET_LAMBDA*1e9:.4f} nm")
    print(f"Source range: {fdtd.getglobalsource('wavelength start')*1e9:.4f} - {fdtd.getglobalsource('wavelength stop')*1e9:.4f} nm")
    print(f"Monitor freq points: {fdtd.getglobalmonitor('frequency points')}")

    # Switch to GPU like the deploy job does.
    try:
        fdtd.setresource("FDTD", 1, "device type", "GPU")
        print("GPU enabled.")
    except Exception as e:
        print(f"GPU enable failed: {e}")

    # Run FDTD.
    print("Running FDTD...")
    fdtd.run()

    # Read modal transmission at the FOM port (Port_2 in baseline) using the
    # SAME 'expansion for port monitor' result lumopt's PortTransmission uses.
    res = fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")
    a_fwd = np.squeeze(res["a"])
    N_real = np.real(np.squeeze(res["N"]))
    print(f"a (forward) shape: {np.shape(a_fwd)}, value: {a_fwd}")
    print(f"N.real: {N_real}")

    # Modal transmission |a * sqrt(N.real)|^2 — same formula PortTransmission uses.
    fwd_coeff = a_fwd * np.sqrt(N_real)
    T_modal = np.real(fwd_coeff * np.conj(fwd_coeff))
    print(f"|a*sqrt(N)|^2 = {T_modal}")

    fdtd.close()


if __name__ == "__main__":
    main()
