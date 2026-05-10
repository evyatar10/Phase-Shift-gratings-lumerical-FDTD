"""
Decisive test #2: take the LUMOPT forward .fsp (which has the polygon and
all the lumopt-specific settings) and modify it to use the BASELINE's
broadband source (10nm scan, 3001 freq points). Run, find peak T.

If peak T ≈ 0.866 (matches baseline):
    → Polygon-rendered geometry is correct; lumopt's single-λ source was
      the source of the discrepancy. Solution: switch FOM to broadband-
      with-Gaussian-weights.
If peak T ≈ 0.75:
    → Polygon-rendered geometry is genuinely producing a different device.
      Need to switch from FunctionDefinedPolygon to a rectangle-based
      ParameterizedGeometry approach.
"""

import sys
import numpy as np

sys.path.insert(0, '/opt/lumerical/v261/api/python')
import lumapi

LUMOPT_FSP = '/work/lumopt_forward.fsp'

# Match the baseline's broadband source range
LAM_CENTER = 1.5601e-6
LAM_HALF_WIDTH = 5e-9   # 10 nm full scan, like baseline
N_FREQ = 3001


def main():
    fdtd = lumapi.FDTD()
    fdtd.load(LUMOPT_FSP)
    fdtd.switchtolayout()

    # Reconfigure to broadband (mirror the baseline source/monitor settings).
    fdtd.setglobalsource("set wavelength", 1)
    fdtd.setglobalsource("wavelength start", LAM_CENTER - LAM_HALF_WIDTH)
    fdtd.setglobalsource("wavelength stop", LAM_CENTER + LAM_HALF_WIDTH)
    fdtd.setglobalmonitor("frequency points", N_FREQ)
    fdtd.setnamed("FDTD::ports", "monitor frequency points", N_FREQ)
    # Re-enable freq-dep profile on both ports to match baseline behavior.
    fdtd.setnamed("FDTD::ports::source", "frequency dependent profile", 1)
    fdtd.setnamed("FDTD::ports::fom", "frequency dependent profile", 1)

    print(f"Source range: {fdtd.getglobalsource('wavelength start')*1e9:.4f} - {fdtd.getglobalsource('wavelength stop')*1e9:.4f} nm")
    print(f"Monitor freq points: {fdtd.getglobalmonitor('frequency points')}")

    try:
        fdtd.setresource("FDTD", 1, "device type", "GPU")
        print("GPU enabled.")
    except Exception as e:
        print(f"GPU enable failed: {e}")

    print("Running FDTD on the LUMOPT geometry with BROADBAND source...")
    fdtd.run()

    # Read S21 (modal transmission) at the FOM port
    res = fdtd.getresult("FDTD::ports::fom", "expansion for port monitor")
    a = np.asarray(np.squeeze(res["a"]))
    b = np.asarray(np.squeeze(res["b"]))
    N = np.asarray(np.squeeze(res["N"]))
    wl = np.asarray(np.squeeze(res["lambda"]))

    print(f"a shape: {a.shape}, |a|² range: {np.min(np.abs(a)**2):.4e} - {np.max(np.abs(a)**2):.4e}")
    print(f"N.real range: {np.min(np.real(N)):.4e} - {np.max(np.real(N)):.4e}")

    # PortTransmission's T = |a*sqrt(N.real)|^2
    fwd_coeff = a * np.sqrt(np.real(N))
    T = np.real(fwd_coeff * np.conj(fwd_coeff))
    T = np.asarray(T).flatten()
    wl = wl.flatten()

    # Find peak
    idx_peak = int(np.argmax(T))
    print(f"\nPeak T (lumopt-geometry, broadband source): {float(T[idx_peak]):.4f} at λ = {wl[idx_peak]*1e9:.4f} nm")

    # T at exactly 1560.254 nm (the lumopt single-λ target)
    target_lam = 1.560253992166315e-6
    idx_target = int(np.argmin(np.abs(wl - target_lam)))
    print(f"T at λ = {wl[idx_target]*1e9:.4f} nm: {float(T[idx_target]):.4f}")

    # Save for downstream inspection
    np.savez('/work/lumopt_broadband_T.npz', wl=wl, T=T)
    print("Saved T(λ) array to /work/lumopt_broadband_T.npz")

    fdtd.close()


if __name__ == "__main__":
    main()
