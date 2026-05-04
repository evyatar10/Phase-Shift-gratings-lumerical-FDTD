"""
Calibrate n_core (Si3N4) of the constant-index FDTD setup against the
experimental Bragg wavelength of the pi-shift Bragg grating.

Usage
-----
Edit LAMBDA_TARGET_M and LAMBDA_PROBE_M below if needed, then run:

    python calibrate_n_core.py

The script:
  1. Computes avg n_eff (wide, narrow cross-sections) for the current
     n_core_const (sanity: should reproduce the simulated Bragg).
  2. Solves for the n_core that makes avg n_eff equal the target n_eff
     (= lambda_exp / (2 * pitch)) at the probe wavelength.
  3. Saves a JSON log of the inputs, intermediate values, and result.

It does NOT modify simulation_config.py — apply the result manually.
"""

import json
import os
import sys
from dataclasses import dataclass

import numpy as np

try:
    import lumapi
except ImportError:
    import importlib.util
    LUMAPI_PATH = r"C:\\Program Files\\Lumerical\\v252\\api\\python\\lumapi.py"
    spec = importlib.util.spec_from_file_location("lumapi", LUMAPI_PATH)
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)

# Make project root importable so we can read the live geometry/material config.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation_config import GeometryConfig, GratingConfig, MaterialConfig  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration target
# ═══════════════════════════════════════════════════════════════════════════════

LAMBDA_EXP_M = 1536e-9      # Measured Bragg wavelength of the left grating
LAMBDA_PROBE_M = 1536e-9    # Wavelength at which FDE evaluates n_eff
N_CORE_BRACKET = (1.85, 1.99)
N_EFF_TOLERANCE = 1e-4      # Stop bisection when |avg_neff - target| < this
MAX_ITERATIONS = 8

OUTPUT_DIR = r"C:\Users\evyat\Lumerical\pi_shifts_FDTD_results\neff_calibration"


@dataclass
class CalibInputs:
    width_wide_m: float
    width_narrow_m: float
    height_m: float
    pitch_m: float
    lambda_exp_m: float
    lambda_probe_m: float
    n_clad: float
    n_core_initial: float

    @property
    def target_avg_neff(self) -> float:
        return self.lambda_exp_m / (2.0 * self.pitch_m)


# ═══════════════════════════════════════════════════════════════════════════════
# FDE driver
# ═══════════════════════════════════════════════════════════════════════════════

class NeffCalibrator:
    """FDE solver wrapper that returns n_eff for a Si3N4/SiO2 strip waveguide
    using constant-index dielectric materials (matches FDTD config)."""

    CORE_MAT = "SiN_Const_Calib"
    CLAD_MAT = "SiO2_Const_Calib"

    def __init__(self, height_m: float, lambda_probe_m: float):
        self.height = height_m
        self.lam = lambda_probe_m
        self.mode = lumapi.MODE()

    def _set_constant_materials(self, n_core: float, n_clad: float) -> None:
        # Use a fresh "Dielectric" material each call (refractive index is
        # frequency-independent — equivalent to FDTD's flat sampled-data path).
        script = f'''
        switchtolayout;
        if (materialexists("{self.CORE_MAT}")) {{ deletematerial("{self.CORE_MAT}"); }}
        if (materialexists("{self.CLAD_MAT}")) {{ deletematerial("{self.CLAD_MAT}"); }}

        addmaterial("Dielectric");
        set("name", "{self.CORE_MAT}");
        set("Refractive Index", {n_core});

        addmaterial("Dielectric");
        set("name", "{self.CLAD_MAT}");
        set("Refractive Index", {n_clad});
        '''
        self.mode.eval(script)

    def _build_geometry(self, width_m: float) -> None:
        m = self.mode
        m.switchtolayout()
        m.deleteall()

        # Rotated convention (matches calc_neff_vs_wl.py): thickness=X, width=Y, length=Z.
        m.addfde()
        m.set("solver type", "2D Z normal")
        m.set("x", 0); m.set("x span", 4e-6)
        m.set("y", 0); m.set("y span", 4e-6)
        m.set("z", 0)
        m.set("wavelength", self.lam)

        m.addrect()
        m.set("name", "substrate")
        m.set("material", self.CLAD_MAT)
        m.set("x span", 10e-6); m.set("y span", 10e-6); m.set("z span", 10e-6)

        m.addrect()
        m.set("name", "core")
        m.set("material", self.CORE_MAT)
        m.set("x", 0); m.set("x span", self.height)
        m.set("y", 0); m.set("y span", width_m)
        m.set("z", 0); m.set("z span", 10e-6)

        m.addmesh()
        m.set("name", "core_mesh")
        m.set("x", 0); m.set("x span", self.height + 0.5e-6)
        m.set("y", 0); m.set("y span", width_m + 0.5e-6)
        m.set("z", 0); m.set("z span", 10e-6)
        m.set("dx", 10e-9); m.set("dy", 10e-9); m.set("dz", 10e-9)

    def _solve_neff(self, width_m: float) -> float:
        # Mirrors calc_neff_vs_wl.py:run_sweep() up through findmodes/selectmode,
        # then reads mode1 neff directly (we only need one wavelength point).
        self._build_geometry(width_m)
        self.mode.run()
        self.mode.setanalysis("wavelength", self.lam)
        self.mode.findmodes()
        self.mode.selectmode(1)
        neff = float(np.real(self.mode.getdata("FDE::data::mode1", "neff")))
        return neff

    def avg_neff(self, n_core: float, n_clad: float,
                 width_wide_m: float, width_narrow_m: float) -> tuple[float, float, float]:
        """Return (avg, neff_wide, neff_narrow) for a given (n_core, n_clad)."""
        self._set_constant_materials(n_core, n_clad)
        n_wide = self._solve_neff(width_wide_m)
        n_narrow = self._solve_neff(width_narrow_m)
        return 0.5 * (n_wide + n_narrow), n_wide, n_narrow

    def close(self) -> None:
        self.mode.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration logic
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate(inp: CalibInputs) -> dict:
    cal = NeffCalibrator(inp.height_m, inp.lambda_probe_m)
    log = {"inputs": inp.__dict__.copy(), "target_avg_neff": inp.target_avg_neff,
           "iterations": []}

    try:
        # Sanity: current n_core
        avg0, w0, nw0 = cal.avg_neff(
            inp.n_core_initial, inp.n_clad, inp.width_wide_m, inp.width_narrow_m,
        )
        sim_lambda_B = 2 * avg0 * inp.pitch_m
        log["sanity"] = {
            "n_core": inp.n_core_initial,
            "neff_wide": w0, "neff_narrow": nw0, "avg_neff": avg0,
            "implied_lambda_B_m": sim_lambda_B,
        }
        print(f"[sanity] n_core={inp.n_core_initial:.4f} -> "
              f"avg n_eff={avg0:.4f} (wide={w0:.4f}, narrow={nw0:.4f}) -> "
              f"lambda_B={sim_lambda_B*1e9:.2f} nm")

        # Bisection over n_core. Higher n_core -> higher n_eff (monotonic).
        lo, hi = N_CORE_BRACKET
        avg_lo, _, _ = cal.avg_neff(lo, inp.n_clad, inp.width_wide_m, inp.width_narrow_m)
        avg_hi, _, _ = cal.avg_neff(hi, inp.n_clad, inp.width_wide_m, inp.width_narrow_m)
        log["iterations"].append({"n_core": lo, "avg_neff": avg_lo})
        log["iterations"].append({"n_core": hi, "avg_neff": avg_hi})
        print(f"[bracket] lo: n_core={lo} -> avg={avg_lo:.4f} | "
              f"hi: n_core={hi} -> avg={avg_hi:.4f}")

        target = inp.target_avg_neff
        if not (min(avg_lo, avg_hi) <= target <= max(avg_lo, avg_hi)):
            raise RuntimeError(
                f"Target avg n_eff {target:.4f} outside bracket "
                f"[{avg_lo:.4f}, {avg_hi:.4f}]; widen N_CORE_BRACKET."
            )

        # Linear interpolation as first guess (often within tolerance).
        n_core_lin = lo + (target - avg_lo) * (hi - lo) / (avg_hi - avg_lo)
        avg_lin, w_lin, nw_lin = cal.avg_neff(
            n_core_lin, inp.n_clad, inp.width_wide_m, inp.width_narrow_m,
        )
        log["iterations"].append({"n_core": n_core_lin, "avg_neff": avg_lin,
                                  "neff_wide": w_lin, "neff_narrow": nw_lin})
        print(f"[lin]  n_core={n_core_lin:.4f} -> avg={avg_lin:.4f} "
              f"(target {target:.4f})")

        n_core_best = n_core_lin
        avg_best = avg_lin
        if abs(avg_lin - target) > N_EFF_TOLERANCE:
            # Tighten with one or two bisection steps.
            if avg_lin < target:
                lo, avg_lo = n_core_lin, avg_lin
            else:
                hi, avg_hi = n_core_lin, avg_lin
            for _ in range(MAX_ITERATIONS):
                if abs(avg_best - target) <= N_EFF_TOLERANCE:
                    break
                mid = 0.5 * (lo + hi)
                avg_mid, w_mid, nw_mid = cal.avg_neff(
                    mid, inp.n_clad, inp.width_wide_m, inp.width_narrow_m,
                )
                log["iterations"].append({"n_core": mid, "avg_neff": avg_mid,
                                          "neff_wide": w_mid, "neff_narrow": nw_mid})
                print(f"[bis]  n_core={mid:.5f} -> avg={avg_mid:.5f}")
                if avg_mid < target:
                    lo, avg_lo = mid, avg_mid
                else:
                    hi, avg_hi = mid, avg_mid
                n_core_best, avg_best = mid, avg_mid

        log["result"] = {
            "n_core_calibrated": n_core_best,
            "avg_neff_at_calibrated": avg_best,
            "residual_neff": avg_best - target,
            "implied_lambda_B_m": 2 * avg_best * inp.pitch_m,
        }
        print(f"\n[result] n_core_calibrated = {n_core_best:.4f} "
              f"(avg n_eff = {avg_best:.4f}, target = {target:.4f})")
        print(f"         implied lambda_B = "
              f"{log['result']['implied_lambda_B_m']*1e9:.2f} nm")
    finally:
        cal.close()

    return log


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def _build_inputs() -> CalibInputs:
    geo = GeometryConfig()
    grat = GratingConfig()
    mat = MaterialConfig()
    return CalibInputs(
        width_wide_m=geo.width_wide_m,
        width_narrow_m=geo.width_narrow_m,
        height_m=geo.core_height_m,
        pitch_m=grat.pitch_m,
        lambda_exp_m=LAMBDA_EXP_M,
        lambda_probe_m=LAMBDA_PROBE_M,
        n_clad=mat.n_clad_const,
        n_core_initial=mat.n_core_const,
    )


def main() -> None:
    inp = _build_inputs()
    print("=" * 72)
    print("Pi-shift Bragg grating: n_core calibration")
    print("=" * 72)
    print(f"  width_wide   = {inp.width_wide_m*1e9:.0f} nm")
    print(f"  width_narrow = {inp.width_narrow_m*1e9:.0f} nm")
    print(f"  height       = {inp.height_m*1e9:.0f} nm")
    print(f"  pitch        = {inp.pitch_m*1e9:.0f} nm")
    print(f"  lambda_exp   = {inp.lambda_exp_m*1e9:.2f} nm")
    print(f"  target avg n_eff = {inp.target_avg_neff:.4f}")
    print(f"  n_clad (fixed)   = {inp.n_clad}")
    print(f"  n_core (initial) = {inp.n_core_initial}")
    print()

    log = calibrate(inp)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "calibrate_n_core_log.json")
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog saved to: {out_path}")


if __name__ == "__main__":
    main()
