"""
Local self-test for the gradient-free-design module: builds the .fsp via
lumapi (license required) and verifies the parametric structure group +
analysis script + sweep object are correctly wired. Saves a debug .fsp.

Run: python -m runners.gradient_free_design.test_geometry
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config as _cfg

sys.path.insert(0, os.path.dirname(_cfg.LUMAPI_PATH))
import lumapi  # type: ignore

from runners.gradient_free_design.gradient_free_design import (
    _build_parametric_fsp,
    _configure_optimization_sweep,
)
from runners.gradient_free_design.smoke_test import SPEC, BASE


def main():
    cfg = BASE
    cfg.grating.n_free_inner_teeth = SPEC.n_free_inner_teeth
    cfg.grating.lengthen_cavity = SPEC.lengthen_cavity
    p0 = list(SPEC.initial_points[0])
    cfg.grating.inner_dw_nm = list(p0[:2])
    cfg.grating.inner_shift_nm = list(p0[2:4])
    cfg.grating.cavity_width_m = float(p0[4]) * 1e-9

    target_lam = 1.560e-6

    print("Opening lumapi.FDTD (hidden)...")
    fdtd = lumapi.FDTD(hide=True)
    try:
        print("Building parametric .fsp...")
        _build_parametric_fsp(fdtd, cfg, SPEC, p0, target_lam)
        print("Parametric .fsp built.")

        # Verify structure group exists
        try:
            n = float(fdtd.getnamed("freed_group", "dw_inner_1_nm"))
            print(f"freed_group::dw_inner_1_nm = {n} (expected {p0[0]})")
        except Exception as e:
            print(f"freed_group user prop check FAILED: {e!r}")

        # Verify rectangles inside the group
        for i in [0, 4, 10]:  # left edge, cavity, right edge
            name = f"freed_seg_{i:02d}"
            try:
                # Inside a structure group, the path is "freed_group::name"
                xs = float(fdtd.getnamed(f"freed_group::{name}", "x span"))
                print(f"  {name}: x_span = {xs*1e9:.2f} nm")
            except Exception as e:
                print(f"  {name}: NOT FOUND ({e!r})")

        # Verify analysis group
        try:
            fdtd.getnamed("fom_extractor", "name")
            print("fom_extractor: present")
        except Exception as e:
            print(f"fom_extractor: NOT FOUND ({e!r})")

        # Configure the optimization sweep
        print("Configuring Optimization sweep...")
        sweep_name = _configure_optimization_sweep(fdtd, SPEC)
        print(f"Sweep configured: {sweep_name}")

        # Save the .fsp for inspection
        out = os.path.abspath("debug_gf_design.fsp")
        fdtd.save(out)
        print(f"Saved: {out}")
    finally:
        fdtd.close()
    print("DONE.")


if __name__ == "__main__":
    main()
