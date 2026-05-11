"""
IT11 — 516 nm pitch batch only.

Submit AFTER it11_devices_500 completes:
    bash athena/deploy_athena.sh --option3 --cards=runners.experiment_comparison.it11_devices_516
    bash athena/deploy_athena.sh --results

See it11_devices.py for parameter conventions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.experiment_comparison.it11_card_builder import build_it11
from simulation_config import SimulationConfig


BASE = SimulationConfig()
BASE.mesh.simulation_mode         = "accurate"
BASE.material.n_core_const        = 1.93024   # Calibrated against IT11 experimental Bragg λ (2026-05-11)
BASE.spectral.scan_width_nm       = 30.0
BASE.spectral.center_wavelength_m = 1.577e-6  # 516 nm pitch ⇒ resonance ~1577 nm


CARDS, RECORDS = build_it11(pitch=516)

if __name__ == "__main__":
    print(f"IT11 516 nm batch: {len(CARDS)} cards")
    for r in RECORDS[:3]:
        print(f"  [{r.idx:3d}] {r.label}  ({r.subname})")
