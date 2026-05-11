"""
IT11 — 500 nm pitch batch only.

Use this for the strict-ordering submission flow:
    bash athena/deploy_athena.sh --option3 --cards=runners.experiment_comparison.it11_devices_500
    # wait for completion + pull
    bash athena/deploy_athena.sh --results

Then submit the 516 nm batch (it11_devices_516.py) so no 516 task ever runs
while a 500 task is still pending.

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
BASE.spectral.center_wavelength_m = 1.538e-6  # 500 nm pitch ⇒ resonance ~1538 nm


CARDS, RECORDS = build_it11(pitch=500)

if __name__ == "__main__":
    print(f"IT11 500 nm batch: {len(CARDS)} cards")
    for r in RECORDS[:3]:
        print(f"  [{r.idx:3d}] {r.label}  ({r.subname})")
