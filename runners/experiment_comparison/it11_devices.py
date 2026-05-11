"""
IT11 fabricated-device sweep — both pitches (500 nm batch first, then 516 nm).

Submission:
    bash athena/deploy_athena.sh --option3 --cards=runners.experiment_comparison.it11_devices

Local smoke test (one device, no FDTD-cluster involvement):
    python -c "from runners.experiment_comparison.it11_devices import CARDS; \\
               from experiment_card import run_card; run_card(CARDS[0])"

Inspection:
    python -c "from runners.experiment_comparison.it11_devices import CARDS, RECORDS; \\
               print(len(CARDS), CARDS[0].pitch_nm, CARDS[-1].pitch_nm)"

The card-list mechanism (kind=cards) is dispatched by athena_run_one.py /
build_sweep_list.py — see those scripts for the wire-up.

NOTE on detuning: cavity_neg_detuning_nm is intentionally left at the
SimulationConfig default (0.0). We are simulating the as-fabricated geometry
verbatim, with no negative-detuning correction.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.experiment_comparison.it11_card_builder import build_it11
from simulation_config import SimulationConfig


BASE = SimulationConfig()
BASE.mesh.simulation_mode    = "accurate"
BASE.material.n_core_const   = 1.93024   # Calibrated against IT11 experimental Bragg λ (2026-05-11)
BASE.spectral.scan_width_nm  = 60.0   # wide enough to span both 1535 nm and 1577 nm windows
BASE.spectral.center_wavelength_m = 1.556e-6  # midpoint between LEFT (1538) and RIGHT (1577) windows


CARDS, RECORDS = build_it11(pitch=None)

# Quick sanity print when imported (cheap; pure Python).
if __name__ == "__main__":
    print(f"IT11 combined sweep: {len(CARDS)} cards "
          f"(first pitch={CARDS[0].pitch_nm:.0f}, last pitch={CARDS[-1].pitch_nm:.0f})")
    for r in RECORDS[:3] + RECORDS[-3:]:
        print(f"  [{r.idx:3d}] {r.label}  ({r.subname})")
