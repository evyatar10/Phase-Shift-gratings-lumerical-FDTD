"""
Period sweep: 528 nm pitch, 500 nm corrugation, tanh apodization (steepness=0.4).

Device parameters:
  - avg waveguide width : 800 nm  (SimulationConfig default)
  - pitch               : 528 nm
  - corrugation depth   : 500 nm
  - center mod depth    : 4 nm    (strong taper — 0.8% of corrugation at cavity)
  - apodization         : tanh, steepness = 0.4
  - cavity length       : pitch/2 - 105 nm = 159 nm

Sweep: n_periods_each_side over [60, 80, 100, 120].

Run from repo root:
    python experiment_examples/sweep_n_periods_528nm_tanh.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiment_card import ExperimentCard
from run_sweep import run_sweep

PITCH_NM = 528.0
CAVITY_LENGTH_NM = PITCH_NM / 2 - 105  # = 264 - 105 = 159 nm

card = ExperimentCard(
    pitch_nm=PITCH_NM,
    corrugation_depth_nm=500,
    center_mod_depth_nm=4,
    apod_method='none',
    tanh_steepness=0.4,
    cavity_length_nm=CAVITY_LENGTH_NM,
    label="528nm-pitch tanh sweep",
)

cfg = card.to_sweep_config("n_periods_each_side", [60])

if __name__ == "__main__":
    run_sweep(cfg)
