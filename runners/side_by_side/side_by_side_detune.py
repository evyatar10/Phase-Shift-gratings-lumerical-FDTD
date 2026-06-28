"""
Study: two side-by-side pi-shift Bragg cavities — corrugation DETUNING knob.

Companion to side_by_side_coupling.py (the gap × stagger distance grid). Here the
geometry (lateral gap + longitudinal stagger) is FIXED at a representative point
and device 2's corrugation depth is swept around device 1's fixed 500 nm. This
detunes the second cavity's resonance/grating strength relative to the first —
the Friedrich–Wintgen knob: destructive interference into the shared radiation
channel can drive one supermode subradiant at a specific detuning (an avoided
crossing), distinct from the Fabry–Pérot phase periodicity of the gap sweep.

  device_gap_nm          : {1000}                          (fixed, nm)
  device_stagger_nm      : {6000}    ← UPDATE to the best Δx from the distance grid
  corrugation_depth_2_nm : {400, 450, 475, 500, 525, 550, 600}   (device-2 depth, nm)
  polarization           : {TE, TM}
  => 7 × 2 = 14 cartesian points (device 1 stays at 500 nm via build_base).

Reuses build_base() from side_by_side_coupling (pitch 500, N=80, n_core 1.97 /
n_clad 1.444, device-1 corrugation 500 nm, y-symmetry off, ports-only monitors,
wide 80 nm scan). Saves the same coupling spectra and geometry to the .mat.

NOTE: set device_stagger_nm to whichever Δx the distance grid found most coupled
(or leave 6000 nm — near the near-axial lobe for a 1 µm gap) before dispatching.

Run locally (sequential):
    python -m runners.side_by_side.side_by_side_detune

Run on Athena as a parallel SLURM array (default partition):
    bash athena/deploy_athena.sh --option3 --spec=runners.side_by_side.side_by_side_detune
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.side_by_side.side_by_side_coupling import build_base


BASE = build_base()


# Representative geometry chosen from the distance grid (job 110724, 2026-06-28):
# TM coupling-into-device-2 peaks at gap 1.0 µm, Δx 8 µm (0.417). See
# results_from_athena/side_by_side_coupling/FINDINGS.md.
REP_GAP_NM     = 1000.0
REP_STAGGER_NM = 8000.0


SPEC = SweepSpec(
    n_devices              = [2],
    device_gap_nm          = [REP_GAP_NM],
    device_stagger_nm      = [REP_STAGGER_NM],
    corrugation_depth_2_nm = [400, 450, 475, 500, 525, 550, 600],   # device 1 fixed at 500
    polarization           = ["TE", "TM"],
    mode  = "cartesian",
    label = "side_by_side_detune",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
