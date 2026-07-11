"""
Study: CLOSED-device-2 recycler — maximize the FIRST π-shift's transmission.

Same side-by-side geometry as side_by_side_coupling.py, but device 2 is CLOSED
(`device2_closed=True`): it has NO access/feed waveguides and NO drain ports, so
the power it catches from device 1's radiation cannot escape as guided power — it
re-radiates, with a fraction returning into device 1's forward guided mode. This
is the actual *recycling* test, after the open-device-2 grid (job 110724) showed
an open device 2 only DRAINS (device-1 T is highest when device 2 is decoupled).

Figure of merit = **device-1 transmission T (|S21|², Port_1→Port_2)** — we want a
(Δy gap, Δx stagger) position where the closed recycler RAISES device-1 T above the
isolated value. Most promising for TM (more in-plane radiation to recover).

Grid is identical to side_by_side_coupling so the two are directly comparable:
  device_gap_nm (Δy)  : {1000, 1500, 2000}
  device_stagger_nm   : {0, 500, 1000, 2000, 3000, 4000, 6000, 8000}
  polarization        : {TE, TM}
  => 48 tasks. Both devices 500 nm corrugation, n_core 1.977.

Device 2 has no ports, so the saved .mat carries device-1 T/R/loss only (no
coupling_* fields). Compare device-1 T here vs the open grid at the same (gap, Δx).

Run on Athena (default partition) — ONLY when no other --option3 job is running
(shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.side_by_side.side_by_side_recycle
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.side_by_side.side_by_side_coupling import build_base


def build_recycle_base():
    cfg = build_base()                 # pitch 500, N=80, 500 nm corr, n_core 1.977, y-sym off, ports-only
    cfg.geometry.device2_closed = True  # closed recycler (no feed waveguides / drain ports)
    return cfg


BASE = build_recycle_base()


SPEC = SweepSpec(
    n_devices              = [2],
    corrugation_depth_2_nm = [500],                  # device 2 = device 1 (identical) recycler
    device_gap_nm          = [1000, 1500, 2000],
    device_stagger_nm      = [0, 500, 1000, 2000, 3000, 4000, 6000, 8000],
    polarization           = ["TE", "TM"],
    mode  = "cartesian",
    label = "side_by_side_recycle",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
