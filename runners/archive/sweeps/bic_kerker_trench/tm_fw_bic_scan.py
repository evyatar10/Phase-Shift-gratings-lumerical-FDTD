"""
BATCH 1b — Friedrich–Wintgen single-resonance BIC scan (route 4.1b, the
headline). Gate verdict: PASS-conditional (docs/phase0_gate_verdicts_2026-07-06.md).

Physics. Two side-coupled pi-shift cavities sharing one radiation continuum
(FW-BIC / Dicke). Device 1 = the driven cavity (rect-1050, the current best
single device, loss 0.077). Device 2 = a PASSIVE pi-shift cavity, DETUNED by
its own average corrugation width (new avg_width_2_nm knob → shifts device 2's
n_eff → its defect resonance), coupled through the edge-to-edge gap. The CMT
(phase0_bic_cmt.py) predicts a radiative-loss collapse of device 1 when the
two resonances interfere at the right (coupling kappa, detuning D): up to
5.3x suppression with the partner pushed >15 nm out of the +-20 nm window ->
ONE usable resonance. kappa is set by the gap (guided lateral decay 1.75/um ->
tail e^{-1.75*gap}); D is set by avg_width_2.

WHAT THIS ARRAY IS. A LOCATE grid at OPTIMIZATION mesh (dx=50 nm) — it maps
where device 2's resonance lands vs avg_width_2 and whether any (gap, D) cell
shows device-1 loss BELOW the isolated reference. It does NOT make accurate
claims; survivors get an accurate-mesh + jitter confirm in a follow-up (per
CLAUDE.md §3). Two-device runs force y-symmetry OFF and enclose both guides,
so they are ~3-4x a single-device run — opt mesh keeps the grid inside the
window.

DOMAIN (physical-sanity gated). n_devices=2 ignores the single-device y=6.8
box; here an ABSOLUTE y-box override (10 um, new two-device path in
simulation_config.y_span) gives each guide ~3.3 um outer standoff (matching
the validated single-device +y half) while span_multiplier_override=5.42 keeps
the TM z-standoff large (z=8.8 um) — WITHOUT inflating the y-pad to ~20 um (the
naive coupling of y-pad to z). Row 0 is the box-sanity/isolated reference:
CHECK IT FIRST (T<=1, loss ~0.077 sane, resonance in window). If row 0 is
unphysical (T>1 / negative loss / off-window), the box is wrong — HALT and run
a two-device domain-convergence check before trusting any FW cell.

REGISTERED PREDICTIONS (honesty gate):
  * Reference rows (gap 1.5 um, kappa~0.07): device-1 loss ~ isolated rect-1050
    (~0.077 at this box), one peak, partner peak (device 2) appears detuned by
    D(avg_2) — this row PAIR maps D vs avg_2.
  * avg_2 = 800 (degenerate, row 13): the two identical-lambda cavities should
    SPLIT into a doublet (the de-prioritized two-resonance case) — a positive
    control that coupling is real. Expect TWO peaks in the window.
  * FW cells (gap 0.4-0.7 x avg_2 1000-1300): IF ρ (radiation-pattern overlap)
    >= ~0.82, at least one cell shows device-1 loss reduced >=1.5x vs the
    reference with the partner peak >=15 nm away (single usable resonance) and
    fwhm_m within ~1%. If NO cell drops below the reference across the whole
    (gap, D) plane, ρ is too small — FW-BIC is dead for this coupling geometry
    (report the null honestly; the CMT flagged ρ as the unknowable risk).

Rows (zipped, 20 tasks):
   0-1   reference gap 1.5 um, avg_2 {1000, 1100}  — isolated dev1 + D-map
   2-4   gap 0.40 um, avg_2 {1000, 1100, 1200}
   5-7   gap 0.55 um, avg_2 {1000, 1100, 1200}
   8-10  gap 0.70 um, avg_2 {1000, 1100, 1200}
   11    gap 0.55, avg_2 900   (smaller detuning)
   12    gap 0.55, avg_2 1150  (intermediate detuning)
   13    gap 0.55, avg_2 1300  (larger detuning)
   14    gap 0.55, avg_2 800   (degenerate → doublet positive control)
   15-17 TE: gap {0.40,0.55,0.70}, avg_2 1100, corr 300 (co-resonant)
   18-19 TE reference gap 1.5, avg_2 {1000,1100}, corr 300
   (jitter partners deferred to the accurate-mesh confirm of survivors: a
    half-cell 25 nm x-jitter snaps to 0 at the opt-mesh dx=50 nm.)

Dispatch (queue must be EMPTY of other --option3 arrays — serialize after
batch-1 job 118618 drains):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_fw_bic_scan
Output -> results/tm_fw_bic_scan/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


Y_BOX_UM = 10.0
BOX_Z_MULT = 5.42


def build_base() -> SimulationConfig:
    """Anchored TM device (pitch 516.83, corr 400, h350, n 1.97/1.444), narrow
    window, two-device pair. Pinned explicitly (no env reads)."""
    cfg = SimulationConfig()
    cfg.grating.pitch_m = 516.83e-9
    cfg.grating.n_periods_each_side = 80
    cfg.apodization.enabled = False
    cfg.geometry.corrugation_depth_m = 400e-9
    cfg.geometry.core_height_m = 350e-9
    cfg.material.use_constant_materials = True
    cfg.material.n_core_const = 1.97
    cfg.material.n_clad_const = 1.444
    cfg.mesh.simulation_mode = "optimization"
    cfg.source.polarization = "TM"

    # Two-device pair: driven device 1 = rect-1050 (cavity_width override set
    # per-row below only for TM; TE uses avg cavity). Device 2 = passive
    # avg-cavity pi-shift grating, corr 400, detuned via avg_width_2.
    cfg.geometry.n_devices = 2
    cfg.geometry.corrugation_depth_2_m = 400e-9

    # Wide window: must hold BOTH the device-1 peak (~1558.6) and the detuned
    # device-2 partner peak (up to +40 nm) so the single-resonance check can see
    # where the partner lands. 60 nm / 4001 pts (~15 pm).
    cfg.spectral.center_wavelength_m = 1.5685e-6
    cfg.spectral.scan_width_nm = 60.0
    cfg.spectral.n_wl_points = 4001

    cfg.span_multiplier_override = BOX_Z_MULT      # z-standoff (TM); y set absolutely
    cfg.y_span_override_m = Y_BOX_UM * 1e-6        # absolute pair y-box (decoupled from z)

    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled = False
    return cfg


BASE = build_base()

# ── Row table ────────────────────────────────────────────────────────────────
# (pol, corr_nm, cavW_nm, gap_um, avg2_nm). Jitter partners are DEFERRED to the
# accurate-mesh confirm of survivors (a half-cell x-jitter of 25 nm snaps to 0
# at the opt-mesh dx=50 nm, so it is meaningless here).
rows = [
    ("TM", 400.0, 1050.0, 1.50, 1000.0),   # 0 reference (CHECK FIRST: box sanity)
    ("TM", 400.0, 1050.0, 1.50, 1100.0),   # 1 reference
    ("TM", 400.0, 1050.0, 0.40, 1000.0),   # 2
    ("TM", 400.0, 1050.0, 0.40, 1100.0),   # 3
    ("TM", 400.0, 1050.0, 0.40, 1200.0),   # 4
    ("TM", 400.0, 1050.0, 0.55, 1000.0),   # 5
    ("TM", 400.0, 1050.0, 0.55, 1100.0),   # 6
    ("TM", 400.0, 1050.0, 0.55, 1200.0),   # 7
    ("TM", 400.0, 1050.0, 0.70, 1000.0),   # 8
    ("TM", 400.0, 1050.0, 0.70, 1100.0),   # 9
    ("TM", 400.0, 1050.0, 0.70, 1200.0),   # 10
    ("TM", 400.0, 1050.0, 0.55, 900.0),    # 11 smaller detuning
    ("TM", 400.0, 1050.0, 0.55, 1150.0),   # 12 intermediate detuning
    ("TM", 400.0, 1050.0, 0.55, 1300.0),   # 13 larger detuning
    ("TM", 400.0, 1050.0, 0.55, 800.0),    # 14 degenerate → doublet positive control
    ("TE", 300.0, None, 0.40, 1100.0),     # 15
    ("TE", 300.0, None, 0.55, 1100.0),     # 16
    ("TE", 300.0, None, 0.70, 1100.0),     # 17
    ("TE", 300.0, None, 1.50, 1000.0),     # 18 TE reference
    ("TE", 300.0, None, 1.50, 1100.0),     # 19 TE reference
]

assert len(rows) == 20, f"task count changed: {len(rows)}"
_sig = [tuple(r) for r in rows]
assert len(set(_sig)) == len(rows), "duplicate row -> file-tag collision"

SPEC = SweepSpec(
    polarization           = [r[0] for r in rows],
    corrugation_depth_nm   = [r[1] for r in rows],
    corrugation_depth_2_nm = [r[1] for r in rows],   # device 2 tracks device 1's grating strength
    cavity_width_nm        = [r[2] for r in rows],
    device_gap_nm          = [r[3] * 1000.0 for r in rows],
    avg_width_2_nm         = [r[4] for r in rows],
    mode  = "zipped",
    label = "tm_fw_bic_scan",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
