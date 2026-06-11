"""
Two apodized pi-shift Bragg resonators with explicit per-tooth widths.

Both devices: 40 periods each side (20 apodized + 20 uniform outer), pitch 500 nm,
500 nm corrugation in the uniform section, scan 1560 ± 100 nm, default materials
(n_core 1.977, n_clad 1.44), accurate mesh.

The 20 apodized teeth per side are given EXPLICITLY (wide + narrow width arrays).
The arrays below are quoted verbatim in the order the user supplied them — FURTHEST
from the cavity -> CLOSEST to the cavity. The device builder expects innermost-first
(d=1 = tooth nearest the cavity), so each array is reversed with [::-1] at the card.
Teeth d=21..40 (outside the apodization) fall back to the uniform full 500 nm
corrugation (wide 1050 / narrow 550 nm) automatically (apod_method='none').

Resonator 1 — "detuning", wide cavity:
    cavity width 800 nm, cavity length 180 nm (= pitch/2 − 70).
    Average tooth width constant at 800 nm; only depth tapers (482 -> 4 nm).

Resonator 2 — "node tuning":
    cavity width 753 nm, cavity length 250 nm (= pitch/2, no detuning).
    Average tooth width tapers 796.5 -> 753 nm toward the cavity; depth tapers too.

Run both as two parallel SLURM-array tasks on Athena:
    bash athena/deploy_athena.sh --option3 --cards=runners.experiment_comparison.two_resonators
    # wait for completion, then pull:
    bash athena/deploy_athena.sh --results
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment_card import ExperimentCard
from runners.experiment_comparison.it11_card_builder import DeviceRecord
from simulation_config import SimulationConfig


# ── Shared base config ───────────────────────────────────────────────────────
BASE = SimulationConfig()
BASE.mesh.simulation_mode         = "accurate"      # dx ≈ 35 nm
BASE.spectral.center_wavelength_m = 1560e-9         # band 1460–1660 nm
BASE.spectral.scan_width_nm       = 200.0
# Materials left at SimulationConfig defaults (n_core 1.977, n_clad 1.44).


# ── Explicit per-tooth widths (nm), ordered FURTHEST -> CLOSEST to center ──────
# Resonator 1 (wide cavity, "detuning")
R1_WIDE = [1041, 1032, 1023, 1013, 1003, 992, 980, 969, 957, 944,
           931, 918, 904, 890, 876, 861, 847, 832, 817, 802]
R1_NARROW = [559, 568, 577, 587, 597, 608, 620, 631, 644, 656,
             669, 682, 696, 710, 724, 739, 753, 768, 783, 798]

# Resonator 2 ("node tuning")
R2_WIDE = [1038, 1026, 1014, 1001, 987, 973, 959, 944, 929, 913,
           898, 882, 866, 850, 834, 818, 802, 786, 771, 755]
R2_NARROW = [555, 561, 568, 575, 582, 590, 598, 607, 616, 626,
             636, 646, 658, 670, 682, 695, 709, 722, 737, 751]


# ── Cards (one parallel task each) ───────────────────────────────────────────
CARDS = [
    ExperimentCard(
        n_periods_each_side       = 40,
        pitch_nm                  = 500,
        corrugation_depth_nm      = 500,        # uniform outer teeth: wide 1050 / narrow 550
        apod_method               = "none",     # explicit arrays encode the taper
        cavity_width_nm           = 800,
        cavity_neg_detuning_nm    = 70,         # cavity length = 250 − 70 = 180 nm
        width_wide_per_tooth_nm   = R1_WIDE[::-1],    # reverse -> innermost-first (d=1)
        width_narrow_per_tooth_nm = R1_NARROW[::-1],
        label                     = "res1_wide_detune",
    ),
    ExperimentCard(
        n_periods_each_side       = 40,
        pitch_nm                  = 500,
        corrugation_depth_nm      = 500,
        apod_method               = "none",
        cavity_width_nm           = 753,
        cavity_neg_detuning_nm    = 0,          # cavity length = pitch/2 = 250 nm
        width_wide_per_tooth_nm   = R2_WIDE[::-1],
        width_narrow_per_tooth_nm = R2_NARROW[::-1],
        label                     = "res2_node_tune",
    ),
]

RECORDS = [
    DeviceRecord(idx=0, device_id="res1", folder="", folder_long="",
                 subname="res1_wide_detune", pitch_nm=500, mat_dir="",
                 label="res1_wide_detune"),
    DeviceRecord(idx=1, device_id="res2", folder="", folder_long="",
                 subname="res2_node_tune", pitch_nm=500, mat_dir="",
                 label="res2_node_tune"),
]


if __name__ == "__main__":
    for card, (wide, narrow) in zip(CARDS, [(R1_WIDE, R1_NARROW), (R2_WIDE, R2_NARROW)]):
        assert len(card.width_wide_per_tooth_nm) == 20, "expected 20 apodized teeth (wide)"
        assert len(card.width_narrow_per_tooth_nm) == 20, "expected 20 apodized teeth (narrow)"
        depths = [w - n for w, n in zip(wide, narrow)]
        assert all(d >= 0 for d in depths), f"negative corrugation in {card.label!r}"
        avg_in  = 0.5 * (wide[-1] + narrow[-1])   # closest to cavity
        avg_out = 0.5 * (wide[0] + narrow[0])     # furthest from cavity
        print(f"{card.label}: cavity_width={card.cavity_width_nm} nm, "
              f"detuning={card.cavity_neg_detuning_nm} nm "
              f"(cavity_len={500/2 - card.cavity_neg_detuning_nm:.0f} nm)")
        print(f"    depth: {depths[-1]} nm (cavity-side) -> {depths[0]} nm (outer apod)")
        print(f"    avg width: {avg_in:.1f} nm (cavity-side) -> {avg_out:.1f} nm (outer apod)")
    print(f"\n{len(CARDS)} cards ready.")
