"""PLAIN TE pi-shift device: cavity-SIZE control knobs (width +-20 nm, length +-20 nm) for the junction-shoulder result.

Study dir: runners/sweeps/ | Created 2026-10-21 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Why (user 2026-10-21: "check that we're not just enlarging the cavity"): the 10-nm shoulder next to the +125 nm step (266507_0)
cut the plain TE leak to 0.958 with overlap eta^2 0.34, but it adds material right beside the cavity segment. These rows measure
what a pure cavity resize does to the far field. Tests (scratch solve_v4.py / a projection script vs the stored round-A control):
(1) fraction of the shoulder's far-field change lying in span{b_width, b_length} (complex) - near 1 => it is a cavity resize;
(2) best reduction from cavity resizing alone (2-knob solve, lambda/W reported) - if >= the shoulder's, the plain cavity was simply
not at its optimal size. Plain TE has never had its cavity size scanned (memory: TE cavity width never measured).
Cards on the expanded round-A TE row (same numerics as the stored control: 150 ps, 1558.8 +- 5 nm, box 6.8, no scatterer):
Dispatch (Athena): bash athena/deploy_athena.sh --cards=runners.sweeps.farfield_sph_15um_cavsize_plain --max-concurrent=1
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "A"
from runners.sweeps.farfield_sph_15um import SPEC as SPEC_A, BASE as BASE_A  # noqa: E402
from experiment_card import ExperimentCard  # noqa: E402

BASE = next(iter(SPEC_A.expand(base=BASE_A)))           # row 0 = plain TE (corr 316, pitch 500, W800, N96)
W0_NM, L0_NM = 800.0, 250.0                              # cavity segment: mean width, pitch/2
CARDS = [ExperimentCard(cavity_width_nm=W0_NM + 20, label="cavW_p20nm"),
         ExperimentCard(cavity_width_nm=W0_NM - 20, label="cavW_m20nm"),
         ExperimentCard(cavity_neg_detuning_nm=-20.0, label="cavL_p20nm"),     # length = pitch/2 - this (cavity_length_nm is IGNORED
         ExperimentCard(cavity_neg_detuning_nm=+20.0, label="cavL_m20nm")]     # by to_device_kwargs - caught pre-dispatch 2026-10-21)
# cards 4-9 APPENDED 2026-10-21 (user: find the OPTIMAL cavity first, then add features on top of it). The TM plain device's optimum
# was a rect cavity of 1050 nm (TM corr 400 / pitch 516.83 / W800: loss -30 %, flat 1040-1075; memory project_loss_exploration_chain)
# - 250 nm away from 800, so +-20 nm (cards 0-1: 0.988 / 0.990) cannot see it. Wide TE ladder; dlambda/dW ~0.0023 nm/nm (MEASURED +-20)
# keeps the resonance well inside the +-5 nm window up to 1100 nm.
CARDS += [ExperimentCard(cavity_width_nm=w, label=f"cavW_{w:.0f}nm") for w in (850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0)]

if __name__ == "__main__":
    expect = {"cavW_p20nm": (820e-9, False), "cavW_m20nm": (780e-9, False), "cavL_p20nm": (None, 270.0), "cavL_m20nm": (None, 230.0)}
    expect.update({f"cavW_{w:.0f}nm": (w * 1e-9, False) for w in (850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0)})
    for card in CARDS:                                   # property asserts (CLAUDE.md 5): the size change must reach the builder kwargs
        kw = card.to_config(base=BASE).to_device_kwargs(); w, L = expect[card.label]
        assert (kw["cavity_width_m"] is None if w is None else abs(kw["cavity_width_m"] - w) < 1e-12), card.label
        assert kw["override_cavity_length_nm"] == L or (L and abs(kw["override_cavity_length_nm"] - L) < 1e-9), (card.label, kw["override_cavity_length_nm"])
    for card in CARDS:
        cfg = card.to_config(base=BASE); kw = cfg.to_device_kwargs()
        print(f"{card.label:11s}: cavity width {kw.get('cavity_width_m')} option {kw.get('cavity_width_option')} | cavity length override "
              f"{kw.get('override_cavity_length_nm')} nm | scatterer {cfg.scatterer.enabled} | corr {kw.get('corrugation_depth_m')} pitch "
              f"{kw.get('pitch')} | sim ps {os.environ.get('TM_SIM_TIME_PS')} short {cfg.farfield.short_run}")
