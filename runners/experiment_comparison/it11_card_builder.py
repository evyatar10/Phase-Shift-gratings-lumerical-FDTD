"""
Parser for the IT11 fabricated-device list.

Reads `device_names.csv` from the experiment-analysis folder and produces a
list of ExperimentCards (one per device × pitch) plus a parallel list of
RECORDS dicts that carry the experiment-side metadata needed at comparison time
(folder, .mat path, raw subname).

Naming convention (cross-checked against analyze_IT11.m:515,956-981):
    corrN     -> geometry.corrugation_depth_m = N * 1e-9
    NpN       -> grating.n_periods_each_side = N
    NtN       -> apodization enabled with n_apod_periods_each_side = N
                 (absent => apodization disabled)
    dwmN      -> apodization.center_mod_depth_nm = N
    tsN       -> grating.innermost_tooth_shift_m = N * 1e-9
                 (absent => 0)

Cavity-width logic (from user):
    apod off              -> cavity_width_option = "narrow"
    apod on,  ts == 0     -> cavity_width_option = "avg"
    apod on,  ts >  0     -> cavity_width_option = "avg_ext"

Each CSV row is emitted twice — once at pitch_nm=500 and once at pitch_nm=516.
The full CARDS list is ordered all-500-first, then all-516, so a SLURM array
with task indices [0..N-1] runs the 500 nm batch before any 516 nm task.

Rows whose Subname contains "tanh_a" are Itay's chirped designs and are
skipped (per user instruction).
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from experiment_card import ExperimentCard


EXPERIMENT_ROOT = r"C:\Users\evyat\MATLAB\bragg_resonator_codes\new_experiment_analysis"
# Prefer the Windows experiment-analysis copy; fall back to a sibling copy
# bundled with this package so the builder works on remote clusters (Athena)
# where the MATLAB folder isn't mounted. Keep the two in sync — re-run
# `cp <experiment-analysis>/device_names.csv runners/experiment_comparison/` if
# devices are added.
_WIN_CSV     = os.path.join(EXPERIMENT_ROOT, "device_names.csv")
_LOCAL_CSV   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "device_names.csv")
EXPERIMENT_CSV = _WIN_CSV if os.path.exists(_WIN_CSV) else _LOCAL_CSV
MAT_ROOT        = os.path.join(EXPERIMENT_ROOT, "2026_04_27_IT11")

# Folder short -> long-name mapping (mirrors analyze_IT11.m:11-15).
FOLDER_LONG = {
    "left_down":  "left_down_311deg",
    "left_up":    "left_up_311deg",
    "right_down": "right_down_311deg",
    "right_up":   "right_up_311deg",
}

PITCHES_NM: Tuple[int, ...] = (500, 516)


@dataclass
class DeviceRecord:
    """Experiment-side metadata for one (device, pitch) pair, parallel to the card list."""
    idx:         int
    device_id:   str
    folder:      str
    folder_long: str
    subname:     str
    pitch_nm:    int
    mat_dir:     str   # directory of the .mat file (filename has timestamp; resolved at compare time)
    label:       str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ─────────────────────────────────────────────────────────────────────────────
# Subname parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_subname(
    subname: str,
    folder:  str,
    id_:     str,
    pitch_nm: int,
) -> Optional[ExperimentCard]:
    """Build an ExperimentCard from one CSV row at a chosen pitch.

    Returns None for Itay rows (tanh_a* — to be skipped).
    """
    if "tanh_a" in subname:
        return None

    m_corr = re.search(r"corr(\d+)", subname)
    m_np   = re.search(r"Np(\d+)",   subname)
    if not m_corr or not m_np:
        raise ValueError(f"Subname missing corr/Np tokens: {subname!r} (id={id_})")
    corr = int(m_corr.group(1))
    Np   = int(m_np.group(1))

    m_nt  = re.search(r"Nt(\d+)",  subname)
    m_dwm = re.search(r"dwm(\d+)", subname)
    m_ts  = re.search(r"ts(\d+)",  subname)
    Nt  = int(m_nt.group(1))    if m_nt  else None
    dwm = float(m_dwm.group(1)) if m_dwm else None
    ts  = float(m_ts.group(1))  if m_ts  else 0.0

    apod_on = Nt is not None and Nt > 0

    if not apod_on:
        cavity_opt = "narrow"
    elif ts == 0:
        cavity_opt = "avg"
    else:
        cavity_opt = "avg_ext"

    return ExperimentCard(
        n_periods_each_side       = Np,
        corrugation_depth_nm      = float(corr),
        n_apod_periods_each_side  = Nt if apod_on else 0,
        center_mod_depth_nm       = dwm if apod_on else None,
        apod_method               = "linear" if apod_on else "none",
        innermost_tooth_shift_nm  = ts,
        pitch_nm                  = float(pitch_nm),
        cavity_width_option       = cavity_opt,
        lengthen_cavity           = (ts > 0),
        label                     = f"{folder}_{id_}_pitch{int(pitch_nm)}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV walker
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv_rows(csv_path: str) -> List[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_it11(
    csv_path: str = EXPERIMENT_CSV,
    pitch:    Optional[int] = None,
) -> Tuple[List[ExperimentCard], List[DeviceRecord]]:
    """
    Build (CARDS, RECORDS) for the IT11 device set.

    Ordering: all 500 nm cards come before any 516 nm card. The two lists are
    parallel — RECORDS[i] describes CARDS[i].

    Args:
        csv_path: device_names.csv path (defaults to EXPERIMENT_CSV).
        pitch:    optional filter: 500, 516, or None (=both, 500 first).
    """
    rows = _read_csv_rows(csv_path)
    pitches = (pitch,) if pitch is not None else PITCHES_NM

    cards:   List[ExperimentCard] = []
    records: List[DeviceRecord]   = []

    # Two passes (or one if filtered): all 500 first, then all 516.
    for p in pitches:
        for row in rows:
            id_     = (row.get("ID")      or "").strip()
            folder  = (row.get("Folder")  or "").strip()
            subname = (row.get("Subname") or "").strip()
            if not id_ or not folder or not subname:
                continue
            if folder not in FOLDER_LONG:
                continue
            card = parse_subname(subname, folder, id_, p)
            if card is None:
                continue
            idx = len(cards)
            mat_dir = os.path.join(MAT_ROOT, FOLDER_LONG[folder])
            records.append(DeviceRecord(
                idx         = idx,
                device_id   = id_,
                folder      = folder,
                folder_long = FOLDER_LONG[folder],
                subname     = subname,
                pitch_nm    = int(p),
                mat_dir     = mat_dir,
                label       = card.label,
            ))
            cards.append(card)

    return cards, records


def build_it11_cards(csv_path: str = EXPERIMENT_CSV, pitch: Optional[int] = None) -> List[ExperimentCard]:
    """Convenience wrapper: just CARDS (drop RECORDS)."""
    cards, _ = build_it11(csv_path, pitch=pitch)
    return cards


def build_it11_records(csv_path: str = EXPERIMENT_CSV, pitch: Optional[int] = None) -> List[DeviceRecord]:
    """Convenience wrapper: just RECORDS."""
    _, records = build_it11(csv_path, pitch=pitch)
    return records
