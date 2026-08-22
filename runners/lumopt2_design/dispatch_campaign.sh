#!/bin/bash
# ONE-COMMAND campaign dispatch for the corr-325 lumopt2 inverse design.
#
#   bash runners/lumopt2_design/dispatch_campaign.sh seedA   # Athena (main)
#   bash runners/lumopt2_design/dispatch_campaign.sh seedB   # IGUM  (2nd seed)
#
# All physics parameters live in campaign_c325_seedA.py / seedB.py (constants
# at the top) — this script only carries the CLUSTER knobs and the dispatch
# checklist. PREREQUISITE (hard): gates B0-B4 PASSED and the gradient
# calibration decision made (see .claude/skills/lumopt2-design/SKILL.md —
# as of 2026-08-15 the campaign is PARKED on the lumopt2 boundary-gradient
# limitation; do not run this until that is resolved/signed off).

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── cluster knobs (the physics is in the campaign runner files) ─────────────
ATHENA_QOS="4d_1g"          # default 24h_1g (23:30) kills the ~28 h driver
ATHENA_TIME="96:00:00"
MEM="160G"                  # measured usage ~6.5 GB — huge margin, cheap
IGUM_TIME="23:30:00"        # seed B ≈ 13-16 h fits the IGUM default window

case "${1:-}" in
  seedA)
    echo "== seed A → Athena =="
    echo "checklist: queue empty? license seats probed from IGUM? gates green?"
    ssh evyatarrubin@athena.technion.ac.il "squeue -u evyatarrubin -h -r" \
        | grep . && { echo "ABORT: Athena queue not empty (serialize rule)"; exit 1; }
    LUMOPT2_QOS="${ATHENA_QOS}" LUMOPT2_TIME="${ATHENA_TIME}" SBATCH_MEM="${MEM}" \
        bash athena/deploy_athena.sh \
        --lumopt2-design=runners.lumopt2_design.campaign_c325_seedA
    ;;
  seedB)
    echo "== seed B → IGUM (dispatch only AFTER seed A runs healthily) =="
    ssh igum "squeue -u \$USER -h" \
        | grep . && { echo "ABORT: IGUM queue not empty (serialize rule)"; exit 1; }
    LUMOPT2_TIME="${IGUM_TIME}" SBATCH_MEM="${MEM}" \
        bash igum/deploy_igum.sh \
        --lumopt2-design=runners.lumopt2_design.campaign_c325_seedB
    ;;
  *)
    echo "usage: $0 seedA|seedB"; exit 1 ;;
esac
