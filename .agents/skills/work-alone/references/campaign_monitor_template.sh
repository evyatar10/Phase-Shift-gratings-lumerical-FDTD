#!/bin/bash
# Gen-5 campaign monitor v2 (2026-08-17). Fixes the v1 defect that a failed ssh
# silently DROPPED a cluster's block, which reads identically to "the job is
# gone" (false alarm raised on seedA at 05:30). v2 emits an explicit
# <CLUSTER>_UNREACHABLE token, which also debounces naturally: the key stays
# constant for the whole outage, so one wake per outage, not per sweep.
# Change key = decision-relevant state ONLY (job states, eval counts, last-eval
# physics, error counts) — never elapsed time (CLAUDE.md / work-alone rule).
ATH=evyatarrubin@athena.technion.ac.il
IGM=evyatarrubin@132.68.58.101
SB=/home/evyatarrubin/research/bragg_sim_igum/results/campaign_c325_seedB/results/lumopt2_c325_seedB/lumopt2_c325_seedB_evals.jsonl
BR=/home/evyatarrubin/research/bragg_sim_igum/results/campaign_c325_bare/results/lumopt2_c325_bare/lumopt2_c325_bare_evals.jsonl
SA=/home/evyatarrubin/bragg_sim_athena/results/campaign_c325_seedA/results/lumopt2_c325_seedA/lumopt2_c325_seedA_evals.jsonl
LAST='import json,sys
rows=[json.loads(l) for l in sys.stdin if l.strip()]
r=rows[-1] if rows else None
print("n=%d"%len(rows), ("last eval=%s T=%.4f lam=%.3f sig=%.3f fom=%.4f"%(r["eval"],r["t_pk"],r["lam_pk_nm"],r["sigma_um"],r["fom"])) if r else "none")'
strip() { grep -vE "post-quantum|openssh|may need to be upgraded"; }
prev=""
while true; do
  a=$(ssh -o ConnectTimeout=25 $ATH "squeue -r -u evyatarrubin -o '%i %T' | tail -5; \
      [ -f $SA ] && python3 -c '$LAST' < $SA; \
      grep -c -iE 'Traceback|TASK FAILED|LumApiError|Unable to checkout|DIVERGED' \
        \$(ls -t ~/bragg_sim_athena/jobs/logs/*133276* 2>/dev/null | head -1) 2>/dev/null" 2>/dev/null | strip)
  [ -z "$a" ] && a="ATHENA_UNREACHABLE"
  i=$(ssh -o ConnectTimeout=25 $IGM "squeue -u evyatarrubin -o '%i %T' | tail -5; \
      [ -f $SB ] && python3 -c '$LAST' < $SB; [ -f $BR ] && python3 -c '$LAST' < $BR; \
      grep -c -iE 'Traceback|TASK FAILED|LumApiError|Unable to checkout|DIVERGED' \
        /home/evyatarrubin/research/bragg_sim_igum/jobs/logs/lum_array-54968_0.out 2>/dev/null" 2>/dev/null | strip)
  [ -z "$i" ] && i="IGUM_UNREACHABLE"
  key="$a|$i"
  if [ "$key" != "$prev" ]; then
    echo "CAMPAIGN v4 -- ATHENA: $a ;; IGUM: $i" | tr '\n' ' '; echo
    prev="$key"
  fi
  # ★IGUM CONNECTION BUDGET (burned 2026-08-17): v2 polled IGUM 24x/h (2 conns
  # per 300 s) and IGUM began refusing our key ~80 min later; ~45 min of ZERO
  # connections restored it — a rate-limit/fail2ban trip. v4 therefore makes
  # ONE IGUM connection per 20 min (3/h, half of v1's 6/h) and folds the
  # license probe into that SAME connection rather than opening a second one.
  sleep 1200
done
