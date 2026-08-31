#!/bin/bash
# ONE-SHOT lumopt2 scratch cleaner - run from cron (a login-node daemon kept
# dying with the session; that is why quota kept climbing unattended).
#
# PASS 1: within each STUDY (*_files dir) keep the newest FOUR *_output.h5
# older than 4 h, so a running iteration's forward + BOTH adjoints always
# survive. ★2026-08-29: the original "keep 2, -mmin +30" KILLED campaign c1
# twice (137985 incarnations 3+4): a slow resumed gradient holds its forward
# h5 live for ~2.5 h, and with port-adj + width-adj outputs newer the forward
# was rank 3 -> deleted mid-gradient -> "Can not find result 'E'".
# PASS 2 (added 2026-08-25): pass 1 ALONE FREED NOTHING. Each study dir holds
# only 2-3 files, so "keep 2" kept essentially everything: 22 dead study dirs
# were sitting on 85 GB with quota at 289G/300G. A study dir whose newest h5 is
# over 24 h old belongs to a job that has already finished, so all of its
# scratch goes.
#
# Only *_output.h5 is ever removed. .mat / .jsonl results are never touched.
CUTOFF=$(( $(date +%s) - 86400 ))
find "$HOME/bragg_sim_athena/results" -type d -name '*_files' 2>/dev/null |
while read -r base; do
  newest=$(find "$base" -name '*_output.h5' -printf '%T@\n' 2>/dev/null |
             sort -rn | head -1 | cut -d. -f1)
  if [ -n "$newest" ] && [ "$newest" -lt "$CUTOFF" ]; then
    # pass 2 - dead study, drop all of its scratch
    find "$base" -name '*_output.h5' -print0 2>/dev/null | xargs -0 -r rm -f
    continue
  fi
  # pass 1 - live study, keep the newest four (fwd + 2 adjoints + margin);
  # 4 h floor > the slowest measured resumed-iterate need-window (~2.5 h)
  find "$base" -name '*_output.h5' -mmin +240 -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | tail -n +5 | cut -d' ' -f2- | xargs -r rm -f
done
