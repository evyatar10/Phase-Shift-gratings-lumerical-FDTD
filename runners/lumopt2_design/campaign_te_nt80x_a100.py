"""Twin of campaign_te_nt80x on the A100 pool (own label/results dir lumopt2_te_nt80x_grad_a100).

Study dir: runners/lumopt2_design/ | Created 2026-10-20
Why: 265121 (the H200 copy) sits behind 35 jobs of another user on the single H200 node; the gradient is the route-A input
for the Nt80x and the user is away. Whichever finishes first is used; the other is to be cancelled by the user.
Dispatch: SBATCH_MEM=256G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=36:00:00 bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_te_nt80x_a100 --gpu=a100
"""
import os
os.environ["NT80X_LABEL_SUFFIX"] = "_a100"
from runners.lumopt2_design.campaign_te_nt80x import *  # noqa: E402,F401,F403
