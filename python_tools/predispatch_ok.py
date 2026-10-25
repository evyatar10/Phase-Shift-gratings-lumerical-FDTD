# predispatch_ok.py (2026-10-25; optional one-command preflight, blocks nothing): runs the dispatch target's own asserts and the
# IGUM seat probe, prints both, and records them in C:/Users/evyat/.claude/dispatch_ok.json; exit 1 if asserts fail or seats > 42/50
# (CLAUDE.md sec. 6 seat bands).
#   python python_tools/predispatch_ok.py runners.sweeps.farfield_sph_15um_postarray_overshoot_x42
import json, os, re, subprocess, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKER = os.path.expanduser("~/.claude/dispatch_ok.json")
target = sys.argv[1]

# asserts: a post-array wrapper only sets FF15_PA_SET and imports; the asserts live in the main runner's __main__
cmd = [sys.executable, "-m", target]
env = dict(os.environ)
wrapper = os.path.join(REPO, *target.split(".")) + ".py"
m = re.search(r'os\.environ\["FF15_PA_SET"\] = "(\w+)"', open(wrapper, encoding="utf-8").read()) if os.path.exists(wrapper) else None
if m:
    env["FF15_PA_SET"] = m.group(1)
    cmd = [sys.executable, "-m", "runners.sweeps.farfield_sph_15um_postarray_overshoot"]
r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
asserts_ok = r.returncode == 0
print(r.stdout[-1500:] if asserts_ok else r.stderr[-3000:])

# seats (from IGUM: Athena's lmstat is a false negative)
s = subprocess.run(["ssh", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "igum",
                    "~/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261/licensingclient/linx64/lmutil lmstat -c 1055@132.68.48.51 -f lum_fdtd_solve"],
                   capture_output=True, text=True)
q = re.search(r"Total of (\d+) licenses issued;\s+Total of (\d+) licenses in use", s.stdout)
seats_used = int(q.group(2)) if q else None

json.dump(dict(target=target, asserts_ok=asserts_ok, seats_used=seats_used, time=time.strftime("%Y-%m-%d %H:%M:%S")), open(MARKER, "w"))
print(f"target {target}: asserts {'OK' if asserts_ok else 'FAILED'}, seats {seats_used}/50 -> marker {MARKER}")
sys.exit(0 if asserts_ok and seats_used is not None and seats_used <= 42 else 1)
