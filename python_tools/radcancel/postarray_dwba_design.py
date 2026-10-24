# postarray_dwba_design.py (study: runners/sweeps/farfield_sph_15um_postarray_overshoot, 2026-10-23): free 2-D post design on a Gram target
# (control C, or a measured-residual target C' from postarray_dwba.py), then save the continuous design
# (the continuous polish interpolates response columns: -11 % |B|^2 at x mid-points) -> re-score the result with direct Born.
#   python postarray_dwba_design.py <target.npz: dwba npz or a Gram npz> <out.npz> X_LIM_NM Y_LO_NM Y_HI_NM GAP_NM [S] seed ...
#   seed = design npz (x,y,r or x_nm,...) | json:<name in postarray_overshoot_designs.json> | empty
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postarray_place2d as pp
tgt_f, out_f, XL, YLO, YHI, GAP = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:7]); rest = sys.argv[7:]
S = float(rest.pop(0)) if rest and rest[0].replace(".", "").isdigit() else 0.70
d = np.load(tgt_f); dwba = "gram" in d.files
gm = pp.Gram(str(d["gram"]) if dwba else tgt_f, s=S); b0, cc0 = gm.b.copy(), gm.cc          # control target (arch Gram)
if dwba:
    assert abs(float(d["s"]) - S) < 1e-9, "the DWBA target was built with a different response scale s"
    gm.b, gm.cc = d["b"], float(d["cc"])
DES = json.load(open(r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/runners/sweeps/postarray_overshoot_designs.json"))
def load_seed(s):
    if s == "empty": return [np.zeros(0)] * 3
    if s.startswith("json:"): D = DES[s[5:]]; return [np.array(D[k], float) for k in ("x_nm", "y_nm", "r_nm")]
    z = np.load(s); return [z[k] if k in z.files else z[k + "_nm"] for k in ("x", "y", "r")]
seeds = [load_seed(s) for s in rest] or [[np.zeros(0)] * 3]
rel = lambda P: P * gm.cc / cc0                                   # leak relative to the CONTROL (what Q_i responds to)
ctrl_P = lambda A: (cc0 + 2 * S * b0 @ A + S * S * A @ gm.mv(A)) / cc0
_cont = out_f.replace(".npz", "_cont.npz")
if os.environ.get("REUSE_CONT") and os.path.exists(_cont):
    z = np.load(_cont); best = pp.Design(z["x"], z["y"], z["r"], float(z["P"]) * cc0 / gm.cc)
else:
    best = pp.design(gm, GAP, XL, YLO, YHI, seeds, label="design")
print(f"continuous design: N {best.N} model P/Pc {rel(best.P):.4f}")
np.savez(out_f.replace(".npz", "_cont.npz"), x=best.x, y=best.y, r=best.r, P=rel(best.P), s=S, target=tgt_f)
# no snapping (GPT 2026-10-23: the snapped-radius solve picked the wrong interpolation corner; continuous designs are verified with
# direct Born instead, scratch direct_dwba.py pattern: postarray_born.posts_ff on the exported x/y/r)
np.savez(out_f, x=best.x, y=best.y, r=best.r, P=rel(best.P), s=S, target=tgt_f, gap=GAP, x_lim=XL, y_lo=YLO, y_hi=YHI)
