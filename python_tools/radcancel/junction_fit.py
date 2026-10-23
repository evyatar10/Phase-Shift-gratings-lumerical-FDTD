# scratch 2026-10-21: two-junction vector-current model of the PLAIN TE leak (GPT 01a15a66 session 21), fitted to the complex sphere field.
# F(n) = H(n_z) sum_j e^{-i k x_j n_x} [A_j cos(k b n_y) P_y + i B_j sin(k b n_y) P_x + C_j cos(k b n_y) (n x e_z)],  P_i = e_i - n_i n.
# Fit on half the directions (alternate phi columns), report explained power on the held-out half. Controls: one source at x = 0,
# junction positions scanned.  usage: python junction_fit.py plain_ff.mat
import sys, numpy as np, scipy.io as sio
sys.path.insert(0, r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/python_tools")
from farfield_surface import sphere_field
m = sio.loadmat(sys.argv[1], squeeze_me=True, struct_as_record=False)
x, w, phi, Eth, Eph, _ = sphere_field(m, 80)
lam = float(np.ravel(m["surface_top"].lam)[0]) * 1e6; k = 2 * np.pi * 1.444 / lam; H = 0.35
ct = x[:, None] * np.ones_like(phi)[None]; st = np.sqrt(1 - ct ** 2); cp = np.cos(phi)[None] * np.ones_like(ct); sp = np.sin(phi)[None] * np.ones_like(ct)
n = np.stack([st * cp, st * sp, ct], -1)
th = np.stack([ct * cp, ct * sp, -st], -1); ph = np.stack([-sp, cp, 0 * ct], -1)
dO = w[:, None] * (2 * np.pi / phi.size) * np.ones_like(ct)
E = np.stack([Eth, Eph], -1)                                           # (nth, nph, 2)
Hz = np.sinc(k * n[..., 2] * H / 2 / np.pi)
def tang(v):                                                           # cartesian (..,3) -> (theta, phi) components
    return np.stack([np.sum(th * v, -1), np.sum(ph * v, -1)], -1)
ey, ex, ez = np.array([0, 1., 0]), np.array([1., 0, 0]), np.array([0, 0, 1.])
Py = tang(ey - n[..., 1:2] * n); Px = tang(ex - n[..., 0:1] * n); Mz = tang(np.cross(n, ez))
train = (np.arange(phi.size) % 2 == 0)[None, :] * np.ones_like(ct, bool); test = ~train


def basis(xs, b):
    cols = []
    for xj in xs:
        e = np.exp(-1j * k * xj * n[..., 0])[..., None] * Hz[..., None]
        cy = np.cos(k * b * n[..., 1])[..., None]; sy = np.sin(k * b * n[..., 1])[..., None]
        cols += [e * cy * Py, 1j * e * sy * Px, e * cy * Mz]
    return cols


def fit(xs, b):
    cols = basis(xs, b); wts = np.sqrt(dO)[..., None]
    A = lambda mask: np.stack([(c * wts)[mask].ravel() for c in cols], 1)
    y = lambda mask: (E * wts)[mask].ravel()
    coef, *_ = np.linalg.lstsq(A(train), y(train), rcond=None)
    out = {}
    for lab, mk in (("train", train), ("test", test)):
        r = y(mk) - A(mk) @ coef; out[lab] = 1 - np.sum(abs(r) ** 2) / np.sum(abs(y(mk)) ** 2)
    return out, coef


print(f"lambda {lam*1e3:.1f} nm, k n_clad {k:.3f}/um; explained power (train / held-out):")
for lab, xs in (("one source x=0", [0.0]), ("junctions +-0.125", [-0.125, 0.125]), ("+-0.25", [-0.25, 0.25]), ("+-0.375", [-0.375, 0.375]),
                ("+-0.5", [-0.5, 0.5]), ("+-0.125 & +-0.625", [-0.625, -0.125, 0.125, 0.625])):
    for b in (0.32, 0.40, 0.48):
        o, c = fit(xs, b)
        print(f"  {lab:20s} b {b:.2f}: {o['train']:.3f} / {o['test']:.3f}")
o, c = fit([-0.125, 0.125], 0.40)
names = ["A_L", "B_L", "C_L", "A_R", "B_R", "C_R"]
print("junction model b 0.40 coefficients (|.|, arg deg): " + "  ".join(f"{nm} {abs(v):.3g} {np.degrees(np.angle(v)):+.0f}" for nm, v in zip(names, c)))
