"""Analytic reach map: point dipoles (+ device-imposed images) projected with the repo's multipoles()."""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\python_tools")
from farfield_multipole import sphere_grid, multipoles

LAM = 1.620; NCLAD = 1.444; K = 2*np.pi*NCLAD/LAM      # 1/um, e^{-i omega t}, outgoing e^{ikr}
LMAX = 60
x, w, phi = sphere_grid(LMAX)
st = np.sqrt(1-x**2)[:, None]; ct = x[:, None]; cp = np.cos(phi)[None, :]; sp = np.sin(phi)[None, :]
NX, NY, NZ = st*cp, st*sp, ct*np.ones_like(cp)
TH = np.stack([ct*cp, ct*sp, -st*np.ones_like(cp)]); PH = np.stack([-sp*np.ones_like(ct), cp*np.ones_like(ct), 0*ct*cp])

def farfield(sources):
    """sources: list of (p(3), r0(3)). E(n) = sum [p - (n.p)n] exp(-i k n.r0)  (transverse part only)."""
    Eth = np.zeros(NX.shape, complex); Eph = np.zeros_like(Eth)
    for p, r0 in sources:
        ph = np.exp(-1j*K*(NX*r0[0] + NY*r0[1] + NZ*r0[2]))
        Eth += (p[0]*TH[0] + p[1]*TH[1] + p[2]*TH[2])*ph
        Eph += (p[0]*PH[0] + p[1]*PH[1] + p[2]*PH[2])*ph
    return Eth, Eph

def spectrum(sources):
    aE, aM = multipoles(x, w, phi, *farfield(sources), LMAX)
    P = np.abs(aE)**2 + np.abs(aM)**2; return aE, aM, P/P.sum()

def cum(P, L): return P[:L+1].sum()
def top(P, n=4):
    out = []
    PE = np.abs(spectrum_cache[0])**2; PM = np.abs(spectrum_cache[1])**2; tot = (PE+PM).sum()
    items = []
    for l in range(LMAX+1):
        for m in range(0, l+1):   # pair (l,+-m)
            e = PE[l, m+LMAX] + (PE[l, -m+LMAX] if m else 0); mm = PM[l, m+LMAX] + (PM[l, -m+LMAX] if m else 0)
            items.append((e/tot, f"E({l},{'+-' if m else ''}{m})")); items.append((mm/tot, f"M({l},{'+-' if m else ''}{m})"))
    items.sort(reverse=True); return items[:n]

def images_y(p, r0):
    """y-mirror pair with TE parity: p_y even, p_x odd (p_z: even in y; z-mirror handled by caller)."""
    px, py, pz = p; x0, y0, z0 = r0
    if abs(y0) < 1e-12: return [(np.array(p, float), np.array(r0, float))]
    return [(np.array([px, py, pz], float), np.array([x0, y0, z0], float)),
            (np.array([-px, py, pz], float), np.array([x0, -y0, z0], float))]

def report(label, sources):
    global spectrum_cache
    aE, aM, P = spectrum(sources); spectrum_cache = (aE, aM)
    Pl = P.sum(1); t = top(P)
    print(f"{label:34s} l<=2 {cum(Pl,2)*100:5.1f}%  l<=5 {cum(Pl,5)*100:5.1f}%  l<=10 {cum(Pl,10)*100:5.1f}%  "
          f"top: " + ", ".join(f"{n} {v*100:.1f}%" for v, n in t))
    return aE, aM, P

print("=== gate: single dipoles at origin (must be pure l=1) ===")
for ax, p in zip("xyz", np.eye(3)):
    aE, aM, P = spectrum([(p, np.zeros(3))]); Pl = P.sum(1)
    print(f"  p_{ax}: l=1 share {Pl[1]*100:.4f}%  E/M split {np.abs(aE).sum()**2/(np.abs(aE).sum()**2+np.abs(aM).sum()**2):.3f}")

Y0 = 0.475
print("\n=== (1) wall pair (x0, +-y0=0.475): p_y even-image, p_x odd-image, p_z even-image (single, no x partner) ===")
for ax, p in zip("yxz", [[0,1,0],[1,0,0],[0,0,1]]):
    for x0 in [0, 0.13, 0.26, 0.5, 1, 2, 4]:
        report(f"p_{ax} x0={x0:<4} (k x0={K*x0:4.1f})", images_y(p, [x0, Y0, 0]))

print("\n=== (1b) same, plus the x-mirror partner (device x-mirror): sign s_x = +1 (x-even p_y) / -1 (x-odd p_y) ===")
for sx in [+1, -1]:
    for x0 in [0.13, 0.26, 0.5, 1, 2]:
        src = images_y([0,1,0],[x0,Y0,0]) + images_y([0,sx,0],[-x0,Y0,0])
        report(f"p_y x-pair s_x={sx:+d} x0={x0:<4}", src)

print("\n=== (3) inner holes on axis (y=0): p_y only (Ex=0 on axis). pair +-x_h, sign s_x ===")
for sx in [+1, -1]:
    for xh in [0.05, 0.1, 0.13, 0.26, 0.39, 0.5, 0.77]:
        report(f"axis hole pair s_x={sx:+d} x_h={xh:<4}", [(np.array([0,1.,0]), np.array([xh,0,0])), (np.array([0,sx,0]), np.array([-xh,0,0]))])
print("--- quartet (+-x_h, +-y_h): p_y even in y (+ small p_x odd in y, ratio eps); s_x as above ---")
for sx in [+1, -1]:
    for xh, yh in [(0.13,0.2),(0.26,0.2),(0.13,0.35),(0.26,0.35),(0.39,0.35)]:
        src = images_y([0,1,0],[xh,yh,0]) + images_y([0,sx,0],[-xh,yh,0])
        report(f"quartet s_x={sx:+d} x_h={xh} y_h={yh}", src)

print("\n=== (2) designed combinations (wall features y0=0.475, each with its y-image) ===")
a = 0.514/4
def wall(xs, signs, comp=[0,1,0]):
    s = []
    for xi, si in zip(xs, signs): s += images_y(list(np.array(comp)*si), [xi, Y0, 0])
    return s
combos = {
 "single step pair x=+-a, same sign (x-even)": wall([a,-a],[1,1]),
 "single step pair x=+-a, opp sign (x-odd)": wall([a,-a],[1,-1]),
 "+-a,+-3a alternating (+,-,+,-) ": wall([-3*a,-a,a,3*a],[1,-1,1,-1]),
 "+-a,+-3a x-odd (-,-,+,+)": wall([-3*a,-a,a,3*a],[-1,-1,1,1]),
 "+-a,+-3a x-odd inner-dom (-,+,-,+)": wall([-3*a,-a,a,3*a],[-1,1,-1,1]),
 "+-a,+-3a all + ": wall([-3*a,-a,a,3*a],[1,1,1,1]),
 "+-a,+-3a,+-5a alternating": wall([-5*a,-3*a,-a,a,3*a,5*a],[1,-1,1,-1,1,-1]),
 "+-a,+-3a,+-5a x-odd Bragg (-,+,-,+,-,+)": wall([-5*a,-3*a,-a,a,3*a,5*a],[-1,1,-1,1,-1,1]),
 "+-3a only x-odd": wall([3*a,-3*a],[1,-1]),
 "+-5a only x-odd": wall([5*a,-5*a],[1,-1]),
 "+-5a only x-even": wall([5*a,-5*a],[1,1]),
 "p_x step pair x=+-a x-odd": wall([a,-a],[1,-1],[1,0,0]),
 "p_x step pair x=+-a x-even": wall([a,-a],[1,1],[1,0,0]),
}
for k_, v in combos.items(): report(k_, v)

print("\n=== (2c) exclusion scan: which (l,m) families each symmetry class can reach ===")
for name, src in {"p_y x-even": wall([a,-a],[1,1]), "p_y x-odd": wall([a,-a],[1,-1]),
                  "p_x x-even": wall([a,-a],[1,1],[1,0,0]), "p_x x-odd": wall([a,-a],[1,-1],[1,0,0])}.items():
    aE, aM, P = spectrum(src)
    PE = np.abs(aE)**2/P.sum()*0+np.abs(aE)**2; PM = np.abs(aM)**2; tot = (PE+PM).sum()
    reachE = sorted({(l%2, m%2) for l in range(LMAX+1) for m in range(-l,l+1) if PE[l,m+LMAX]/tot > 1e-8})
    reachM = sorted({(l%2, m%2) for l in range(LMAX+1) for m in range(-l,l+1) if PM[l,m+LMAX]/tot > 1e-8})
    print(f"  {name:12s} E(l%2,m%2) reachable: {reachE}   M: {reachM}")
