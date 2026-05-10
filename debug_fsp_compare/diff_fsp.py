"""
Diagnostic: open the baseline .fsp and the lumopt forward .fsp side-by-side
and print every setting that could explain a T discrepancy at λ=1560.254 nm.

Run on Athena (uploaded by ssh):
    apptainer exec ~/containers/lumerical-2026R1.sif \\
        /opt/lumerical/v261/python/bin/python diff_fsp.py
"""

import sys
import os

sys.path.insert(0, '/opt/lumerical/v261/api/python')
import lumapi

BASELINE = '/work/baseline.fsp'
LUMOPT   = '/work/lumopt_forward.fsp'


def header(s):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def safe(call):
    try:
        return call()
    except Exception as e:
        return f"<err: {e}>"


def report(fdtd, label):
    header(label)

    print("\n--- Global source ---")
    print(f"  wavelength start: {safe(lambda: fdtd.getglobalsource('wavelength start'))}")
    print(f"  wavelength stop:  {safe(lambda: fdtd.getglobalsource('wavelength stop'))}")
    print(f"  set wavelength:   {safe(lambda: fdtd.getglobalsource('set wavelength'))}")

    print("\n--- Global monitor ---")
    print(f"  frequency points:  {safe(lambda: fdtd.getglobalmonitor('frequency points'))}")
    print(f"  sample spacing:    {safe(lambda: fdtd.getglobalmonitor('sample spacing'))}")
    print(f"  use source limits: {safe(lambda: fdtd.getglobalmonitor('use source limits'))}")

    print("\n--- FDTD region ---")
    print(f"  x span:    {safe(lambda: fdtd.getnamed('FDTD','x span'))}")
    print(f"  y span:    {safe(lambda: fdtd.getnamed('FDTD','y span'))}")
    print(f"  z span:    {safe(lambda: fdtd.getnamed('FDTD','z span'))}")
    print(f"  dx:        {safe(lambda: fdtd.getnamed('FDTD','dx'))}")
    print(f"  dy:        {safe(lambda: fdtd.getnamed('FDTD','dy'))}")
    print(f"  dz:        {safe(lambda: fdtd.getnamed('FDTD','dz'))}")
    print(f"  sim time:  {safe(lambda: fdtd.getnamed('FDTD','simulation time'))}")
    print(f"  shutoff:   {safe(lambda: fdtd.getnamed('FDTD','auto shutoff min'))}")

    print("\n--- Mesh override ---")
    print(f"  x span:    {safe(lambda: fdtd.getnamed('mesh_override','x span'))}")
    print(f"  y span:    {safe(lambda: fdtd.getnamed('mesh_override','y span'))}")
    print(f"  z span:    {safe(lambda: fdtd.getnamed('mesh_override','z span'))}")
    print(f"  dx:        {safe(lambda: fdtd.getnamed('mesh_override','dx'))}")
    print(f"  dy:        {safe(lambda: fdtd.getnamed('mesh_override','dy'))}")
    print(f"  dz:        {safe(lambda: fdtd.getnamed('mesh_override','dz'))}")

    # Find the source port — try both names
    for src_name in ['FDTD::ports::source', 'FDTD::ports::Port_1']:
        try:
            inj = fdtd.getnamed(src_name, 'injection axis')
            print(f"\n--- Source port: {src_name} ---")
            print(f"  exists, injection axis: {inj}")
            print(f"  direction:          {safe(lambda: fdtd.getnamed(src_name, 'direction'))}")
            print(f"  mode selection:     {safe(lambda: fdtd.getnamed(src_name, 'mode selection'))}")
            print(f"  freq dep profile:   {safe(lambda: fdtd.getnamed(src_name, 'frequency dependent profile'))}")
            print(f"  x:                  {safe(lambda: fdtd.getnamed(src_name, 'x'))}")
            print(f"  y span:             {safe(lambda: fdtd.getnamed(src_name, 'y span'))}")
            print(f"  z span:             {safe(lambda: fdtd.getnamed(src_name, 'z span'))}")
            break
        except Exception:
            continue

    # Active source-port flag on FDTD::ports
    print(f"\n--- FDTD::ports active source ---")
    print(f"  source port:  {safe(lambda: fdtd.getnamed('FDTD::ports', 'source port'))}")
    print(f"  source mode:  {safe(lambda: fdtd.getnamed('FDTD::ports', 'source mode'))}")
    print(f"  monitor freq points: {safe(lambda: fdtd.getnamed('FDTD::ports', 'monitor frequency points'))}")

    # FOM port
    for fom_name in ['FDTD::ports::fom', 'FDTD::ports::Port_2']:
        try:
            inj = fdtd.getnamed(fom_name, 'injection axis')
            print(f"\n--- FOM port: {fom_name} ---")
            print(f"  exists, injection axis: {inj}")
            print(f"  direction:          {safe(lambda: fdtd.getnamed(fom_name, 'direction'))}")
            print(f"  mode selection:     {safe(lambda: fdtd.getnamed(fom_name, 'mode selection'))}")
            print(f"  x:                  {safe(lambda: fdtd.getnamed(fom_name, 'x'))}")
            print(f"  y span:             {safe(lambda: fdtd.getnamed(fom_name, 'y span'))}")
            print(f"  z span:             {safe(lambda: fdtd.getnamed(fom_name, 'z span'))}")
            break
        except Exception:
            continue

    # Geometry inventory via getnamednumber (partial matching)
    print(f"\n--- Geometry inventory ---")
    for prefix in ("L_narrow", "L_wide", "R_narrow", "R_wide", "cavity",
                   "wg_left_inf", "wg_right_inf", "polygon", "opt_fields"):
        try:
            n = int(fdtd.getnamednumber(prefix))
        except Exception:
            n = 0
        print(f"  '{prefix}' matches: {n}")


def main():
    fdtd = lumapi.FDTD()

    fdtd.load(BASELINE)
    report(fdtd, "BASELINE  (gives T = 0.866 at 1560.254 nm)")

    fdtd.load(LUMOPT)
    report(fdtd, "LUMOPT FORWARD  (gives T = 0.752 at same wavelength)")

    fdtd.close()


if __name__ == "__main__":
    main()
