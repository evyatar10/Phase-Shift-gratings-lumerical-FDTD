import os
import sys
import numpy as np

try:
    import lumapi
except ImportError:
    LUMAPI_PATH = r"C:\Program Files\Lumerical\v252\api\python\lumapi.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("lumapi", LUMAPI_PATH)
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)

fsp_path = r"C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles_5_long\layouts\layout_80_periods_CONST.fsp"

print(f"Loading {fsp_path}")
fdtd = lumapi.FDTD(filename=fsp_path, hide=False)

try:
    script = """
    idx_f = 1; # just test with 1
    n_pts = 361;
    theta_deg = linspace(0, 360, n_pts);
    theta_rad = theta_deg * pi / 180;
    R = 1.0;
    
    x = theta_rad * 0;
    y = R * cos(theta_rad);
    z = R * sin(theta_rad);
    
    ETop = farfieldexact3d("monitor_ff_z_top", x, y, z, idx_f);
    """
    fdtd.eval(script)
    print("Test 1 (farfieldexact3d with x,y,z arrays) succeeded!")
except Exception as e:
    print("Test 1 failed:", e)

    try:
        script = """
        idx_f = 1;
        R = 1.0;
        x = [0]; y = [0]; z = [1];
        ETop = farfieldexact3d("monitor_ff_z_top", x, y, z, idx_f);
        """
        fdtd.eval(script)
        print("Test 2 (farfieldexact3d with explicit arrays) succeeded!")
    except Exception as e2:
        print("Test 2 failed:", e2)

fdtd.close()
