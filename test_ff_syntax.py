import os
import importlib.util

try:
    import lumapi
except ImportError:
    LUMAPI_PATH = r"C:\Program Files\Lumerical\v252\api\python\lumapi.py"
    if not os.path.exists(LUMAPI_PATH):
        LUMAPI_PATH = r"C:\Program Files\Lumerical\v242\api\python\lumapi.py"
    if not os.path.exists(LUMAPI_PATH):
        LUMAPI_PATH = r"C:\Program Files\Lumerical\v241\api\python\lumapi.py"
    import glob
    candidates = glob.glob(r"C:\Program Files\Lumerical\v*\api\python\lumapi.py")
    if candidates:
        LUMAPI_PATH = candidates[0]
    spec = importlib.util.spec_from_file_location("lumapi", LUMAPI_PATH)
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)

fdtd = lumapi.FDTD(hide=True)

test_script = """
addfdtd;
addprofile;
set("x", 0); set("x span", 1e-6);
set("y", 0); set("y span", 1e-6);
varname = "monitor1";
set("name", varname);
set("monitor type", "2D Z-normal");
# we don't have data, just checking if function 'farfieldexact3d' parses with x,y,z arrays.
"""
fdtd.eval(test_script)
print("FDTD started and monitor added.")

# We can't actually run farfieldexact3d without running the simulation.
# Let's run a tiny 5-cell simulation.
run_script = """
setnamed("FDTD", "x span", 2e-6);
setnamed("FDTD", "y span", 2e-6);
setnamed("FDTD", "z span", 2e-6);
adddipole;
run;
n_pts = 10;
theta = linspace(0, 2*pi, n_pts);
x = theta * 0;
y = cos(theta);
z = sin(theta);
E1 = farfieldexact3d(varname, x, y, z, 1);
sz = size(E1);
"""
fdtd.eval(run_script)
sz = fdtd.getv("sz")
print("Size of E1:", sz)
fdtd.close()
