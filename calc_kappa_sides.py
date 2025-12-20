import numpy as np
import importlib.util
import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -----------------------------------------------------------------------------
# 0. SETUP LUMAPI
# -----------------------------------------------------------------------------
try:
    import lumapi
except ImportError:
    LUMAPI_PATH = r"C:\\Program Files\\Lumerical\\v252\\api\\python\\lumapi.py"
    if os.path.exists(LUMAPI_PATH):
        spec = importlib.util.spec_from_file_location("lumapi", LUMAPI_PATH)
        lumapi = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lumapi)
    else:
        raise ImportError("Could not find lumapi.py. Check installation path.")


# -----------------------------------------------------------------------------
# PART 1: CALCULATE GROUP INDEX (ng)
# -----------------------------------------------------------------------------
class GroupIndexSolver:
    def __init__(self, target_wavelength=1.55e-6, width_avg=800e-9, core_height=350e-9):
        self.mode = lumapi.MODE()
        self.wl = target_wavelength
        self.width = width_avg
        self.height = core_height

    def calculate_ng(self):
        mode = self.mode
        mode.switchtolayout()
        mode.deleteall()

        mode.addrect(name="core", x=0, y=0, z=0, x_span=10e-6, y_span=self.width, z_span=self.height)
        mode.set("material", "Si3N4 (Silicon Nitride) - Luke")

        mode.addfde()
        mode.set("solver type", "2D X normal")
        mode.set("x", 0)
        mode.set("y", 0);
        mode.set("y span", self.width + 3.0e-6)
        mode.set("z", 0);
        mode.set("z span", self.height + 3.0e-6)
        mode.set("wavelength", self.wl)
        mode.set("background material", "SiO2 (Glass) - Palik")

        mode.setanalysis("calculate group index", True)
        mode.setanalysis("number of trial modes", 5)
        mode.setanalysis("search", "near n")
        mode.setanalysis("use max index", True)

        print("Calculating modes in FDE...")
        mode.mesh()
        mode.findmodes()

        ng = mode.getdata("FDE::data::mode1", "ng")
        if isinstance(ng, np.ndarray): ng = ng.item()

        mode.close()
        return np.real(ng)


# -----------------------------------------------------------------------------
# PART 2: FDTD UNIT CELL SIMULATION (With Exact Mesh)
# -----------------------------------------------------------------------------
class BraggKappaFDTD:
    def __init__(self, ng, pitch=500e-9, w_narrow=700e-9, w_wide=900e-9, core_h=350e-9):
        self.ng = ng
        self.pitch = pitch
        self.fdtd = lumapi.FDTD()
        self.w_narrow = w_narrow
        self.w_wide = w_wide
        self.core_h = core_h
        self.sim_time_fs = 10000

    def run_simulation(self):
        fdtd = self.fdtd
        fdtd.switchtolayout()
        fdtd.deleteall()

        # --- 1. GEOMETRY ---
        fdtd.addrect(name="narrow", x=-self.pitch / 4, y=0, z=0,
                     x_span=self.pitch / 2, y_span=self.w_narrow, z_span=self.core_h)
        fdtd.set("material", "Si3N4 (Silicon Nitride) - Luke")

        fdtd.addrect(name="wide", x=self.pitch / 4, y=0, z=0,
                     x_span=self.pitch / 2, y_span=self.w_wide, z_span=self.core_h)
        fdtd.set("material", "Si3N4 (Silicon Nitride) - Luke")

        # --- 2. FDTD REGION ---
        fdtd.addfdtd(dimension="3D", x=0, x_span=self.pitch, y=0, y_span=3e-6, z=0, z_span=3e-6)
        fdtd.set("background material", "SiO2 (Glass) - Palik")
        fdtd.set("x min bc", "Bloch");
        fdtd.set("x max bc", "Bloch")
        fdtd.set("y min bc", "PML");
        fdtd.set("y max bc", "PML")
        fdtd.set("z min bc", "PML");
        fdtd.set("z max bc", "PML")

        # We can leave auto mesh at 2 because we will use an override
        fdtd.set("mesh accuracy", 2)
        fdtd.set("simulation time", self.sim_time_fs * 1e-15)
        fdtd.set("use early shutoff", 0)

        # --- 3. MESH OVERRIDE (CRITICAL FIX) ---
        # This forces the mesh to align EXACTLY with the period.
        # pitch = 500nm. dx = 10nm. This gives exactly 50 cells per period.
        fdtd.addmesh()
        fdtd.set("name", "mesh_periodic")
        fdtd.set("x", 0);
        fdtd.set("x span", self.pitch)
        fdtd.set("y", 0);
        fdtd.set("y span", self.w_wide + 1e-6)  # Cover width
        fdtd.set("z", 0);
        fdtd.set("z span", self.core_h + 1e-6)  # Cover height

        # Force X mesh steps
        fdtd.set("override x mesh", 1)
        fdtd.set("dx", 10e-9)  # 10 nm step (500/10 = 50 cells)

        # Optional: Force Y/Z if you want high precision on mode profile
        fdtd.set("override y mesh", 1);
        fdtd.set("dy", 10e-9)
        fdtd.set("override z mesh", 1);
        fdtd.set("dz", 10e-9)

        # --- 4. BLOCH SETTINGS ---
        fdtd.set("set based on source angle", 0)
        fdtd.set("bloch units", "bandstructure")
        fdtd.set("kx", 0.5)

        # --- 5. SOURCE ---
        fdtd.addmode()
        fdtd.set("name", "source")
        fdtd.set("injection axis", "x")
        fdtd.set("direction", "Forward")
        fdtd.set("x", 0)
        fdtd.set("y", 0);
        fdtd.set("y span", 2e-6)
        fdtd.set("z", 0);
        fdtd.set("z span", 2e-6)
        fdtd.set("wavelength start", 1.45e-6)
        fdtd.set("wavelength stop", 1.65e-6)

        # --- 6. MONITORS ---
        np.random.seed(123)
        n_monitors = 10
        print(f"Adding {n_monitors} random monitors...")

        for i in range(n_monitors):
            name = f"mon_{i}"
            fdtd.addtime()
            fdtd.set("name", name)
            fdtd.set("x", (np.random.rand() - 0.5) * self.pitch)
            fdtd.set("y", (np.random.rand() - 0.5) * self.w_wide)
            fdtd.set("z", (np.random.rand() - 0.5) * self.core_h)

        # --- 7. SAVE & RUN ---
        save_path = os.path.join(os.getcwd(), "unit_cell_exact_mesh.fsp")
        fdtd.save(save_path)
        print(f"Layout saved to: {save_path}")
        print("Running FDTD simulation...")
        fdtd.run()

        # --- 8. EXTRACT ---
        print("Extracting time signals...")
        f_vec = None
        spectrum_sum = None
        monitors_found = 0

        for i in range(n_monitors):
            name = f"mon_{i}"
            try:
                if fdtd.havedata(name, "Ex"):
                    Ex = np.squeeze(fdtd.getdata(name, "Ex"))
                    t = np.squeeze(fdtd.getdata(name, "t"))

                    dt = t[1] - t[0]
                    n_samp = len(Ex)

                    # Apply Hanning Window for smooth peaks
                    window = np.hanning(n_samp)
                    Ex_smooth = Ex * window

                    Y = np.fft.fft(Ex_smooth)
                    f = np.fft.fftfreq(n_samp, d=dt)

                    mask = f > 0
                    f = f[mask]
                    Y = np.abs(Y[mask])

                    if f_vec is None:
                        f_vec = f
                        spectrum_sum = Y
                    else:
                        spectrum_sum += Y
                    monitors_found += 1
            except Exception:
                pass

        if monitors_found == 0:
            raise RuntimeError("CRITICAL: Failed to extract data.")

        return f_vec, spectrum_sum, self.sim_time_fs


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    pitch = 500e-9
    w_narrow = 700e-9
    w_wide = 900e-9
    w_avg = (w_narrow + w_wide) / 2.0

    print("--- STEP 1: CALCULATING GROUP INDEX ---")
    fde = GroupIndexSolver(width_avg=w_avg)
    ng = fde.calculate_ng()

    print(f"\n--- STEP 2: RUNNING FDTD (EXACT MESH) ---")
    sim = BraggKappaFDTD(ng=ng, pitch=pitch, w_narrow=w_narrow, w_wide=w_wide)
    freq, intensity, t_max_fs = sim.run_simulation()

    # Analyze
    c = 299792458
    f_min, f_max = 180e12, 210e12
    mask = (freq > f_min) & (freq < f_max)
    f_roi = freq[mask]
    i_roi = intensity[mask]

    peaks, _ = find_peaks(i_roi, height=np.max(i_roi) * 0.1, distance=5)

    if len(peaks) >= 2:
        peak_inds = peaks[np.argsort(i_roi[peaks])][-2:]
        f1, f2 = sorted(f_roi[peak_inds])

        lam1 = c / f2;
        lam2 = c / f1
        delta_lam = lam2 - lam1
        lam_bragg = (lam1 + lam2) / 2.0
        kappa = (np.pi * ng * delta_lam) / (lam_bragg ** 2)
        dn = (kappa * lam_bragg) / np.pi

        print(f"\n" + "=" * 50)
        print(f"       RESULTS (EXACT MESH ALIGNMENT)")
        print(f"=" * 50)
        print(f"Resonance 1:          {lam1 * 1e9:.2f} nm")
        print(f"Resonance 2:          {lam2 * 1e9:.2f} nm")
        print(f"Bragg Wavelength:     {lam_bragg * 1e9:.2f} nm")
        print(f"-" * 50)
        print(f"KAPPA:                {kappa:.2e} m^-1")
        print(f"Index Perturbation:   {dn:.4f}")
        print(f"=" * 50)

        lam_roi_nm = (c / f_roi) * 1e9
        plt.figure(figsize=(10, 6))
        plt.plot(lam_roi_nm, i_roi, 'b-', label="Spectrum")
        plt.plot([lam1 * 1e9, lam2 * 1e9], [i_roi[peak_inds[1]], i_roi[peak_inds[0]]], "rx")
        plt.axvline(lam_bragg * 1e9, color='k', linestyle='--', label=f"Center: {lam_bragg * 1e9:.1f}nm")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Intensity [a.u.]")
        plt.title(f"Kappa = {kappa:.2e} m$^{{-1}}$")
        plt.legend()
        plt.show()
    else:
        print("Could not find peaks.")