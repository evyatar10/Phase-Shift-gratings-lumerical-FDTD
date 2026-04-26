import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt

# Try to import lumapi normally
try:
    import lumapi
except ImportError:
    import importlib.util

    LUMAPI_PATH = r"C:\\Program Files\\Lumerical\\v252\\api\\python\\lumapi.py"
    spec = importlib.util.spec_from_file_location("lumapi", LUMAPI_PATH)
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)


class NeffSweeper:
    def __init__(self,
                 width=1000e-9,
                 height=350e-9,
                 wl_start=1.5e-6,
                 wl_stop=1.6e-6,
                 n_points=100,
                 material_db_path=None,
                 core_material="Si3N4 (Silicon Nitride) - Luke",
                 clad_material="SiO2 (Glass) - Palik"):

        self.width = width
        self.height = height
        self.wl_start = wl_start
        self.wl_stop = wl_stop
        self.n_points = n_points

        self.material_db_path = material_db_path
        self.core_material = core_material
        self.clad_material = clad_material

        self.mode = lumapi.MODE()
        self._setup_materials()

    def _setup_materials(self):
        if self.material_db_path and os.path.exists(self.material_db_path):
            print(f"Importing material DB from: {self.material_db_path}")
            self.mode.importmaterialdb(self.material_db_path)

        src_core = self.core_material
        src_clad = self.clad_material
        custom_sin = "SiN_custom"
        custom_sio2 = "SiO2_custom"

        script = f'''
        if (haveresult("{custom_sin}", "material")) {{ deletematerial("{custom_sin}"); }}
        if (haveresult("{custom_sio2}", "material")) {{ deletematerial("{custom_sio2}"); }}

        if (!haveresult("{src_core}", "material")) {{ print("Error: Core '{src_core}' not found!"); }}
        if (!haveresult("{src_clad}", "material")) {{ print("Error: Clad '{src_clad}' not found!"); }}

        m1 = copymaterial("{src_core}"); setmaterial(m1, "name", "{custom_sin}");
        m2 = copymaterial("{src_clad}"); setmaterial(m2, "name", "{custom_sio2}");

        setmaterial("{custom_sin}", "specify fit range", 1);
        setmaterial("{custom_sio2}", "specify fit range", 1);
        setmaterial("{custom_sin}", "wavelength min", {self.wl_start});
        setmaterial("{custom_sin}", "wavelength max", {self.wl_stop});
        setmaterial("{custom_sio2}", "wavelength min", {self.wl_start});
        setmaterial("{custom_sio2}", "wavelength max", {self.wl_stop});
        setmaterial("{custom_sin}", "tolerance", 0.001);
        setmaterial("{custom_sio2}", "tolerance", 0.001);
        setmaterial("{custom_sin}", "make fit passive", 1);
        setmaterial("{custom_sio2}", "make fit passive", 1);
        '''
        self.mode.eval(script)
        self.core_material = custom_sin
        self.clad_material = custom_sio2

    def build_sim(self):
        self.mode.switchtolayout()
        self.mode.deleteall()

        # 1. Simulation Region (FDE) -> 2D Z normal
        self.mode.addfde()
        self.mode.set("solver type", "2D Z normal")
        self.mode.set("x", 0)
        self.mode.set("x span", 4e-6)
        self.mode.set("y", 0)
        self.mode.set("y span", 4e-6)
        self.mode.set("z", 0)
        self.mode.set("wavelength", (self.wl_start + self.wl_stop) / 2)

        # 2. Substrate
        self.mode.addrect()
        self.mode.set("name", "substrate")
        self.mode.set("material", self.clad_material)
        self.mode.set("x span", 10e-6)
        self.mode.set("y span", 10e-6)
        self.mode.set("z span", 10e-6)

        # 3. Core (Rotated: Thickness=X, Width=Y, Length=Z)
        self.mode.addrect()
        self.mode.set("name", "core")
        self.mode.set("material", self.core_material)
        self.mode.set("x", 0)
        self.mode.set("x span", self.height)  # 350nm
        self.mode.set("y", 0)
        self.mode.set("y span", self.width)  # 1000nm
        self.mode.set("z", 0)
        self.mode.set("z span", 10e-6)

        # 4. Mesh override
        self.mode.addmesh()
        self.mode.set("name", "core_mesh")
        self.mode.set("x", 0)
        self.mode.set("x span", self.height + 0.5e-6)
        self.mode.set("y", 0)
        self.mode.set("y span", self.width + 0.5e-6)
        self.mode.set("z", 0)
        self.mode.set("z span", 10e-6)

        # --- FIX: Changed dx from 20e-9 to 10e-9 to perfectly divide 350nm ---
        self.mode.set("dx", 10e-9)
        self.mode.set("dy", 10e-9)  # Keeping dy consistent, though 20nm was fine for 1000nm width
        self.mode.set("dz", 10e-9)

    def run_sweep(self):
        self.mode.run()
        self.mode.setanalysis("wavelength", self.wl_start)
        self.mode.findmodes()
        self.mode.selectmode(1)

        self.mode.setanalysis("track selected mode", 1)
        self.mode.setanalysis("detailed dispersion calculation", 0)
        self.mode.setanalysis("stop wavelength", self.wl_stop)
        self.mode.setanalysis("number of points", self.n_points)
        self.mode.frequencysweep()

        res = self.mode.getdata("frequencysweep", "neff")
        f = self.mode.getdata("frequencysweep", "f")
        wl = 299792458 / f

        neff_complex = np.squeeze(res)
        wl = np.squeeze(wl)

        if wl[0] > wl[-1]:
            wl = np.flip(wl)
            neff_complex = np.flip(neff_complex)

        return wl, neff_complex

    def save_results(self, wl, neff, filename):
        data = {
            "wavelengths": wl,
            "neff_complex": neff,
            "neff_real": np.real(neff),
            "neff_imag": np.imag(neff)
        }
        sio.savemat(filename, data, do_compression=True)
        print(f"Saved Neff data to {filename}")

    def save_sim_file(self, filepath):
        self.mode.save(filepath)
        print(f"Saved simulation layout to: {filepath}")

    def close(self):
        self.mode.close()


if __name__ == "__main__":
    base_dir = r"C:\Users\evyat\Lumerical\pi_shifts_FDTD_results\neff_vs_wl_new"
    if not os.path.exists(base_dir): os.makedirs(base_dir)

    mdf_path = r'C:\Users\evyat\Lumerical\ron_lumerical_codes\lgt_materials.mdf'
    core_mat_name = "Si3N4 (Silicon Nitride) - Luke"
    clad_mat_name = "SiO2 (Glass) - Palik"

    sweeper = NeffSweeper(
        width=1000e-9,
        height=350e-9,
        wl_start=1.5e-6,
        wl_stop=1.7e-6,
        n_points=400,
        material_db_path=mdf_path,
        core_material=core_mat_name,
        clad_material=clad_mat_name
    )

    sweeper.build_sim()
    wl, neff = sweeper.run_sweep()

    sweeper.save_results(wl, neff, os.path.join(base_dir, "FDE_sweep_results.mat"))
    sweeper.save_sim_file(os.path.join(base_dir, "FDE_sweep_layout_rotated.lms"))

    plt.figure()
    plt.plot(wl * 1e9, np.real(neff))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Real(Neff)")
    plt.title("Effective Index vs Wavelength (Rotated + Fixed Mesh)")
    plt.grid(True)
    plt.show()

    sweeper.close()