"""No-solve S2 provenance probe. Creates a NEW output directory; never saves originals.

Coordinates use getdata(..., option=2), Ansys's documented symmetry-unfolded
raw coordinate API. No electric/magnetic field arrays are requested. A successful
execution is a provenance measurement, not a gradient/calibration physics gate.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

C0 = 299792458.0
FORBIDDEN = {"run", "runjobs", "runparallel", "runsweep", "runanalysis",
             "addjob", "savejob", "system", "feval", "eval", "save", "export"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def compare(left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        return {"equal": False, "shape_left": list(a.shape), "shape_right": list(b.shape)}
    if a.dtype.kind not in "biufc" or b.dtype.kind not in "biufc":
        return {"equal": bool(np.array_equal(a, b)), "shape": list(a.shape)}
    finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
    return {"equal": finite and bool(np.array_equal(a, b)), "finite": finite,
            "shape": list(a.shape),
            "max_abs_difference": float(np.max(np.abs(a-b))) if a.size and finite else None}


class NoSolveCAD:
    """Guard normal lumapi paths; all sessions hidden, saves restricted to new output.

    This is an accidental-call guard for this reviewed script, not a security
    sandbox for arbitrary hostile Python code or modified installation sources.
    """
    def __init__(self, raw, output_root):
        self._raw = raw
        self._root = Path(output_root).resolve()
        self.audit = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._raw.close()

    def __getattr__(self, name):
        if name.lower() in FORBIDDEN or name.lower().startswith("run"):
            raise RuntimeError("Forbidden no-solve CAD method: " + name)
        return getattr(self._raw, name)

    def eval(self, script):
        if re.search(r'getresult\s*\([^;\n]*,[\s]*["\'](?:E|H|Ex|Ey|Ez|Hx|Hy|Hz)["\']', script, re.I):
            raise RuntimeError('Full field dataset request in script is forbidden')
        # Strip string literals and comments before recognizing command calls.
        stripped = re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', '""', script)
        stripped = re.sub(r"#[^\n]*", "", stripped)
        calls = re.findall(r"\b([A-Za-z_]\w*)\s*(?:\(|;|$)", stripped)
        bad = [c for c in calls if c.lower() in FORBIDDEN or c.lower().startswith("run")]
        if bad:
            raise RuntimeError("Forbidden command in no-solve CAD script: " + repr(bad))
        return self._raw.eval(script)

    def getdata(self, monitor, key, option=2):
        if key not in ("x", "y", "z", "f", "lambda"):
            raise RuntimeError("Only coordinate getdata is permitted")
        return self._raw.getdata(monitor, key, option)

    def getresult(self, monitor, result):
        if result.lower() in ("e", "h", "ex", "ey", "ez", "hx", "hy", "hz"):
            raise RuntimeError("Full field extraction is forbidden")
        return self._raw.getresult(monitor, result)

    def save(self, path=None):
        if path is None:
            raise RuntimeError("Implicit save is forbidden")
        target = Path(path).resolve()
        if not target.is_relative_to(self._root):
            raise RuntimeError("Save outside NEW output directory is forbidden")
        if target.exists():
            raise RuntimeError("Overwriting a saved file is forbidden: " + str(target))
        self.audit.append({"save_new_file": str(target)})
        return self._raw.save(str(target))

    def load(self, path):
        self.audit.append({"load": str(Path(path).resolve())})
        return self._raw.load(str(path))


def attempt(fn):
    try:
        value = fn()
        array = np.asarray(value)
        if array.dtype.kind in "fcu" and not np.all(np.isfinite(array)):
            raise ValueError("Nonfinite returned value")
        return {"available": True, "value": plain(value)}
    except Exception as exc:
        return {"available": False, "error": type(exc).__name__ + ": " + str(exc),
                "next_action": "Inspect this property's/API's availability on scheduled hidden CAD; do not infer equality."}


def install_constructor_guard(fdtd_class, output_root, sessions):
    """Keep FDTD's class symbol intact: vendor uses super(FDTD,self).

    Wrap __init__, capture original bound API methods after construction, and
    guard each instance. No subclass/global class-symbol replacement is used.
    """
    from types import SimpleNamespace
    original_init = fdtd_class.__init__
    def blocked(name):
        def refuse(*args, **kwargs):
            raise RuntimeError('Forbidden no-solve CAD method: ' + name)
        return refuse
    def hidden_init(instance, filename=None, key=None, hide=True, serverArgs=None, remoteArgs=None, **kwargs):
        if filename is not None:
            raise RuntimeError('Constructor filename is forbidden; use explicit guarded load')
        original_init(instance, filename=filename, key=key, hide=True,
                      serverArgs=serverArgs or {}, remoteArgs=remoteArgs or {}, **kwargs)
        names = ('eval', 'getdata', 'getresult', 'save', 'load', 'close')
        captured = SimpleNamespace(**{name: getattr(instance, name) for name in names})
        guard = NoSolveCAD(captured, output_root)
        sessions.append(guard)
        for name in names[:-1]:
            setattr(instance, name, getattr(guard, name))
        forbidden_names = FORBIDDEN | {name for name in dir(instance) if name.lower().startswith('run')}
        for name in forbidden_names - {'eval', 'save'}:
            setattr(instance, name, blocked(name))
    fdtd_class.__init__ = hidden_init
    def restore():
        fdtd_class.__init__ = original_init
    return restore


def file_identity(path, hash_contents=True):
    path = Path(path)
    out = {"path": str(path)}
    if not path.exists():
        return dict(out, available=False)
    stat = path.stat()
    out.update(available=True, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    out["sha256"] = sha256(path) if hash_contents else None
    out["hash_status"] = "measured" if hash_contents else "not_requested_large_companion"
    return out


def coords(cad, monitor):
    out = {axis: attempt(lambda a=axis: np.asarray(cad.getdata(monitor, a, 2)).ravel())
           for axis in ("x", "y", "z", "f")}
    if out["f"]["available"]:
        f = np.asarray(out["f"]["value"], dtype=float)
        out["lambda"] = attempt(lambda: C0 / f)
    else:
        out["lambda"] = {"available": False, "error": "No saved frequency coordinates"}
    out["coordinate_api"] = "getdata(monitor,axis,2); lambda=c/f; no E/H transfer"
    out["expected_field_array_shape_from_axes"] = ([len(out[a]["value"]) for a in ("x", "y", "z", "f")] + [3]
                                                  if all(out[a]["available"] for a in ("x", "y", "z", "f")) else None)
    out["field_shape_status"] = "inferred_from_axes_only_not_an_E_array_shape_measurement"
    return out


def index_coords(cad):
    # Vendor store_mesh_info reads getresult('optimization_index','index').
    # Read it within CAD, transfer only its axes, then clear the temporary dataset.
    out = {}
    try:
        cad.eval("s2_probe_index = getresult('optimization_index','index');")
        for axis in ("x", "y", "z", "lambda"):
            def read_axis(a=axis):
                cad.eval("s2_probe_axis = s2_probe_index." + a + ";")
                return np.asarray(cad.getv("s2_probe_axis")).ravel()
            out[axis] = attempt(read_axis)
    except Exception as exc:
        out["dataset_error"] = str(exc)
    finally:
        attempt(lambda: cad.eval("clear(s2_probe_index,s2_probe_axis);"))
    return out


FDTD_PROPS = ["mesh refinement", "mesh type", "x", "y", "z", "x span", "y span", "z span",
              "mesh accuracy", "dt", "x min bc", "x max bc", "y min bc", "y max bc", "z min bc", "z max bc"]
MON_PROPS = ["x", "y", "z", "x span", "y span", "z span", "spatial interpolation",
             "override global monitor settings", "use source limits", "frequency points", "enabled"]


def cad_snapshot(cad, expected):
    props, diffs = {}, {}
    for key, expected_value in expected.items():
        obj, prop = key.rsplit("::", 1)
        result = attempt(lambda o=obj, p=prop: cad.getnamed(o, p))
        props[key] = result
        diffs[key] = compare(np.asarray(result["value"]).squeeze(), expected_value) if result["available"] else {"equal": None}
    settings = {"FDTD::" + p: attempt(lambda p=p: cad.getnamed("FDTD", p)) for p in FDTD_PROPS}
    for axis in ("x", "y", "z"):
        settings["FDTD::user specified mesh " + axis] = attempt(lambda a=axis: cad.getnamed("FDTD", "user specified mesh " + a))
        settings["FDTD::result " + axis] = attempt(lambda a=axis: cad.getresult("FDTD", a))
    for mon in ("optimization_dft", "optimization_index"):
        for prop in MON_PROPS:
            settings[mon + "::" + prop] = attempt(lambda m=mon, p=prop: cad.getnamed(m, p))
    for port in ("Port_1", "Port_2"):
        for prop in ("x", "y", "z", "phase", "injection axis", "direction", "enabled"):
            key = "FDTD::ports::" + port
            settings[key + "::" + prop] = attempt(lambda k=key, p=prop: cad.getnamed(k, p))
    return {"mapped_properties": props, "against_te_point": diffs,
            "mapped_property_count": len(props), "settings": settings}


def coord_comparison(a, b, axes):
    return {axis: compare(a[axis]["value"], b[axis]["value"])
            if axis in a and axis in b and a[axis].get("available") and b[axis].get("available")
            else {"equal": None, "error": "missing coordinate evidence"} for axis in axes}


def probe(args):
    root = Path(args.output_root).resolve()
    repo = Path(args.repo).resolve()
    if root.is_relative_to(repo):
        raise ValueError("Output directory must be outside repository")
    if args.old_root and root.is_relative_to(Path(args.old_root).resolve()):
        raise ValueError("Output directory must be outside original results")
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo))
    import config
    if args.lumapi:
        config.LUMAPI_PATH = args.lumapi
    config.BASE_SAVE_DIR = str(root / "generated")
    config.USE_GPU = False
    from runners.lumopt2_design import recover_te_s2 as recovery
    if root.is_relative_to(Path(args.old_root or recovery.OLD).resolve()):
        raise ValueError("Output directory must be outside original results")
    eng, val = recovery.eng, recovery.val
    lmpt = eng.import_lumopt2()
    import lumapi
    original_fdtd = lumapi.FDTD
    sessions = []
    restore_constructor = install_constructor_guard(lumapi.FDTD, root, sessions)
    assert lumapi.FDTD is original_fdtd
    p = val.te_point(recovery.SPEC)
    expected = eng.make_func(recovery.SPEC)(p)
    api_root = Path(lumapi.__file__).resolve().parent
    report = {"schema": 1, "probe_source_sha256": sha256(__file__), "no_solve": True,
              "not_a_calibration_gate": True, "params_nm": p.tolist(),
              "expected_properties_SI": plain(expected), "spec": plain(__import__('dataclasses').asdict(recovery.SPEC)),
              "api_root": str(api_root), "vendor_sources": {str(f.relative_to(api_root)): sha256(f)
                  for f in sorted((api_root / "lumopt2").rglob("*.py"))},
              "lumapi_sha256": sha256(lumapi.__file__), "cases": {}, "comparisons": {},
              "coordinate_reference": "https://optics.ansys.com/hc/en-us/articles/360034409834-getdata-Script-command"}
    paths = {}
    for quad, dirname in (("re", "lumopt2_te_s2_cport_fd_c_files"), ("im", "lumopt2_te_s2_cport_im_c_files")):
        for role, basename in (("fwd", "fwd_default"), ("adj", "adj_default_Port_2")):
            paths[quad + "_" + role] = Path(args.old_root or recovery.OLD) / dirname / (basename + ".fsp")
    originals = list(paths.values()) + [f.parent / f.stem / (f.stem + "_output.h5") for f in paths.values()]
    report["originals_before"] = [file_identity(f, f.suffix != ".h5" or args.hash_h5) for f in originals]
    project = None
    try:
        project, wl = eng.make_project(recovery.SPEC, str(root / "generated"), lmpt)
        project.generate()  # CAD/mesh only: vendor generate() inspected; no set_fwd/run calls.
        report["requested_wavelengths_m"] = (np.asarray(wl) * 1e-9).tolist()
        mesh = project.parametrization.structure["mesh_coordinates"]
        report["generated_seed_mesh"] = {a: {"available": True, "value": plain(np.asarray(v).ravel())} for a, v in mesh.items()}
        cad = project.fdtd_session.fdtd
        report["generated_seed_settings"] = cad_snapshot(cad, {})["settings"]
        report["FDTD_version"] = attempt(cad.version)
        for label, path in paths.items():
            case = {"source": str(path)}
            report["cases"][label] = case
            try:
                cad.load(str(path))
                # Saved coordinates and CAD geometry MUST precede switchtolayout.
                case["saved_dft"] = coords(cad, "optimization_dft")
                case["saved_layout"] = cad_snapshot(cad, expected)
                if label.endswith("adj"):
                    cad.switchtolayout()
                    project.parametrization.update_structure(project.fdtd_session, p)
                    project.fdtd_session.invalidate_loaded_state()
                    case["nominal_index_after_update"] = index_coords(cad)
                    report["comparisons"][label + "_dft_vs_nominal_index"] = coord_comparison(case["saved_dft"], case["nominal_index_after_update"], ("x", "y", "z"))
                report["comparisons"][label + "_dft_vs_generated_seed_mesh"] = coord_comparison(case["saved_dft"], report["generated_seed_mesh"], ("x", "y", "z"))
                case["lambda_vs_requested"] = compare(case["saved_dft"]["lambda"]["value"], report["requested_wavelengths_m"]) if case["saved_dft"]["lambda"].get("available") else {"equal": None}
            except Exception as exc:
                case["error"] = type(exc).__name__ + ": " + str(exc)
        for left, right in (("re_fwd", "re_adj"), ("im_fwd", "im_adj"), ("re_fwd", "im_fwd")):
            report["comparisons"][left + "_vs_" + right] = coord_comparison(report["cases"][left].get("saved_dft", {}), report["cases"][right].get("saved_dft", {}), ("x", "y", "z", "lambda"))
            lset = report["cases"][left].get("saved_layout", {}).get("settings", {})
            rset = report["cases"][right].get("saved_layout", {}).get("settings", {})
            report["comparisons"][left + "_settings_vs_" + right] = coord_comparison(lset, rset, sorted(set(lset) | set(rset)))
    except Exception as exc:
        report["execution_error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        report["originals_after"] = [file_identity(f, f.suffix != ".h5" or args.hash_h5) for f in originals]
        report["original_file_identity_unchanged"] = report["originals_before"] == report["originals_after"]
        report["CAD_call_audits"] = [s.audit for s in sessions]
        report["critical_evidence_available"] = (len(report["cases"]) == 4 and all(
            all(case.get("saved_dft", {}).get(a, {}).get("available", False) for a in ("x", "y", "z", "lambda"))
            and all(v.get("available", False) for v in case.get("saved_layout", {}).get("mapped_properties", {}).values())
            and len(case.get("saved_layout", {}).get("mapped_properties", {})) == len(expected)
            and (not label.endswith("adj") or all(case.get("nominal_index_after_update", {}).get(a, {}).get("available", False) for a in ("x", "y", "z")))
            for label, case in report["cases"].items()))
        for s in sessions:
            attempt(s.close)
        restore_constructor()
        out = root / "provenance.json"
        with out.open("x", encoding="utf-8") as stream:
            json.dump(plain(report), stream, indent=2, allow_nan=False)
        print(out, flush=True)
    return 2 if report.get("execution_error") or any("error" in c for c in report["cases"].values()) or not report["original_file_identity_unchanged"] or not report["critical_evidence_available"] else 0


def self_test():
    from contextlib import nullcontext
    import tempfile
    class Fake:
        def __getattr__(self, name):
            def called(*a, **k):
                return (name, a, k)
            return called
    with nullcontext(tempfile.mkdtemp(prefix="guard_tests_", dir=Path(__file__).parent)) as tmp:
        guard = NoSolveCAD(Fake(), tmp)
        bad = [lambda: guard.run(), lambda: guard.runjobs(), lambda: guard.addjob('a'),
               lambda: guard.eval('run;'), lambda: guard.eval('run'),
               lambda: guard.eval('run; runjobs("FDTD");'), lambda: guard.eval('rUnparallel();'),
               lambda: guard.eval('system("bad");'), lambda: guard.eval('save("old.fsp");'),
               lambda: guard.eval('a=getresult("optimization_dft","E");'),
               lambda: guard.save(), lambda: guard.save(Path(tmp).parent / 'old.fsp'),
               lambda: guard.getresult('optimization_dft', 'E'), lambda: guard.getdata('optimization_dft', 'Ex')]
        for fn in bad:
            try:
                fn()
            except RuntimeError:
                continue
            raise AssertionError("negative control was not blocked")
        guard.eval('a=getresult("optimization_index","index"); # runjobs("FDTD");\nb=a.x;')
        guard.getdata('optimization_dft', 'x', 2)
        target = Path(tmp) / 'new.fsp'
        guard.save(target)
        target.write_text('existing')
        try:
            guard.save(target)
        except RuntimeError:
            pass
        else:
            raise AssertionError("overwrite not blocked")
        assert compare([0., 1.], [1., 2.])["equal"] is False  # same shape, wrong coordinates
        assert compare([0., 1.], [0., 1.])["equal"] is True
        assert compare([float('nan')], [float('nan')])["equal"] is False
        assert coord_comparison({}, {}, ('x',))["x"]["equal"] is None
        assert attempt(lambda: np.array([float('nan')]))["available"] is False
        scalar = attempt(lambda: np.array([[5.0]]))
        assert compare(np.asarray(scalar["value"]).squeeze(), 5.0)["equal"] is True
        # Test the diagnostic callback itself independently of any CAD driver.
        def unavailable():
            raise RuntimeError('unsupported coordinate')
        assert attempt(unavailable)["available"] is False
        class FakeBase(Fake):
            def __init__(self, **kwargs):
                self.constructor_options = kwargs
        class FDTD(FakeBase):
            def __init__(self, **kwargs):
                super(FDTD, self).__init__(**kwargs)  # actual vendor superclass lookup pattern
        original_class = FDTD
        original_init = FDTD.__init__
        sessions = []
        restore = install_constructor_guard(FDTD, tmp, sessions)
        try:
            instance = FDTD(hide=False)
            assert FDTD is original_class
            assert instance.constructor_options['hide'] is True
            assert instance.getdata('monitor', 'x', 2)[0] == 'getdata'
            for action in (lambda: instance.run(), lambda: instance.eval('run;'),
                           lambda: instance.save(Path(tmp).parent / 'protected.fsp')):
                try:
                    action()
                except RuntimeError:
                    continue
                raise AssertionError('instance guard bypassed')
        finally:
            restore()
        assert FDTD.__init__ is original_init
    print("PASS no-CAD guard, coordinate and missing-evidence tests")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--repo')
    parser.add_argument('--lumapi', help='Actual cluster lumapi.py path; do not use Windows default remotely')
    parser.add_argument('--old-root', help='Default /work/results/validate_te/results/validate_te_s2 from recovery.OLD')
    parser.add_argument('--output-root', help='NEW exclusive directory, outside source and original results')
    parser.add_argument('--hash-h5', action='store_true', help='Stream large companion hashes; default records stat only')
    options = parser.parse_args()
    if options.self_test:
        self_test()
    else:
        if not options.repo or not options.output_root or not options.lumapi:
            parser.error('--repo, --lumapi and --output-root are required')
        sys.exit(probe(options))
