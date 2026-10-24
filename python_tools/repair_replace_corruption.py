"""Repair `.replace` that got corrupted into `_$` in Python files -- only where the code cannot compile without the fix.

Study dir: python_tools/ | Created 2026-10-22 | Purpose: undo an external tool's `.replace` -> `_$` rewrite safely.
Rule: in Python, `$` is never valid outside a string, so a real corruption makes the file fail to compile. For every `_$`
in a file that fails to compile, try `.replace`; KEEP a change only if undoing it alone breaks compilation again. A `_$`
inside a string (bash `xkb_$$`, `_$(date)`, `${VAR}_$`) compiles either way, so it is always left untouched.
Files that already compile are never changed. Non-Python files are only REPORTED (no compiler to prove a fix).
Usage: python python_tools/repair_replace_corruption.py [root ...]            (dry run: prints the diff)
       python python_tools/repair_replace_corruption.py [root ...] --apply    (writes; a .orig copy is kept next to each file)
"""
import difflib
import os
import re
import sys

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "site-packages", "node_modules"}
REPORT_EXT = (".m", ".sh", ".md", ".ipynb", ".json", ".txt")
BAD = "_$"


def compiles(src):
    try:
        compile(src, "<check>", "exec")
        return True
    except SyntaxError:
        return False


def apply_at(src, positions):
    """Replace `_$` -> `.replace` at the given character offsets (offsets refer to the original text)."""
    out, last = [], 0
    for p in sorted(positions):
        out += [src[last:p], ".replace"]
        last = p + len(BAD)
    return "".join(out) + src[last:]


def repair(src):
    """Return (fixed source, number of fixes) or (None, 0) when no provable repair exists."""
    if compiles(src) or BAD not in src:
        return src, 0
    hits = [m.start() for m in re.finditer(re.escape(BAD), src)]
    keep = set(hits)
    if not compiles(apply_at(src, keep)):
        return None, 0                                   # broken for another reason too: touch nothing
    for p in hits:                                       # drop every change the file does not need to compile
        if compiles(apply_at(src, keep - {p})):
            keep.discard(p)
    return apply_at(src, keep), len(keep)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--apply" in sys.argv
    roots = args or ["."]
    n_files = n_fix = 0
    for root in roots:
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for f in files:
                path = os.path.join(d, f)
                if f.endswith(REPORT_EXT):
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        for i, line in enumerate(fh, 1):
                            if re.search(r"_\$(?![{$])", line):
                                print(f"REPORT ONLY (not Python, check by hand): {path}:{i}: {line.rstrip()}")
                    continue
                if not f.endswith(".py"):
                    continue
                with open(path, encoding="utf-8", newline="") as fh:
                    src = fh.read()
                fixed, k = repair(src)
                if fixed is None:
                    print(f"SKIPPED (does not compile even after the fix -- other damage): {path}")
                    continue
                if k == 0:
                    continue
                n_files += 1; n_fix += k
                sys.stdout.writelines(difflib.unified_diff(src.splitlines(True), fixed.splitlines(True), path, path + " (repaired)"))
                if write:
                    with open(path + ".orig", "w", encoding="utf-8", newline="") as fh:
                        fh.write(src)
                    with open(path, "w", encoding="utf-8", newline="") as fh:
                        fh.write(fixed)
    print(f"\n{n_fix} fix(es) in {n_files} file(s) -- {'WRITTEN (.orig backups kept)' if write else 'dry run, nothing written (add --apply)'}")


if __name__ == "__main__":
    main()
