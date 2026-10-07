"""PostToolUse hook: AST syntax check on any edited/written .py file.

Reads the hook JSON from stdin, no-ops (exit 0) for non-Python files, and
exits 2 with the error on stderr so Claude sees the failure immediately.
Uses ast.parse (not py_compile) to avoid dropping .pyc files in __pycache__.
"""
import ast
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)

file_path = (payload.get("tool_input") or {}).get("file_path", "")
if not file_path.lower().endswith(".py"):
    sys.exit(0)

try:
    with open(file_path, encoding="utf-8") as fh:
        source = fh.read()
except OSError:
    sys.exit(0)

try:
    ast.parse(source, filename=file_path)
except SyntaxError as exc:
    print(
        f"Python syntax error in {file_path}: line {exc.lineno}: {exc.msg}",
        file=sys.stderr,
    )
    sys.exit(2)
