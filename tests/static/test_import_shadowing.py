"""Catches a function-local import that shadows a module already used above it.

`import mlflow.pytorch` inside an `except` block binds `mlflow` as a *function-local*
name for the whole function, so an earlier `mlflow.onnx.load_model(...)` in the same
function raises UnboundLocalError. In batch_inference.py that quietly disabled the
entire ONNX serving path and made every model look like it had no ONNX artifact.

The rule is general: within one function, a name may not be used before the line that
imports it locally.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_DIRS = ["dags", "jobs"]


def _python_files():
    for directory in SOURCE_DIRS:
        yield from sorted((ROOT / directory).rglob("*.py"))


def _shadowing_violations(tree):
    violations = []

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        local_imports = {}
        for node in ast.walk(func):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound = (alias.asname or alias.name).split(".")[0]
                    local_imports.setdefault(bound, node.lineno)

        if not local_imports:
            continue

        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                import_line = local_imports.get(node.id)
                if import_line is not None and node.lineno < import_line:
                    violations.append(
                        f"{func.name}(): '{node.id}' used on line {node.lineno} but "
                        f"imported locally on line {import_line}"
                    )

    return violations


def test_no_local_import_shadows_earlier_use():
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for violation in _shadowing_violations(tree):
            offenders.append(f"{path.relative_to(ROOT)}: {violation}")

    assert not offenders, (
        "Function-local imports shadow a name already used earlier in the same "
        "function (UnboundLocalError at runtime):\n  " + "\n  ".join(offenders)
    )
