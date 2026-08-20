"""Every source file must at least parse. Cheapest possible regression net."""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_all_sources_parse():
    failures = []
    for directory in ("dags", "jobs", "tests"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            try:
                ast.parse(path.read_text())
            except SyntaxError as e:
                failures.append(f"{path.relative_to(ROOT)}: {e}")
    assert not failures, "Syntax errors:\n  " + "\n  ".join(failures)
