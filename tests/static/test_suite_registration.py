"""Every unit test file must actually be executed by something.

A test that runs nowhere is worse than no test: it reports nothing while everyone
believes it covers something. Not hypothetical here - the training suites sat in
tests/unit for their whole life running only on one developer's laptop, because
`dev.sh` and `ci.yml` each enumerate by hand the files they execute. tests/static and
tests/airflow are run whole, by directory, so only tests/unit needs guarding.

Pure stdlib on purpose: this has to run before anything is built.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
UNIT_DIR = ROOT / "tests" / "unit"
RUNNERS = [ROOT / "dev.sh", ROOT / ".github" / "workflows" / "ci.yml"]


def test_every_unit_test_file_is_registered_in_every_runner():
    missing = []
    for runner in RUNNERS:
        text = runner.read_text()
        for path in sorted(UNIT_DIR.glob("test_*.py")):
            if path.name not in text:
                missing.append(f"{runner.relative_to(ROOT)} never runs {path.name}")

    assert not missing, (
        "Unit tests that no runner executes - add them to the suite that has their "
        "dependencies:\n  " + "\n  ".join(missing)
    )
