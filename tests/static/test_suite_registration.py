"""Every test file must actually be executed by something.

A test that runs nowhere is worse than no test: it reports nothing while everyone
believes it covers something. Not hypothetical here - the training suites sat in
tests/unit for their whole life running only on one developer's laptop, because
`dev.sh` and `ci.yml` each enumerate by hand the files they execute.

Two different exposures, so two tests. tests/unit is enumerated file by file, so a
new file there can be forgotten one runner at a time. Every other suite is run whole
by directory, where the way to run nowhere is for no runner to name the directory
at all.

Pure stdlib on purpose: this has to run before anything is built.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
UNIT_DIR = ROOT / "tests" / "unit"
RUNNERS = [ROOT / "dev.sh", ROOT / ".github" / "workflows" / "ci.yml"]
TESTS_DIR = ROOT / "tests"


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


def test_every_test_directory_is_named_by_a_runner():
    """dev.sh is the only runner checked here. CI cannot run every suite - the
    integration suite needs Nessie, MinIO and MLflow up, and the training image it
    runs in is 11 GB - so requiring ci.yml to name them all would just be a lie the
    test enforces."""
    dev_sh = (ROOT / "dev.sh").read_text()

    unrun = [
        directory.name
        for directory in sorted(TESTS_DIR.iterdir())
        if directory.is_dir()
        and not directory.name.startswith("__")
        and any(directory.glob("test_*.py"))
        and f"tests/{directory.name}" not in dev_sh
        and f"/tests/{directory.name}" not in dev_sh
    ]

    assert not unrun, (
        "test directories dev.sh never runs - add a command for them:\n  "
        + "\n  ".join(unrun)
    )
