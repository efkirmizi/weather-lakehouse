"""Epoch budget resolution. Runs inside dag-pytorch-model-training (PYTHONPATH=/app)."""
import os

import pytest

from trainer import resolve_epochs


@pytest.fixture(autouse=True)
def clear_override():
    os.environ.pop("TRAINING_EPOCHS", None)
    yield
    os.environ.pop("TRAINING_EPOCHS", None)


def test_defaults_when_unset():
    assert resolve_epochs(is_incremental=False, scratch=10, incremental=2) == 10
    assert resolve_epochs(is_incremental=True, scratch=10, incremental=2) == 2


def test_override_is_honoured():
    os.environ["TRAINING_EPOCHS"] = "1"
    assert resolve_epochs(False, scratch=10, incremental=2) == 1


def test_override_tolerates_whitespace():
    os.environ["TRAINING_EPOCHS"] = "  3 "
    assert resolve_epochs(False, scratch=10, incremental=2) == 3


@pytest.mark.parametrize("bad", ["abc", "0", "-5", ""])
def test_bad_override_falls_back_to_the_default(bad):
    """A typo in a DAG conf must never silently weaken the production budget."""
    os.environ["TRAINING_EPOCHS"] = bad
    assert resolve_epochs(False, scratch=10, incremental=2) == 10
