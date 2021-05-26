import pytest
from unittest.mock import patch

from app.sample import Sample


def dummy_func(self) -> int:
    return 2


@pytest.fixture
def sample() -> Sample:
    return Sample()


def test_original(sample) -> None:
    assert sample.test()


@patch("app.sample.Sample._load_param", dummy_func)
def test_use_patch(sample) -> None:
    assert sample.test()


def test_original_again(sample) -> None:
    assert sample.test()