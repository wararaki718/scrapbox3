import math
from unittest.mock import patch, MagicMock

from app.sample import Sample


def load_dummy(filename: str) -> MagicMock:
    mock = MagicMock()
    mock.transform = MagicMock(return_value=1.0)
    return mock


@patch("app.sample.joblib.load", load_dummy)
def test_sample():
    sample = Sample()
    assert math.isclose(sample.method("hello"), 1.0)
