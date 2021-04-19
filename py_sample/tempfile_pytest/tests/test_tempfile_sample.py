import json
from typing import Generator
from tempfile import NamedTemporaryFile

import pytest


@pytest.fixture(scope="module")
def sample_json() -> Generator[str, None, None]:
    with NamedTemporaryFile("w") as f:
        f.write(
            json.dumps({"sample": "text"})
        )
        f.seek(0)
        yield f.name


def test_json_load(sample_json: str) -> None:
    with open(sample_json) as f:
        data = json.load(f)
    assert data == {"sample": "text"}
