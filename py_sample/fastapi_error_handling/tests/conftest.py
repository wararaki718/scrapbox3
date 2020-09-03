from fastapi.testclient import TestClient
import pytest

from src.app import app
from src.calculator import CalculatorService


@pytest.fixture
def calculator():
    return CalculatorService

@pytest.fixture
def web_client():
    return TestClient(app)
