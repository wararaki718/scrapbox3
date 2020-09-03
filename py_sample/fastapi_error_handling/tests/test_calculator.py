'''
src.calculator
'''
import math

def test_calculator(calculator):
    result = calculator.divide(10, 2)
    assert math.isclose(result, 5.0)
