class CalculatorService:
    @classmethod
    def divide(cls, x: int, y: int) -> float:
        try:
            result = x / y
        except ZeroDivisionError as e:
            raise e
        return result
