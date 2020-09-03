from fastapi import FastAPI, HTTPException

from .calculator import CalculatorService
from .models import Result, Variables


app = FastAPI()


@app.post('/divide', response_model=Result)
def divide(variables: Variables):
    try:
        result = CalculatorService.divide(**(variables.dict()))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail='Failed to divide error'
        )
    return Result(result=result)
