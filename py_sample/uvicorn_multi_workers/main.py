from fastapi import FastAPI

api = FastAPI()

@api.get('/hello')
def hello():
    return 'hello'
