import logging

from fastapi import FastAPI

logger = logging.getLogger("uvicorn")

api = FastAPI()


@api.get("/ping")
def ping():
    logger.info("ping")
    return "ping"
