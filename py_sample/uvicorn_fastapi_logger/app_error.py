import logging

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api = FastAPI()


@api.get("/ping")
def ping():
    logger.info("ping")
    return "ping"
