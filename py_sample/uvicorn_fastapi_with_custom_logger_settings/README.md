# uvicorn fastapi logger

## setup environment

```shell
pip install fastapi uvicorn
```

## run

```shell
uvicorn app:api --log-config log_config.json
```

check

```shell
curl localhost:8000/ping
```
