# sample uvicorn multiple workers

## setup environment

```shell
pip install -r requirements.txt
```

## run

```shell
uvicorn main:api --workers 4
```

```shell
gunicorn main:api -w 4
```

recommend

```shell
gunicorn main:api -w 4 -k uvicorn.workers.UvicornWorker
```
