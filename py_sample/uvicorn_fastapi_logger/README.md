# uvicorn fastapi logger

## setup environment

```shell
pip install fastapi uvicorn
```

## run

```shell
uvicorn app_error:api
```

```shell
uvicorn app:api
```

check

```shell
curl localhost:8000/ping
```

## output

エラーケース

```shell
 $ uvicorn app_error:api
 INFO:     Started server process [10380]
 INFO:uvicorn.error:Started server process [10380]
 INFO:     Waiting for application startup.
 INFO:uvicorn.error:Waiting for application startup.
 INFO:     Application startup complete.
 INFO:uvicorn.error:Application startup complete.
 INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
 INFO:uvicorn.error:Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
 INFO:app_error:ping
 INFO:     127.0.0.1:50773 - "GET /ping HTTP/1.1" 200 OK
```

通常

```shell
 $ uvicorn app:api
 INFO:     Started server process [10508]
 INFO:     Waiting for application startup.
 INFO:     Application startup complete.
 INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
 INFO:     ping
 INFO:     127.0.0.1:50786 - "GET /ping HTTP/1.1" 200 OK
```
