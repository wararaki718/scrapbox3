# sample rest application by using django

## setup environment

```shell
pip install Django==3.1.2 djangorestframework==3.12.1
```

## run application

```shell
python manage.py runserver
```

check response

```shell
curl -H 'Accept: application/json; indent=4' -u root:password http://127.0.0.1:8000/users/
```
