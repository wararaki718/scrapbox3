# spring boot tutorial

## run

```shell
docker-compose up
```

```shell
./gradlew bootRun
```

## store key

```shell
curl --header "Content-Type: application/json" \
    --request POST \
    --data '{"description":"configuration","details":"congratulations, you have set up JDBC correctly!","done": "true"}' \
    http://localhost:8080
```

## get data

```shell
curl localhost:8080
```
