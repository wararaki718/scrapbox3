# sample mypy readme

## setup

```shell
pip install mypy
```

## use mypy

```shell
mypy main.py
```

except import modules

```shell
mypy --follow-imports skip main.py
```

must define type-hint

```shell
mypy --disallow-untyped-defs main.py
```
