FROM python:3.8

RUN apt-get update && \
    apt-get install -y git sudo mecab libmecab-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/neologd/mecab-unidic-neologd && \
    ./mecab-unidic-neologd/bin/install-mecab-unidic-neologd -n -y && \
    rm -rf mecab-unidic-neologd

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir && rm /tmp/requirements.txt

COPY src /app
WORKDIR /app

CMD ["python", "main.py"]
