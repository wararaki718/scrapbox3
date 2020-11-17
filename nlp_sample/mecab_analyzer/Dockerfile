FROM python:3.8

RUN apt-get update && \
    apt-get install -y git sudo mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8 && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/neologd/mecab-ipadic-neologd -b v0.0.7 && \
    ./mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -y && \
    rm -rf mecab-ipadic-neologd

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir && rm /tmp/requirements.txt

COPY src /app
WORKDIR /app

CMD ["python", "main.py"]
