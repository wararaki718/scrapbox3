# text similarity approaches

## download dataset

```shell
bash download.sh
```

## download model

https://www.dropbox.com/s/j75s0eq4eeuyt5n/jawiki.doc2vec.dbow300d.tar.bz2?dl=0

after download the model, move to model directory.

```shell
cd model
mv ~/Downloads/jawiki.doc2vec.dbow300d.tar.bz2 .
tar -jxvf jawiki.doc2vec.dbow300d.tar.bz2
```

## setup environment

```shell
pip install transforms fugashi unidic-lite numpy scikit-learn gensim spacy ginza
```
