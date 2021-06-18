#!/bin/bash

wget http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/data/20170201.tar.bz2
tar xf 20170201.tar.bz2
rm 20170201.tar.bz2
rm entity_vector/entity_vector.model.txt

echo "DONE"
