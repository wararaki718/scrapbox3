#!/bin/bash

wget http://sudachi.s3-website-ap-northeast-1.amazonaws.com/sudachidict/sudachi-dictionary-20201223-small.zip
unzip sudachi-dictionary-20201223-small.zip
rm sudachi-dictionary-20201223-small.zip

mkdir -p src/main/resources
mv sudachi-dictionary-20201223/system_small.dic src/main/resources/system_core.dic
rm -rf sudachi-dictionary-20201223
echo "DONE"
