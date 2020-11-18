#!/bin/bash
sudo yum install -y  bzip2 bzip2-devel gcc gcc-c++ git make wget curl openssl-devel readline-devel zlib-devel patch file

## mecab setup
mkdir -p ~/source/mecab
cd ~/source/mecab
wget 'https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7cENtOXlicTFaRUE' -O mecab-0.996.tar.gz
tar zxvf mecab-0.996.tar.gz
cd mecab-0.996

sudo mkdir -p /opt/mecab
./configure --prefix=/opt/mecab --with-charset=utf8 --enable-utf8-only
make
sudo make install

echo "export PATH=/opt/mecab/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
mecab-config --libs-only-L | sudo tee /etc/ld.so.conf.d/mecab.conf
sudo ldconfig

## mecab-ipadic setup
mkdir ~/source/mecab-ipadic
cd ~/source/mecab-ipadic
wget 'https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7MWVlSDBCSXZMTXM' -O mecab-ipadic-2.7.0-20070801.tar.gz
tar zxvf mecab-ipadic-2.7.0-20070801.tar.gz
cd mecab-ipadic-2.7.0-20070801

./configure --with-mecab-config=/opt/mecab/bin/mecab-config --with-charset=utf8
make
sudo make install

## neologd setup
cd ~/source
git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git

./mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -y -p /opt/mecab/lib/mecab/dic/neologd

echo "setup done"
