#!/bin/bash
# MeCab-ko 및 MeCab-ko-dic 설치 스크립트 (Ubuntu/Debian)

set -e

echo "Installing MeCab-ko for Linux..."

# 필수 패키지 설치
apt-get update
apt-get install -y \
    curl \
    git \
    make \
    g++ \
    autoconf \
    automake \
    libtool \
    pkg-config

# MeCab 설치
cd /tmp
curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar zxfv mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
make install
ldconfig

# MeCab-ko-dic 설치
cd /tmp
curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar -zxvf mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
./autogen.sh
./configure
make
make install

# Python 패키지 설치
pip install konlpy jpype1

echo "MeCab-ko installation completed!"
echo "Test: mecab --version"
mecab --version
