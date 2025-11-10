FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    git \
    make \
    g++ \
    autoconf \
    automake \
    libtool \
    pkg-config \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Java 환경변수 설정 (konlpy 필요)
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# MeCab-ko 설치
RUN cd /tmp && \
    curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz && \
    tar zxfv mecab-0.996-ko-0.9.2.tar.gz && \
    cd mecab-0.996-ko-0.9.2 && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    rm -rf /tmp/mecab-*

# MeCab-ko-dic 설치
RUN cd /tmp && \
    curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz && \
    tar -zxvf mecab-ko-dic-2.1.1-20180720.tar.gz && \
    cd mecab-ko-dic-2.1.1-20180720 && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    rm -rf /tmp/mecab-*

# MeCab 환경변수 설정
ENV MECAB_PATH=/usr/local/lib/libmecab.so
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Python 패키지 설치 (JPype1 먼저 설치)
COPY requirements.txt .
RUN pip install --no-cache-dir jpype1
RUN pip install --no-cache-dir konlpy
RUN pip install --no-cache-dir -r requirements.txt

# konlpy Mecab 테스트
RUN python3 -c "from konlpy.tag import Mecab; m = Mecab(); print('✅ Mecab 로드 성공:', m.morphs('테스트'))" 2>&1 || \
    echo "⚠️  Mecab 로드 실패, regex fallback 모드 사용"

# 애플리케이션 파일 복사
COPY . .

# 포트 노출
EXPOSE 5555

# 실행
CMD ["python", "dgr_version/get_title_api.py"]
