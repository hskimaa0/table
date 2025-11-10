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

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Okt 형태소 분석기 테스트
RUN python3 -c "from konlpy.tag import Okt; okt = Okt(); print('✅ Okt 로드 성공:', okt.morphs('형태소분석 테스트'))" || \
    echo "⚠️  형태소 분석기 로드 실패, regex fallback 모드 사용"

# 애플리케이션 파일 복사
COPY . .

# 포트 노출
EXPOSE 5555

# 실행
CMD ["python", "dgr_version/get_title_api.py"]
