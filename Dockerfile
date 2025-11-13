FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY dgr_version/ ./dgr_version/

# Python 경로에 dgr_version 추가
ENV PYTHONPATH=/app:/app/dgr_version:$PYTHONPATH

# 작업 디렉토리를 dgr_version으로 변경
WORKDIR /app/dgr_version

# 복사된 파일 확인 (디버깅용)
RUN ls -la /app/dgr_version/

# 포트 노출
EXPOSE 5555

# API 서버 실행
CMD ["python", "get_title_api.py"]
