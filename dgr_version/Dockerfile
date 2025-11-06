FROM python:3.11-slim

WORKDIR /app

# 의존성 파일 복사 및 설치
COPY get_title_api.py .

# 필요한 패키지 설치
RUN pip install --no-cache-dir \
    flask \
    transformers \
    torch \
    sentencepiece \
    sentence-transformers \
    numpy

# 환경 변수 설정
ENV TOKENIZERS_PARALLELISM=false

# 포트 노출
EXPOSE 5555

# 실행
CMD ["python", "get_title_api.py"]
