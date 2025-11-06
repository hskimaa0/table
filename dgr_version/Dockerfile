FROM python:3.11-slim

WORKDIR /app

# 의존성 파일만 먼저 복사 (레이어 캐싱 최적화)
COPY requirements.txt .

# 필요한 패키지 설치 (requirements.txt가 변경되지 않으면 캐시 사용)
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사 (코드 변경 시 이 레이어만 재빌드)
COPY get_title_api.py .

# 환경 변수 설정
ENV TOKENIZERS_PARALLELISM=false

# 포트 노출
EXPOSE 5555

# 실행
CMD ["python", "get_title_api.py"]
