FROM python:3.11-slim

WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt .

# 파이썬 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY get_title_api.py .

# 포트 노출
EXPOSE 5555

# 실행
CMD ["python", "get_title_api.py"]
