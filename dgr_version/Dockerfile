FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 기본 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 파일만 먼저 복사 (레이어 캐싱 최적화)
COPY requirements.txt .

# 파이썬 패키지 (CUDA 12.1용 PyTorch)
RUN pip3 install --upgrade pip && \
    pip3 install --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio && \
    pip3 install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사 (코드 변경 시 이 레이어만 재빌드)
COPY get_title_api.py .

# 환경 변수 설정
ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES=0

# 포트 노출
EXPOSE 5555

# 실행
CMD ["python3", "get_title_api.py"]
