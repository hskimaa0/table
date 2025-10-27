# GPU 가속 최적화 가이드

## 현재 상태 확인

### GPU 사용 가능 여부 확인
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Docling GPU 설정 확인
개선된 `pdf_to_json.py`는 자동으로 GPU를 감지하고 사용합니다.

```bash
python pdf_to_json.py
```

출력 예시:
```
🖥️  Device: CUDA
   GPU: NVIDIA GeForce RTX 3060
   CUDA Version: 11.8
✓ Converter initialized (GPU will be used automatically if available)
```

## 성능 최적화 팁

### 1. CUDA 메모리 관리
큰 PDF 파일 처리 시 메모리 부족이 발생할 수 있습니다.

```python
# pdf_to_json.py에서 배치 크기 조정
# 메모리가 부족하면 한 번에 처리하는 파일 수를 줄이세요
```

### 2. TableFormer 모드 선택

현재 설정: `TableFormerMode.ACCURATE` (가장 정확하지만 느림)

**빠른 처리가 필요한 경우:**
```python
pipeline_options.table_structure_options.mode = TableFormerMode.FAST
```

**옵션 비교:**
- `FAST`: 빠르지만 정확도 낮음
- `ACCURATE`: 느리지만 높은 정확도 (기본값)

### 3. 멀티스레딩
여러 PDF를 병렬로 처리하려면:

```python
# 시스템 코어 수에 따라 조정
pipeline_options.accelerator_options.num_threads = 4
```

## GPU 메모리 모니터링

### 실시간 GPU 사용량 확인
```bash
# Windows (NVIDIA)
nvidia-smi -l 1

# 또는 Python으로
python -c "import torch; print(f'Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB'); print(f'Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB')"
```

## 문제 해결

### GPU를 사용하지 않는 경우

1. **PyTorch CUDA 버전 확인**
```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

2. **CUDA 드라이버 업데이트**
- NVIDIA 드라이버 최신 버전 설치
- [NVIDIA 드라이버 다운로드](https://www.nvidia.com/Download/index.aspx)

3. **PyTorch 재설치 (CUDA 지원 버전)**
```bash
# CUDA 11.8 버전
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 버전
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 메모리 부족 오류

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결 방법:**
1. 한 번에 처리하는 PDF 수 줄이기
2. 배치 크기 감소
3. 테이블 모드를 FAST로 변경
4. 작은 PDF부터 처리

## 성능 비교

### CPU vs GPU (예상)
- **CPU**: ~30-60초/페이지
- **GPU**: ~5-10초/페이지 (3-6배 빠름)

실제 성능은 다음에 따라 달라집니다:
- PDF 복잡도
- 테이블 수
- GPU 성능
- 이미지 해상도

## 권장 사양

### 최소 사양
- GPU: NVIDIA GTX 1060 이상
- VRAM: 6GB 이상
- RAM: 16GB 이상

### 권장 사양
- GPU: NVIDIA RTX 3060 이상
- VRAM: 8GB 이상
- RAM: 32GB 이상

## 현재 설정 요약

`pdf_to_json.py`의 최적화된 설정:
```python
✓ GPU 자동 감지 및 사용
✓ TableFormer ACCURATE 모드 (높은 정확도)
✓ 테이블 구조 인식 활성화
✓ 에러 로깅
✓ 이미 처리된 파일 스킵
```
