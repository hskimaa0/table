# Docker로 KoBERT 타이틀 추출 API 실행하기

## 📋 필수 파일 확인

실행 전 다음 파일들이 있는지 확인:

```
표_연속성체크_04/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── dgr_version/
│   ├── get_title_api.py
│   ├── kobert_classifier.py
│   └── kobert_table_classifier.pt  # ⚠️ 학습 완료 후 생성된 파일 필수!
```

## 🚀 실행 방법

### 1. 이미지 빌드

```bash
# 프로젝트 루트 디렉토리에서 실행
cd d:\표_연속성체크_04

# Docker 이미지 빌드
docker-compose build
```

### 2. 컨테이너 실행

```bash
# 백그라운드 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 3. API 테스트

```bash
curl -X POST http://localhost:5555/get_title \
  -H "Content-Type: application/json" \
  -d '{
    "tables": [...],
    "texts": [...]
  }'
```

### 4. 중지 및 재시작

```bash
# 중지
docker-compose down

# 재시작
docker-compose restart
```

## 🔧 환경 설정

### GPU 사용 (선택)

GPU를 사용하려면 `docker-compose.yml` 수정:

```yaml
services:
  title-api:
    build: .
    runtime: nvidia  # 추가
    environment:
      - NVIDIA_VISIBLE_DEVICES=all  # 추가
```

### 포트 변경

`docker-compose.yml`에서 포트 수정:

```yaml
ports:
  - "8080:5555"  # 호스트:컨테이너
```

## 📊 리소스 요구사항

- **최소**:
  - CPU: 2 코어
  - RAM: 4GB
  - 디스크: 5GB

- **권장**:
  - CPU: 4 코어
  - RAM: 8GB
  - 디스크: 10GB
  - GPU: NVIDIA GPU (선택)

## 🐛 트러블슈팅

### 1. 모델 파일 없음 오류

```bash
⚠️  KoBERT 모델 파일 없음: kobert_table_classifier.pt
```

**해결**: 먼저 학습 실행
```bash
cd dgr_version
python train_kobert.py
```

### 2. 메모리 부족

`docker-compose.yml`에 메모리 제한 추가:

```yaml
services:
  title-api:
    mem_limit: 8g
```

### 3. 컨테이너 재빌드

코드 수정 후:

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📝 로그 확인

```bash
# 실시간 로그
docker-compose logs -f

# 최근 100줄
docker-compose logs --tail=100

# 특정 컨테이너만
docker logs kobert-title-api
```

## 🔄 업데이트 방법

1. 코드 수정
2. 컨테이너 재시작

```bash
docker-compose restart
```

모델 파일이나 의존성 변경 시:

```bash
docker-compose down
docker-compose build
docker-compose up -d
```
