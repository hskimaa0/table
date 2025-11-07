# 표_연속성체크_04

## 타이틀 추출 API (get_title_api.py)

### 개요
하이브리드 방식(규칙 기반 + NLI + 임베딩 유사도 + 레이아웃)으로 PDF 문서에서 표의 제목을 자동 추출하는 Flask API

---

## 사용 라이브러리 및 원리

### 1. **Flask** ([dgr_version/get_title_api.py:5](dgr_version/get_title_api.py#L5))
- **역할**: REST API 서버 프레임워크
- **원리**: HTTP POST 요청으로 표/텍스트 데이터를 받아 처리 후 JSON 반환
- **엔드포인트**:  - 표 배열과 텍스트 배열을 받아 각 표의 제목을 찾아 반환

### 2. **torch** ([dgr_version/get_title_api.py:45](dgr_version/get_title_api.py#L45))
- **역할**: 딥러닝 프레임워크 (GPU 가속 지원)
- **원리**:
  - CUDA 가용성 체크 및 디바이스 할당 (GPU/CPU)
  - FP16/FP32 정밀도 선택으로 메모리 최적화
  - TF32 연산 활성화로 A100/RTX40 GPU 성능 최적화

### 3. **sentence-transformers** ([dgr_version/get_title_api.py:66](dgr_version/get_title_api.py#L66), [dgr_version/get_title_api.py:80](dgr_version/get_title_api.py#L80))
#### 3-1. SentenceTransformer (임베딩 모델)
- **모델**: `BAAI/bge-m3` (다국어 임베딩 모델)
- **역할**: 텍스트를 고차원 벡터로 변환하여 의미적 유사도 계산
- **원리**:
  - 후보 제목과 표 문맥을 각각 벡터화
  - 코사인 유사도로 관련성 측정 ([dgr_version/get_title_api.py:194](dgr_version/get_title_api.py#L194))
  - 후보가 많을 때 프리랭킹에 사용 ([dgr_version/get_title_api.py:840](dgr_version/get_title_api.py#L840))

#### 3-2. CrossEncoder (리랭커)
- **모델**: `BAAI/bge-reranker-v2-m3` (다국어 SOTA 리랭커)
- **역할**: 후보 제목 집합 내에서 상대적 순위 결정
- **원리**:
  - 후보 제목과 표 문맥의 쌍을 입력으로 받아 로짓(logits) 반환
  - 소프트맥스로 정규화하여 상대 확률 계산 ([dgr_version/get_title_api.py:612](dgr_version/get_title_api.py#L612))
  - 온도 스케일링(τ=0.6)으로 후보 간 대비 강화

### 4. **transformers** ([dgr_version/get_title_api.py:100](dgr_version/get_title_api.py#L100))
#### Zero-shot Classification
- **모델**: `joeddav/xlm-roberta-large-xnli` (다국어 NLI 모델)
- **역할**: 후보 텍스트가 '표 제목'인지 절대적 확률 판정
- **원리**:
  - 6가지 라벨로 분류 ([dgr_version/get_title_api.py:650](dgr_version/get_title_api.py#L650)):
    - 표의 제목/표제/캡션 (가중치: +0.85)
    - 표 위쪽 소제목 (가중치: +0.25)
    - 상위 섹션 헤더 (가중치: -0.25)
    - 본문 설명문 (가중치: -0.60)
    - 단위 표기 (가중치: -0.65)
    - 그림/도 제목 (가중치: -0.25)
  - 3가지 한글 템플릿으로 앙상블 ([dgr_version/get_title_api.py:660](dgr_version/get_title_api.py#L660))
  - 표 상단 근접도(≤120px)에 따라 가중치 동적 조정

### 5. **sentencepiece** ([dgr_version/get_title_api.py:101](dgr_version/get_title_api.py#L101))
- **역할**: XLM-RoBERTa 토크나이저 (slow tokenizer)
- **원리**: 서브워드 기반 토큰화로 다국어 텍스트 처리

### 6. **numpy** ([dgr_version/get_title_api.py:8](dgr_version/get_title_api.py#L8))
- **역할**: 수치 연산 및 벡터 연산
- **원리**:
  - 코사인 유사도 계산 ([dgr_version/get_title_api.py:194](dgr_version/get_title_api.py#L194))
  - 소프트맥스 정규화 ([dgr_version/get_title_api.py:612](dgr_version/get_title_api.py#L612))
  - 배치 점수 계산

### 7. **re (정규표현식)** ([dgr_version/get_title_api.py:7](dgr_version/get_title_api.py#L7))
- **역할**: 패턴 매칭 기반 규칙 필터링
- **원리**:
  - 표 제목 패턴 감지: `표 A.8`, `표 3-2` 등 ([dgr_version/get_title_api.py:245](dgr_version/get_title_api.py#L245))
  - 소제목 패턴: `①`, `(1)`, `•` 등 ([dgr_version/get_title_api.py:210](dgr_version/get_title_api.py#L210))
  - 노이즈 제거: 페이지 번호, 저작권, 단위 표기 ([dgr_version/get_title_api.py:143](dgr_version/get_title_api.py#L143))
  - 교차 참조 감지: `표 A.20에 의하면` ([dgr_version/get_title_api.py:255](dgr_version/get_title_api.py#L255))
﻿
---

## 제목 추출 원리 및 파이프라인

### 전체 프로세스

```
[입력] tables + texts
    ↓
┌─────────────────────────────────────────────────────┐
│ 1. 후보 수집 (규칙 기반 필터링)                       │
│    - 표 위쪽 탐색 영역 설정 (표 높이의 1.5배)         │
│    - 수평 근접도 체크 (IoU + 800px 허용 범위)         │
│    - 무의미 텍스트 제거 (페이지 번호, 저작권 등)        │
│    - 같은 줄 텍스트 병합 (y좌표 ±100px)              │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 2. 표 문맥 구축                                      │
│    - 표의 헤더와 첫 행 추출                          │
│    - 예: "헤더: [날짜, 온도, 습도] / 첫행: [..."     │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 3. 프리랭킹 (후보 > 8개일 때)                         │
│    - 임베딩 유사도(0.8) + 레이아웃 점수(0.2)          │
│    - 상위 8개 후보만 선택                            │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 4. 하드 필터링                                       │
│    - 유닛 라인 제거 (표 제목 패턴이 있을 때)          │
│    - 교차 참조 문장 제거                             │
│    - 긴 설명문 제거 (40자 이상 + 종결어미)            │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 5. 소제목 우선 모드 (표 상단 220px 이내)              │
│    - 소제목 패턴 후보가 있으면 소제목만 사용          │
│    - 최근접 소제목을 앞으로 정렬                      │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 6. ML 기반 점수 계산 (하이브리드)                     │
│                                                     │
│  [6-1] Zero-shot 분류 (가중치: 0.50)                │
│    - 6개 라벨 multi-label 분류                      │
│    - 3개 한글 템플릿 앙상블                          │
│    - 표 근접도 기반 가중치 동적 조정                  │
│    - 표 제목 패턴 하한 보정 (min 0.80)              │
│                                                     │
│  [6-2] 리랭커 점수 (가중치: 0.42)                    │
│    - 후보-표 문맥 쌍으로 로짓 계산                   │
│    - 온도 소프트맥스(τ=0.6)로 상대 확률 산출         │
│                                                     │
│  [6-3] Prior 점수 (가중치: 0.06)                    │
│    - 표 제목 패턴: +0.45                            │
│    - 소제목 패턴: +0.35                             │
│    - 섹션 헤더: -0.35                               │
│    - 단위 표기: -0.65                               │
│    - 교차 참조: -0.60                               │
│    - 설명문: -0.55                                 │
│    - 근접도 보너스 (≤120px: +0.08~0.30)            │
│                                                     │
│  [6-4] 임베딩 유사도 (가중치: 0.04)                  │
│    - 후보 ↔ 표 문맥 코사인 유사도                   │
│                                                     │
│  [6-5] 레이아웃 점수 (가중치: 0.04)                  │
│    - 표 상단과의 거리 기반 (0~3000px 선형)          │
│                                                     │
│  [6-6] 보너스                                       │
│    - 표 제목 패턴: +0.03                            │
│    - 단위/주석: -0.08                               │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 7. 타이브레이커 (점수차 ≤0.03일 때)                   │
│    - 우선순위: 표 제목 패턴 > 소제목 > 기타          │
│    - 동률 시: 더 가까운 후보 선택                     │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 8. 최종 검증                                         │
│    - 표 제목 패턴 없으면 prior < 0.4인 경우 기각     │
│    - 최고 점수가 임계값(0.01) 미만이면 기각          │
└─────────────────────────────────────────────────────┘
    ↓
[출력] 각 표에 title + title_bbox 추가
```

---

## 핵심 동작 순서

### Step 1: 후보 수집 ([dgr_version/get_title_api.py:510](dgr_version/get_title_api.py#L510))
1. 표 위쪽 탐색 영역 계산: `y_min = table_top - (table_height × 1.5)`
2. 텍스트를 줄 단위로 병합 (y좌표 ±100px 이내는 같은 줄)
3. 수평 근접도 체크: IoU > 0 또는 거리 ≤ 800px
4. 무의미 텍스트 필터링 (`is_trivial()`)

### Step 2: 표 문맥 구축 ([dgr_version/get_title_api.py:725](dgr_version/get_title_api.py#L725))
- 표의 첫 행(헤더)과 두 번째 행(첫 데이터)을 추출하여 문맥 문자열 생성

### Step 3: 하드 필터링 ([dgr_version/get_title_api.py:857](dgr_version/get_title_api.py#L857))
- 교차 참조 제거 (`is_cross_reference()`)
- 긴 설명문 제거 (≥40자 + 종결어미)

### Step 4: 소제목 우선 모드 ([dgr_version/get_title_api.py:877](dgr_version/get_title_api.py#L877))
- 표 상단 220px 이내에 소제목 패턴이 있으면 소제목만 사용

### Step 5: ML 기반 스코어링 ([dgr_version/get_title_api.py:749](dgr_version/get_title_api.py#L749))
- **Zero-shot**: 각 후보가 '표 제목'일 절대 확률 (0~1)
- **리랭커**: 후보 집합 내 상대 순위 (소프트맥스 정규화)
- **Prior**: 패턴 기반 사전 확률
- **임베딩/레이아웃**: 타이브레이커용 보조 점수

### Step 6: 타이브레이커 ([dgr_version/get_title_api.py:914](dgr_version/get_title_api.py#L914))
- 1, 2위 점수차가 0.03 이하일 때:
  1. 표 제목 패턴 > 소제목 > 기타 우선
  2. 동률이면 더 가까운 후보 선택

### Step 7: 최종 검증 ([dgr_version/get_title_api.py:929](dgr_version/get_title_api.py#L929))
- 제목 패턴이 없고 prior < 0.4이면 기각
- 최고 점수 < 0.01이면 기각

---

## 하이브리드 접근의 장점

### 1. **규칙 기반 필터링** (사전 노이즈 제거)
- 페이지 번호, 저작권, 단위 표기 등 명확한 노이즈 제거
- 패턴 매칭으로 표 제목 후보 우선 순위 부여

### 2. **Zero-shot 분류** (절대적 판정)
- 후보가 '표 제목'인지 아닌지 절대적 확률 제공
- 다중 라벨로 세밀한 분류 (제목/소제목/섹션/설명/단위)

### 3. **리랭커** (상대적 순위)
- 후보 집합 내에서 가장 적합한 제목 선택
- 표 문맥과의 관련성 기반 순위 결정

### 4. **임베딩 유사도** (의미적 관련성)
- 후보가 많을 때 프리랭킹으로 성능 향상
- 타이브레이커로 동점 해소

### 5. **레이아웃 점수** (공간적 근접성)
- 표와 가까울수록 제목일 확률 높음
- 타이브레이커로 최종 결정 보조

---

## 주요 상수 및 가중치

### 거리 임계값
- `Y_LINE_TOLERANCE = 100`: 같은 줄로 간주할 y 좌표 허용 오차
- `UP_MULTIPLIER = 1.5`: 표 위쪽 탐색 범위 (표 높이의 1.5배)
- `X_TOLERANCE = 800`: 수평 근접 허용 거리

### 최종 점수 가중치 ([dgr_version/get_title_api.py:32](dgr_version/get_title_api.py#L32))
- `WEIGHT_ZEROSHOT = 0.50`: Zero-shot 분류 점수
- `WEIGHT_RERANKER = 0.42`: 리랭커 점수
- `WEIGHT_PRIOR = 0.06`: Prior 점수 (패턴 기반)
- `WEIGHT_EMBEDDING = 0.04`: 임베딩 유사도
- `WEIGHT_LAYOUT = 0.04`: 레이아웃 점수

### 모델 설정
- `EMBEDDING_MODEL = "BAAI/bge-m3"`: 문장 임베딩
- `RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"`: 리랭커
- `ZEROSHOT_MODEL = "joeddav/xlm-roberta-large-xnli"`: Zero-shot 분류

---

## API 사용법

### 요청
```json
POST /get_title
{
  "tables": [
    {
      "bbox": [100, 200, 500, 400],
      "rows": [...]
    }
  ],
  "texts": [
    {
      "bbox": [150, 180, 450, 195],
      "text": "표 A.8 월별 기온 현황"
    }
  ]
}
```

### 응답
```json
[
  {
    "bbox": [100, 200, 500, 400],
    "rows": [...],
    "title": "표 A.8 월별 기온 현황",
    "title_bbox": [150, 180, 450, 195]
  }
]
```

---

## 성능 최적화

### GPU 최적화 ([dgr_version/get_title_api.py:55](dgr_version/get_title_api.py#L55))
- TF32 연산 활성화 (A100/RTX40 GPU)
- FP16 모델 로딩 (GPU 메모리 절감)
- CUDA 워밍업으로 콜드스타트 단축

### 배치 처리
- 리랭커 배치 추론 ([dgr_version/get_title_api.py:596](dgr_version/get_title_api.py#L596))
- Zero-shot 배치 분류 ([dgr_version/get_title_api.py:636](dgr_version/get_title_api.py#L636))

### 프리랭킹 ([dgr_version/get_title_api.py:836](dgr_version/get_title_api.py#L836))
- 후보 > 8개일 때 임베딩 유사도로 상위 8개 선택
- ML 추론 비용 절감

---

## Docker 실행 방법

### 사전 요구사항
- Docker 설치
- Docker Compose 설치
- NVIDIA Docker Runtime 설치 (GPU 사용 시)
- NVIDIA GPU 드라이버 설치 (GPU 사용 시)

### 1. Docker Compose로 실행 (권장)

```bash
# dgr_version 디렉토리로 이동
cd dgr_version

# Docker 이미지 빌드 및 컨테이너 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 컨테이너 중지
docker-compose down
```

### 2. Docker 명령어로 직접 실행

#### 2-1. 이미지 빌드
```bash
cd dgr_version

# GPU 지원 이미지 빌드
docker build -t title-extractor:latest .
```

#### 2-2. 컨테이너 실행 (GPU 사용)
```bash
docker run -d \
  --name title-api \
  --gpus all \
  -p 5555:5555 \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --restart always \
  title-extractor:latest
```

#### 2-3. 컨테이너 실행 (CPU만 사용)
```bash
docker run -d \
  --name title-api \
  -p 5555:5555 \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e CUDA_VISIBLE_DEVICES=-1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --restart always \
  title-extractor:latest
```

### 3. 로컬 실행 (Docker 없이)

#### 3-1. 의존성 설치
```bash
cd dgr_version

# Python 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# GPU 사용 시 (CUDA 12.1)
pip install torch==2.4.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3-2. API 서버 실행
```bash
# GPU 사용
python get_title_api.py

# CPU 사용
ML_DEVICE=-1 python get_title_api.py
```

### 4. API 테스트

```bash
# 서버 헬스체크
curl http://localhost:5555/

# 제목 추출 테스트
curl -X POST http://localhost:5555/get_title \
  -H "Content-Type: application/json" \
  -d '{
    "tables": [
      {
        "bbox": [100, 200, 500, 400],
        "rows": [[{"texts": [{"v": "날짜"}]}]]
      }
    ],
    "texts": [
      {
        "bbox": [150, 180, 450, 195],
        "text": "표 A.8 월별 기온 현황"
      }
    ]
  }'
```

---

## Docker 설정 파일 설명

### Dockerfile ([dgr_version/Dockerfile](dgr_version/Dockerfile))
- **베이스 이미지**: `nvidia/cuda:12.1.1-runtime-ubuntu22.04` (GPU 지원)
- **Python 버전**: Python 3 (Ubuntu 22.04 기본)
- **PyTorch**: 2.4.0+cu121 (CUDA 12.1)
- **포트**: 5555
- **환경변수**:
  - `HF_HUB_ENABLE_HF_TRANSFER=1`: Hugging Face 모델 다운로드 가속
  - `TOKENIZERS_PARALLELISM=false`: 토크나이저 병렬화 경고 비활성화
  - `CUDA_VISIBLE_DEVICES=0`: GPU 0번 사용

### docker-compose.yml ([dgr_version/docker-compose.yml](dgr_version/docker-compose.yml))
- **서비스명**: `api`
- **포트 매핑**: 5555:5555
- **재시작 정책**: `always`
- **볼륨 마운트**: `~/.cache/huggingface` (모델 캐시 공유)
- **GPU 설정**: 모든 GPU 사용 (`count: all`)

### requirements.txt ([dgr_version/requirements.txt](dgr_version/requirements.txt))
- `flask>=3.0.0`: REST API 서버
- `transformers>=4.35.0`: Hugging Face Transformers
- `torch>=2.0.0`: PyTorch
- `sentencepiece>=0.1.99`: 토크나이저
- `sentence-transformers>=2.2.0`: 임베딩 및 리랭커
- `numpy>=1.24.0`: 수치 연산

---

## 환경 변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `ML_DEVICE` | `0` | ML 모델 디바이스 (-1: CPU, 0: GPU 0번) |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Hugging Face 모델 다운로드 가속 |
| `TOKENIZERS_PARALLELISM` | `false` | 토크나이저 병렬화 (false 권장) |
| `CUDA_VISIBLE_DEVICES` | `0` | 사용할 GPU 번호 (-1: CPU) |

---

## 컨테이너 관리 명령어

```bash
# 로그 확인
docker logs -f title-api

# 컨테이너 상태 확인
docker ps -a | grep title-api

# 컨테이너 재시작
docker restart title-api

# 컨테이너 중지
docker stop title-api

# 컨테이너 삭제
docker rm title-api

# 이미지 삭제
docker rmi title-extractor:latest

# 실행 중인 컨테이너 내부 접속
docker exec -it title-api bash
```

---

## 모델 캐시 관리

첫 실행 시 Hugging Face Hub에서 모델을 다운로드합니다 (약 3-5GB):
- `BAAI/bge-m3` (~2GB)
- `BAAI/bge-reranker-v2-m3` (~1.5GB)
- `joeddav/xlm-roberta-large-xnli` (~1.5GB)

**캐시 경로**:
- 로컬: `~/.cache/huggingface`
- 컨테이너: `/root/.cache/huggingface`
- Docker 볼륨 마운트로 캐시 공유 가능

**수동 다운로드** (선택사항):
```bash
# 컨테이너 내부에서
python3 -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

SentenceTransformer('BAAI/bge-m3')
CrossEncoder('BAAI/bge-reranker-v2-m3')
pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli')
"
```

---

## 성능 및 리소스

### GPU 사용 시 (권장)
- **GPU 메모리**: 최소 6GB (RTX 3060 이상 권장)
- **추론 속도**: 표 1개당 약 0.5-1초
- **배치 처리**: 후보 8개 동시 처리

### CPU 사용 시
- **RAM**: 최소 8GB (16GB 권장)
- **추론 속도**: 표 1개당 약 3-5초
- **FP32 정밀도**: GPU 대비 느림

---

## 트러블슈팅

### GPU 인식 안 됨
```bash
# NVIDIA Docker Runtime 확인
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# CUDA 버전 확인
nvidia-smi
```

### 포트 충돌
```bash
# 포트 변경 (docker-compose.yml)
ports:
  - "5556:5555"  # 호스트 포트를 5556으로 변경
```

### 모델 다운로드 실패
```bash
# Hugging Face 토큰 설정 (private 모델인 경우)
docker run -e HUGGING_FACE_HUB_TOKEN=your_token ...
```

### 메모리 부족
```bash
# Docker 메모리 제한 증가 (docker-compose.yml)
deploy:
  resources:
    limits:
      memory: 16G
```
