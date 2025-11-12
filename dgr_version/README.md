# 표 제목 추출 API (get_title_api.py)

## 개요

하이브리드 AI 기반 표 제목 자동 추출 Flask API입니다. 규칙 기반 필터링, Zero-shot NLI 분류, 크로스인코더 리랭커, 임베딩 유사도, 레이아웃 분석을 결합하여 PDF 문서에서 표의 제목을 정확하게 추출합니다.

### 핵심 특징

- **하이브리드 접근**: 규칙 기반 + 딥러닝 모델 결합으로 높은 정확도
- **다층 검증**: 6가지 스코어링 방식의 가중 평균
- **GPU 가속**: CUDA, FP16, TF32 지원으로 빠른 추론
- **배치 처리**: 여러 후보를 동시에 평가하여 효율성 향상
- **다국어 지원**: 한국어, 영어 등 다국어 처리 가능

---

## 시스템 아키텍처

### 전체 파이프라인

```
[입력] tables + texts (JSON)
    ↓
┌─────────────────────────────────────────────┐
│ 1. 후보 수집 (규칙 기반 필터링)              │
│    - 표 위쪽 영역 탐색 (표 높이×1.5)          │
│    - 수평 근접도 검증 (IoU + 800px 허용)      │
│    - 무의미 텍스트 제거                       │
│    - 같은 줄 텍스트 병합                      │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 2. 표 문맥 구축                              │
│    - 헤더: [열1, 열2, ...]                   │
│    - 첫행: [데이터1, 데이터2, ...]            │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 3. 하드 필터링                               │
│    - 유닛 라인 제거 (제목 패턴 있을 때만)     │
│    - 교차 참조 문장 제거                      │
│    - 긴 설명문 제거 (≥40자 + 종결어미)        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 4. ML 기반 하이브리드 스코어링               │
│                                              │
│  [4-1] Zero-shot 분류 (가중치: 40%)         │
│    - 6개 라벨 multi-label 분류               │
│    - 3개 한글 템플릿 앙상블                  │
│    - 표 근접도 기반 동적 가중치               │
│    - 제목 패턴 하한 보정 (min 0.80)          │
│                                              │
│  [4-2] 리랭커 점수 (가중치: 40%)             │
│    - CrossEncoder 로짓 계산                  │
│    - 온도 소프트맥스(τ=0.6) 정규화            │
│    - 후보 집합 내 상대 순위                   │
│                                              │
│  [4-3] 레이아웃 점수 (가중치: 15%)           │
│    - 표 상단 거리 기반 (0~3000px 선형)       │
│                                              │
│  [4-4] 보너스 (가중치: 5%)                   │
│    - 제목 패턴: +5%                          │
│    - 단위/주석: -5%                          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 5. 타이브레이커 (점수차 ≤0.03)               │
│    - 우선순위: 제목 패턴 > 소제목 > 기타     │
│    - 동률 시: 더 가까운 후보 선택             │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 6. 최종 검증                                 │
│    - 제목 패턴 없고 zeroshot < 0.5 → 기각    │
│    - 최고 점수 < 0.01 → 기각                 │
└─────────────────────────────────────────────┘
    ↓
[출력] tables with title + title_bbox
```

---

## 사용된 ML 모델

### 1. 리랭커: BAAI/bge-reranker-v2-m3
- **용도**: 후보 집합 내 상대 순위 결정
- **크기**: ~1.5GB
- **특징**:
  - CrossEncoder 아키텍처
  - 로짓 출력 모드 (softmax 미적용)
  - 온도 스케일링(τ=0.6)으로 대비 강화

### 2. Zero-shot 분류기: joeddav/xlm-roberta-large-xnli
- **용도**: 후보가 '표 제목'인지 절대적 확률 판정
- **크기**: ~1.5GB
- **특징**:
  - XLM-RoBERTa 기반 NLI 모델
  - Multi-label 분류 지원
  - 한국어 성능 우수
  - FP16 지원 (GPU 메모리 절감)

---

## 주요 상수 및 가중치

### 거리 임계값
```python
Y_LINE_TOLERANCE = 100  # 같은 줄로 간주할 y좌표 허용 오차
X_GAP_THRESHOLD = 100   # 텍스트 사이 공백 추가 기준 간격
UP_MULTIPLIER = 1.5     # 표 위쪽 탐색 범위 (표 높이의 배수)
X_TOLERANCE = 800       # 수평 근접 허용 거리
```

### ML 모델 설정
```python
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
ZEROSHOT_MODEL = "joeddav/xlm-roberta-large-xnli"
ML_DEVICE = 0                    # -1: CPU, 0: GPU
MAX_TEXT_INPUT_LENGTH = 512      # ML 모델 입력 최대 길이
```

### 최종 점수 가중치
```python
WEIGHT_ZEROSHOT = 0.40   # Zero-shot 분류 점수 (40%)
WEIGHT_RERANKER = 0.40   # 리랭커 점수 (40%)
WEIGHT_LAYOUT = 0.15     # 레이아웃 점수 (15%)
WEIGHT_BONUS = 0.05      # 보너스 점수 (5%)
SCORE_THRESHOLD = 0.01   # 제목 판정 최소 점수
```

---

## 규칙 기반 패턴 인식

### 1. 표 제목 패턴 (`is_table_title_like`)
**감지 패턴:**
- `표 B.8 월별 기온`
- `표 A.6 토지이용현황`
- `표 B .4` (공백 포함)
- `표 3-2 연간 실적`
- `1.2.3 제목` (섹션 번호)

**정규식:**
```python
r"^표\s*[A-Za-z]?\s*[\.\-]?\s*\d+([\-\.]\d+)?"
r"^\d+(\.\d+){0,2}\s+[^\[\(]{2,}"
```

### 2. 소제목 패턴 (`is_subtitle_like`)
**감지 패턴:**
- `① 항목명` (원문자)
- `(1) 항목명`, `1)`, `1.` (괄호/점 번호)
- `• 항목명`, `- 항목명` (불릿)

**조건:**
- 길이: 4~40자
- 패턴 시작 + 실제 내용 포함

### 3. 단위 표기 패턴 (`is_unit_like`)
**감지 패턴:**
- `[단위: ℃]`, `(단위 : %)`, `<단위: mm>`
- 짧은 텍스트(≤12자) + 단위 토큰만 포함

**단위 토큰:**
```python
["단위", "unit", "u.", "℃", "°c", "%", "mm", "kg", "km", "원", "개", "회"]
```

### 4. 교차 참조 패턴 (`is_cross_reference`)
**감지 패턴:**
- `표 A.20에 의하면`
- `표 B.4에서와 같이`
- `다음 표 B.12와 같다`
- `상세한 내용은 아래 표 참조`

**목적:** 표를 설명하는 본문 문장을 제목으로 오인하지 않도록 필터링

### 5. 섹션 헤더 패턴 (`is_section_header_like`)
**감지 패턴:**
- `1.`, `1.1`, `1.1.1` (계층 구조)
- `제3장`, `제2절` (문서 구조)

**특징:** 표 상단 근접 시(≤120px)만 약한 가점, 그 외는 감점

### 6. 설명문 패턴 (`is_sentence_like`)
**감지 패턴:**
- 한국어 종결 어미: `~이다`, `~였다`, `~나타난다` 등
- 긴 문장(≥30자) + 마침표
- 분사 절(≥18자) + `~이며`, `~면서` 등

---

## 세부 로직 설명

### Step 1: 후보 수집 (`collect_candidates_for_table`)

**탐색 영역 계산:**
```python
y_min = table_top - (table_height × 1.5)
```

**필터링 조건:**
1. **위치 검증**: 텍스트가 표 위쪽 탐색 영역 내에 있어야 함
2. **수평 근접도**: IoU > 0 또는 수평 거리 ≤ 800px
3. **무의미 텍스트 제거** (`is_trivial`):
   - 1~3자리 숫자만 (페이지 번호)
   - 저작권 표시 (`©`, `all rights reserved`)
   - 2자 이하 짧은 텍스트
   - 숫자/특수문자만 포함

**같은 줄 텍스트 병합:**
- y좌표 차이 ≤ 100px → 같은 줄로 간주
- x좌표 순서대로 공백으로 연결
- 중복 제거

### Step 2: 표 문맥 구축 (`build_table_context_rich`)

**추출 정보:**
```python
헤더: [열1, 열2, 열3, ...]
첫행: [데이터1, 데이터2, 데이터3, ...]
```

**용도:**
- 리랭커 입력 쌍 생성: `(후보 제목, 표 문맥)`
- 임베딩 유사도 계산 기준

### Step 3: 하드 필터링

**제거 대상:**
1. **교차 참조** (`is_cross_reference`): `표 A.20에 의하면` 등
2. **긴 설명문**: 길이 ≥ 40자 + 종결어미(`~다.`)
3. **유닛 라인** (`is_unit_like`): 표 제목 패턴이 있을 때만

### Step 4: ML 기반 스코어링 (`score_candidates_with_logits`)

#### 4-1. Zero-shot 분류

**6개 라벨:**
1. `표의 제목/표제/캡션` → 가중치: **+0.85**
2. `소제목(섹션 내 소단락 제목)` → 가중치: **+0.25**
3. `섹션 헤더(장/절/항 제목)` → 가중치: **-0.25** (표 상단 근접 시 +0.10)
4. `본문 설명문/서술문` → 가중치: **-0.60**
5. `단위 표기/비고/주:` → 가중치: **-0.65**
6. `그림/도 제목·캡션` → 가중치: **-0.25**

**3개 한글 템플릿 앙상블:**
```python
"이 텍스트는 {}이다."
"다음 문구의 용도는 {}이다."
"이 문장은 {}에 해당한다."
```

**동적 가중치 조정:**
- 표 상단 근접(≤120px) → 섹션 헤더 가중치 +0.10으로 상향

**하한 보정:**
- 표 제목 패턴 매칭 시 → Zero-shot 점수 최소 0.80 보장

#### 4-2. 리랭커

**입력 쌍 생성:**
```python
query = "[표제목 후보] 표 A.8 월별 기온 현황"
passage = "[표 문맥] 헤더: [날짜, 온도, 습도] / 첫행: [2024-01, 15.3, 65]"
```

**온도 소프트맥스:**
```python
tau = 0.6  # < 1 → 후보 간 대비 강화
prob[i] = exp(logit[i] / tau) / Σ exp(logit[j] / tau)
```

**효과:** 후보 집합 내 상대적 순위 명확화

#### 4-3. 레이아웃 점수

```python
distance = max(0, table_top - text_bottom)
layout_score = max(0, 1.0 - min(distance, 3000) / 3000)
```

**스케일:** 0~3000px 구간에서 선형 감소

**가중치:** 15% (표와의 물리적 거리 반영)

#### 4-4. 보너스

```python
bonus = 0.0
if is_table_title_like(후보):
    bonus += 1.0  # +5% (만점)
if is_unit_like(후보) or "주:|비고|참고" in 후보:
    bonus -= 1.0  # -5%
```

**용도:** 명확한 패턴 기반 미세 조정

**최종 점수:**
```python
final_score = (0.40 × zeroshot
             + 0.40 × reranker
             + 0.15 × layout
             + 0.05 × bonus)
```

### Step 5: 타이브레이커

**조건:** 1위와 2위 점수 차이 ≤ 0.03

**우선순위:**
1. **패턴 우선**: 표 제목 패턴 > 소제목 > 기타
2. **근접도**: 동률 시 표와 더 가까운 후보 선택

```python
if score_gap <= 0.03:
    if pattern_rank(2위) < pattern_rank(1위):
        선택 = 2위
    elif pattern_rank(2위) == pattern_rank(1위) and distance(2위) < distance(1위):
        선택 = 2위
```

### Step 6: 최종 검증

**기각 조건:**
1. 모든 후보에 표 제목 패턴 없음 **AND** Zero-shot 점수 < 0.5
2. 최고 점수 < 0.01

**목적:** 낮은 신뢰도 결과 제거

---

## GPU 최적화

### 디바이스 설정
```python
ML_DEVICE = 0  # -1: CPU, 0: GPU
device = "cuda:0" if (ML_DEVICE == 0 and torch.cuda.is_available()) else "cpu"
```

### A100/RTX40 계열 최적화
```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**효과:** TF32 연산으로 FP32 정확도 유지하면서 속도 향상

### FP16 모델 로딩
```python
dtype = torch.float16 if device.startswith("cuda") else torch.float32
model = AutoModelForSequenceClassification.from_pretrained(
    ZEROSHOT_MODEL,
    torch_dtype=dtype
)
```

**효과:** GPU 메모리 사용량 약 50% 절감

### CUDA 워밍업
```python
_ = embedder.encode(["warmup"], normalize_embeddings=True)
```

**효과:** 첫 요청 지연 시간 단축 (콜드스타트 방지)

---

## API 사용법

### 엔드포인트

**POST** `/get_title`

### 요청 형식

```json
{
  "tables": [
    {
      "bbox": [100, 200, 500, 400],
      "rows": [
        [
          {"texts": [{"v": "날짜"}]},
          {"texts": [{"v": "온도"}]}
        ],
        [
          {"texts": [{"v": "2024-01"}]},
          {"texts": [{"v": "15.3"}]}
        ]
      ]
    }
  ],
  "texts": [
    {
      "bbox": [150, 180, 450, 195],
      "text": "표 A.8 월별 기온 현황"
    },
    {
      "bbox": [150, 160, 300, 175],
      "text": "① 기온 데이터"
    }
  ]
}
```

### 응답 형식

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

### 헬스체크

**GET** `/`

---

## Docker 실행 방법

### 1. Docker Compose (권장)

```bash
cd dgr_version
docker-compose up -d
```

**설정 파일:** `docker-compose.yml`

### 2. Docker 직접 실행

#### 이미지 빌드
```bash
docker build -t title-extractor:latest .
```

#### 컨테이너 실행 (GPU)
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

#### 컨테이너 실행 (CPU)
```bash
docker run -d \
  --name title-api \
  -p 5555:5555 \
  -e CUDA_VISIBLE_DEVICES=-1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --restart always \
  title-extractor:latest
```

### 3. 로컬 실행 (Docker 없이)

```bash
cd dgr_version
pip install -r requirements.txt

# GPU 사용
python get_title_api.py

# CPU 사용
ML_DEVICE=-1 python get_title_api.py
```

### 4. API 테스트

```bash
# 헬스체크
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

## 파일 구조

```
dgr_version/
├── get_title_api.py          # 메인 API 서버 (980줄)
├── get_title_api_backup.py   # 백업 파일
├── Dockerfile                # Docker 이미지 설정
├── docker-compose.yml        # Docker Compose 설정
├── requirements.txt          # Python 패키지
└── README.md                 # 이 문서
```

---

## 환경 변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `ML_DEVICE` | `0` | ML 모델 디바이스 (-1: CPU, 0: GPU) |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Hugging Face 모델 다운로드 가속 |
| `TOKENIZERS_PARALLELISM` | `false` | 토크나이저 병렬화 경고 비활성화 |
| `CUDA_VISIBLE_DEVICES` | `0` | 사용할 GPU 번호 (-1: CPU) |

---

## 모델 캐시 관리

### 첫 실행 시 자동 다운로드

총 약 **5GB** 모델 다운로드:
- `BAAI/bge-m3`: ~2GB
- `BAAI/bge-reranker-v2-m3`: ~1.5GB
- `joeddav/xlm-roberta-large-xnli`: ~1.5GB

### 캐시 경로

- **로컬**: `~/.cache/huggingface`
- **Docker 컨테이너**: `/root/.cache/huggingface`
- **볼륨 마운트**: 캐시 공유 가능 (재다운로드 방지)

### 수동 다운로드 (선택)

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

SentenceTransformer('BAAI/bge-m3')
CrossEncoder('BAAI/bge-reranker-v2-m3')
pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli')
```

---

## 성능 벤치마크

### GPU 사용 시 (RTX 3060 기준)
- **추론 속도**: 표 1개당 약 **0.5~1초**
- **배치 처리**: 후보 8개 동시 처리
- **메모리**: 약 **4~6GB** VRAM

### CPU 사용 시 (i7-10700K 기준)
- **추론 속도**: 표 1개당 약 **3~5초**
- **메모리**: 약 **8~12GB** RAM
- **정밀도**: FP32 (GPU 대비 느림)

### 병렬 처리
- 단일 표 내 후보 8개 → 배치 추론
- 여러 표 → 순차 처리 (API 레벨에서 병렬화 가능)

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

```yaml
# docker-compose.yml 수정
ports:
  - "5556:5555"  # 호스트 포트를 5556으로 변경
```

### 모델 다운로드 실패

```bash
# Hugging Face 토큰 설정 (private 모델 시)
docker run -e HUGGING_FACE_HUB_TOKEN=your_token ...
```

### 메모리 부족

```yaml
# docker-compose.yml 메모리 제한 증가
deploy:
  resources:
    limits:
      memory: 16G
```

### Zero-shot 로딩 실패

```bash
# sentencepiece 라이브러리 설치 확인
pip install sentencepiece transformers
```

---

## 개선 포인트 및 향후 계획

### 현재 한계점
1. **단일 표 순차 처리**: 여러 표를 병렬로 처리하지 않음 → API 레벨 병렬화 필요
2. **고정 가중치**: 도메인별 최적 가중치가 다를 수 있음 → 자동 튜닝 필요
3. **하드코딩된 패턴**: 새로운 표 형식에 대응 어려움 → 학습 데이터 확장

### 개선 방안
1. **비동기 처리**: FastAPI + async/await로 API 성능 향상
2. **가중치 자동 조정**: 베이지안 최적화로 도메인별 튜닝
3. **파인튜닝**: 한국어 표 제목 데이터셋으로 Zero-shot 모델 파인튜닝
4. **앙상블 확장**: 추가 모델(BERT, T5) 결합

---

## 라이선스 및 인용

### 사용 모델 라이선스

- **BAAI/bge-m3**: MIT License
- **BAAI/bge-reranker-v2-m3**: MIT License
- **joeddav/xlm-roberta-large-xnli**: MIT License

### 참고 논문

1. **BGE**: BAAI General Embedding (2023)
2. **XLM-RoBERTa**: Cross-lingual Language Model (2019)
3. **XNLI**: Cross-lingual Natural Language Inference (2018)

---

## 문의 및 기여

- **이슈 리포트**: [GitHub Issues](https://github.com/your-repo/issues)
- **기여 가이드**: Pull Request 환영
- **문의**: your-email@example.com

---

## 버전 히스토리

### v1.0 (현재)
- 하이브리드 스코어링 시스템 구현
- Zero-shot + 리랭커 + 규칙 기반 결합
- GPU 가속 및 FP16 지원
- Docker 환경 구축
