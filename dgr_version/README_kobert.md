# KoBERT 표 텍스트 분류기

표의 제목/설명/제목아님을 분류하는 KoBERT 기반 분류기

## 📋 파일 구성

- `kobert_classifier.py` - KoBERT 분류 모델 코드
- `train_kobert.py` - 학습 실행 스크립트
- `train_data_example.json` - 라벨링 데이터 예시

## 🚀 사용 방법

### 1. 패키지 설치

```bash
pip install torch transformers sentencepiece
```

### 2. 학습 데이터 준비

두 가지 방법 중 선택:

#### 방법 A: JSON 파일 사용

`train_data_example.json` 파일에 데이터 추가:

```json
{
  "train_data": [
    {"text": "표 3.2 연간 실적", "label": 0, "note": "제목"},
    {"text": "상세 내용은 다음과 같다", "label": 1, "note": "설명"},
    {"text": "단위: ℃", "label": 2, "note": "제목아님"}
  ]
}
```

**라벨 정의:**
- `0` = 제목 (표 번호, 섹션 제목 등)
- `1` = 설명 (표 참조 문장, 주석 등)
- `2` = 제목아님 (페이지 번호, 단위, 저작권 등)

#### 방법 B: Python 코드로 직접 작성

`train_kobert.py`의 `create_sample_data()` 함수 수정:

```python
train_data = [
    ("표 3.2 연간 실적", 0),  # 제목
    ("상세 내용은 다음과 같다", 1),  # 설명
    ("단위: ℃", 2),  # 제목아님
]
```

### 3. 학습 실행

```bash
python train_kobert.py
```

**학습 파라미터 (train_kobert.py에서 수정 가능):**
- `epochs` - 에폭 수 (기본: 5)
- `batch_size` - 배치 크기 (기본: 8, GPU 메모리에 따라 조절)
- `lr` - 학습률 (기본: 2e-5)

### 4. 추론 (학습된 모델 사용)

```python
from kobert_classifier import TableTextClassifier

# 모델 로드
classifier = TableTextClassifier(model_path='kobert_table_classifier.pt')

# 단일 예측
label = classifier.predict("표 4.21 예산 현황")
print(label)  # "제목"

# 확률값 포함 예측
label, prob, all_probs = classifier.predict("표 4.21 예산 현황", return_probs=True)
print(f"{label} (확신도: {prob:.3f})")
# 제목 (확신도: 0.952)

# 배치 예측
texts = ["표 3.2 현황", "다음 표와 같다", "단위: %"]
results = classifier.predict_batch(texts)
for text, label, prob in results:
    print(f"{text} → {label} ({prob:.3f})")
```

## 📊 라벨링 가이드

### 0: 제목
표의 제목으로 사용되는 텍스트

**예시:**
- `표 3.2 연간 실적`
- `표 B.8 월별 기온`
- `□ 추진조직 구성`
- `■ 사업개요`
- `ㅇ 주요 추진 내용`
- `① 기본 방향`
- `(1) 사업 추진 방향`
- `3.2 토지이용현황`

### 1: 설명
표를 설명하거나 참조하는 문장

**예시:**
- `상세한 내용은 다음 표 B.12와 같다`
- `표 A.20에 의하면 다음과 같은 결과를 얻었다`
- `※ 본 자료는 참고용입니다`
- `아래 표에서 보는 바와 같이 전년 대비 증가하였다`

### 2: 제목아님
페이지 번호, 단위, 저작권 등 노이즈

**예시:**
- `123` (페이지 번호)
- `단위: ℃`
- `© 2024 All Rights Reserved`
- `-` (특수문자)

## 💡 권장 학습 데이터 크기

- **최소**: 각 클래스당 20개 (총 60개)
- **권장**: 각 클래스당 100개 (총 300개)
- **이상적**: 각 클래스당 500개 이상 (총 1500개 이상)

## ⚙️ 학습 팁

1. **데이터 불균형 해결**: 각 클래스의 개수를 비슷하게 맞추기
2. **에폭 조절**: 데이터가 적으면 에폭 증가 (5-10)
3. **학습률 조절**: 수렴이 안 되면 학습률 낮추기 (1e-5)
4. **배치 크기**: GPU 메모리 부족 시 줄이기 (4, 2)

## 🔧 트러블슈팅

### CUDA out of memory
```python
# train_kobert.py에서 batch_size 줄이기
batch_size = 4  # 또는 2
```

### 정확도가 낮음
- 학습 데이터 추가 (각 클래스당 100개 이상)
- 에폭 증가 (10-20)
- 데이터 품질 확인 (라벨링 오류 체크)
