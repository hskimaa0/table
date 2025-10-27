# 테이블 연결성 분석 및 시각화 도구

PDF 문서에서 여러 페이지에 걸쳐 이어지는 테이블들을 자동으로 찾아서 병합하고 시각화하는 도구입니다.

## 주요 기능

- ✅ **텍스트 연결성 분석**: 테이블 간 텍스트가 자연스럽게 이어지는지 감지
- ✅ **헤더 비교**: 동일한 헤더를 가진 테이블 자동 병합
- ✅ **헤더+데이터 구조 인식**: 헤더가 있는 테이블과 데이터만 있는 테이블 연결
- ✅ **원본 PDF 시각화**: 연결된 테이블을 색깔별로 표시
- ✅ **상세 보고서**: 각 연결의 근거가 되는 텍스트 비교 정보 제공

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 0. GPU 가속 상태 확인 (선택)

```bash
python check_gpu.py
```

현재 시스템의 GPU 사용 가능 여부와 성능을 확인합니다.

### 1. 테이블 데이터 준비

**옵션 A: 직접 준비**
`table_output/` 디렉토리에 분석할 JSON 파일들을 위치시킵니다.
- 파일 형식: `*_tables.json`
- 파일은 DocLayNet 형식의 테이블 데이터를 포함해야 합니다.

**옵션 B: output 폴더에서 추출**
```bash
python extract_tables_from_json.py
```
- `output/` 폴더의 JSON 파일에서 테이블 데이터만 추출하여 `table_output/` 폴더에 저장합니다.

### 2. 테이블 병합 실행

```bash
python merge_connected_tables.py
```

**출력 결과:**
- `merged_tables_output/*_merged.json`: 각 파일별 병합 결과
- `merged_tables_output/merge_summary.json`: 전체 요약

### 3. 시각화 생성

```bash
python visualize_connected_tables.py
```

**출력 결과:**
- `visualized_pdfs/*_connected_tables.pdf`: 원본 PDF에 연결 테이블 표시
- `visualized_pdfs/연결된_테이블_설명.pdf`: 상세 설명 및 텍스트 비교

## 연결 조건

### 1. 텍스트 연결 🔗
- **숫자/한글 순서**: `1)` → `2)`, `가.` → `나.`
- **단어 잘림**: "국가재난관리" → "국가재난관리시스템" (최소 5자 이상)
- **불완전한 문장**: "~의", "~를", "~및" 등으로 끝나는 경우

### 2. 헤더 동일 📋
- 80% 이상 일치
- 최소 2개 이상의 공통 헤더
- 두 테이블 모두 최소 2개 이상의 헤더 보유

### 3. 헤더 + 데이터 📊
- 첫 번째 테이블: 헤더 있음
- 두 번째 테이블: 헤더 없음 (데이터만)
- 열 개수 정확히 동일
- 연속 페이지 (1페이지 차이)

## 제약 조건

❌ **연결되지 않는 경우:**
- 같은 페이지에 있는 테이블들
- 순차적이지 않은 테이블 (건너뛰기 불가)
- 페이지가 2페이지 이상 떨어진 경우
- 구조가 다른 테이블 (열 개수 차이 2개 초과)

## 디렉토리 구조

```
표검출/
├── input/                          # 원본 PDF 파일
├── output/                         # DocLayNet JSON 파일 (원본)
├── table_output/                   # 테이블 JSON 데이터 (추출됨)
├── merged_tables_output/           # 병합 결과
│   ├── *_merged.json              # 각 파일별 병합 결과
│   └── merge_summary.json         # 전체 요약
├── visualized_pdfs/               # 시각화 결과
│   ├── *_connected_tables.pdf    # 오버레이된 PDF
│   └── 연결된_테이블_설명.pdf      # 상세 설명
├── extract_tables_from_json.py    # 테이블 추출 스크립트
├── merge_connected_tables.py      # 병합 스크립트
└── visualize_connected_tables.py  # 시각화 스크립트
```

## 출력 예시

### 병합 결과 (콘솔)
```
그룹 1:
  테이블 인덱스: [3, 4]
  페이지: [4, 5]
  - Table 3 -> 4: 헤더가 동일함 (4개 일치)

그룹 2:
  테이블 인덱스: [6, 7, 8]
  페이지: [6, 7, 8]
  - Table 6 -> 7: 헤더가 동일함 (2개 일치)
  - Table 7 -> 8: 헤더 테이블 뒤 데이터 테이블 (열 2개 동일)
```

### 시각화 PDF
- 같은 색상 = 연결된 테이블 그룹
- 각 박스에 "그룹 N" 레이블 표시
- 원본 내용 유지 (오버레이 방식)

### 상세 설명 PDF
```
그룹 1 (빨강색)
테이블 인덱스: [1, 2]
페이지: [1, 2]

연결 상세:
• Table 1 -> 2: 헤더가 동일함 (3개 일치)
  └ Table 1 끝: 항목1 | 항목2 | 항목3...
  └ Table 2 시작: 데이터1 | 데이터2 | 데이터3...
```

## 요구사항

- Python 3.8+
- reportlab: PDF 생성
- PyPDF2: PDF 처리
- docling: PDF 파싱 (GPU 가속 지원)
- torch: GPU 가속 (선택사항)

### GPU 가속 (선택사항, 권장)

Docling은 GPU를 사용하여 3-6배 빠른 처리가 가능합니다.

**GPU 사용 확인:**
```bash
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

**자세한 내용:** [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md) 참고

## 참고

- 이 도구는 DocLayNet 형식의 테이블 데이터를 사용합니다
- 한글 폰트는 Windows의 `malgun.ttf`를 사용합니다
- PDF 생성 시 A4 크기 기준으로 작성됩니다
