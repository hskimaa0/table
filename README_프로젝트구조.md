# 테이블 연결성 분석 프로젝트 구조

PDF 문서에서 여러 페이지에 걸쳐 이어지는 테이블들을 자동으로 찾아서 병합하고 시각화하는 도구입니다.

## 🏗️ 프로젝트 구조

이 프로젝트는 두 가지 버전으로 구성되어 있습니다:

### 1. docling 버전 (GPU 가속 지원)
- **위치**: `docling_version/`
- **파싱 엔진**: docling (DocLayNet 형식)
- **특징**: GPU 가속으로 3-6배 빠른 처리
- **출력**: `merged_tables_output/`, `visualized_pdfs/`

### 2. opendataloader 버전 (Java 기반)
- **위치**: `opendataloader_version/`
- **파싱 엔진**: opendataloader-pdf
- **특징**: Java 기반, 안정적인 테이블 추출
- **출력**: `merged_output_v2/`, `visualized_pdfs_v2/`, `overlayed_pdfs_v2/`

## 📁 전체 디렉토리 구조

```
d:\표_연속성체크_04\
│
├── 📁 docling_version/              # docling 기반 버전
│   ├── merge_connected_tables.py   # 병합 스크립트
│   ├── visualize_connected_tables.py # 시각화 스크립트
│   ├── extract_tables_from_json.py # 테이블 추출
│   ├── check_gpu.py                # GPU 확인
│   └── README.md                   # 사용 설명서
│
├── 📁 opendataloader_version/       # opendataloader 기반 버전
│   ├── merge_tables_v2.py          # 병합 스크립트 (전체 기능)
│   ├── visualize_v2.py             # 시각화 스크립트
│   ├── overlay_connected_tables.py # 원본 PDF 오버레이
│   ├── run_all_v2.py               # 전체 파이프라인
│   └── README.md                   # 사용 설명서
│
├── 📁 input/                        # 입력 PDF 파일 (공통)
│   ├── 삼성물산.pdf
│   ├── 재난원인조사.pdf
│   └── ...
│
├── 📁 output/                       # docling JSON 출력
│
├── 📁 table_output/                 # docling 테이블 추출 결과
│
├── 📁 merged_tables_output/         # docling 병합 결과
│   ├── *_merged.json
│   ├── merge_summary.json
│   └── merged_tables_visualization.pdf
│
├── 📁 merged_output_v2/             # opendataloader 병합 결과
│   └── *_merged.json
│
├── 📁 visualized_pdfs/              # docling 시각화 결과
│   ├── *_connected_tables.pdf
│   └── 연결된_테이블_설명.pdf
│
├── 📁 visualized_pdfs_v2/           # opendataloader 시각화 결과
│   └── *_visualized.pdf
│
├── 📁 overlayed_pdfs_v2/            # opendataloader 오버레이 결과
│   └── *_overlayed.pdf              # 원본 PDF + 색상 오버레이
│
├── 📁 temp_opendataloader/          # opendataloader 임시 파일
│
└── README_프로젝트구조.md            # 이 파일
```

## 🚀 어느 버전을 사용해야 할까요?

### docling 버전을 선택하세요:
- ✅ GPU가 있고 빠른 처리를 원할 때
- ✅ DocLayNet 형식의 정확한 레이아웃 분석이 필요할 때
- ✅ 대량의 PDF를 처리해야 할 때
- ⚠️ Python 3.8+, torch, docling 설치 필요

### opendataloader 버전을 선택하세요:
- ✅ GPU가 없을 때
- ✅ Java 환경을 사용할 때
- ✅ 원본 PDF에 오버레이가 필요할 때
- ✅ 안정적인 테이블 추출이 필요할 때
- ⚠️ Python 3.9+, Java 11+, opendataloader-pdf 설치 필요

## 📋 공통 기능

두 버전 모두 다음 기능을 지원합니다:

1. **텍스트 연결성 분석**
   - 마지막 5개 셀 vs 첫 5개 셀 비교
   - 숫자/한글 순서, 단어 잘림, 불완전한 문장 감지

2. **헤더 비교**
   - 유사도 60% 이상
   - 공통 헤더 2개 이상
   - 30% 키값 중복 체크

3. **타이틀 기반 분리**
   - 완전 일치/포함 관계만 허용
   - 그 외는 엄격하게 분리

4. **멀티 로우 헤더 처리**
   - 각 열에서 가장 긴 텍스트를 헤더로 선택

5. **원본 PDF 시각화**
   - 연결된 테이블을 색상으로 표시
   - 그룹 라벨 추가

## 🔧 빠른 시작

### docling 버전:
```bash
cd docling_version
python merge_connected_tables.py
python visualize_connected_tables.py
```

### opendataloader 버전:
```bash
cd opendataloader_version
python run_all_v2.py
```

## 📊 출력 비교

| 출력 | docling | opendataloader |
|------|---------|----------------|
| 병합 JSON | merged_tables_output/ | merged_output_v2/ |
| 시각화 PDF | visualized_pdfs/ | visualized_pdfs_v2/ |
| 원본 오버레이 | visualized_pdfs/*_connected_tables.pdf | overlayed_pdfs_v2/*_overlayed.pdf |
| 상세 설명 | visualized_pdfs/연결된_테이블_설명.pdf | - |

## ⚙️ 설치

### docling 버전:
```bash
pip install docling torch reportlab PyPDF2
```

### opendataloader 버전:
```bash
pip install opendataloader-pdf reportlab PyPDF2
```

Java 11+ 필요:
```bash
java -version  # 확인
```

## 📖 더 자세한 정보

- docling 버전: `docling_version/README.md`
- opendataloader 버전: `opendataloader_version/README.md`

## 🔍 버전별 특화 기능

### docling 전용:
- GPU 가속 (3-6배 빠름)
- DocLayNet 형식 지원
- `table_output/`에서 테이블 추출

### opendataloader 전용:
- 원본 PDF 색상 오버레이 (투명도 15%)
- `G{그룹}-T{테이블}` 라벨 자동 추가
- run_all_v2.py로 전체 파이프라인 실행

## 📝 참고사항

- 두 버전은 `input/` 폴더를 공유합니다
- 출력 폴더는 각각 독립적입니다
- 병합 규칙은 동일합니다 (README.md 참조)
- 한글 폰트는 Windows `malgun.ttf` 사용
