"""
타이틀 추출 API
하이브리드 방식: 규칙 기반 필터링 + NLI + 임베딩 유사도 + 레이아웃 점수
"""
from flask import Flask, jsonify, request
import copy
import re
import numpy as np

app = Flask(__name__)

# ========== 상수 정의 ==========
# 거리 및 필터링 관련
Y_LINE_TOLERANCE = 100  # 같은 줄로 간주할 y 좌표 허용 오차 (px)
X_GAP_THRESHOLD = 100  # 텍스트 사이 공백 추가 기준 간격 (px)
UP_MULTIPLIER = 1.5  # 표 위쪽 탐색 범위 (표 높이의 배수)
X_TOLERANCE = 800  # 수평 근접 허용 거리 (px)

# ML 모델 관련
EMBEDDING_MODEL = "BAAI/bge-m3"  # 문장 임베딩 (다국어)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 크로스-인코더 리랭커 (다국어 SOTA)
ZEROSHOT_MODEL = "joeddav/xlm-roberta-large-xnli"  # Zero-shot 분류 (다국어, 한국어 우수)
ML_DEVICE = 0  # -1: CPU, 0: GPU
MAX_TEXT_INPUT_LENGTH = 512  # ML 모델 입력 최대 길이

# 리랭커 설정
USE_RERANKER = True  # 리랭커 사용 여부
USE_ZEROSHOT = True  # Zero-shot 분류 사용 여부
TOPK_CANDIDATES = 8  # 표당 리랭커에 보낼 최대 후보 수

# 최종 점수 가중치
WEIGHT_ZEROSHOT = 0.50   # Zero-shot 분류 점수 (제목 vs 비제목 판별)
WEIGHT_RERANKER = 0.42   # 리랭커 점수 (상대 순위)
WEIGHT_PRIOR = 0.06      # Prior 점수 (패턴 기반 규칙)
WEIGHT_EMBEDDING = 0.04  # 임베딩 유사도 (타이브레이커)
WEIGHT_LAYOUT = 0.04     # 레이아웃 점수 (타이브레이커)
SCORE_THRESHOLD = 0.01   # 제목 판정 최소 점수

# ML 모델 로드
embedder = None
reranker = None
zeroshot_classifier = None

# 디바이스 설정
import torch

def _resolve_device():
    """ML_DEVICE 설정과 CUDA 가용성에 따라 디바이스 결정"""
    use_gpu = (ML_DEVICE == 0 and torch.cuda.is_available())
    return "cuda:0" if use_gpu else "cpu"

DEVICE_STR = _resolve_device()
print(f"▶ Inference device = {DEVICE_STR}")

# GPU 성능 최적화 (A100/RTX40 계열)
if DEVICE_STR.startswith("cuda"):
    torch.set_float32_matmul_precision("high")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# 임베딩 모델 로드
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE_STR)
    # 워밍업 (CUDA 상주, 콜드스타트 단축)
    _ = embedder.encode(["warmup"], normalize_embeddings=True, convert_to_numpy=True)
    print(f"✅ 임베딩 모델 로드 완료 ({EMBEDDING_MODEL}, device={DEVICE_STR})")
except ImportError:
    print("⚠️  sentence-transformers 라이브러리 없음")
    embedder = None
except Exception as e:
    print(f"⚠️  임베딩 모델 로드 실패: {e}")
    embedder = None

# 리랭커 모델 로드
try:
    from sentence_transformers import CrossEncoder
    try:
        reranker = CrossEncoder(
            RERANKER_MODEL,
            device=DEVICE_STR,
            default_activation_function=None  # logits 모드 (softmax 미적용)
        )
    except TypeError:
        # 일부 구버전은 인자 없이도 logits 반환 가능
        reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE_STR)
    print(f"✅ 리랭커 로드 완료 ({RERANKER_MODEL}, device={DEVICE_STR})")
except ImportError:
    print("⚠️  sentence-transformers 라이브러리 없음 (CrossEncoder)")
    reranker = None
except Exception as e:
    print(f"⚠️  리랭커 로드 실패: {e}")
    reranker = None

# Zero-shot 분류기 로드 (FP16 지원)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import sentencepiece  # noqa: F401

    # SentencePiece 기반 slow tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ZEROSHOT_MODEL, use_fast=False)

    # FP16 로드 (GPU면 FP16, CPU면 FP32)
    dtype = torch.float16 if DEVICE_STR.startswith("cuda") else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        ZEROSHOT_MODEL,
        torch_dtype=dtype
    )

    device_id = 0 if DEVICE_STR.startswith("cuda") else -1
    zeroshot_classifier = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device=device_id
    )
    print(f"✅ Zero-shot 로드 완료 ({ZEROSHOT_MODEL}, device={DEVICE_STR}, dtype={dtype})")
    print(f"✅ Zero-shot 분류 활성화 (가중치: {WEIGHT_ZEROSHOT:.0%})")
except ImportError as e:
    print(f"⚠️  라이브러리 없음: {e}")
    print("   → pip install sentencepiece transformers 실행 필요")
    zeroshot_classifier = None
    USE_ZEROSHOT = False
except Exception as e:
    print(f"⚠️  Zero-shot 로드 실패: {e}")
    zeroshot_classifier = None
    USE_ZEROSHOT = False

# ========== 유틸리티 함수 ==========
def clean_text(s: str) -> str:
    """텍스트 정리"""
    return re.sub(r"\s+", " ", s).strip()

def clamp_text_len(s: str, max_chars=MAX_TEXT_INPUT_LENGTH):
    """텍스트 길이 제한"""
    s = clean_text(s)
    return s[:max_chars] if len(s) > max_chars else s


def is_trivial(text: str) -> bool:
    """무의미한 텍스트 필터링 (페이지 번호, 저작권 등)"""
    s = text.strip()

    # 1~3자리 숫자만
    if re.fullmatch(r"\d{1,3}", s):
        return True

    # 저작권 표시
    if "all rights reserved" in s.lower():
        return True
    if s.startswith("©"):
        return True

    # 너무 짧음
    if len(s) <= 2:
        return True

    # 숫자만 포함
    if re.match(r'^[\d\s\.\-]+$', s):
        return True

    # 특수문자만
    if len(re.sub(r"[\W_]+", "", s)) <= 1:
        return True

    return False

def iou_1d(a: tuple, b: tuple) -> float:
    """1차원 IoU (수평 겹침 계산)"""
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - overlap
    return overlap / union if union > 0 else 0.0

def horizontally_near(table_x: tuple, text_x: tuple, tol: int = X_TOLERANCE) -> bool:
    """수평으로 겹치거나 근접한지 확인"""
    if iou_1d(table_x, text_x) > 0:
        return True
    return (text_x[1] >= table_x[0] - tol and text_x[0] <= table_x[1] + tol)

def layout_score(table_bbox, text_bbox) -> float:
    """레이아웃 점수: 표와의 세로 거리 기반 (가까울수록 1)"""
    _, ty1, _, ty2 = table_bbox
    x1, y1, x2, y2 = text_bbox

    # 텍스트가 표 위에 있을 때의 거리
    dist = max(0, ty1 - y2)

    # 0~3000px 구간에 대해 선형 스케일링
    return max(0.0, 1.0 - min(dist, 3000) / 3000.0)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ========== 패턴 스코어/페널티 ==========
UNIT_TOKENS = ["단위", "unit", "u.", "℃", "°c", "%", "mm", "kg", "km", "원", "개", "회"]

# 소제목/섹션 패턴
CIRCLED_RX = r"[\u2460-\u2473\u3251-\u325F]"  # ①-⑳, 21-35
BULLET_RX = r"[•∙·\-–—]"  # 불릿/대시
PAREN_NUM_RX = r"\(?\d{1,2}\)?[.)]"  # (1) 1) 1.

def is_subtitle_like(s: str) -> bool:
    """테이블 바로 위에 자주 오는 '소제목' 패턴:
    - ① ② … / (1) 1) 1. / • - 등 불릿 시작
    - 길이 4~40자 권장
    """
    t = s.strip()
    if re.match(rf"^({CIRCLED_RX}|{PAREN_NUM_RX}|{BULLET_RX})\s*\S+", t):
        if 4 <= len(t) <= 40:
            return True
    return False

def is_section_header_like(s: str) -> bool:
    """상위 섹션 헤더(문서 구조용):
    - 1. 1.1 1.1.1 형태
    - 제3장, 제2절 등
    """
    t = s.strip()
    if re.match(r"^\d+(\.\d+){0,3}\s+\S+", t):
        return True
    if re.match(r"^제\s*\d+\s*(장|절|항)\b", t):
        return True
    return False

def is_unit_like(s: str) -> bool:
    """단위 표기 라인 판별"""
    t = s.strip().lower()
    # [단위: ℃], (단위 : %), <단위: mm> 등
    if re.match(r"^[\[\(\<]?\s*(단위|unit)\s*[:：]?\s*[\w%°℃㎜\-/]+[\]\)\>]?\s*$", t):
        return True
    # 길이 아주 짧고(<=12) 유닛 토큰만 있는 경우
    if len(t) <= 12 and any(tok in t for tok in UNIT_TOKENS):
        letters = re.sub(r"[^가-힣a-zA-Z]", "", t)
        return len(letters) <= 1
    return False

def is_table_title_like(s: str) -> bool:
    """표 제목 패턴 판별"""
    # '표 B.8 월별 기온', '표 A.6 토지이용현황', '표 B .4' (공백 포함), '표 3-2 연간 실적' 등
    if re.search(r"^표\s*[A-Za-z]?\s*[\.\-]?\s*\d+([\-\.]\d+)?", s.strip()):
        return True
    # 섹션/표 제목 형태(숫자.숫자 제목)
    if re.search(r"^\d+(\.\d+){0,2}\s+[^\[\(]{2,}", s.strip()):
        return True
    return False

def is_cross_reference(s: str) -> bool:
    """교차 참조/설명 문장 판별 (강화)"""
    t = s.strip().replace(" ", "")

    # '표 A.20에의하면', '표 B .4에서', '표 3.2에따르면' 등 (표 번호로 시작하는 참조 문장)
    if re.search(r"^표[A-Za-z]?[\.\-]?\d+([\-\.]\d+)?(에의하면|에따르면|에서|을보면|에나타난|에서와같이|과같이)", t):
        return True

    # '상세한 내용은 다음 표 B.12와 같다' 류
    if re.search(r"(다음표|하기표|본표|아래표)[A-Za-z]?\d+.*(같다|나타낸다|보인다|정리|참조)", t):
        return True

    # '상세한내용은다음표B.12와같다' 같이 붙어쓴 OCR도 커버
    if re.search(r"(상세한내용|자세한내용).*(다음표|아래표).*(같다|나타난다|참조)", t):
        return True

    # 명확한 설명 문장만 (동사 어미로 끝나는 긴 문장)
    if len(t) >= 35 and re.search(r"(바랍니다|바란다|협의대로|안내하|요청)", t):
        return True

    return False

# === 일반 '문장' 판별 (설명/서술형) ===
KOREAN_SENT_END_RX = r"(이다|였다|하였다|했다|된다|되었다|나타난다|나타났다|보인다|보였다|이다\.|다\.|다,)$"
CLAUSE_TOKENS = r"(이며|면서|면서도|고서|고 있으며|으로|로써|으로서|에 따라|에 의하면)"

def is_sentence_like(s: str) -> bool:
    """일반 문장(설명/서술형) 판별

    주의: 제목 패턴 체크 후 사용할 것 (표 A.3 ... 같은 제목을 문장으로 오판 방지)
    """
    t = s.strip()

    # 한국어 서술어/종결 어미 (가장 확실한 문장 패턴)
    if re.search(KOREAN_SENT_END_RX, t):
        return True

    # 분사/이어주는 절 표시 & 길이
    if len(t) >= 18 and re.search(CLAUSE_TOKENS, t):
        return True

    # 마침표로 끝나고 길면 문장 (쉼표는 제목에도 많아서 제외)
    if len(t) >= 30 and (t.endswith(".") or "。" in t or "．" in t):
        return True

    # 라인 내 공백 거의 없고 길이 긴 경우(붙어쓴 문장 OCR)
    if len(t) >= 40 and re.search(r"[가-힣]{15,}", t):
        return True

    return False

def prior_score(cand_text: str, cand_bbox, table_bbox) -> float:
    """룰 기반 사전확률(0~1). 유닛/주석/설명문 강한 감점, 표제/소제목 패턴 가점"""
    s = cand_text.strip()
    prior = 0.5  # 기본값

    # 표제 패턴 가점
    if is_table_title_like(s):
        prior += 0.45        # ★ +0.40 → +0.45

    # === 소제목/섹션 헤더 처리 ===
    if is_subtitle_like(s):
        prior += 0.35
    if is_section_header_like(s):
        prior -= 0.35

    # 유닛/주석 강한 감점
    if is_unit_like(s):
        prior -= 0.65        # ★ 강화

    # 교차 참조/설명 문장 강한 감점
    if is_cross_reference(s):
        prior -= 0.60        # ★ 강화

    # ★ NEW: 설명문 강감점
    if is_sentence_like(s):
        prior -= 0.55

    # 근접도 기반 보정(표 바로 위일수록 ↑)
    _, ty1, _, _ = table_bbox
    _, _, _, cy2 = cand_bbox
    dy = max(0, ty1 - cy2)

    # ★ 근접 보너스
    if dy <= 120:
        prior += 0.08 + (0.22 if is_subtitle_like(s) else 0.0)
    elif dy <= 300:
        prior += 0.04 + (0.10 if is_subtitle_like(s) else 0.0)
    elif dy >= 800:
        prior -= 0.10

    # '대괄호 한 줄' & 표와 초근접(유닛 가능성↑) 감점
    bracket_line = bool(re.match(r"^[\[\(＜〈].+[\]\)＞〉]$", s))
    if bracket_line and dy <= 80:
        prior -= 0.25

    # 길이 보정: 제목은 보통 5~60자
    L = len(s)
    if L < 5:
        prior -= 0.15
    if L > 60:
        prior -= 0.15

    # 거의 기호/숫자뿐
    if re.search(r"^[\d\W_]+$", s):
        prior -= 0.20

    return max(0.0, min(1.0, prior))

# ========== bbox 추출 ==========
def get_bbox_from_text(text):
    """text 객체에서 bbox 정보 추출 [l, t, r, b] 형식으로 반환"""
    if 'rect' in text and isinstance(text['rect'], list) and len(text['rect']) >= 4:
        return text['rect']
    elif 'bbox' in text and isinstance(text['bbox'], list) and len(text['bbox']) >= 4:
        return text['bbox']
    elif 'bbox' in text and isinstance(text['bbox'], dict):
        bbox = text['bbox']
        return [bbox.get('l', 0), bbox.get('t', 0), bbox.get('r', 0), bbox.get('b', 0)]
    elif 'merged_bbox' in text:
        return text['merged_bbox']
    return None

def get_bbox_from_table(table):
    """table 객체에서 bbox 정보 추출 [l, t, r, b] 형식으로 반환"""
    if 'bbox' in table and isinstance(table['bbox'], list) and len(table['bbox']) >= 4:
        return table['bbox']
    elif 'bbox' in table and isinstance(table['bbox'], dict):
        bbox = table['bbox']
        return [bbox.get('l', 0), bbox.get('t', 0), bbox.get('r', 0), bbox.get('b', 0)]
    return None

# ========== 텍스트 추출 및 병합 ==========
def extract_text_content(text_obj):
    """text 객체에서 실제 텍스트 추출"""
    if 'merged_text' in text_obj:
        return text_obj['merged_text']

    if 't' in text_obj and isinstance(text_obj['t'], list) and len(text_obj['t']) > 0:
        t_items = text_obj['t']
        sorted_items = sorted(t_items, key=lambda item: item.get('tid', 0))
        texts = []
        for t_item in sorted_items:
            if 'text' in t_item:
                texts.append(t_item['text'])
        if texts:
            return ' '.join(texts)  # 공백으로 연결

    if 'text' in text_obj:
        return text_obj['text']
    elif 'v' in text_obj:
        return text_obj['v']
    return ""

def flatten_text_objects(texts):
    """paraIndex로 그룹화된 텍스트를 개별 't' 요소로 분해"""
    flattened = []
    for text_obj in texts:
        if 't' in text_obj and isinstance(text_obj['t'], list):
            for t_item in text_obj['t']:
                if 'bbox' in t_item and 'text' in t_item:
                    flattened.append({
                        'bbox': t_item['bbox'],
                        'text': t_item['text'],
                        'tid': t_item.get('tid', 0)
                    })
        else:
            flattened.append(text_obj)
    return flattened

def group_texts_by_line(texts, y_tolerance=Y_LINE_TOLERANCE):
    """같은 줄의 텍스트들을 그룹화"""
    if not texts:
        return []

    flattened = flatten_text_objects(texts)
    sorted_texts = sorted(flattened, key=lambda t: get_bbox_from_text(t)[1] if get_bbox_from_text(t) else float('inf'))

    grouped = []
    current_group = []
    group_y_min = None

    for text in sorted_texts:
        bbox = get_bbox_from_text(text)
        if not bbox:
            continue

        y_center = (bbox[1] + bbox[3]) / 2

        if group_y_min is None or abs(y_center - group_y_min) <= y_tolerance:
            current_group.append(text)
            if group_y_min is None:
                group_y_min = y_center
        else:
            if current_group:
                merged = merge_text_group(current_group)
                if merged:
                    grouped.append(merged)
            current_group = [text]
            group_y_min = y_center

    if current_group:
        merged = merge_text_group(current_group)
        if merged:
            grouped.append(merged)

    grouped.sort(key=lambda g: g.get('merged_bbox', [0, 0, 0, 0])[1] if g else float('inf'))
    return grouped

def merge_text_group(text_group):
    """같은 줄의 텍스트들을 x 좌표 순서대로 병합"""
    if not text_group:
        return None

    sorted_group = sorted(text_group, key=lambda t: get_bbox_from_text(t)[0] if get_bbox_from_text(t) else 0)

    text_parts = []
    prev_bbox = None

    for t in sorted_group:
        text_content = extract_text_content(t)
        if not text_content:
            continue

        current_bbox = get_bbox_from_text(t)

        # 텍스트 조각 사이에 항상 공백 추가 (첫 조각 제외)
        if prev_bbox and current_bbox:
            text_parts.append(' ')

        text_parts.append(text_content)
        prev_bbox = current_bbox

    merged_text = ''.join(text_parts)

    bboxes = [get_bbox_from_text(t) for t in sorted_group]
    bboxes = [b for b in bboxes if b]

    if not bboxes:
        return None

    merged_bbox = [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes)
    ]

    merged_obj = text_group[0].copy() if text_group else {}
    merged_obj['merged_text'] = merged_text
    merged_obj['merged_bbox'] = merged_bbox

    return merged_obj

# ========== 후보 수집 ==========
def collect_candidates_for_table(table, texts, all_tables=None):
    """표 위쪽 + 아래쪽 텍스트 후보 수집 (규칙 기반 필터링)

    표 위: 모든 후보 수집
    표 아래: 가장 가까운 후보 1개만 수집
    """
    table_bbox = get_bbox_from_table(table)
    if not table_bbox:
        return []

    tbx1, tby1, tbx2, tby2 = table_bbox
    h = tby2 - tby1
    y_min = max(0, tby1 - int(UP_MULTIPLIER * h))
    y_max = tby2 + int(UP_MULTIPLIER * h)  # 표 아래쪽 범위

    # 그룹화된 텍스트
    grouped_texts = group_texts_by_line(texts, y_tolerance=Y_LINE_TOLERANCE)

    candidates_above = []  # 표 위 후보
    candidates_below = []  # 표 아래 후보

    for text in grouped_texts:
        if not text:
            continue

        text_bbox = text.get('merged_bbox') or get_bbox_from_text(text)
        if not text_bbox:
            continue

        px1, py1, px2, py2 = text_bbox

        # 수평으로 겹치거나 근접한지 확인
        if not horizontally_near((tbx1, tbx2), (px1, px2), tol=X_TOLERANCE):
            continue

        text_content = clean_text(text.get('merged_text') or extract_text_content(text))

        # 무의미한 텍스트 필터링
        if not text_content or is_trivial(text_content):
            continue

        cand = {
            'text': text_content,
            'bbox': text_bbox
        }

        # 표 위쪽에 있는지 확인
        if py2 <= tby1 and py1 >= y_min:
            candidates_above.append(cand)
        # 표 아래쪽에 있는지 확인
        elif py1 >= tby2 and py2 <= y_max:
            candidates_below.append(cand)

    # 표 아래 후보 중 가장 가까운 2개만 선택
    if candidates_below:
        candidates_below.sort(key=lambda c: c['bbox'][1] - tby2)  # 표 아래와의 거리 기준
        candidates_below = candidates_below[:2]  # 가장 가까운 2개만

    # 위 + 아래 후보 병합
    candidates = candidates_above + candidates_below

    # 중복 제거
    unique = {}
    for c in candidates:
        unique.setdefault(c['text'], c)

    return list(unique.values())

# ========== 표 문맥 구축 ==========
def build_table_context(table, max_cells=10):
    """표의 헤더와 첫 행으로 문맥 구축"""
    headers = []
    if 'rows' in table and table['rows']:
        for cell in table['rows'][0]:
            cell_texts = [t['v'] for t in cell.get('texts', []) if t.get('v')]
            if cell_texts:
                headers.append(clean_text(" ".join(cell_texts)))

    header_str = " | ".join(headers[:max_cells]) if headers else ""

    first_row = []
    if 'rows' in table and len(table['rows']) >= 2:
        for cell in table['rows'][1]:
            cell_texts = [t['v'] for t in cell.get('texts', []) if t.get('v')]
            if cell_texts:
                first_row.append(clean_text(" ".join(cell_texts)))

    first_row_str = " | ".join(first_row[:max_cells]) if first_row else ""

    parts = []
    if header_str:
        parts.append(f"헤더: {header_str}")
    if first_row_str:
        parts.append(f"첫행: {first_row_str}")

    return " / ".join(parts) if parts else "표 정보 없음"

# ========== ML 스코어링 ==========
def make_reranker_pair(cand_text: str, table_ctx: str):
    """리랭커 입력 쌍 생성 (명시적 역할 프롬프트)"""
    q = f"[표제목 후보] {clamp_text_len(cand_text)}"
    p = f"[표 문맥] {clamp_text_len(table_ctx)}"
    return (q, p)

def reranker_logits_batch(pairs):
    """배치 리랭커 로짓 반환
    pairs: [(cand_text, table_ctx), ...]
    반환: numpy array of logits (float)
    """
    import numpy as np
    if not reranker or not pairs:
        return np.zeros((len(pairs),), dtype=float)
    try:
        out = reranker.predict(pairs, convert_to_numpy=True)  # shape (N,1) or (N,)
        logits = np.asarray(out).reshape(-1)
        return logits
    except Exception as e:
        print(f"  리랭커 오류: {e}")
        return np.zeros((len(pairs),), dtype=float)

def softmax_with_temp_from_logits(logits, tau=0.6):
    """로짓에 온도를 적용한 소프트맥스 (후보 집합 내 대비 ↑)
    tau < 1 -> 대비 강화, tau ~0.5~0.7 권장
    """
    import numpy as np
    x = np.asarray(logits, dtype=float) / max(tau, 1e-6)
    x = x - x.max()  # overflow 방지
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)

def embedding_similarity(text_a: str, text_b: str) -> float:
    """임베딩 유사도 계산"""
    if not embedder:
        return 0.0

    try:
        a = clamp_text_len(text_a)
        b = clamp_text_len(text_b)
        vecs = embedder.encode([a, b], normalize_embeddings=True)
        return cosine_similarity(vecs[0], vecs[1])
    except Exception as e:
        print(f"  임베딩 오류: {e}")
        return 0.0

def zeroshot_title_score_batch(candidates):
    """Zero-shot 분류: 각 후보가 '표 제목'일 확률 반환

    Returns:
        numpy array of scores (0~1), 높을수록 제목일 확률 높음
    """
    import numpy as np
    if not zeroshot_classifier or not USE_ZEROSHOT or not candidates:
        return np.ones(len(candidates)) * 0.5  # 중립 점수

    try:
        texts = [clamp_text_len(c['text']) for c in candidates]

        # 세분화된 다중 라벨 (긍정 + 부정)
        labels = [
            "표의 제목/표제/캡션",
            "표 위쪽 소제목(섹션 내 소단락 제목)",
            "상위 섹션 헤더(장/절/항 제목)",
            "본문 설명문/서술문 (표에 대한 설명)",
            "단위 표기/비고/주석 라인",
            "그림/도/이미지 제목·캡션",
        ]

        # 3가지 다양한 한글 템플릿 (✅ 반드시 '{}' 만 사용)
        templates = [
            "이 텍스트는 {}이다.",
            "다음 문구의 용도는 {}이다.",
            "이 문장은 {}에 해당한다.",
        ]

        # 라벨별 가중치 (긍정: +, 부정: -)
        base_weights = {
            "표의 제목/표제/캡션": 0.85,
            "표 위쪽 소제목(섹션 내 소단락 제목)": 0.25,
            "상위 섹션 헤더(장/절/항 제목)": -0.25,  # near-top일 때만 +0.05으로 조정
            "본문 설명문/서술문 (표에 대한 설명)": -0.60,
            "단위 표기/비고/주석 라인": -0.65,
            "그림/도/이미지 제목·캡션": -0.25,
        }

        # 표 상단 근접 마스크 (≤120px)
        table_bbox = globals().get("_current_table_bbox")
        if table_bbox:
            top = table_bbox[1]
            near_top_mask = np.array([
                max(0, top - (c.get("bbox", [0, 0, 0, 0])[3])) <= 120 for c in candidates
            ], dtype=bool)
        else:
            near_top_mask = np.ones(len(candidates), dtype=bool)

        def score_once(tmpl: str) -> np.ndarray:
            """특정 템플릿으로 multi-label 분류 후 가중합 계산"""
            try:
                res = zeroshot_classifier(
                    texts, labels,
                    hypothesis_template=tmpl,
                    multi_label=True,
                    truncation=True  # 긴 텍스트 안전 처리
                )
            except Exception as e:
                print(f"  Zero-shot 템플릿 '{tmpl[:20]}...' 예외: {e}")
                return np.ones(len(candidates)) * 0.5

            sc = np.zeros(len(candidates), dtype=float)
            for i, r in enumerate(res):
                probs = {lab: float(p) for lab, p in zip(r["labels"], r["scores"])}
                # 동적 가중치 조정 (근접도 기반)
                w = base_weights.copy()
                if near_top_mask[i]:
                    # 표 상단 근접 시: 소제목 가중↑, 섹션헤더 약간 가산
                    w["표 위쪽 소제목(섹션 내 소단락 제목)"] = 0.45
                    w["상위 섹션 헤더(장/절/항 제목)"] = 0.05
                else:
                    # 표와 멀리 떨어진 경우: 소제목 가중↓, 섹션헤더 감점
                    w["표 위쪽 소제목(섹션 내 소단락 제목)"] = 0.15
                    w["상위 섹션 헤더(장/절/항 제목)"] = -0.30

                s = sum(w[k] * probs.get(k, 0.0) for k in w)
                sc[i] = np.clip(s + 0.35, 0.0, 1.0)  # 약간의 오프셋 후 [0,1] 클립
            return sc

        # 3개 템플릿 평균
        scores = np.mean([score_once(t) for t in templates], axis=0)
        return scores

    except Exception as e:
        print(f"  Zero-shot 오류: {e}")
        return np.ones(len(candidates)) * 0.5

def build_table_context_rich(table, max_cells=10):
    """표 문맥 구축 (헤더 레이블 명시)"""
    headers = []
    if 'rows' in table and table['rows']:
        for cell in table['rows'][0]:
            cell_texts = [t['v'] for t in cell.get('texts', []) if t.get('v')]
            if cell_texts:
                headers.append(clean_text(" ".join(cell_texts)))

    first_row = []
    if 'rows' in table and len(table['rows']) >= 2:
        for cell in table['rows'][1]:
            cell_texts = [t['v'] for t in cell.get('texts', []) if t.get('v')]
            if cell_texts:
                first_row.append(clean_text(" ".join(cell_texts)))

    parts = []
    if headers:
        parts.append(f"헤더: [{', '.join(headers[:max_cells])}]")
    if first_row:
        parts.append(f"첫행: [{', '.join(first_row[:max_cells])}]")

    return " / ".join(parts) if parts else "표 정보 없음"

def score_candidates_with_logits(candidates, table_ctx, table_bbox):
    """하이브리드 스코어링: Zero-shot + 리랭커 + prior + 보조항"""
    import numpy as np

    # 1) Zero-shot 분류: 각 후보가 '표 제목'일 절대 확률
    # table_bbox를 전역으로 전달 (근접도 판단용)
    globals()["_current_table_bbox"] = table_bbox
    zs_scores = zeroshot_title_score_batch(candidates) if USE_ZEROSHOT else np.ones(len(candidates)) * 0.5

    # 2) 리랭커: 후보 집합 내 상대 순위
    pairs = [make_reranker_pair(c['text'], table_ctx) for c in candidates]
    logits = reranker_logits_batch(pairs) if USE_RERANKER else np.zeros(len(candidates))
    rer_prob = softmax_with_temp_from_logits(logits, tau=0.6)

    scored = []
    for i, c in enumerate(candidates):
        txt, bb = c['text'], c['bbox']

        # prior (0~1): 패턴/유닛/참조/길이 보정
        p = prior_score(txt, bb, table_bbox)

        # 보조 점수 (미세 타이브레이커)
        emb = embedding_similarity(txt, table_ctx)
        lay = layout_score(table_bbox, bb)

        # Zero-shot 점수 + 하한 보정
        zs = float(zs_scores[i])
        # 표 번호/패턴 매칭 시 ZS 하한 보정 (패턴이 명확하면 최소 0.80 보장)
        if is_table_title_like(txt):
            zs = max(zs, 0.80)

        # 게이팅/가산 방식: 제목 패턴 보너스, 유닛/주석 강감점
        bonus = 0.0
        if is_table_title_like(txt):
            bonus += 0.03
        if is_unit_like(txt) or re.search(r"(주:|비고|참고)\b", txt):
            bonus -= 0.08

        # 최종 점수: Zero-shot(하한 보정) + 리랭커(상대) + 보조항
        final = (WEIGHT_ZEROSHOT * zs
                 + WEIGHT_RERANKER * float(rer_prob[i])
                 + WEIGHT_PRIOR * p
                 + WEIGHT_EMBEDDING * emb
                 + WEIGHT_LAYOUT * lay
                 + bonus)

        scored.append({
            "text": txt, "bbox": bb, "score": final,
            "details": {
                "zeroshot": zs,
                "rer_logits": float(logits[i]),
                "rer_prob_norm": float(rer_prob[i]),
                "prior": p, "emb": emb, "lay": lay, "bonus": bonus
            }
        })
    return scored

# ========== 메인 로직 ==========
def find_title_for_table(table, texts, all_tables=None, used_titles=None):
    """하이브리드 방식으로 표 제목 찾기"""
    table_bbox = get_bbox_from_table(table)
    if not table_bbox:
        print("  테이블 bbox 없음")
        return "", None

    print(f"  테이블 bbox: y={table_bbox[1]}")

    if used_titles is None:
        used_titles = set()

    # Step 1: 후보 수집 (규칙 기반 필터링)
    candidates = collect_candidates_for_table(table, texts, all_tables)

    # 이미 사용된 제목 제외
    candidates = [c for c in candidates if c['text'] not in used_titles]

    print(f"  후보 수집: {len(candidates)}개")

    if not candidates:
        print("  ❌ 후보 없음")
        return "", None

    # Step 2: 표 문맥 구축 (풍부한 레이블링)
    table_ctx = build_table_context_rich(table)
    print(f"  표 문맥: {table_ctx[:80]}")

    # Step 2.5: 후보 프리랭킹 (후보가 많을 경우)
    if len(candidates) > TOPK_CANDIDATES and embedder:
        print(f"  후보 프리랭킹: {len(candidates)}개 → {TOPK_CANDIDATES}개")
        prelim = []
        for c in candidates:
            prelim_score = (0.8 * embedding_similarity(c['text'], table_ctx) +
                           0.2 * layout_score(table_bbox, c['bbox']))
            prelim.append((prelim_score, c))
        prelim.sort(key=lambda x: x[0], reverse=True)
        candidates = [c for _, c in prelim[:TOPK_CANDIDATES]]

    # Step 2.9: 유닛 후보 삭제
    if any(is_table_title_like(c['text']) for c in candidates):
        candidates = [c for c in candidates if not is_unit_like(c['text'])]
        print(f"  유닛 필터링 후: {len(candidates)}개")

    # ★ 제목 패턴 통계 출력 (필터링 안 함, ML이 판단)
    titles = [c for c in candidates if is_table_title_like(c['text'])]
    if len(titles) >= 1:
        print(f"  제목패턴 후보: {len(titles)}개 (ML 기반 점수 적용)")

    # ★ 하드 게이트: 명확한 노이즈만 제거 (교차참조, 긴 설명문)
    before = len(candidates)
    filtered_out = []
    kept = []
    for c in candidates:
        txt = c['text']
        # 교차 참조는 확실히 제목 아님
        if is_cross_reference(txt):
            filtered_out.append(f"{txt[:40]}... (교차참조)")
        # 길고 명확한 설명문 (40자 이상 + 종결어미)
        elif len(txt) >= 40 and re.search(r"(다|였다|한다|였다)\.$", txt.strip()):
            filtered_out.append(f"{txt[:40]}... (긴 설명문)")
        else:
            kept.append(c)
    candidates = kept
    if len(candidates) != before:
        print(f"  노이즈 제거: {before}→{len(candidates)}개")
        for fo in filtered_out[:3]:  # 최대 3개만 출력
            print(f"    제거: {fo}")

    # ★ 소제목 우선 모드: 창 내 소제목이 있으면 소제목만 사용
    subtitle_priority_window = 220  # px
    tbx1, tby1, tbx2, tby2 = table_bbox

    def dy_to_table_top(bb):
        return max(0, tby1 - bb[3])

    subs_in_win = [c for c in candidates if is_subtitle_like(c['text']) and dy_to_table_top(c['bbox']) <= subtitle_priority_window]
    if subs_in_win:
        # 섹션/기타 제거하고 소제목만 남김
        before = len(candidates)
        candidates = subs_in_win
        print(f"  소제목 우선 모드: {before}→{len(candidates)}개 (≤{subtitle_priority_window}px)")

        # 최근접 소제목을 맨 앞으로(동률 시 tie-break에 유리)
        candidates.sort(key=lambda c: dy_to_table_top(c['bbox']))
    # 섹션 헤더 필터링 제거: ML이 판단하도록 함

    if not candidates:
        print("  ❌ 필터링 후 후보 없음")
        return "", None

    # Step 3: 배치 리랭커(로짓) → 소프트맥스 확률 → prior/보조 결합
    print("\n  후보 점수:")
    scored = score_candidates_with_logits(candidates, table_ctx, table_bbox)
    for x in scored:
        t = x["text"][:50] if len(x["text"]) > 50 else x["text"]
        d = x["details"]
        print(f"    '{t}'")
        print(f"      zs: {d['zeroshot']:.3f}, logit: {d['rer_logits']:.3f}, prob*: {d['rer_prob_norm']:.3f}, "
              f"prior: {d['prior']:.3f}, emb: {d['emb']:.3f}, lay: {d['lay']:.3f}, "
              f"bonus: {d['bonus']:+.2f}, Final: {x['score']:.3f}")

    # 최고 점수 선택
    scored.sort(key=lambda x: x['score'], reverse=True)
    best = scored[0]

    # ★ 타이브레이커: 제목패턴 > 소제목 > 기타, 그리고 더 가까운 쪽
    if len(scored) >= 2:
        second = scored[1]
        gap = best['score'] - second['score']
        if gap <= 0.03:
            def rank(c):
                t = c['text']
                if is_table_title_like(t): return 0
                if is_subtitle_like(t):    return 1
                return 2
            def dy(bb): return max(0, table_bbox[1] - bb[3])
            bR, sR = rank(best), rank(second)
            if (sR < bR) or (sR == bR and dy(second['bbox']) < dy(best['bbox'])):
                print("  타이브레이커 적용: 제목/소제목 및 근접도 우선")
                best = second

    # 제목 패턴 검증: 모든 후보가 일반 문장이면 제목 없음으로 처리
    has_title_pattern = any(is_table_title_like(s['text']) for s in scored)
    if not has_title_pattern:
        # Prior 점수가 낮으면(유닛/주석/일반문장) 제목 없음
        if best['details']['prior'] < 0.4:
            print(f"  ⚠️  제목 패턴 없음 (최고 prior: {best['details']['prior']:.3f})")
            return "", None

    # 임계값 체크
    if best['score'] < SCORE_THRESHOLD:
        print(f"  ⚠️  최고 점수({best['score']:.3f})가 임계값({SCORE_THRESHOLD}) 미만")
        return "", None

    print(f"\n  ✅ 선택: '{best['text']}' (점수: {best['score']:.3f})")
    return best['text'], best['bbox']

@app.route('/get_title', methods=['POST'])
def get_title():
    """받은 데이터(tables, texts)에 각 테이블마다 title 프로퍼티를 추가해서 되돌려주는 API"""
    data = request.get_json()

    if isinstance(data, dict):
        tables = data.get('tables', [])
        texts = data.get('texts', [])

        print(f"받은 테이블 수: {len(tables)}")
        print(f"받은 텍스트 수: {len(texts)}")

        result_tables = []
        used_titles = set()

        for idx, table in enumerate(tables):
            table_with_title = copy.deepcopy(table)
            title, title_bbox = find_title_for_table(table, texts, all_tables=tables, used_titles=used_titles)
            print(f"테이블 {idx} 타이틀: '{title}'")
            table_with_title['title'] = title
            table_with_title['title_bbox'] = title_bbox

            if title:
                used_titles.add(title)

            result_tables.append(table_with_title)

        print(f"\n최종 반환: {len(result_tables)}개 테이블")
        return jsonify(result_tables)

    elif isinstance(data, list):
        result = []
        for idx, table in enumerate(data):
            table_with_title = copy.deepcopy(table)
            table_with_title['title'] = f'테이블 {idx + 1}의 타이틀'
            result.append(table_with_title)
        return jsonify(result)

    else:
        return jsonify({'error': 'Data must be an object with tables and texts, or an array of tables'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)
