"""
타이틀 추출 API
하이브리드 방식: 리랭커 + 레이아웃 점수
"""
from flask import Flask, jsonify, request
import copy
import re
import numpy as np

app = Flask(__name__)

# ========== 상수 정의 ==========
# 거리 및 필터링 관련
Y_LINE_TOLERANCE = 100  # 같은 줄로 간주할 y 좌표 허용 오차 (px)
UP_MULTIPLIER = 1.5  # 표 위쪽 탐색 범위 (표 높이의 배수)
X_TOLERANCE = 800  # 수평 근접 허용 거리 (px)

# ML 모델 관련
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 크로스-인코더 리랭커 (다국어 SOTA)
ML_DEVICE = 0  # -1: CPU, 0: GPU
MAX_TEXT_INPUT_LENGTH = 512  # ML 모델 입력 최대 길이

# 모델 사용 설정
USE_RERANKER = True   # 리랭커 사용 여부

SCORE_THRESHOLD = 0.01   # 제목 판정 최소 점수

# 리랭커 설정
RERANKER_TEMPERATURE = 0.6  # 온도 소프트맥스 tau 값 (< 1 → 대비 강화)
RERANKER_BATCH_SIZE = 32    # 리랭커 배치 크기


# 패턴 관련 상수
SUBTITLE_MIN_LENGTH = 4  # 소제목 최소 길이
SUBTITLE_MAX_LENGTH = 40  # 소제목 최대 길이
CROSS_REF_MIN_LENGTH = 35  # 교차 참조 최소 길이
LONG_SENTENCE_MIN_LENGTH = 40  # 긴 설명문 최소 길이

# 텍스트 출력 관련
MAX_DISPLAY_TEXT_LENGTH = 50  # 콘솔 출력 시 텍스트 최대 길이
MAX_CONTEXT_DISPLAY_LENGTH = 80  # 표 문맥 출력 시 최대 길이
MAX_FILTER_DISPLAY = 3  # 필터링된 항목 최대 출력 개수

# API 서버 설정
API_HOST = '0.0.0.0'
API_PORT = 5555
API_DEBUG = True

# ML 모델 변수
reranker = None

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

# ========== 패턴 스코어/페널티 ==========
UNIT_TOKENS = ["단위", "unit", "u.", "℃", "°c", "%", "mm", "kg", "km", "원", "개", "회"]

# 소제목/섹션 패턴
CIRCLED_RX = r"[\u2460-\u2473\u3251-\u325F]"  # ①-⑳, 21-35
BULLET_RX = r"[•∙·\-–—]"  # 불릿/대시
PAREN_NUM_RX = r"\(?\d{1,2}\)?[.)]"  # (1) 1) 1.

def is_subtitle_like(s: str) -> bool:
    """테이블 바로 위에 자주 오는 '소제목' 패턴:
    - ① ② … / (1) 1) 1. / • - 등 불릿 시작
    - 길이 제한 확인
    """
    t = s.strip()
    if re.match(rf"^({CIRCLED_RX}|{PAREN_NUM_RX}|{BULLET_RX})\s*\S+", t):
        if SUBTITLE_MIN_LENGTH <= len(t) <= SUBTITLE_MAX_LENGTH:
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
    """교차 참조/설명 문장 판별"""
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
    if len(t) >= CROSS_REF_MIN_LENGTH and re.search(r"(바랍니다|바란다|협의대로|안내하|요청)", t):
        return True

    return False

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
    """표 위/아래쪽에 있는 텍스트 후보 수집 (규칙 기반 필터링)"""
    table_bbox = get_bbox_from_table(table)
    if not table_bbox:
        return []

    tbx1, tby1, tbx2, tby2 = table_bbox
    h = tby2 - tby1
    y_min_up = max(0, tby1 - int(UP_MULTIPLIER * h))  # 위쪽 탐색 범위
    y_max_down = tby2 + int(UP_MULTIPLIER * h)  # 아래쪽 탐색 범위

    # 그룹화된 텍스트
    grouped_texts = group_texts_by_line(texts, y_tolerance=Y_LINE_TOLERANCE)

    candidates = []
    for text in grouped_texts:
        if not text:
            continue

        text_bbox = text.get('merged_bbox') or get_bbox_from_text(text)
        if not text_bbox:
            continue

        px1, py1, px2, py2 = text_bbox

        # 표 위쪽 또는 아래쪽에 있는지 확인
        is_above = (py2 <= tby1 and py1 >= y_min_up)
        is_below = (py1 >= tby2 and py2 <= y_max_down)

        if not (is_above or is_below):
            continue

        # 수평으로 겹치거나 근접한지 확인
        if not horizontally_near((tbx1, tbx2), (px1, px2), tol=X_TOLERANCE):
            continue

        text_content = clean_text(text.get('merged_text') or extract_text_content(text))

        # 무의미한 텍스트 필터링
        if not text_content or is_trivial(text_content):
            continue

        candidates.append({
            'text': text_content,
            'bbox': text_bbox
        })

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
        out = reranker.predict(pairs, convert_to_numpy=True, batch_size=RERANKER_BATCH_SIZE)
        logits = np.asarray(out).reshape(-1)

        # 디버깅: 로짓 출력
        for i, (pair, logit) in enumerate(zip(pairs, logits)):
            cand_text = pair[0].replace("[표제목 후보] ", "")[:30]
            print(f"      [RR-DEBUG] '{cand_text}' → logit={logit:.3f}")

        return logits
    except Exception as e:
        print(f"  리랭커 오류: {e}")
        return np.zeros((len(pairs),), dtype=float)

def softmax_with_temp_from_logits(logits, tau=RERANKER_TEMPERATURE):
    """로짓에 온도를 적용한 소프트맥스 (후보 집합 내 대비 ↑)"""
    import numpy as np
    x = np.asarray(logits, dtype=float) / max(tau, 1e-6)
    x = x - x.max()  # overflow 방지
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)


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
    """리랭커 기반 스코어링"""
    import numpy as np

    # 리랭커: 후보 집합 내 상대 순위
    pairs = [make_reranker_pair(c['text'], table_ctx) for c in candidates]
    logits = reranker_logits_batch(pairs) if USE_RERANKER else np.zeros(len(candidates))
    rer_prob = softmax_with_temp_from_logits(logits)

    scored = []
    for i, c in enumerate(candidates):
        txt, bb = c['text'], c['bbox']

        # 최종 점수 = 리랭커 점수
        final = float(rer_prob[i])

        scored.append({
            "text": txt, "bbox": bb, "score": final,
            "details": {
                "reranker": float(rer_prob[i])
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

    # Step 2: 표 문맥 구축
    table_ctx = build_table_context_rich(table)
    print(f"  표 문맥: {table_ctx[:MAX_CONTEXT_DISPLAY_LENGTH]}")

    # Step 2.9: 유닛 후보 삭제
    if any(is_table_title_like(c['text']) for c in candidates):
        candidates = [c for c in candidates if not is_unit_like(c['text'])]
        print(f"  유닛 필터링 후: {len(candidates)}개")

    # 제목 패턴 통계
    titles = [c for c in candidates if is_table_title_like(c['text'])]
    if len(titles) >= 1:
        print(f"  제목패턴 후보: {len(titles)}개 (ML 기반 점수 적용)")

    # 하드 필터링: 명확한 노이즈만 제거
    before = len(candidates)
    filtered_out = []
    kept = []
    for c in candidates:
        txt = c['text']
        # 교차 참조
        if is_cross_reference(txt):
            filtered_out.append(f"{txt[:MAX_DISPLAY_TEXT_LENGTH]}... (교차참조)")
        # 긴 설명문 (종결어미로 끝남)
        elif len(txt) >= LONG_SENTENCE_MIN_LENGTH and re.search(r"(다|였다|한다|였다)\.$", txt.strip()):
            filtered_out.append(f"{txt[:MAX_DISPLAY_TEXT_LENGTH]}... (긴 설명문)")
        else:
            kept.append(c)
    candidates = kept
    if len(candidates) != before:
        print(f"  노이즈 제거: {before}→{len(candidates)}개")
        for fo in filtered_out[:MAX_FILTER_DISPLAY]:
            print(f"    제거: {fo}")

    if not candidates:
        print("  ❌ 필터링 후 후보 없음")
        return "", None

    # Step 3: ML 기반 스코어링
    print("\n  후보 점수:")
    scored = score_candidates_with_logits(candidates, table_ctx, table_bbox)
    for x in scored:
        t = x["text"][:MAX_DISPLAY_TEXT_LENGTH] if len(x["text"]) > MAX_DISPLAY_TEXT_LENGTH else x["text"]
        d = x["details"]
        print(f"    '{t}'")
        print(f"      reranker: {d['reranker']:.3f}, Final: {x['score']:.3f}")

    # 최고 점수 선택
    scored.sort(key=lambda x: x['score'], reverse=True)
    best = scored[0]

    # 최종 검증
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
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
