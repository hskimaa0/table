"""
타이틀 추출 API
LLM 방식: Ollama gemma2:2b로 표+텍스트 분석
"""
from flask import Flask, jsonify, request
import copy
import re
import requests
import json
import os

app = Flask(__name__)

# ========== 상수 정의 ==========
# 거리 및 필터링 관련
Y_LINE_TOLERANCE = 100  # 같은 줄로 간주할 y 좌표 허용 오차 (px)
UP_MULTIPLIER = 1.5  # 표 위쪽 탐색 범위 (표 높이의 배수)
X_TOLERANCE = 800  # 수평 근접 허용 거리 (px)

# LLM 설정 (환경변수 지원)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma2:2b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "100"))

MAX_TEXT_INPUT_LENGTH = 512  # LLM 입력 최대 길이
SCORE_THRESHOLD = 0.40   # 제목 판정 최소 점수 (더 이상 사용 안 함)


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

print(f"[OK] LLM API: {OLLAMA_URL}")
print(f"[OK] 사용 모델: {LLM_MODEL}")

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

# ========== LLM 함수 ==========
def call_ollama_llm(prompt: str) -> str:
    """Ollama API를 호출하여 LLM 응답 받기"""
    try:
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "temperature": LLM_TEMPERATURE,
            "stream": False,
            "options": {
                "num_predict": LLM_MAX_TOKENS
            }
        }

        print(f"    [API] 요청 전송 중 - 모델: {LLM_MODEL}, 온도: {LLM_TEMPERATURE}")
        import time
        start = time.time()

        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()

        elapsed = time.time() - start
        print(f"    [API] 응답 수신 완료 ({elapsed:.2f}초)")

        result = response.json()
        llm_response = result.get("response", "").strip()

        print(f"    [API] 응답 길이: {len(llm_response)}자")

        return llm_response

    except requests.exceptions.RequestException as e:
        print(f"  [!] Ollama API 오류: {e}")
        return ""
    except Exception as e:
        print(f"  [!] LLM 호출 오류: {e}")
        return ""

def build_table_context_full(table):
    """표 전체 내용 구축 - LLM 프롬프트용"""
    all_texts = []

    if 'rows' in table and table['rows']:
        for row_idx, row in enumerate(table['rows'][:10]):  # 최대 10행만 (토큰 제한)
            row_texts = []
            for cell in row:
                cell_texts = [t['v'] for t in cell.get('texts', []) if t.get('v')]
                if cell_texts:
                    row_texts.append(clean_text(" ".join(cell_texts)))
            if row_texts:
                all_texts.append(" | ".join(row_texts))

    return "\n".join(all_texts) if all_texts else "표 정보 없음"

def select_title_with_llm(candidates, table_content):
    """LLM을 사용하여 후보 중 가장 적절한 제목 선택"""
    if not candidates:
        return None

    # 프롬프트 구성
    candidates_text = "\n".join([f"{i+1}. {c['text']}" for i, c in enumerate(candidates)])

    prompt = f"""다음은 표의 내용입니다:
{table_content[:1000]}

위 표의 제목으로 가장 적절한 것을 아래 후보 중에서 골라주세요:
{candidates_text}

규칙:
- 표의 내용과 가장 관련 있는 것을 선택
- 단위 표기(예: (단위:원))는 제목이 아님
- 교차 참조 문장(예: 표 X에 의하면)은 제목이 아님
- 긴 설명문은 제목이 아님
- 표 번호가 포함된 것을 우선 선택(예: 표 3-2 연간 실적)
- 제목이 없다고 판단되면 "없음"이라고만 답하세요

답변은 반드시 숫자만 출력하세요 (예: 1 또는 2 또는 없음). 설명 없이 숫자만."""

    print(f"\n  [DEBUG-LLM] 프롬프트 생성 완료")
    print(f"    - 표 내용 길이: {len(table_content[:1000])}자")
    print(f"    - 후보 개수: {len(candidates)}개")
    print(f"    - 프롬프트 전체 길이: {len(prompt)}자")
    print(f"\n  [DEBUG-LLM] 프롬프트 미리보기:")
    print(f"    {prompt[:300]}...")

    print(f"\n  [DEBUG-LLM] Ollama API 호출 중...")
    response = call_ollama_llm(prompt)
    print(f"  [DEBUG-LLM] 응답 수신: '{response}'")

    # 응답 파싱
    print(f"\n  [DEBUG-LLM] 응답 파싱 시작")
    response = response.strip()

    # "없음" 또는 "제목 없음" 등
    if "없음" in response.lower() or "no" in response.lower():
        print(f"    - '없음' 감지: LLM이 제목 없음으로 판단")
        return None

    # 숫자 추출
    import re
    numbers = re.findall(r'\d+', response)
    print(f"    - 추출된 숫자: {numbers}")

    if numbers:
        idx = int(numbers[0]) - 1  # 1-based → 0-based
        print(f"    - 인덱스 변환: {numbers[0]} → {idx}")

        if 0 <= idx < len(candidates):
            print(f"    - 유효한 인덱스, 후보 반환: '{candidates[idx]['text']}'")
            return candidates[idx]
        else:
            print(f"    - 인덱스 범위 초과 ({idx} >= {len(candidates)})")

    # 파싱 실패 시 첫 번째 후보 반환
    print(f"  [!] LLM 응답 파싱 실패, 첫 번째 후보 반환")
    return candidates[0] if candidates else None

# ========== 메인 로직 ==========
def find_title_for_table(table, texts, all_tables=None, used_titles=None):
    """LLM 방식으로 표 제목 찾기"""
    table_bbox = get_bbox_from_table(table)
    if not table_bbox:
        print("  테이블 bbox 없음")
        return "", None

    print(f"  테이블 bbox: y={table_bbox[1]}")

    if used_titles is None:
        used_titles = set()

    # Step 1: 후보 수집 (규칙 기반 필터링)
    print("\n[DEBUG] Step 1: 후보 수집 (규칙 기반)")
    candidates = collect_candidates_for_table(table, texts, all_tables)
    print(f"  - 위치/거리 필터링 후: {len(candidates)}개")
    for i, c in enumerate(candidates[:5]):  # 처음 5개만
        print(f"    {i+1}. '{c['text'][:50]}'")

    # 이미 사용된 제목 제외
    before_used = len(candidates)
    candidates = [c for c in candidates if c['text'] not in used_titles]
    if before_used != len(candidates):
        print(f"  - 이미 사용된 제목 제외: {before_used}→{len(candidates)}개")

    print(f"  총 후보: {len(candidates)}개")

    if not candidates:
        print("  [X] 후보 없음")
        return "", None

    # Step 2: 유닛 후보 삭제
    print("\n[DEBUG] Step 2: 유닛 필터링")
    before_unit = len(candidates)
    if any(is_table_title_like(c['text']) for c in candidates):
        unit_removed = [c for c in candidates if is_unit_like(c['text'])]
        candidates = [c for c in candidates if not is_unit_like(c['text'])]
        print(f"  - 유닛 제거: {before_unit}→{len(candidates)}개")
        for u in unit_removed[:3]:
            print(f"    제거: '{u['text']}'")
    else:
        print(f"  - 제목 패턴 없음, 유닛 필터링 스킵")

    # Step 3: 하드 필터링 - 명확한 노이즈만 제거
    print("\n[DEBUG] Step 3: 노이즈 필터링")
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
    print(f"  - 노이즈 제거: {before}→{len(candidates)}개")
    if filtered_out:
        for fo in filtered_out[:MAX_FILTER_DISPLAY]:
            print(f"    제거: {fo}")

    if not candidates:
        print("  [X] 필터링 후 후보 없음")
        return "", None

    # Step 4: 표 전체 내용 구축
    print("\n[DEBUG] Step 4: 표 내용 구축")
    table_content = build_table_context_full(table)
    print(f"  - 표 내용 (처음 {MAX_CONTEXT_DISPLAY_LENGTH}자): {table_content[:MAX_CONTEXT_DISPLAY_LENGTH]}...")
    print(f"  - 전체 길이: {len(table_content)}자")

    # Step 5: LLM으로 제목 선택
    print("\n[DEBUG] Step 5: LLM으로 제목 선택")
    print(f"  - 최종 후보 목록:")
    for i, c in enumerate(candidates):
        print(f"    {i+1}. '{c['text']}'")

    selected = select_title_with_llm(candidates, table_content)

    if selected is None:
        print("  [X] LLM이 제목 없음으로 판단")
        return "", None

    print(f"\n[DEBUG] 최종 선택: '{selected['text']}'")
    return selected['text'], selected['bbox']

@app.route('/get_title', methods=['POST'])
def get_title():
    """받은 데이터(tables, texts)에 각 테이블마다 title 프로퍼티를 추가해서 되돌려주는 API"""
    print("\n" + "="*80)
    print("[API 요청 수신]")
    print("="*80)

    data = request.get_json()

    if isinstance(data, dict):
        tables = data.get('tables', [])
        texts = data.get('texts', [])

        print(f"\n[요청 데이터]")
        print(f"  - 테이블 수: {len(tables)}개")
        print(f"  - 텍스트 수: {len(texts)}개")

        result_tables = []
        used_titles = set()

        for idx, table in enumerate(tables):
            print(f"\n{'='*80}")
            print(f"[테이블 {idx+1}/{len(tables)} 처리 시작]")
            print(f"{'='*80}")

            table_with_title = copy.deepcopy(table)
            title, title_bbox = find_title_for_table(table, texts, all_tables=tables, used_titles=used_titles)

            print(f"\n[테이블 {idx+1} 결과] 타이틀: '{title}'")

            table_with_title['title'] = title
            table_with_title['title_bbox'] = title_bbox

            if title:
                used_titles.add(title)

            result_tables.append(table_with_title)

        print(f"\n{'='*80}")
        print(f"[API 처리 완료]")
        print(f"  - 반환 테이블: {len(result_tables)}개")
        print(f"  - 발견된 제목: {len(used_titles)}개")
        print(f"{'='*80}\n")

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
