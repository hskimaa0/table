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
KOBERT_MODEL_PATH = "kobert_table_classifier.pt"  # KoBERT 분류 모델 경로
ML_DEVICE = 0  # -1: CPU, 0: GPU
MAX_TEXT_INPUT_LENGTH = 512  # ML 모델 입력 최대 길이

# 모델 사용 설정
USE_KOBERT = True  # KoBERT 분류기 사용 여부 (제목/설명/제목아님 필터링)


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
kobert_classifier = None

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

# KoBERT 분류기 로드
if USE_KOBERT:
    try:
        import sys
        import os

        # 현재 디렉토리를 Python 경로에 추가
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        from kobert_classifier import TableTextClassifier

        if os.path.exists(KOBERT_MODEL_PATH):
            kobert_classifier = TableTextClassifier(model_path=KOBERT_MODEL_PATH, device=DEVICE_STR)
            print(f"✅ KoBERT 분류기 로드 완료 ({KOBERT_MODEL_PATH}, device={DEVICE_STR})")
        else:
            print(f"⚠️  KoBERT 모델 파일 없음: {KOBERT_MODEL_PATH}")
            kobert_classifier = None
    except ImportError as e:
        print(f"⚠️  kobert_classifier 모듈 없음: {e}")
        kobert_classifier = None
    except Exception as e:
        print(f"⚠️  KoBERT 분류기 로드 실패: {e}")
        kobert_classifier = None
else:
    print("ℹ️  KoBERT 분류기 비활성화 (USE_KOBERT=False)")

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

    # 'ㅇ 제목' 형식도 추가 - 단, 너무 짧거나 불완전한 문장은 제외
    if re.match(r"^[ㅇo]\s+\S{2,}", t) and len(t) <= SUBTITLE_MAX_LENGTH:
        # "ㅇ 요구사항", "ㅇ 제약" 같은 불완전한 제목 제외
        incomplete_patterns = [
            r"^[ㅇo]\s+(요구사항|제약|조건|사항)$",  # 너무 짧음
            r"^[ㅇo]\s+제약\s*요구사항$",  # "ㅇ 제약 요구사항" (문맥 없음)
        ]
        for pattern in incomplete_patterns:
            if re.search(pattern, t):
                return False
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
    t = s.strip()

    # 1. '표 B.8 월별 기온', '표 A.6 토지이용현황', '표 B .4' (공백 포함), '표 3-2 연간 실적' 등
    if re.search(r"^표\s*[A-Za-z]?\s*[\.\-]?\s*\d+([\-\.]\d+)?", t):
        return True

    # 2. '□ 추진조직 구성', '■ 사업개요' 등 (박스 기호 + 제목)
    if re.search(r"^[□■◆◇▪▫●○◉◎]\s*\S", t):
        return True

    # 3. '<표 제목>', '【표 제목】' 등
    if re.search(r"^[<《\[【\(]\s*표?\s*\d*\s*[\]】\)>》]", t):
        return True

    # 4. 섹션/표 제목 형태(숫자.숫자 제목) - 단, 너무 짧지 않아야 함
    if re.search(r"^\d+(\.\d+){0,2}\s+\S{3,}", t) and len(t) <= 50:
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

    # '※' 시작하는 주석/부가설명
    if t.startswith("※") and len(t) >= 20:
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
    """페이지의 모든 텍스트를 후보로 수집 (위치 조건 없음, KoBERT만 의존)"""
    # 그룹화된 텍스트
    grouped_texts = group_texts_by_line(texts, y_tolerance=Y_LINE_TOLERANCE)

    candidates = []
    for text in grouped_texts:
        if not text:
            continue

        text_bbox = text.get('merged_bbox') or get_bbox_from_text(text)
        if not text_bbox:
            continue

        text_content = clean_text(text.get('merged_text') or extract_text_content(text))

        # 빈 텍스트만 제외
        if not text_content or len(text_content.strip()) == 0:
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



# ========== 메인 로직 ==========
# 이 함수는 더 이상 사용되지 않음 (get_title()에서 직접 처리)

def calculate_distance_between_title_and_table(title_bbox, table_bbox):
    """제목과 표 사이의 거리 계산"""
    tx1, ty1, tx2, ty2 = title_bbox
    tbx1, tby1, tbx2, tby2 = table_bbox

    # 표 위쪽에 있는 경우: 표 상단과 제목 하단의 거리
    if ty2 <= tby1:
        return tby1 - ty2
    # 표 아래쪽에 있는 경우: 제목 상단과 표 하단의 거리
    elif ty1 >= tby2:
        return ty1 - tby2
    # 겹치는 경우
    else:
        return 0

@app.route('/get_title', methods=['POST'])
def get_title():
    """받은 데이터(tables, texts)에 각 테이블마다 title 프로퍼티를 추가해서 되돌려주는 API"""
    data = request.get_json()

    if isinstance(data, dict):
        tables = data.get('tables', [])
        texts = data.get('texts', [])

        print(f"받은 테이블 수: {len(tables)}")
        print(f"받은 텍스트 수: {len(texts)}")

        # 테이블이 없으면 빈 배열 반환
        if not tables:
            print("테이블이 없어서 분류 생략")
            return jsonify([])

        # Step 1: 모든 텍스트에서 KoBERT로 제목 후보 추출
        grouped_texts = group_texts_by_line(texts, y_tolerance=Y_LINE_TOLERANCE)

        all_title_candidates = []
        if USE_KOBERT and kobert_classifier:
            print("\n[모든 텍스트 KoBERT 분류]")
            for text in grouped_texts:
                if not text:
                    continue

                text_bbox = text.get('merged_bbox') or get_bbox_from_text(text)
                if not text_bbox:
                    continue

                text_content = clean_text(text.get('merged_text') or extract_text_content(text))
                if not text_content or len(text_content.strip()) == 0:
                    continue

                label = kobert_classifier.predict(text_content)
                if label == "제목":
                    all_title_candidates.append({
                        'text': text_content,
                        'bbox': text_bbox
                    })
                    print(f"  ✓ '{text_content[:40]}' → 제목")

        print(f"\n총 제목 후보: {len(all_title_candidates)}개")

        # Step 2: 각 제목을 가장 가까운 테이블에 할당
        title_assignments = {}  # {table_idx: {'text': ..., 'bbox': ..., 'distance': ...}}

        for title_cand in all_title_candidates:
            closest_table_idx = None
            min_distance = float('inf')

            for idx, table in enumerate(tables):
                table_bbox = get_bbox_from_table(table)
                if not table_bbox:
                    continue

                distance = calculate_distance_between_title_and_table(title_cand['bbox'], table_bbox)

                if distance < min_distance:
                    min_distance = distance
                    closest_table_idx = idx

            # 가장 가까운 테이블에 할당 (더 가까운 제목이 있으면 교체)
            if closest_table_idx is not None:
                if closest_table_idx not in title_assignments or min_distance < title_assignments[closest_table_idx]['distance']:
                    title_assignments[closest_table_idx] = {
                        'text': title_cand['text'],
                        'bbox': title_cand['bbox'],
                        'distance': min_distance
                    }

        # Step 3: 결과 생성
        result_tables = []
        for idx, table in enumerate(tables):
            table_with_title = copy.deepcopy(table)

            if idx in title_assignments:
                assignment = title_assignments[idx]
                table_with_title['title'] = assignment['text']
                table_with_title['title_bbox'] = assignment['bbox']
                print(f"테이블 {idx} → 제목: '{assignment['text']}' (거리: {assignment['distance']:.1f})")
            else:
                table_with_title['title'] = ""
                table_with_title['title_bbox'] = None
                print(f"테이블 {idx} → 제목 없음")

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
