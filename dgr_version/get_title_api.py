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
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # Zero-shot NLI (다국어)
EMBEDDING_MODEL = "BAAI/bge-m3"  # 문장 임베딩 (다국어)
ML_DEVICE = -1  # -1: CPU, 0: GPU
MAX_TEXT_INPUT_LENGTH = 512  # ML 모델 입력 최대 길이

# 최종 점수 가중치
WEIGHT_NLI = 0.55  # Zero-shot NLI (제목 확률)
WEIGHT_EMBEDDING = 0.35  # 임베딩 유사도
WEIGHT_LAYOUT = 0.10  # 레이아웃 점수
SCORE_THRESHOLD = 0  # 제목 판정 최소 점수

# ML 모델 로드
nli_classifier = None
embedder = None

# Zero-shot NLI 모델 로드
try:
    from transformers import pipeline
    nli_classifier = pipeline(
        "zero-shot-classification",
        model=NLI_MODEL,
        device=ML_DEVICE
    )
    print(f"✅ NLI 모델 로드 완료 ({NLI_MODEL})")
except ImportError:
    print("⚠️  transformers 라이브러리 없음")
    nli_classifier = None
except Exception as e:
    print(f"⚠️  NLI 모델 로드 실패: {e}")
    nli_classifier = None

# 임베딩 모델 로드
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✅ 임베딩 모델 로드 완료 ({EMBEDDING_MODEL})")
except ImportError:
    print("⚠️  sentence-transformers 라이브러리 없음")
    embedder = None
except Exception as e:
    print(f"⚠️  임베딩 모델 로드 실패: {e}")
    embedder = None

# ========== 유틸리티 함수 ==========
def clean_text(s: str) -> str:
    """텍스트 정리"""
    return re.sub(r"\s+", " ", s).strip()

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

    # 단위 표기 (다양한 형태)
    if re.search(r'^\s*\(?단위\s*[:：]', s, re.IGNORECASE):
        return True
    if re.match(r'^\s*\(\s*단위\s*[:：]?.*\)\s*$', s, re.IGNORECASE):
        return True
    if re.match(r'^\s*\(\s*단위\s+[a-zA-Z가-힣%]+\s*\)\s*$', s, re.IGNORECASE):
        return True

    # 목록 항목 (1., 2., ①, ② 등으로 시작)
    if re.match(r'^[\d①-⑳]\.\s+[A-Z_]+\s*[=:]+', s):
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
            return ''.join(texts)

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

        if prev_bbox and current_bbox:
            gap = current_bbox[0] - prev_bbox[2]
            if gap > X_GAP_THRESHOLD:
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
    """표 위쪽에 있는 텍스트 후보 수집 (규칙 기반 필터링)"""
    table_bbox = get_bbox_from_table(table)
    if not table_bbox:
        return []

    tbx1, tby1, tbx2, tby2 = table_bbox
    h = tby2 - tby1
    y_min = max(0, tby1 - int(UP_MULTIPLIER * h))

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

        # 표 위쪽에 있는지 확인
        if not (py2 <= tby1 and py1 >= y_min):
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
def nli_title_prob(text: str) -> float:
    """Zero-shot Classification으로 제목 확률 계산"""
    if not nli_classifier:
        return 0.0

    try:
        result = nli_classifier(
            text,
            candidate_labels=["table title", "not a title"],
            hypothesis_template="This text is a {}."
        )
        labels = result["labels"]
        scores = result["scores"]
        score_dict = {l: s for l, s in zip(labels, scores)}
        return float(score_dict.get("table title", 0.0))
    except Exception as e:
        print(f"  NLI 오류: {e}")
        return 0.0

def embedding_similarity(text_a: str, text_b: str) -> float:
    """임베딩 유사도 계산"""
    if not embedder:
        return 0.0

    try:
        vecs = embedder.encode([text_a, text_b])
        return cosine_similarity(vecs[0], vecs[1])
    except Exception as e:
        print(f"  임베딩 오류: {e}")
        return 0.0

def score_candidate(cand_text: str, cand_bbox, table_bbox, table_ctx: str) -> dict:
    """후보 점수 계산 (NLI + 임베딩 + 레이아웃 + 보너스)"""
    # NLI 점수
    nli_score = nli_title_prob(cand_text)

    # 임베딩 유사도
    emb_score = embedding_similarity(cand_text, table_ctx)

    # 레이아웃 점수
    lay_score = layout_score(table_bbox, cand_bbox)

    # 제목 패턴 보너스
    title_bonus = 0.0
    # "표 X.X ..." 또는 "Table X.X ..." 패턴
    if re.match(r'^(표|table)\s*[\d\.]+', cand_text, re.IGNORECASE):
        title_bonus = 0.15
    # "<그림>" 등은 페널티
    elif re.match(r'^<.*>$', cand_text):
        title_bonus = -0.2
    # 단위 표기는 페널티
    elif re.search(r'\(\s*단위', cand_text, re.IGNORECASE):
        title_bonus = -0.3

    # 최종 점수
    final_score = (WEIGHT_NLI * nli_score +
                   WEIGHT_EMBEDDING * emb_score +
                   WEIGHT_LAYOUT * lay_score +
                   title_bonus)

    return {
        'final_score': final_score,
        'nli': nli_score,
        'embedding': emb_score,
        'layout': lay_score,
        'bonus': title_bonus
    }

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
    table_ctx = build_table_context(table)
    print(f"  표 문맥: {table_ctx[:80]}")

    # Step 3: ML 스코어링
    print("\n  후보 점수:")
    scored = []
    for c in candidates:
        scores = score_candidate(c['text'], c['bbox'], table_bbox, table_ctx)
        text_preview = c['text'][:50] if len(c['text']) > 50 else c['text']
        print(f"    '{text_preview}'")
        print(f"      NLI: {scores['nli']:.3f}, Emb: {scores['embedding']:.3f}, Layout: {scores['layout']:.3f}, Bonus: {scores['bonus']:.3f}, Final: {scores['final_score']:.3f}")

        scored.append({
            'text': c['text'],
            'bbox': c['bbox'],
            'score': scores['final_score'],
            'details': scores
        })

    # 최고 점수 선택
    scored.sort(key=lambda x: x['score'], reverse=True)
    best = scored[0]

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
