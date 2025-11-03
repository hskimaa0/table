"""
하이브리드 타이틀 추출 API
규칙 기반 필터링 + ML 모델 선택
"""
from flask import Flask, jsonify, request
import copy
import re

app = Flask(__name__)

# ========== 상수 정의 ==========
# 거리 및 필터링 관련
MAX_DISTANCE_THRESHOLD = 10000  # 타이틀 후보로 고려할 최대 거리 (px)
MAX_TEXT_LENGTH = 150  # 타이틀로 고려할 최대 텍스트 길이
Y_LINE_TOLERANCE = 50  # 같은 줄로 간주할 y 좌표 허용 오차 (px)
X_GAP_THRESHOLD = 100  # 텍스트 사이 공백 추가 기준 간격 (px)
HORIZONTAL_WEIGHT = 0.1  # 수평 거리 가중치 (수직 거리 대비)

# ML 모델 관련
ML_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # Zero-shot classification 모델 (280MB, 다국어)
ML_DEVICE = -1  # -1: CPU, 0: GPU
ML_CONFIDENCE_THRESHOLD = 0.5  # ML 모델 제목 판단 임계값
MAX_TEXT_INPUT_LENGTH = 512  # ML 모델 입력 최대 길이
TOP_CANDIDATES_COUNT = 2  # ML 모델에 전달할 상위 후보 개수
ML_CANDIDATE_LABELS = ["substantive title conveying the table's main subject matter", "parenthetical notes, units, symbols, or reference markers"]
ML_HYPOTHESIS_TEMPLATE = "This is {}."  # 가설 템플릿

# 거리 점수 계산 관련
DISTANCE_SCORE_BASE = 100  # 거리 0일 때 기본 점수
DISTANCE_SCORE_DIVIDER = 50  # 거리 점수 감소율

# ML 모델 로드 (Zero-shot Classification)
USE_ML_MODEL = True
classifier = None

try:
    from transformers import pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model=ML_MODEL_NAME,
        device=ML_DEVICE
    )
    print(f"✅ Zero-shot Classification 모델 로드 완료 ({ML_MODEL_NAME})")
except ImportError:
    print("⚠️  transformers 라이브러리 없음, 거리 기반만 사용")
except Exception as e:
    print(f"⚠️  모델 로드 실패: {e}, 거리 기반만 사용")

def get_bbox_from_text(text):
    """text 객체에서 bbox 정보 추출 [l, t, r, b] 형식으로 반환"""
    # rect가 있는 경우 (배열 형태: [l, t, r, b])
    if 'rect' in text and isinstance(text['rect'], list) and len(text['rect']) >= 4:
        return text['rect']
    # bbox가 있는 경우 (객체 형태: {l, t, r, b})
    elif 'bbox' in text and isinstance(text['bbox'], dict):
        bbox = text['bbox']
        return [bbox.get('l', 0), bbox.get('t', 0), bbox.get('r', 0), bbox.get('b', 0)]
    # bbox가 배열 형태인 경우
    elif 'bbox' in text and isinstance(text['bbox'], list) and len(text['bbox']) >= 4:
        return text['bbox']
    return None

def get_bbox_from_table(table):
    """table 객체에서 bbox 정보 추출 [l, t, r, b] 형식으로 반환"""
    if 'bbox' in table and isinstance(table['bbox'], list) and len(table['bbox']) >= 4:
        return table['bbox']
    elif 'bbox' in table and isinstance(table['bbox'], dict):
        bbox = table['bbox']
        return [bbox.get('l', 0), bbox.get('t', 0), bbox.get('r', 0), bbox.get('b', 0)]
    return None

def calculate_distance(text_bbox, table_bbox):
    """text와 table 사이의 거리 계산 (table 위쪽에 있는 text만 고려)"""
    if not text_bbox or not table_bbox:
        return float('inf')

    # bbox 형식: [l, t, r, b]
    # y 좌표는 아래로 갈수록 증가 (top < bottom)
    text_bottom = text_bbox[3]  # text의 아래쪽
    table_top = table_bbox[1]   # table의 위쪽

    # text가 table 위에 있는지 확인 (text의 bottom이 table의 top보다 작아야 함)
    if text_bottom >= table_top:
        return float('inf')  # table 아래에 있거나 겹치면 무한대 거리

    # 수직 거리 계산
    vertical_distance = table_top - text_bottom

    # 수평 정렬도 고려 (테이블과 텍스트의 중심이 가까울수록 좋음)
    text_center_x = (text_bbox[0] + text_bbox[2]) / 2
    table_center_x = (table_bbox[0] + table_bbox[2]) / 2
    horizontal_distance = abs(text_center_x - table_center_x)

    # 수직 거리에 가중치를 더 주되, 수평 정렬도 약간 고려
    return vertical_distance + (horizontal_distance * HORIZONTAL_WEIGHT)

def extract_text_content(text_obj):
    """text 객체에서 실제 텍스트 추출 (x 좌표 순서대로)"""
    # 't' 배열 내부의 'text' 속성 (주어진 데이터 형식)
    if 't' in text_obj and isinstance(text_obj['t'], list) and len(text_obj['t']) > 0:
        # 't' 배열을 x 좌표(bbox.l) 순서대로 정렬
        t_items = text_obj['t']

        # bbox 정보가 있으면 x 좌표로 정렬
        sorted_items = sorted(t_items, key=lambda item: item.get('bbox', [0, 0, 0, 0])[0] if item.get('bbox') else 0)

        # 정렬된 순서대로 텍스트 추출
        texts = []
        for t_item in sorted_items:
            if 'text' in t_item:
                texts.append(t_item['text'])

        if texts:
            return ''.join(texts)  # 공백 없이 붙임 (원본 텍스트 그대로)

    # 'v' 속성에 텍스트가 있는 경우
    if 'v' in text_obj:
        return text_obj['v']
    # 'text' 속성에 텍스트가 있는 경우
    elif 'text' in text_obj:
        return text_obj['text']
    return ""

def group_texts_by_line(texts, y_tolerance=50):
    """같은 줄(y 좌표 비슷)에 있는 텍스트들을 그룹화하고 위에서 아래 순서로 정렬"""
    if not texts:
        return []

    # y 좌표 기준으로 정렬 (위에서 아래로)
    sorted_texts = sorted(texts, key=lambda t: get_bbox_from_text(t)[1] if get_bbox_from_text(t) else float('inf'))

    grouped = []
    current_group = []
    current_y = None

    for text in sorted_texts:
        bbox = get_bbox_from_text(text)
        if not bbox:
            continue

        y_top = bbox[1]

        # 첫 텍스트이거나 y 좌표가 비슷하면 같은 그룹
        if current_y is None or abs(y_top - current_y) <= y_tolerance:
            current_group.append(text)
            if current_y is None:
                current_y = y_top
        else:
            # 새로운 줄 시작 - 이전 그룹 저장
            if current_group:
                merged = merge_text_group(current_group)
                if merged:
                    grouped.append(merged)
            current_group = [text]
            current_y = y_top

    # 마지막 그룹 추가
    if current_group:
        merged = merge_text_group(current_group)
        if merged:
            grouped.append(merged)

    # 그룹화된 결과를 다시 y 좌표 순서로 정렬 (위에서 아래로)
    grouped.sort(key=lambda g: g.get('merged_bbox', [0, 0, 0, 0])[1] if g else float('inf'))

    return grouped

def merge_text_group(text_group):
    """같은 줄의 텍스트들을 x 좌표 순서대로 병합"""
    if not text_group:
        return None

    # x 좌표 기준으로 정렬 (왼쪽에서 오른쪽)
    sorted_group = sorted(text_group, key=lambda t: get_bbox_from_text(t)[0] if get_bbox_from_text(t) else 0)

    # 텍스트 추출 및 병합 (각 텍스트는 이미 내부적으로 정렬됨)
    text_parts = []
    prev_bbox = None

    for t in sorted_group:
        text_content = extract_text_content(t)
        if not text_content:
            continue

        current_bbox = get_bbox_from_text(t)

        # 이전 텍스트와의 간격 확인 (x 좌표 차이)
        if prev_bbox and current_bbox:
            gap = current_bbox[0] - prev_bbox[2]  # 현재 left - 이전 right
            # 간격이 크면 공백 추가
            if gap > X_GAP_THRESHOLD:
                text_parts.append(' ')

        text_parts.append(text_content)
        prev_bbox = current_bbox

    merged_text = ''.join(text_parts)

    # bbox 계산 (그룹 전체를 포함하는 영역)
    bboxes = [get_bbox_from_text(t) for t in sorted_group]
    bboxes = [b for b in bboxes if b]

    if not bboxes:
        return None

    merged_bbox = [
        min(b[0] for b in bboxes),  # left
        min(b[1] for b in bboxes),  # top
        max(b[2] for b in bboxes),  # right
        max(b[3] for b in bboxes)   # bottom
    ]

    # 원본 객체 정보 유지하되, 병합된 정보로 업데이트
    merged_obj = text_group[0].copy() if text_group else {}
    merged_obj['merged_text'] = merged_text
    merged_obj['merged_bbox'] = merged_bbox

    return merged_obj

def is_title_candidate(text_content, distance):
    """규칙 기반 타이틀 후보 필터링 - 명백히 제목이 아닌 것만 제외"""
    if not text_content or len(text_content.strip()) == 0:
        return False, "빈 텍스트"

    text = text_content.strip()

    # 1. 너무 멀리 있는 텍스트 제외
    if distance > MAX_DISTANCE_THRESHOLD:
        return False, "너무 멀리 떨어짐"

    # 2. 너무 긴 텍스트 제외
    if len(text) > MAX_TEXT_LENGTH:
        return False, "너무 긴 텍스트"

    # 3. 단일 문자 제외 (마침표, 쉼표 등)
    if len(text) == 1:
        return False, "단일 문자"

    # 4. 페이지 번호 패턴 제외 (예: "- 8 -", "184")
    if re.match(r'^-?\s*\d+\s*-?$', text):
        return False, "페이지 번호"

    # 5. 단위만 있는 경우 우선순위 낮춤 (완전 제외는 안 함)
    # "(단위: cm)" 같은 건 후보로 남기되 점수를 낮춤

    return True, "후보"

def score_title_candidate(text_obj, text_content, distance):
    """타이틀 후보에 점수 부여 - 거리 기반만 사용"""
    # 거리 점수만 사용 (가까울수록 높은 점수)
    score = max(0, DISTANCE_SCORE_BASE - (distance / DISTANCE_SCORE_DIVIDER))

    return score

def select_best_title_ml(candidates):
    """Zero-shot Classification으로 최적의 타이틀 선택 (배치 처리)"""
    if not USE_ML_MODEL or not classifier or not candidates:
        return None

    try:
        # 텍스트 전처리: 너무 긴 텍스트는 잘라서 처리
        candidate_texts = [
            c['text'][:MAX_TEXT_INPUT_LENGTH] if len(c['text']) > MAX_TEXT_INPUT_LENGTH else c['text']
            for c in candidates
        ]

        print("  Zero-shot Classification 점수 (배치 처리):")

        # 배치 처리: 모든 후보를 한번에 처리
        results = classifier(
            candidate_texts,
            candidate_labels=ML_CANDIDATE_LABELS,
            hypothesis_template=ML_HYPOTHESIS_TEMPLATE
        )

        # 단일 결과인 경우 리스트로 변환
        if not isinstance(results, list):
            results = [results]

        best_idx = -1
        best_score = -1

        # 각 결과에서 첫 번째 라벨(제목) 확률 추출
        for i, result in enumerate(results):
            title_score = result['scores'][result['labels'].index(ML_CANDIDATE_LABELS[0])]

            text_preview = candidate_texts[i][:40] if len(candidate_texts[i]) > 40 else candidate_texts[i]
            print(f"    {i+1}. '{text_preview}' → 제목 확률: {title_score:.3f}")

            if title_score > best_score:
                best_score = title_score
                best_idx = i

        print(f"  ML 최고 확률: {best_score:.3f} (임계값: {ML_CONFIDENCE_THRESHOLD})")

        # 임계값 이상이어야 제목으로 판단
        if best_idx >= 0 and best_score > ML_CONFIDENCE_THRESHOLD:
            return candidates[best_idx]
        else:
            print(f"  ML 임계값 미달 → 거리 기반 사용")
            return None

    except Exception as e:
        print(f"  ML 모델 오류: {e}")
        import traceback
        traceback.print_exc()

    return None

def find_title_for_table(table, texts):
    """하이브리드 방식으로 table의 title 찾기"""
    table_bbox = get_bbox_from_table(table)
    if not table_bbox:
        print("  테이블 bbox 없음")
        return "", None

    print(f"  테이블 bbox: y={table_bbox[1]}")

    # Step 0: 같은 줄의 텍스트들을 그룹화
    grouped_texts = group_texts_by_line(texts, y_tolerance=Y_LINE_TOLERANCE)
    print(f"  원본 텍스트: {len(texts)}개 → 그룹화: {len(grouped_texts)}개")

    # 그룹화된 텍스트 처음 5개 출력
    print("  그룹화된 텍스트 샘플 (위에서 아래 순서):")
    for i, gt in enumerate(grouped_texts[:5]):
        if gt:
            text_preview = (gt.get('merged_text') or extract_text_content(gt))[:40]
            bbox = gt.get('merged_bbox') or get_bbox_from_text(gt)
            print(f"    {i+1}. y={bbox[1]}: '{text_preview}'")

    # Step 1: 테이블 위쪽 텍스트만 수집 (거리 무관, 상위 2개)
    candidates = []

    print("\n  필터링 상세 로그:")
    for i, text in enumerate(grouped_texts):
        if not text:
            continue

        # 병합된 텍스트 사용
        text_bbox = text.get('merged_bbox') or get_bbox_from_text(text)
        if not text_bbox:
            continue

        distance = calculate_distance(text_bbox, table_bbox)

        # 병합된 텍스트 우선 사용
        text_content = text.get('merged_text') or extract_text_content(text)
        text_preview = text_content[:40] if len(text_content) > 40 else text_content

        if distance == float('inf'):
            print(f"    {i+1}. ❌ '{text_preview}' - 테이블 아래 또는 겹침")
            continue

        # 빈 텍스트만 제외
        if not text_content or len(text_content.strip()) == 0:
            print(f"    {i+1}. ❌ '{text_preview}' - 빈 텍스트")
            continue

        # 점수 계산
        score = score_title_candidate(text, text_content, distance)

        print(f"    {i+1}. ✅ '{text_preview}' - 후보 (거리: {distance:.0f}, 길이: {len(text_content)}, bbox: {text_bbox})")
        candidates.append({
            'text': text_content,
            'distance': distance,
            'score': score,
            'bbox': text_bbox
        })

    if not candidates:
        print("  ❌ 타이틀 후보 없음")
        return "", None

    # Step 2: 거리 순으로 정렬하고 상위 2개만 선택 (테이블에 가장 가까운 것들)
    candidates.sort(key=lambda x: x['distance'])  # 거리 오름차순 정렬
    top_candidates = candidates[:2]

    print(f"  거리 기반 상위 후보 {len(top_candidates)}개 선택 (전체 {len(candidates)}개 중):")
    for i, c in enumerate(top_candidates):
        text_preview = c['text'][:50] if len(c['text']) > 50 else c['text']
        print(f"    {i+1}. '{text_preview}' (거리: {c['distance']:.0f}, bbox: {c['bbox']})")

    # Step 3: ML 모델로 최종 선택 (후보가 1개 이상일 때)
    if USE_ML_MODEL and len(top_candidates) >= 1:
        print(f"\n  ML 모델에 {len(top_candidates)}개 후보 전달:")
        ml_result = select_best_title_ml(top_candidates)
        if ml_result:
            print(f"  ✅ ML 선택: '{ml_result['text']}'")
            return ml_result['text'], ml_result['bbox']
        else:
            print(f"  ❌ ML 임계값 미달 - 타이틀 없음")
            return "", None

    # Step 4: ML 사용 안 하는 경우에만 거리 기반 선택
    print("  ⚠️  ML 모델 미사용, 거리 기반 대체")
    best = top_candidates[0]
    print(f"  ✅ 거리 기반 선택: '{best['text']}' (점수: {best['score']:.1f}, bbox: {best['bbox']})")
    return best['text'], best['bbox']

@app.route('/get_title', methods=['POST'])
def get_title():
    """받은 데이터(tables, texts)에 각 테이블마다 title 프로퍼티를 추가해서 되돌려주는 API"""
    data = request.get_json()

    # 데이터가 딕셔너리인지 확인 (tables와 texts를 포함)
    if isinstance(data, dict):
        tables = data.get('tables', [])
        texts = data.get('texts', [])

        print(f"받은 테이블 수: {len(tables)}")
        print(f"받은 텍스트 수: {len(texts)}")

        # 첫 번째 테이블과 텍스트 샘플 출력
        if tables:
            print(f"첫 번째 테이블 bbox: {get_bbox_from_table(tables[0])}")
        if texts:
            print(f"첫 번째 텍스트: {texts[0]}")
            print(f"첫 번째 텍스트 bbox: {get_bbox_from_text(texts[0])}")

        # 각 테이블에 title 추가
        result_tables = []
        for idx, table in enumerate(tables):
            table_with_title = copy.deepcopy(table)
            title, title_bbox = find_title_for_table(table, texts)
            print(f"테이블 {idx} 타이틀: '{title}'")
            table_with_title['title'] = title
            table_with_title['title_bbox'] = title_bbox

            # title이 제대로 추가되었는지 확인
            if 'title' in table_with_title:
                print(f"  ✓ title 프로퍼티 추가 확인: '{table_with_title['title']}'")
                print(f"  ✓ title_bbox 프로퍼티 추가 확인: {table_with_title['title_bbox']}")
            else:
                print(f"  ✗ title 프로퍼티 추가 실패!")

            result_tables.append(table_with_title)

        print(f"\n최종 반환: {len(result_tables)}개 테이블")
        # 첫 번째 테이블의 키 확인
        if result_tables:
            print(f"첫 번째 테이블 키: {list(result_tables[0].keys())}")

        return jsonify(result_tables)

    # 하위 호환성: 배열만 오는 경우 (기존 방식)
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
    app.run(host='0.0.0.0', port=5000, debug=True)
