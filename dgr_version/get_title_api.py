"""
하이브리드 타이틀 추출 API
규칙 기반 필터링 + ML 모델 선택
"""
from flask import Flask, jsonify, request
import copy
import re

app = Flask(__name__)

# ML 모델 로드 (Zero-shot Classification)
USE_ML_MODEL = False
classifier = None

try:
    from transformers import pipeline
    # Zero-shot classification 모델 로드 (다국어 지원, 속도와 정확도 균형)
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",  # 280MB, 다국어 최적
        device=-1  # CPU 사용 (GPU 사용하려면 0)
    )
    USE_ML_MODEL = True
    print("✅ Zero-shot Classification 모델 로드 완료 (mDeBERTa-v3, 다국어)")
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
    return vertical_distance + (horizontal_distance * 0.1)

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
            # 간격이 크면 공백 추가 (100 픽셀 이상)
            if gap > 100:
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

    # 1. 너무 멀리 있는 텍스트 제외 (10000px 이상)
    if distance > 10000:
        return False, "너무 멀리 떨어짐"

    # 2. 너무 긴 텍스트 제외 (150자 이상은 보통 본문)
    if len(text) > 150:
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
    # 거리 0 = 100점, 거리가 멀어질수록 점수 감소
    score = max(0, 100 - (distance / 50))

    return score

def select_best_title_ml(candidates):
    """Zero-shot Classification으로 최적의 타이틀 선택"""
    if not USE_ML_MODEL or not classifier or not candidates:
        return None

    try:
        candidate_texts = [c['text'] for c in candidates]

        print("  Zero-shot Classification 점수:")

        best_idx = -1
        best_score = -1

        for i, text in enumerate(candidate_texts):
            # 너무 긴 텍스트는 잘라서 처리 (모델 제한)
            text_input = text[:512] if len(text) > 512 else text

            result = classifier(
                text_input,
                candidate_labels=["테이블 제목", "일반 텍스트"],
                hypothesis_template="이 텍스트는 {}이다."
            )

            # "테이블 제목" 라벨의 확률
            title_score = result['scores'][result['labels'].index("테이블 제목")]

            text_preview = text[:40] if len(text) > 40 else text
            print(f"    {i+1}. '{text_preview}' → 제목 확률: {title_score:.3f}")

            if title_score > best_score:
                best_score = title_score
                best_idx = i

        print(f"  ML 최고 확률: {best_score:.3f} (임계값: 0.5)")

        # 임계값: 0.5 이상이어야 제목으로 판단
        if best_idx >= 0 and best_score > 0.5:
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
        return ""

    print(f"  테이블 bbox: y={table_bbox[1]}")

    # Step 0: 같은 줄의 텍스트들을 그룹화
    grouped_texts = group_texts_by_line(texts, y_tolerance=50)
    print(f"  원본 텍스트: {len(texts)}개 → 그룹화: {len(grouped_texts)}개")

    # 그룹화된 텍스트 처음 5개 출력
    print("  그룹화된 텍스트 샘플 (위에서 아래 순서):")
    for i, gt in enumerate(grouped_texts[:5]):
        if gt:
            text_preview = (gt.get('merged_text') or extract_text_content(gt))[:40]
            bbox = gt.get('merged_bbox') or get_bbox_from_text(gt)
            print(f"    {i+1}. y={bbox[1]}: '{text_preview}'")

    # Step 1: 거리 기반 후보 수집
    candidates = []
    filtered_out = []

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
            filtered_out.append(f"'{text_content[:30]}' - 테이블 아래 또는 겹침")
            continue

        # 규칙 기반 필터링
        is_candidate, reason = is_title_candidate(text_content, distance)
        if not is_candidate:
            print(f"    {i+1}. ❌ '{text_preview}' - {reason} (거리: {distance:.0f}, 길이: {len(text_content)})")
            filtered_out.append(f"'{text_content[:30]}' - {reason}")
            continue

        # 점수 계산
        score = score_title_candidate(text, text_content, distance)

        print(f"    {i+1}. ✅ '{text_preview}' - 후보 (거리: {distance:.0f}, 길이: {len(text_content)})")
        candidates.append({
            'text': text_content,
            'distance': distance,
            'score': score,
            'bbox': text_bbox
        })

    if not candidates:
        print("  ❌ 타이틀 후보 없음")
        # 제외된 텍스트 샘플 출력
        if filtered_out:
            print("  제외된 텍스트 샘플:")
            for reason in filtered_out[:5]:
                print(f"    - {reason}")
        return ""

    # Step 2: 거리 순으로 정렬하고 상위 3개만 선택 (거리가 가장 가까운 것들)
    candidates.sort(key=lambda x: x['distance'])  # 거리 오름차순 정렬
    top_candidates = candidates[:3]  # 가장 가까운 3개만

    print(f"  거리 기반 상위 후보 {len(top_candidates)}개 선택 (전체 {len(candidates)}개 중):")
    for i, c in enumerate(top_candidates):
        text_preview = c['text'][:50] if len(c['text']) > 50 else c['text']
        print(f"    {i+1}. '{text_preview}' (거리: {c['distance']:.0f})")

    # Step 3: ML 모델로 최종 선택 (후보가 2개 이상일 때만)
    if USE_ML_MODEL and len(top_candidates) > 1:
        print(f"\n  ML 모델에 {len(top_candidates)}개 후보 전달:")
        ml_result = select_best_title_ml(top_candidates)
        if ml_result:
            print(f"  ✅ ML 선택: '{ml_result['text']}'")
            return ml_result['text']

    # Step 4: 거리 기반 최종 선택 (점수 최고)
    best = top_candidates[0]
    print(f"  ✅ 거리 기반 선택: '{best['text']}' (점수: {best['score']:.1f})")
    return best['text']

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
            title = find_title_for_table(table, texts)
            print(f"테이블 {idx} 타이틀: '{title}'")
            table_with_title['title'] = title

            # title이 제대로 추가되었는지 확인
            if 'title' in table_with_title:
                print(f"  ✓ title 프로퍼티 추가 확인: '{table_with_title['title']}'")
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
