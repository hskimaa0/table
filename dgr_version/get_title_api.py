"""
타이틀 추출 API
규칙 기반 필터링 + Zero-shot Classification
"""
from flask import Flask, jsonify, request
import copy
import re

app = Flask(__name__)

# ========== 상수 정의 ==========
# 거리 및 필터링 관련
MAX_DISTANCE_THRESHOLD = 10000  # 타이틀 후보로 고려할 최대 거리 (px)
MAX_TEXT_LENGTH = 150  # 타이틀로 고려할 최대 텍스트 길이
Y_LINE_TOLERANCE = 100  # 같은 줄로 간주할 y 좌표 허용 오차 (px)
X_GAP_THRESHOLD = 100  # 텍스트 사이 공백 추가 기준 간격 (px)
HORIZONTAL_WEIGHT = 0.1  # 수평 거리 가중치 (수직 거리 대비)

# ML 모델 관련
ZERO_SHOT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # Zero-shot classification 모델 (다국어)
ML_DEVICE = -1  # -1: CPU, 0: GPU
MAX_TEXT_INPUT_LENGTH = 512  # ML 모델 입력 최대 길이
TOP_CANDIDATES_COUNT = 5  # ML 모델에 전달할 상위 후보 개수
ML_CANDIDATE_LABELS = ["table title", "not a title"]  # 단순 양/음 라벨

ML_HYPOTHESIS_TEMPLATES = [
    "This text is a {}.",
    "이 텍스트는 {}이다."
]

# ML 모델 로드
classifier = None

# Zero-shot Classification 모델 로드
try:
    from transformers import pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model=ZERO_SHOT_MODEL,
        device=ML_DEVICE
    )
    print(f"✅ Zero-shot Classification 모델 로드 완료 ({ZERO_SHOT_MODEL})")
except ImportError:
    print("⚠️  transformers 라이브러리 없음")
    classifier = None
except Exception as e:
    print(f"⚠️  Zero-shot 모델 로드 실패: {e}")
    classifier = None

def get_bbox_from_text(text):
    """text 객체에서 bbox 정보 추출 [l, t, r, b] 형식으로 반환"""
    # rect가 있는 경우 (배열 형태: [l, t, r, b])
    if 'rect' in text and isinstance(text['rect'], list) and len(text['rect']) >= 4:
        return text['rect']
    # bbox가 있는 경우 (배열 형태: [l, t, r, b])
    elif 'bbox' in text and isinstance(text['bbox'], list) and len(text['bbox']) >= 4:
        return text['bbox']
    # bbox가 있는 경우 (객체 형태: {l, t, r, b})
    elif 'bbox' in text and isinstance(text['bbox'], dict):
        bbox = text['bbox']
        return [bbox.get('l', 0), bbox.get('t', 0), bbox.get('r', 0), bbox.get('b', 0)]
    # merged_bbox가 있는 경우 (병합된 텍스트)
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
    """text 객체에서 실제 텍스트 추출 (tid 순서대로)"""
    # merged_text가 있는 경우 (이미 병합된 텍스트)
    if 'merged_text' in text_obj:
        return text_obj['merged_text']

    # 't' 배열 내부의 'text' 속성 (주어진 데이터 형식)
    if 't' in text_obj and isinstance(text_obj['t'], list) and len(text_obj['t']) > 0:
        # 't' 배열을 tid 순서대로 정렬 (원본 문서 순서)
        t_items = text_obj['t']

        # tid로 정렬
        sorted_items = sorted(t_items, key=lambda item: item.get('tid', 0))

        # 정렬된 순서대로 텍스트 추출
        texts = []
        for t_item in sorted_items:
            if 'text' in t_item:
                texts.append(t_item['text'])

        if texts:
            return ''.join(texts)  # 공백 없이 붙임 (원본 텍스트 그대로)

    # 'text' 속성에 텍스트가 있는 경우 (flatten된 개별 텍스트)
    if 'text' in text_obj:
        return text_obj['text']
    # 'v' 속성에 텍스트가 있는 경우
    elif 'v' in text_obj:
        return text_obj['v']
    return ""

def flatten_text_objects(texts):
    """
    paraIndex로 이미 그룹화된 텍스트를 개별 't' 요소로 분해
    각 't' 요소를 독립적인 텍스트 객체로 변환
    """
    flattened = []
    for text_obj in texts:
        if 't' in text_obj and isinstance(text_obj['t'], list):
            # 't' 배열의 각 항목을 개별 텍스트로 분리
            for t_item in text_obj['t']:
                if 'bbox' in t_item and 'text' in t_item:
                    flattened.append({
                        'bbox': t_item['bbox'],
                        'text': t_item['text'],
                        'tid': t_item.get('tid', 0)
                    })
        else:
            # 't' 배열이 없으면 원본 그대로 사용
            flattened.append(text_obj)
    return flattened

def group_texts_by_line(texts, y_tolerance=50):
    """같은 줄(y 좌표 비슷)에 있는 텍스트들을 그룹화하고 위에서 아래 순서로 정렬"""
    if not texts:
        return []

    # Step 1: paraIndex로 묶인 텍스트를 개별 't' 요소로 분해
    flattened = flatten_text_objects(texts)

    # Step 2: y 좌표 기준으로 정렬 (위에서 아래로)
    sorted_texts = sorted(flattened, key=lambda t: get_bbox_from_text(t)[1] if get_bbox_from_text(t) else float('inf'))

    grouped = []
    current_group = []
    group_y_min = None  # 그룹의 첫 번째 텍스트 y 중심값

    for text in sorted_texts:
        bbox = get_bbox_from_text(text)
        if not bbox:
            continue

        # y 좌표 중심값 사용 (top + bottom) / 2
        y_center = (bbox[1] + bbox[3]) / 2

        # 첫 텍스트이거나 그룹의 첫 텍스트와의 차이가 허용범위 내면 같은 그룹
        if group_y_min is None or abs(y_center - group_y_min) <= y_tolerance:
            current_group.append(text)
            if group_y_min is None:
                group_y_min = y_center
        else:
            # 새로운 줄 시작 - 이전 그룹 저장
            if current_group:
                merged = merge_text_group(current_group)
                if merged:
                    grouped.append(merged)
            current_group = [text]
            group_y_min = y_center

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

def is_valid_title_candidate(text_content):
    """텍스트가 타이틀 후보로 유효한지 확인"""
    if not text_content or len(text_content.strip()) == 0:
        return False, "빈 텍스트"

    text = text_content.strip()

    # 1. 단일 문자/숫자 제외
    if len(text) <= 2:
        return False, "너무 짧음 (2자 이하)"

    # 2. 숫자만 있는 경우 제외
    if re.match(r'^[\d\s\.\-]+$', text):
        return False, "숫자만 포함"

    # 3. IP 주소 패턴 제외
    if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', text):
        return False, "IP 주소"

    # 4. 단위 표기 제외
    if re.search(r'^\s*\(?단위\s*[:：]', text, re.IGNORECASE):
        return False, "단위 표기"

    # 5. "다." 단독으로 끝나는 경우 제외
    if re.search(r'다\.\s*$', text):
        return False, "'다.'로 끝나는 문장"

    # 6. 동사/형용사 어미로 끝나는 문장 제외
    if re.search(r'([가-힣]+(다|며|고|자|라|ㄴ다|는다|ㄹ다|ㅂ니다|습니다|이다|한다|된다|있다|없다|같다|왔다|났다|하였다|되었다|였다|았다|었다))[\.?!]?\s*$', text):
        return False, "동사로 끝나는 문장"

    # 7. 너무 긴 텍스트 제외
    if len(text) > 100:
        return False, f"너무 긴 텍스트 ({len(text)}자)"

    # 8. 특수문자만 있거나 특수문자로 시작하는 경우
    if re.match(r'^[^\w가-힣]', text):
        return False, "특수문자로 시작"

    return True, "후보"

def select_best_title_ml(candidates):
    """Zero-shot Classification으로 최적의 타이틀 선택 (다국어 가설)"""
    if not candidates or not classifier:
        print("\n  ML 모델 없음, 거리 기반 선택")
        return None

    try:
        # 텍스트 전처리: 너무 긴 텍스트는 잘라서 처리
        candidate_texts = [
            c['text'][:MAX_TEXT_INPUT_LENGTH] if len(c['text']) > MAX_TEXT_INPUT_LENGTH else c['text']
            for c in candidates
        ]

        best_idx, best_score = None, -1.0
        all_scores = {i: [] for i in range(len(candidates))}  # 각 후보의 모든 가설 점수 저장

        print("  Zero-shot Classification 점수:")

        # 각 가설 템플릿으로 점수 계산
        for hyp_idx, hyp in enumerate(ML_HYPOTHESIS_TEMPLATES):
            hyp_name = "영어" if hyp_idx == 0 else "한국어"
            print(f"\n  [{hyp_name} 가설] '{hyp}'")

            results = classifier(
                candidate_texts,
                candidate_labels=ML_CANDIDATE_LABELS,
                hypothesis_template=hyp,
                truncation=True
            )

            if not isinstance(results, list):
                results = [results]

            # "table title" 라벨의 점수 추출
            for i, r in enumerate(results):
                try:
                    idx = r['labels'].index(ML_CANDIDATE_LABELS[0])
                    score = float(r['scores'][idx])
                except Exception:
                    score = 0.0

                all_scores[i].append(score)
                text_preview = candidate_texts[i][:40] if len(candidate_texts[i]) > 40 else candidate_texts[i]
                print(f"    {i+1}. '{text_preview}' → {score:.3f}")

                # 현재까지의 최고 점수 업데이트 (동점이면 거리가 가까운 것, 그다음 짧은 것 우선)
                tie_break = (-candidates[i]['distance'], -len(candidate_texts[i]))
                if best_idx is None:
                    best_idx, best_score = i, score
                else:
                    current_tie_break = (-candidates[best_idx]['distance'], -len(candidate_texts[best_idx]))
                    if (score, *tie_break) > (best_score, *current_tie_break):
                        best_idx, best_score = i, score

        # 평균 점수 계산 및 출력
        print(f"\n  [평균 점수]")
        avg_scores = []
        for i in range(len(candidates)):
            avg = sum(all_scores[i]) / len(all_scores[i]) if all_scores[i] else 0.0
            avg_scores.append(avg)
            text_preview = candidate_texts[i][:40] if len(candidate_texts[i]) > 40 else candidate_texts[i]
            print(f"    {i+1}. '{text_preview}' → {avg:.3f} (거리: {candidates[i]['distance']:.0f})")

        # 평균 점수로 최종 선택 (동점이면 거리가 가까운 것 우선)
        best_idx = max(range(len(avg_scores)), key=lambda i: (avg_scores[i], -candidates[i]['distance'], -len(candidate_texts[i])))
        best_score = avg_scores[best_idx]

        print(f"  최종 선택: 후보 {best_idx+1}, 평균 점수: {best_score:.3f}, 거리: {candidates[best_idx]['distance']:.0f}")

        return candidates[best_idx] if best_idx is not None else None

    except Exception as e:
        print(f"  ML 모델 오류: {e}")
        import traceback
        traceback.print_exc()

    return None

def is_bbox_overlapping(text_bbox, table_bbox):
    """두 bbox가 겹치는지 확인 (겹치면 True)"""
    if not text_bbox or not table_bbox:
        return False

    # bbox 형식: [l, t, r, b]
    # 겹치지 않는 조건: text가 table의 완전히 왼쪽/오른쪽/위/아래에 있음
    # - text가 table 왼쪽: text_r <= table_l
    # - text가 table 오른쪽: text_l >= table_r
    # - text가 table 위: text_b <= table_t
    # - text가 table 아래: text_t >= table_b

    text_l, text_t, text_r, text_b = text_bbox
    table_l, table_t, table_r, table_b = table_bbox

    # 겹치지 않으면 False, 겹치면 True
    if text_r <= table_l or text_l >= table_r or text_b <= table_t or text_t >= table_b:
        return False
    return True

def find_title_for_table(table, texts, all_tables=None):
    """Zero-shot Classification으로 table의 title 찾기"""
    table_bbox = get_bbox_from_table(table)
    if not table_bbox:
        print("  테이블 bbox 없음")
        return "", None

    print(f"  테이블 bbox: y={table_bbox[1]}")

    # all_tables가 없으면 현재 테이블만 포함
    if all_tables is None:
        all_tables = [table]

    # Step 1: 테이블 내부에 있는 텍스트 필터링
    filtered_texts = []
    for text in texts:
        text_bbox = get_bbox_from_text(text)
        if not text_bbox:
            continue

        # 모든 테이블과 겹치는지 체크
        overlaps_any = any(
            is_bbox_overlapping(text_bbox, get_bbox_from_table(tbl))
            for tbl in all_tables
            if get_bbox_from_table(tbl)
        )

        if not overlaps_any:
            filtered_texts.append(text)

    print(f"  원본 텍스트: {len(texts)}개 → 테이블 외부: {len(filtered_texts)}개")

    # Step 2: 같은 줄의 텍스트들을 그룹화
    grouped_texts = group_texts_by_line(filtered_texts, y_tolerance=Y_LINE_TOLERANCE)
    print(f"  그룹화: {len(grouped_texts)}개")

    # 그룹화된 텍스트 샘플 출력
    print("  그룹화된 텍스트 샘플 (위에서 아래 순서):")
    for i, gt in enumerate(grouped_texts[:5]):
        if gt:
            text_preview = (gt.get('merged_text') or extract_text_content(gt))[:40]
            bbox = gt.get('merged_bbox') or get_bbox_from_text(gt)
            print(f"    {i+1}. y={bbox[1]}: '{text_preview}'")

    # Step 3: 테이블 위쪽 텍스트 후보 수집
    candidates = []

    print("\n  필터링 상세 로그:")
    for i, text in enumerate(grouped_texts):
        if not text:
            continue

        text_bbox = text.get('merged_bbox') or get_bbox_from_text(text)
        if not text_bbox:
            continue

        text_content = text.get('merged_text') or extract_text_content(text)
        text_preview = text_content[:40] if len(text_content) > 40 else text_content

        # 거리 계산
        distance = calculate_distance(text_bbox, table_bbox)

        if distance == float('inf'):
            print(f"    {i+1}. ❌ '{text_preview}' - 테이블 아래")
            continue

        # 유효성 검사
        is_valid, reason = is_valid_title_candidate(text_content)
        if not is_valid:
            print(f"    {i+1}. ❌ '{text_preview}' - {reason}")
            continue

        print(f"    {i+1}. ✅ '{text_preview}' - 후보 (거리: {distance:.0f}, 길이: {len(text_content)}, bbox: {text_bbox})")
        candidates.append({
            'text': text_content,
            'distance': distance,
            'bbox': text_bbox
        })

    if not candidates:
        print("  ❌ 타이틀 후보 없음")
        return "", None

    # Step 4: 거리 순으로 정렬하고 상위 N개 선택
    candidates.sort(key=lambda x: x['distance'])
    top_candidates = candidates[:TOP_CANDIDATES_COUNT]

    print(f"  거리 기반 상위 후보 {len(top_candidates)}개 선택 (전체 {len(candidates)}개 중):")
    for i, c in enumerate(top_candidates):
        text_preview = c['text'][:50] if len(c['text']) > 50 else c['text']
        print(f"    {i+1}. '{text_preview}' (거리: {c['distance']:.0f}, bbox: {c['bbox']})")

    # Step 5: Zero-shot Classification으로 최종 선택
    if classifier and len(top_candidates) >= 1:
        print(f"\n  Zero-shot Classification에 {len(top_candidates)}개 후보 전달:")
        ml_result = select_best_title_ml(top_candidates)
        if ml_result:
            print(f"  ✅ 선택: '{ml_result['text']}'")
            return ml_result['text'], ml_result['bbox']

    # Step 6: ML 미사용 시 거리 기반 선택
    print("  ⚠️  ML 모델 미사용, 거리 기반 선택")
    best = top_candidates[0]
    print(f"  ✅ 선택: '{best['text']}' (bbox: {best['bbox']})")
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
            title, title_bbox = find_title_for_table(table, texts, all_tables=tables)
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
    app.run(host='0.0.0.0', port=5555, debug=True)
