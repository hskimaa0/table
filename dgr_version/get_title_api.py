"""
표 제목/설명 추출 API - 간단 버전
"""
from flask import Flask, jsonify, request
import copy
import re

app = Flask(__name__)

# ========== 설정 ==========
X_TOLERANCE = 800  # 수평 근접 허용 거리 (px)

# 모델 설정
ZEROSHOT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
EMBEDDING_MODEL = "BAAI/bge-m3"
NSP_MODEL = "google-bert/bert-base-multilingual-cased"
USE_GPU = True  # GPU 사용 여부

# 모델 로드
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForNextSentencePrediction

device = 0 if USE_GPU and torch.cuda.is_available() else -1
device_str = "cuda" if device == 0 else "cpu"
print(f"▶ Device: {device_str}")

# Zero-shot 분류기 로드
try:
    zeroshot_classifier = pipeline(
        "zero-shot-classification",
        model=ZEROSHOT_MODEL,
        device=device
    )
    print(f"✅ Zero-shot 모델 로드 완료: {ZEROSHOT_MODEL}")
except Exception as e:
    print(f"⚠️ Zero-shot 모델 로드 실패: {e}")
    zeroshot_classifier = None

# Embedding 모델 로드
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device_str)
    print(f"✅ Embedding 모델 로드 완료: {EMBEDDING_MODEL}")
except Exception as e:
    print(f"⚠️ Embedding 모델 로드 실패: {e}")
    embedding_model = None

# NSP (Next Sentence Prediction) 모델 로드
try:
    nsp_tokenizer = AutoTokenizer.from_pretrained(NSP_MODEL)
    nsp_model = AutoModelForNextSentencePrediction.from_pretrained(NSP_MODEL).to(device_str)
    nsp_model.eval()
    print(f"✅ NSP 모델 로드 완료: {NSP_MODEL}")
except Exception as e:
    print(f"⚠️ NSP 모델 로드 실패: {e}")
    nsp_tokenizer = None
    nsp_model = None

# 형태소 분석기 로드 (kiwipiepy)
try:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    print(f"✅ 형태소 분석기 로드 완료: kiwipiepy")
except Exception as e:
    print(f"⚠️ 형태소 분석기 로드 실패: {e}")
    print(f"   pip install kiwipiepy 실행 필요")
    kiwi = None

# ========== 유틸리티 함수 ==========
def get_bbox_from_table(table):
    """table 객체에서 bbox 추출 [l, t, r, b]"""
    if 'bbox' in table:
        bbox = table['bbox']
        if isinstance(bbox, list) and len(bbox) >= 4:
            return bbox[:4]
        elif isinstance(bbox, dict):
            return [bbox.get('l', 0), bbox.get('t', 0), bbox.get('r', 0), bbox.get('b', 0)]
    return None

def horizontally_near(table_x, text_x, tol=X_TOLERANCE):
    """수평으로 겹치거나 근접한지 확인"""
    # IoU 계산
    overlap = max(0, min(table_x[1], text_x[1]) - max(table_x[0], text_x[0]))
    if overlap > 0:
        return True
    # 근접 확인
    return (text_x[1] >= table_x[0] - tol and text_x[0] <= table_x[1] + tol)

def is_trivial(text):
    """무의미한 텍스트 필터링"""
    s = text.strip()

    # 1~3자리 숫자만
    if re.fullmatch(r"\d{1,3}", s):
        return True

    # 너무 짧음
    if len(s) <= 2:
        return True

    # 숫자만
    if re.match(r'^[\d\s\.\-,]+$', s):
        return True

    # 특수문자만
    if len(re.sub(r"[\W_]+", "", s)) <= 1:
        return True

    return False

def group_texts_by_line(texts, y_tolerance=150):
    """texts를 y좌표 기준으로 같은 줄끼리 그룹화"""
    if not texts:
        return []

    print(f"  [디버그-원본] 전체 texts 개수: {len(texts)}")
    # 첫 3개 texts 객체의 구조 확인
    for i, text_obj in enumerate(texts[:3]):
        print(f"    texts[{i}] keys: {text_obj.keys()}")
        if 't' in text_obj and isinstance(text_obj['t'], list):
            print(f"      t 배열 길이: {len(text_obj['t'])}")
            if text_obj['t']:
                print(f"      t[0] keys: {text_obj['t'][0].keys()}")
                print(f"      t[0] tid: {text_obj['t'][0].get('tid', 'N/A')}")

    # 모든 개별 텍스트 항목 추출 (t 배열 안의 tid 사용)
    all_items = []
    for text_obj in texts:
        if 't' in text_obj and isinstance(text_obj['t'], list):
            for t_item in text_obj['t']:
                if 'text' in t_item and 'bbox' in t_item:
                    # t_item에 이미 tid가 있음
                    all_items.append(t_item.copy())

    if not all_items:
        return []

    # tid 순으로 정렬
    all_items.sort(key=lambda x: x.get('tid', 0))

    print(f"  [디버그-정렬후] 추출된 개별 텍스트 항목: {len(all_items)}")
    for i, item in enumerate(all_items[:20]):  # 첫 20개만 출력
        tid = item.get('tid', 'N/A')
        text = item.get('text', '')
        bbox = item.get('bbox', [])
        y_center = (bbox[1] + bbox[3]) / 2 if len(bbox) >= 4 else 0
        print(f"    {i+1}. tid={tid}, y={y_center:.1f}, text='{text[:30]}'")

    # 같은 줄끼리 그룹화
    lines = []
    current_line = []
    current_y_center = None

    for item in all_items:
        y_center = (item['bbox'][1] + item['bbox'][3]) / 2

        if current_y_center is None or abs(y_center - current_y_center) <= y_tolerance:
            current_line.append(item)
            if current_y_center is None:
                current_y_center = y_center
        else:
            if current_line:
                lines.append(current_line)
            current_line = [item]
            current_y_center = y_center

    if current_line:
        lines.append(current_line)

    # 각 줄을 병합
    merged_lines = []
    for line in lines:
        # x 좌표 순으로 정렬
        line.sort(key=lambda x: x['bbox'][0])

        # 텍스트 병합
        text_parts = [item['text'] for item in line]
        merged_text = ' '.join(text_parts)

        # bbox 병합
        merged_bbox = [
            min(item['bbox'][0] for item in line),  # l
            min(item['bbox'][1] for item in line),  # t
            max(item['bbox'][2] for item in line),  # r
            max(item['bbox'][3] for item in line)   # b
        ]

        merged_lines.append({
            'text': merged_text,
            'bbox': merged_bbox
        })

    print(f"  [디버그-줄병합] 병합된 줄 개수: {len(merged_lines)}")
    for i, line in enumerate(merged_lines[:10]):
        print(f"    줄{i+1}: '{line['text'][:50]}'")

    # NSP를 사용해서 연결된 줄 병합
    if nsp_model and nsp_tokenizer and len(merged_lines) > 1:
        print(f"  [디버그-NSP] NSP 기반 줄 병합 시작...")
        merged_lines = merge_lines_with_nsp(merged_lines)
        print(f"  [디버그-NSP] NSP 병합 후 줄 개수: {len(merged_lines)}")
        for i, line in enumerate(merged_lines[:10]):
            print(f"    줄{i+1}: '{line['text'][:50]}'")

    return merged_lines

def merge_lines_with_nsp(lines, nsp_threshold=0.95):
    """NSP를 사용해서 문맥적으로 연결된 줄들을 병합

    Args:
        lines: [{"text": "...", "bbox": [...]}, ...]
        nsp_threshold: NSP 확률 임계값 (이 값보다 높으면 연결)

    Returns:
        병합된 줄 리스트
    """
    if not lines or not nsp_model or not nsp_tokenizer:
        return lines

    merged = []
    i = 0

    while i < len(lines):
        current = lines[i].copy()
        merge_count = 0

        # 다음 줄과 연결 가능한지 확인
        while i + 1 < len(lines):
            next_line = lines[i + 1]

            # NSP 점수 계산
            try:
                inputs = nsp_tokenizer(
                    current['text'],
                    next_line['text'],
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                ).to(device_str)

                with torch.no_grad():
                    outputs = nsp_model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    # 0: IsNext, 1: NotNext
                    is_next_prob = probs[0][0].item()

                print(f"    NSP: '{current['text'][-20:]}' -> '{next_line['text'][:20]}' = {is_next_prob:.3f}")

                # 연결 가능하면 병합
                if is_next_prob >= nsp_threshold:
                    # 텍스트 병합
                    current['text'] = current['text'] + ' ' + next_line['text']
                    # bbox 병합
                    current['bbox'] = [
                        min(current['bbox'][0], next_line['bbox'][0]),
                        min(current['bbox'][1], next_line['bbox'][1]),
                        max(current['bbox'][2], next_line['bbox'][2]),
                        max(current['bbox'][3], next_line['bbox'][3])
                    ]
                    i += 1  # 다음 줄도 처리했으므로 건너뜀
                    merge_count += 1
                    print(f"      ✅ 병합됨 (누적: {merge_count}개)")
                else:
                    print(f"      ❌ 병합 안됨 (점수 부족)")
                    break

            except Exception as e:
                print(f"    NSP 오류: {e}")
                break

        merged.append(current)
        i += 1

    return merged

def collect_candidates_for_table(table, texts, all_tables=None):
    """표 위/아래 텍스트 후보 수집

    Args:
        table: 표 객체
        texts: [{"bbox": [...], "t": [{"text": "...", "bbox": [...]}, ...], "paraIndex": N}, ...]
        all_tables: 전체 표 리스트

    Returns:
        list: [{"text": "...", "bbox": [l, t, r, b]}, ...]
    """
    table_bbox = get_bbox_from_table(table)
    if not table_bbox:
        print("  ⚠️  표 bbox 없음")
        return []

    tbx1, tby1, tbx2, tby2 = table_bbox
    print(f"  [후보 수집] 표 bbox: {table_bbox}")

    # 다른 표들의 경계 찾기
    upper_limit = 0  # 표 위쪽 탐색 최대 범위
    lower_limit = float('inf')  # 표 아래쪽 탐색 최대 범위

    if all_tables:
        for other_table in all_tables:
            other_bbox = get_bbox_from_table(other_table)
            if not other_bbox or other_bbox == table_bbox:
                continue

            ox1, oy1, ox2, oy2 = other_bbox

            # 수평으로 겹치는 다른 표만 고려
            if not horizontally_near((tbx1, tbx2), (ox1, ox2), tol=X_TOLERANCE):
                continue

            # 현재 표 위에 있는 표
            if oy2 <= tby1:
                upper_limit = max(upper_limit, oy2)

            # 현재 표 아래에 있는 표
            elif oy1 >= tby2:
                lower_limit = min(lower_limit, oy1)

    print(f"  [후보 수집] 탐색 범위: 위쪽={upper_limit}~{tby1}, 아래쪽={tby2}~{lower_limit}")

    # 같은 줄끼리 병합
    merged_lines = group_texts_by_line(texts)
    print(f"  [디버그] 병합된 줄 개수: {len(merged_lines)}")

    # 후보 수집
    candidates = []
    skipped_horizontal = 0
    skipped_trivial = 0
    skipped_position = 0

    for line in merged_lines:
        merged_text = line['text'].strip()
        para_bbox = line['bbox']

        if not merged_text:
            continue

        px1, py1, px2, py2 = para_bbox

        # 수평으로 겹치거나 근접한지 확인
        if not horizontally_near((tbx1, tbx2), (px1, px2), tol=X_TOLERANCE):
            skipped_horizontal += 1
            if skipped_horizontal <= 3:  # 첫 3개만 출력
                print(f"  [스킵-수평] '{merged_text[:30]}' 표x:[{tbx1},{tbx2}] 텍스트x:[{px1},{px2}]")
            continue

        # 무의미한 텍스트 필터링
        if is_trivial(merged_text):
            skipped_trivial += 1
            continue

        # 표 위쪽에 있는지 확인
        if py2 <= tby1 and py1 >= upper_limit:
            candidates.append({
                'text': merged_text,
                'bbox': para_bbox,
                'position': 'above'
            })
        # 표 아래쪽에 있는지 확인
        elif py1 >= tby2 and py2 <= lower_limit:
            candidates.append({
                'text': merged_text,
                'bbox': para_bbox,
                'position': 'below'
            })
        else:
            skipped_position += 1
            if skipped_position <= 3:  # 첫 3개만 출력
                print(f"  [스킵-위치] '{merged_text[:30]}' 텍스트y:[{py1},{py2}] 표y:[{tby1},{tby2}]")

    print(f"  [스킵 통계] 수평:{skipped_horizontal} 무의미:{skipped_trivial} 위치:{skipped_position}")

    # 중복 제거
    unique = {}
    for c in candidates:
        unique.setdefault(c['text'], c)

    result = list(unique.values())

    print(f"  [후보 수집] 최종 후보 개수: {len(result)}")
    for i, cand in enumerate(result[:5]):  # 첫 5개만 출력
        pos = "위" if cand['position'] == 'above' else "아래"
        print(f"    {i+1}. [{pos}] '{cand['text']}'")

    return result

# ========== 표 내용 추출 ==========
def extract_table_text(table):
    """표 안의 모든 텍스트 추출"""
    all_texts = []

    if 'rows' not in table:
        return ""

    for row in table['rows']:
        if not isinstance(row, list):
            continue
        for cell in row:
            if 'texts' in cell and isinstance(cell['texts'], list):
                for text_obj in cell['texts']:
                    if 'v' in text_obj:
                        all_texts.append(text_obj['v'])

    return ' '.join(all_texts)

# ========== Embedding 유사도 계산 ==========
def compute_embedding_similarity(table_text, candidate_text):
    """Embedding 유사도 계산 (표 내용과 후보 텍스트의 의미적 유사도)

    Returns:
        float: 코사인 유사도 (0~1)
    """
    if not embedding_model or not table_text or not candidate_text:
        return 0.5

    try:
        import numpy as np
        embeddings = embedding_model.encode([table_text, candidate_text])

        # 코사인 유사도
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return float(similarity)
    except Exception as e:
        print(f"  ⚠️ Embedding 오류: {e}")
        return 0.5

# ========== Zero-shot + Embedding 판단 로직 ==========
def score_candidates_with_models(candidates, table_text):
    """Zero-shot + Embedding으로 각 후보 점수 계산

    Args:
        candidates: [{"text": "...", "bbox": [...], "position": "above/below"}, ...]
        table_text: 표 내용 텍스트

    Returns:
        list: [{"text": "...", "bbox": [...], "position": "...", "zeroshot_score": 0.X, "embedding_score": 0.X, "final_score": 0.X}, ...]
    """
    if not candidates:
        print("  ⚠️ 후보 없음")
        return []

    # 표 내용 요약 (처음 300자만 사용)
    table_text_short = table_text[:300] if len(table_text) > 300 else table_text

    # Zero-shot 라벨
    labels = ["표 제목", "관련 없음"]

    results = []

    for cand in candidates:
        txt = cand['text']
        bbox = cand['bbox']
        position = cand['position']

        # 1. Zero-shot 점수
        zeroshot_score = 0.5
        if zeroshot_classifier:
            try:
                # 가설: "이 텍스트는 다음 표의 {label}이다: {표내용}"
                hypothesis_template = f"이 텍스트는 다음 표의 {{}}이다: {table_text_short}"

                result = zeroshot_classifier(
                    txt,
                    candidate_labels=labels,
                    hypothesis_template=hypothesis_template,
                    multi_label=False
                )

                # "표 제목" 라벨의 점수
                title_idx = result['labels'].index("표 제목")
                zeroshot_score = result['scores'][title_idx]

            except Exception as e:
                print(f"  ⚠️ Zero-shot 오류: {e}")

        # 2. Embedding 유사도 (전체 표 내용 사용)
        embedding_score = compute_embedding_similarity(table_text, txt)

        # 3. 최종 점수: Zero-shot 60% + Embedding 40%
        final_score = zeroshot_score * 0.6 + embedding_score * 0.4

        results.append({
            'text': txt,
            'bbox': bbox,
            'position': position,
            'zeroshot_score': zeroshot_score,
            'embedding_score': embedding_score,
            'final_score': final_score
        })

    # 최종 점수 순으로 정렬
    results.sort(key=lambda x: -x['final_score'])

    return results

# ========== 제목 판단 로직 ==========
def is_table_title_pattern(text):
    """표 제목 패턴 판별: '표 X.X ...', '표 A.10 ...' 등"""
    if re.search(r"^표\s*[A-Za-z]?\s*[\.\-]?\s*\d+([\-\.]\d+)?", text.strip()):
        return True
    return False

def is_source_pattern(text):
    """출처/자료 패턴: '출처:', '자료:', '주:' 등"""
    t = text.strip()
    if re.search(r"(출처|자료|근거|제공|주)\s*[:：]", t):
        return True
    if t.startswith("※"):
        return True
    return False

def is_noun_phrase(text):
    """명사형 종결인지 판별 (형태소 분석기 사용)

    Returns:
        True: 명사형으로 끝남 (제목 가능)
        False: 동사/형용사로 끝남 (제목 불가)
    """
    if not kiwi:
        # 형태소 분석기 없으면 간단한 규칙 사용
        t = text.strip()
        # 동사 어미로 끝나면 False
        if re.search(r"(한다|된다|있다|없다|였다|했다|나타나|비교해|보면|따르면|의하면|하고|되고)$", t):
            return False
        return True

    try:
        t = text.strip().rstrip(".")

        # 형태소 분석
        result = kiwi.analyze(t)
        if not result or not result[0] or not result[0][0]:
            return True  # 분석 실패시 허용

        tokens = result[0][0]

        # 마지막 토큰의 품사 확인
        if not tokens:
            return True

        last_token = tokens[-1]
        last_tag = last_token.tag

        # 명사형 품사
        noun_tags = ['NNG', 'NNP', 'NNB', 'NR', 'SL', 'SH', 'SN']
        # 동사/형용사 품사
        verb_tags = ['VV', 'VA', 'VX', 'VCP', 'VCN']
        # 어미 품사
        ending_tags = ['EC', 'EF', 'ETN', 'ETM']
        # 조사 품사 (제목으로 부적합)
        josa_tags = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC']

        # 기호는 무시하고 앞 토큰 확인
        if last_tag in ['SF', 'SP', 'SS', 'SE', 'SO', 'SW']:
            if len(tokens) >= 2:
                last_token = tokens[-2]
                last_tag = last_token.tag

        # 조사로 끝나면 False (예: "표에", "하천의")
        if last_tag in josa_tags:
            return False

        # 명사형이면 True
        if last_tag in noun_tags:
            return True

        # 동사/형용사/어미로 끝나면 False
        if last_tag in verb_tags or last_tag in ending_tags:
            return False

        # 기타는 허용
        return True

    except Exception as e:
        print(f"  ⚠️ 형태소 분석 오류: {e}")
        return True  # 오류시 허용

def compute_distance_score(text_bbox, table_bbox):
    """텍스트와 표 사이의 거리 점수 계산 (0~1, 가까울수록 높음)"""
    tx1, ty1, tx2, ty2 = text_bbox
    tbx1, tby1, tbx2, tby2 = table_bbox

    # 표 위쪽에 있는 경우
    if ty2 <= tby1:
        distance = tby1 - ty2
        # 0~100px: 1.0~0.9, 100~200px: 0.9~0.7, 200px+: 0.7~0
        if distance <= 100:
            return 1.0 - (distance / 100) * 0.1  # 0.9~1.0
        elif distance <= 200:
            return 0.9 - ((distance - 100) / 100) * 0.2  # 0.7~0.9
        else:
            return max(0.0, 0.7 - ((distance - 200) / 200) * 0.7)  # 0~0.7
    # 표 아래쪽에 있는 경우
    elif ty1 >= tby2:
        return 0.2  # 기본적으로 낮은 점수
    else:
        return 0.0  # 표와 겹침

def find_title_from_candidates(candidates, table_bbox, table):
    """Zero-shot + Embedding으로 후보 중에서 제목 선택

    Returns:
        tuple: (title, title_bbox, descriptions, desc_bboxes)
    """
    if not candidates:
        return "", None, [], []

    # 표 내용 추출
    table_text = extract_table_text(table)
    print(f"  [표 내용] {len(table_text)}자: '{table_text}'")

    # Zero-shot + Embedding으로 점수 계산
    scored = score_candidates_with_models(candidates, table_text)

    if not scored:
        return "", None, [], []

    # 디버깅 출력
    print(f"\n  [모델 판단 결과]")
    for i, item in enumerate(scored[:5]):
        pos = "위" if item['position'] == 'above' else "아래"
        is_noun = is_noun_phrase(item['text'])
        noun_str = "명사○" if is_noun else "동사✗"
        print(f"    {i+1}. [{pos}] Zero:{item['zeroshot_score']:.3f} Embed:{item['embedding_score']:.3f} 최종:{item['final_score']:.3f} [{noun_str}] '{item['text']}'")

    # 명사형 종결인 후보 중에서 최고 점수 선택
    best = None
    for item in scored:
        if is_noun_phrase(item['text']):
            best = item
            break

    if not best:
        # 명사형이 없으면 최고 점수 선택
        best = scored[0]
        print(f"  ⚠️ 명사형 후보 없음, 최고 점수 선택")

    title = best['text']
    title_bbox = best['bbox']

    # 나머지는 설명 후보 (출처 패턴이면 설명으로)
    descriptions = []
    desc_bboxes = []

    for item in scored[1:]:
        if is_source_pattern(item['text']):
            descriptions.append(item['text'])
            desc_bboxes.append(item['bbox'])

    return title, title_bbox, descriptions, desc_bboxes

# ========== API 엔드포인트 ==========
@app.route('/get_title', methods=['POST'])
def get_title():
    """표 제목/설명 추출 API"""
    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({'error': 'Data must be an object with tables and texts'}), 400

    tables = data.get('tables', [])
    texts = data.get('texts', [])

    print(f"\n{'='*60}")
    print(f"받은 표 개수: {len(tables)}")
    print(f"받은 텍스트 개수: {len(texts)}")
    print(f"{'='*60}\n")

    result_tables = []

    for idx, table in enumerate(tables):
        print(f"\n[표 {idx+1}/{len(tables)}] 처리 중...")

        table_with_title = copy.deepcopy(table)

        # 후보 수집
        candidates = collect_candidates_for_table(table, texts, all_tables=tables)

        # 제목 판단 (Zero-shot)
        table_bbox = get_bbox_from_table(table)
        title, title_bbox, descriptions, desc_bboxes = find_title_from_candidates(candidates, table_bbox, table)

        table_with_title['title'] = title
        table_with_title['title_bbox'] = title_bbox
        table_with_title['descriptions'] = descriptions
        table_with_title['description_bboxes'] = desc_bboxes

        if title:
            print(f"  ✅ 제목: '{title}'")
        else:
            print(f"  ⚠️  제목 없음")

        result_tables.append(table_with_title)

    return jsonify(result_tables)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)
