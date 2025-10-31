"""
페이지를 넘어가는 테이블을 찾아서 병합하는 스크립트
텍스트 연결성을 기반으로 같은 테이블로 판단
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import re
# from difflib import SequenceMatcher  # 더 이상 사용하지 않음 (sentence-transformers로 대체)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# 타이틀 분류기: ML 모델 사용 (transformers 필요)
# fallback으로 규칙 기반 시스템도 포함
try:
    from title_classifier_ml import select_best_title_with_model
    USE_ML_CLASSIFIER = True
except ImportError:
    print("경고: transformers 라이브러리가 없어 규칙 기반 타이틀 선택 사용")
    from title_classifier import select_best_title
    USE_ML_CLASSIFIER = False

# 타이틀 유사도 비교용 sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _similarity_model = None

    def get_similarity_model():
        """전역 sentence-transformer 모델 (lazy loading)"""
        global _similarity_model
        if _similarity_model is None:
            print("  📊 타이틀 유사도 모델 로딩 중... (최초 1회)", flush=True)
            _similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # 다국어 지원, 50MB
            print("  ✅ 유사도 모델 로드 완료", flush=True)
        return _similarity_model

    USE_SIMILARITY_MODEL = True
except ImportError:
    print("경고: sentence-transformers 없음, 기본 유사도 사용")
    USE_SIMILARITY_MODEL = False


def register_korean_font():
    """한글 폰트 등록"""
    try:
        # Windows 기본 폰트 사용
        pdfmetrics.registerFont(TTFont('Korean', 'malgun.ttf'))
        return 'Korean'
    except:
        try:
            pdfmetrics.registerFont(TTFont('Korean', 'C:/Windows/Fonts/malgun.ttf'))
            return 'Korean'
        except:
            print("Warning: Could not load Korean font. Using default font.")
            return 'Helvetica'


def is_point_in_bbox(x: float, y: float, bbox: Dict) -> bool:
    """점이 bbox 내부에 있는지 확인"""
    if not bbox:
        return False
    left = bbox.get('l', 0)
    right = bbox.get('r', 0)
    top = bbox.get('t', 0)
    bottom = bbox.get('b', 0)
    coord_origin = bbox.get('coord_origin', 'BOTTOMLEFT')

    if coord_origin == 'BOTTOMLEFT':
        # BOTTOMLEFT: y가 위로 갈수록 증가
        return left <= x <= right and bottom <= y <= top
    else:  # TOPLEFT
        # TOPLEFT: y가 아래로 갈수록 증가
        return left <= x <= right and top <= y <= bottom


def is_text_in_any_table(text_bbox: Dict, all_tables: List[Dict], exclude_table: Dict) -> bool:
    """텍스트가 다른 테이블 안에 있는지 확인"""
    if not text_bbox:
        return False

    # 텍스트의 중심점 계산
    text_center_x = (text_bbox.get('l', 0) + text_bbox.get('r', 0)) / 2
    text_center_y = (text_bbox.get('t', 0) + text_bbox.get('b', 0)) / 2

    exclude_bbox = exclude_table.get('prov', [{}])[0].get('bbox', {})

    for table in all_tables:
        table_bbox = table.get('prov', [{}])[0].get('bbox', {})

        # 현재 테이블 제외
        if table_bbox == exclude_bbox:
            continue

        # 텍스트 중심점이 테이블 내부에 있는지 확인
        if is_point_in_bbox(text_center_x, text_center_y, table_bbox):
            return True

    return False


def detect_repeated_headers_footers(all_texts: List[Dict], min_pages: int = 3) -> set:
    """문서 내에서 반복되는 header/footer 텍스트 패턴 감지"""
    # 페이지별 텍스트 위치 수집 (상단 100px, 하단 100px)
    page_top_texts = {}  # {page_no: [(text, y_position)]}
    page_bottom_texts = {}

    for text_obj in all_texts:
        text_content = text_obj.get('text', '').strip()
        if not text_content or len(text_content) < 2:
            continue

        page_no = text_obj.get('prov', [{}])[0].get('page_no', -1)
        if page_no == -1:
            continue

        text_bbox = text_obj.get('prov', [{}])[0].get('bbox', {})
        if not text_bbox:
            continue

        coord_origin = text_bbox.get('coord_origin', 'BOTTOMLEFT')

        if coord_origin == 'BOTTOMLEFT':
            y_pos = text_bbox.get('t', 0)  # 상단 y 좌표
            # 상단 100px 내의 텍스트 (y > 692 for 792 height)
            if y_pos > 692:
                if page_no not in page_top_texts:
                    page_top_texts[page_no] = []
                page_top_texts[page_no].append(text_content)
            # 하단 100px 내의 텍스트 (y < 100)
            elif y_pos < 100:
                if page_no not in page_bottom_texts:
                    page_bottom_texts[page_no] = []
                page_bottom_texts[page_no].append(text_content)
        else:  # TOPLEFT
            y_pos = text_bbox.get('t', 0)
            # 상단 100px 내의 텍스트 (y < 100)
            if y_pos < 100:
                if page_no not in page_top_texts:
                    page_top_texts[page_no] = []
                page_top_texts[page_no].append(text_content)
            # 하단 100px 내의 텍스트 (y > 692 for 792 height)
            elif y_pos > 692:
                if page_no not in page_bottom_texts:
                    page_bottom_texts[page_no] = []
                page_bottom_texts[page_no].append(text_content)

    # 반복되는 텍스트 찾기
    repeated_texts = set()

    # 텍스트 빈도 계산
    from collections import Counter

    # 상단 텍스트 중 여러 페이지에서 반복되는 것
    all_top_texts = [text for texts in page_top_texts.values() for text in texts]
    top_counter = Counter(all_top_texts)
    for text, count in top_counter.items():
        if count >= min_pages:  # 최소 N개 페이지에서 반복
            repeated_texts.add(text)

    # 하단 텍스트 중 여러 페이지에서 반복되는 것
    all_bottom_texts = [text for texts in page_bottom_texts.values() for text in texts]
    bottom_counter = Counter(all_bottom_texts)
    for text, count in bottom_counter.items():
        if count >= min_pages:
            repeated_texts.add(text)

    return repeated_texts


def find_table_title(table: Dict, all_texts: List[Dict], all_tables: List[Dict] = None, page_height: float = 792.0, repeated_patterns: set = None) -> str:
    """테이블 위 300px 내의 텍스트를 타이틀로 추출 (다른 테이블 안의 텍스트 제외)"""
    table_bbox = table.get('prov', [{}])[0].get('bbox', {})
    table_page = table.get('prov', [{}])[0].get('page_no', -1)

    if not table_bbox or table_page == -1:
        return ""

    # 좌표계 확인
    coord_origin = table_bbox.get('coord_origin', 'BOTTOMLEFT')

    if coord_origin == 'BOTTOMLEFT':
        # BOTTOMLEFT: y가 위로 갈수록 증가
        # 테이블의 상단 y 좌표
        table_top_y = table_bbox.get('t', 0)
    else:  # TOPLEFT
        # TOPLEFT: y가 아래로 갈수록 증가
        # 테이블의 상단 y 좌표 (작은 값)
        table_top_y = table_bbox.get('t', 0)

    # 테이블 위 300px 영역의 텍스트 찾기
    title_candidates = []

    for text_obj in all_texts:
        text_page = text_obj.get('prov', [{}])[0].get('page_no', -1)

        # 같은 페이지의 텍스트만 확인
        if text_page != table_page:
            continue

        text_bbox = text_obj.get('prov', [{}])[0].get('bbox', {})
        if not text_bbox:
            continue

        text_coord_origin = text_bbox.get('coord_origin', 'BOTTOMLEFT')

        if coord_origin == 'BOTTOMLEFT' and text_coord_origin == 'BOTTOMLEFT':
            # BOTTOMLEFT: 테이블 위에 있는 텍스트는 y값이 더 큼
            text_bottom_y = text_bbox.get('b', 0)
            text_top_y = text_bbox.get('t', 0)

            # 텍스트의 하단이 테이블 상단보다 위에 있는지 확인
            if text_bottom_y > table_top_y:
                distance = text_bottom_y - table_top_y
                if distance <= 300:
                    text_content = text_obj.get('text', '').strip()
                    if text_content and len(text_content) > 0:
                        title_candidates.append({
                            'text': text_content,
                            'distance': distance,
                            'y_position': text_bottom_y
                        })

        elif coord_origin == 'TOPLEFT' and text_coord_origin == 'TOPLEFT':
            # TOPLEFT: 테이블 위에 있는 텍스트는 y값이 더 작음
            text_bottom_y = text_bbox.get('b', 0)

            # 텍스트의 하단이 테이블 상단보다 위에 있는지 확인
            if text_bottom_y < table_top_y:
                distance = table_top_y - text_bottom_y
                if distance <= 300:
                    text_content = text_obj.get('text', '').strip()
                    if text_content and len(text_content) > 0:
                        title_candidates.append({
                            'text': text_content,
                            'distance': distance,
                            'y_position': text_bottom_y
                        })

    # ML 모델을 사용하여 최적의 타이틀 선택
    if title_candidates:
        # 반복 패턴 필터링 (전처리)
        filtered_candidates = []
        for candidate in title_candidates:
            title_text = candidate['text']
            distance = candidate['distance']

            # 숫자만 있는 경우 제외
            if title_text.isdigit():
                continue

            # 너무 짧은 텍스트 제외 (2자 미만)
            if len(title_text.strip()) < 2:
                continue

            # 반복되는 header/footer 패턴 제외
            if repeated_patterns and title_text in repeated_patterns:
                continue

            # 페이지 헤더 패턴 제외
            import re
            if '│' in title_text or '|' in title_text:
                if re.search(r'\d+\s*[│|]', title_text) or re.search(r'[│|]\s*\d+', title_text):
                    continue
            if re.match(r'^제\d+장', title_text):
                continue

            # 다른 테이블 안에 있는지 확인
            if distance > 50 and all_tables:
                text_bbox = None
                for text_obj in all_texts:
                    if text_obj.get('text', '').strip() == title_text:
                        text_bbox = text_obj.get('prov', [{}])[0].get('bbox', {})
                        break

                if text_bbox and is_text_in_any_table(text_bbox, all_tables, table):
                    continue

            filtered_candidates.append(candidate)

        # ML 모델 또는 규칙 기반으로 최적의 타이틀 선택
        if USE_ML_CLASSIFIER:
            return select_best_title_with_model(filtered_candidates)
        else:
            return select_best_title(filtered_candidates)

    return ""


def extract_table_info(table: Dict, all_texts: List[Dict] = None, all_tables: List[Dict] = None, repeated_patterns: set = None) -> Dict:
    """테이블에서 필요한 정보 추출"""
    import time
    start = time.time()

    cells = table.get('data', {}).get('table_cells', [])
    page_no = table.get('prov', [{}])[0].get('page_no', -1)

    # 텍스트 추출
    texts = [cell.get('text', '').strip() for cell in cells if cell.get('text', '').strip()]

    # 헤더 셀 찾기
    # 각 열마다 가장 긴(구체적인) 텍스트를 헤더로 선택
    header_cells = [cell for cell in cells if cell.get('column_header', False)]

    if header_cells:
        # 헤더는 보통 상단 몇 개 행에만 있으므로, 최대 행 번호 확인
        max_header_row = max([cell.get('end_row_offset_idx', 0) for cell in header_cells]) if header_cells else 0

        # 상단 5개 행 이내의 헤더만 사용 (중간 소계 행 등 제외)
        # 단, 전체 헤더 행이 5개 이하면 그대로 사용
        header_row_threshold = min(5, max_header_row) if max_header_row > 0 else 5

        # 열별로 헤더 셀 그룹화
        headers_by_col = {}
        merged_headers = []  # 병합된 헤더 셀 (여러 열에 걸친 것)

        for cell in header_cells:
            start_col = cell.get('start_col_offset_idx', -1)
            end_col = cell.get('end_col_offset_idx', -1)
            start_row = cell.get('start_row_offset_idx', -1)
            text = cell.get('text', '').strip()

            # 상단 헤더 영역의 셀만 사용
            if start_row > header_row_threshold:
                continue

            if text and start_col >= 0:
                span = end_col - start_col

                # 단일 열 헤더
                if span == 1:
                    if start_col not in headers_by_col:
                        headers_by_col[start_col] = []
                    headers_by_col[start_col].append(text)
                # 병합된 헤더 (여러 열에 걸친 것)
                elif span > 1:
                    merged_headers.append({
                        'text': text,
                        'start_col': start_col,
                        'end_col': end_col,
                        'span': span
                    })
                    # 병합된 헤더가 포함하는 각 열에 추가
                    for col in range(start_col, end_col):
                        if col not in headers_by_col:
                            headers_by_col[col] = []
                        headers_by_col[col].append(text)

        # 각 열마다 가장 긴 텍스트를 헤더로 선택
        headers = []
        for col in sorted(headers_by_col.keys()):
            col_headers = headers_by_col[col]
            # 가장 긴 텍스트 선택 (가장 구체적인 헤더)
            longest_header = max(col_headers, key=len)
            headers.append(longest_header)

        # 헤더가 1개뿐이면 헤더가 아닐 가능성이 높음 (데이터 행일 수 있음)
        if len(headers) == 1:
            headers = []
    else:
        headers = []

    # 키값 추출 (첫 번째 열의 데이터, 헤더 제외)
    key_values = []
    for cell in cells:
        if (not cell.get('column_header', False) and
            not cell.get('row_header', False) and
            cell.get('start_col_offset_idx', -1) == 0 and
            cell.get('text', '').strip()):
            key_values.append(cell.get('text', '').strip())

    # 테이블 구조 정보
    rows = max([cell.get('end_row_offset_idx', 0) for cell in cells]) if cells else 0
    cols = max([cell.get('end_col_offset_idx', 0) for cell in cells]) if cells else 0

    # 타이틀 추출
    title = ""
    if all_texts:
        title_start = time.time()
        title = find_table_title(table, all_texts, all_tables, repeated_patterns=repeated_patterns)
        title_time = time.time() - title_start
        if title_time > 0.1:
            print(f"      [타이틀 추출 느림: {title_time:.2f}초, 페이지 {page_no}]")

    total_time = time.time() - start
    if total_time > 0.5:
        print(f"      [extract_table_info 느림: {total_time:.2f}초, 페이지 {page_no}]")

    return {
        'page_no': page_no,
        'cells': cells,
        'texts': texts,
        'headers': headers,
        'key_values': key_values,
        'rows': rows,
        'cols': cols,
        'bbox': table.get('prov', [{}])[0].get('bbox', {}),
        'title': title,
        'original_table': table
    }


def normalize_text(text: str) -> str:
    """텍스트 정규화 (비교용)"""
    # 공백, 특수문자 제거
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[·\-]+$', '', text)  # 끝의 점선 제거
    return text.strip()


def is_continuation_text(prev_text: str, curr_text: str) -> bool:
    """이전 텍스트와 현재 텍스트가 이어지는지 확인"""
    if not prev_text or not curr_text:
        return False

    prev_norm = normalize_text(prev_text)
    curr_norm = normalize_text(curr_text)

    # 너무 짧은 텍스트는 제외 (단일 문자나 기호)
    if len(prev_norm) < 2 or len(curr_norm) < 2:
        return False

    # 1. 숫자 연속성 체크 (가장 명확한 연결성)
    # 예: "1)" -> "2)", "1." -> "2.", "가." -> "나."
    prev_num = re.search(r'(\d+)[).]\s*$', prev_text)
    curr_num = re.search(r'^(\d+)[).]', curr_text)
    if prev_num and curr_num:
        try:
            if int(curr_num.group(1)) == int(prev_num.group(1)) + 1:
                return True
        except:
            pass

    # 가나다 순서
    prev_hangul = re.search(r'([가-힣])[).]\s*$', prev_text)
    curr_hangul = re.search(r'^([가-힣])[).]', curr_text)
    if prev_hangul and curr_hangul:
        hangul_order = "가나다라마바사아자차카타파하"
        prev_char = prev_hangul.group(1)
        curr_char = curr_hangul.group(1)
        if prev_char in hangul_order and curr_char in hangul_order:
            if hangul_order.index(curr_char) == hangul_order.index(prev_char) + 1:
                return True

    # 2. 단어의 일부가 잘린 경우 (단, 최소 5자 이상의 공통 부분)
    # 예: "국가재난관리" -> "국가재난관리시스템"
    if len(prev_norm) >= 5:
        # 마지막 5~15자가 다음 텍스트의 시작과 일치하는지
        for length in range(min(15, len(prev_norm)), 4, -1):
            if curr_norm.startswith(prev_norm[-length:]):
                # 추가로 의미 있는 연결인지 확인 (최소 길이)
                if len(prev_norm[-length:]) >= 5:
                    return True

    # 3. 테이블 row 연속성 (같은 단어가 반복되는 패턴)
    # 예: 마지막이 "합계" -> 다음 시작이 "합계"
    # 하지만 너무 일반적인 단어는 제외
    common_words = {'합계', '소계', '계', '총계', '비고', '기타', '구분', '항목', '내용',
                    '번호', '연번', '순번', '-', '·', '※', '○', '●', '□', '■'}

    # 일반적인 단어가 아니고, 3자 이상이며, 완전히 일치하는 경우
    if prev_norm == curr_norm and len(prev_norm) >= 3 and prev_norm not in common_words:
        return True

    # 4. 문장이 중간에 끊긴 경우
    # 마지막이 조사나 불완전한 문장으로 끝나는 경우
    incomplete_endings = ['의', '를', '을', '가', '이', '에', '에서', '으로', '로',
                         '와', '과', '하', '되', '한', '된', '및', '등', '또는']
    for ending in incomplete_endings:
        if prev_text.strip().endswith(ending) and len(prev_text.strip()) > len(ending) + 2:
            # 다음 텍스트가 명사나 동사로 시작하면 연결 가능
            if len(curr_text.strip()) > 2 and not curr_text.strip()[0].isdigit():
                return True

    return False


def has_similar_structure(table1_info: Dict, table2_info: Dict) -> bool:
    """두 테이블이 비슷한 구조를 가지는지 확인"""
    # 헤더가 비슷한지 확인
    if table1_info['headers'] and table2_info['headers']:
        headers1 = [normalize_text(h) for h in table1_info['headers']]
        headers2 = [normalize_text(h) for h in table2_info['headers']]

        # 헤더 일치도 확인
        common = set(headers1) & set(headers2)
        if len(common) / max(len(headers1), len(headers2)) > 0.5:
            return True

        # 부분 일치 확인 (헤더 포함 관계)
        # 예: "구 분"이 "구"를 포함하거나, "미적용시 유 및 대체"가 "부분적용/ 미적용시 사 유 및 대체 기술"에 포함
        match_count = 0
        for h1 in headers1:
            for h2 in headers2:
                # 길이가 2자 이상인 경우에만 부분 일치 확인
                if len(h1) >= 2 and len(h2) >= 2:
                    if h1 in h2 or h2 in h1:
                        match_count += 1
                        break

        # 최소 2개 이상의 헤더가 부분 일치하면 유사한 구조로 판단
        if match_count >= 2:
            return True

    return True  # 구조 정보만으로는 판단하기 어려우면 True


def calculate_title_similarity(title1: str, title2: str) -> float:
    """두 타이틀의 유사도 계산 (0~1 사이 값) - ML 모델 기반"""
    if not title1 or not title2:
        return 0.0

    # 완전 일치
    if title1 == title2:
        return 1.0

    if USE_SIMILARITY_MODEL:
        try:
            # Sentence-Transformers로 의미적 유사도 계산
            model = get_similarity_model()
            embeddings = model.encode([title1, title2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            # 모델이 계산한 유사도를 그대로 사용 (하드코딩 금지)
            # 0.7 이상이면 유사, 0.85 이상이면 거의 동일
            if similarity >= 0.85:
                return 1.0
            elif similarity >= 0.7:
                return 0.7
            else:
                return 0.0

        except Exception as e:
            print(f"  ⚠️  유사도 계산 실패: {e}, fallback 사용", flush=True)
            # Fallback: 단순 문자열 비교
            pass

    # Fallback: 정규화 후 단순 비교
    t1 = normalize_text(title1)
    t2 = normalize_text(title2)

    if t1 == t2:
        return 1.0
    elif t1 in t2 or t2 in t1:
        return 0.7
    else:
        return 0.0


def _check_text_continuation(table1_info: Dict, table2_info: Dict) -> Tuple[bool, str]:
    """텍스트 연결성 확인 (병렬 처리용)"""
    if table1_info['texts'] and table2_info['texts']:
        last_texts = table1_info['texts'][-5:]
        first_texts = table2_info['texts'][:5]

        for prev_text in last_texts:
            for curr_text in first_texts:
                if is_continuation_text(prev_text, curr_text):
                    return True, f"텍스트 연결: '{prev_text}' -> '{curr_text}'"
    return False, ""


def _check_header_similarity(table1_info: Dict, table2_info: Dict) -> Tuple[bool, str, int, float]:
    """헤더 유사도 확인 (병렬 처리용)"""
    if table1_info['headers'] and table2_info['headers']:
        headers1 = [normalize_text(h) for h in table1_info['headers'] if len(normalize_text(h)) > 0]
        headers2 = [normalize_text(h) for h in table2_info['headers'] if len(normalize_text(h)) > 0]

        if len(headers1) >= 2 and len(headers2) >= 2:
            headers1_set = set(headers1)
            headers2_set = set(headers2)

            common = headers1_set & headers2_set
            similarity = len(common) / max(len(headers1_set), len(headers2_set))

            partial_match_count = 0
            for h1 in headers1:
                for h2 in headers2:
                    if len(h1) >= 3 and len(h2) >= 3:
                        if len(h1) > len(h2) + 2 and h2 in h1:
                            partial_match_count += 1
                            break
                        elif len(h2) > len(h1) + 2 and h1 in h2:
                            partial_match_count += 1
                            break

            partial_similarity = partial_match_count / max(len(headers1), len(headers2))

            if (similarity >= 1.0 and len(common) >= 2) or (partial_similarity >= 0.5 and partial_match_count >= 2):
                return True, "", len(common), partial_match_count

    return False, "", 0, 0


def _check_width_similarity(table1_info: Dict, table2_info: Dict) -> Tuple[bool, float, float, float]:
    """테이블 너비 유사도 확인 (병렬 처리용)"""
    bbox1 = table1_info['bbox']
    bbox2 = table2_info['bbox']

    if bbox1 and bbox2:
        width1 = abs(bbox1.get('r', 0) - bbox1.get('l', 0))
        width2 = abs(bbox2.get('r', 0) - bbox2.get('l', 0))

        if width1 > 0 and width2 > 0:
            width_diff_ratio = abs(width1 - width2) / min(width1, width2)
            return True, width1, width2, width_diff_ratio

    return False, 0, 0, 0


def check_table_connection(table1_info: Dict, table2_info: Dict) -> Tuple[bool, str]:
    """두 테이블이 연결되어 있는지 확인 (조건 병렬 처리)"""
    # 페이지 차이 확인
    page_diff = table2_info['page_no'] - table1_info['page_no']

    # 페이지가 너무 멀리 떨어져 있으면 연결 안됨
    if page_diff < 0 or page_diff > 2:
        return False, "페이지가 너무 멀리 떨어져 있음"

    # 구조 유사성 확인
    if not has_similar_structure(table1_info, table2_info):
        return False, "테이블 구조가 다름"

    # 타이틀 정보
    title1 = table1_info.get('title', '')
    title2 = table2_info.get('title', '')

    # 병렬로 처리할 조건들
    from concurrent.futures import ThreadPoolExecutor

    results = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        # 모든 태스크를 먼저 제출 (병렬 실행 시작)
        if title1 and title2:
            futures['title'] = executor.submit(calculate_title_similarity, title1, title2)

        futures['text'] = executor.submit(_check_text_continuation, table1_info, table2_info)
        futures['header'] = executor.submit(_check_header_similarity, table1_info, table2_info)
        futures['width'] = executor.submit(_check_width_similarity, table1_info, table2_info)

        # 모든 결과를 한번에 수집 (병렬 실행 완료 대기)
        if 'title' in futures:
            results['title_similarity'] = futures['title'].result()

        results['text_connected'], results['text_reason'] = futures['text'].result()
        results['header_similar'], _, results['header_common_count'], results['header_partial_count'] = futures['header'].result()
        results['width_available'], results['width1'], results['width2'], results['width_diff_ratio'] = futures['width'].result()

    # === 타이틀 체크 (텍스트 연결보다 우선) ===
    if title1 and title2:
        title_similarity = results.get('title_similarity', 0)

        # 같은 페이지에 있으면서 타이틀이 다른 경우
        if page_diff == 0 and title_similarity < 0.8:
            return False, f"같은 페이지의 다른 테이블 ('{title1}' vs '{title2}', 유사도: {title_similarity:.2f})"

        # 다른 페이지인 경우 유사도 85% 미만일 때만 분리
        if title_similarity < 0.85:
            return False, f"타이틀이 다름 ('{title1}' vs '{title2}', 유사도: {title_similarity:.2f})"

    # 두 번째 테이블에만 타이틀이 있는 경우
    elif not title1 and title2:
        return False, f"두 번째 테이블에 새로운 타이틀 시작 ('{title2}')"

    # === 텍스트 연결성 확인 ===
    # 주의: 텍스트 연결성만으로는 불충분 (우연히 비슷한 단어로 시작/끝날 수 있음)
    # 타이틀이나 헤더 등 더 명확한 증거가 필요
    # if results['text_connected']:
    #     return True, results['text_reason']

    # === 타이틀 있음 → 타이틀 없음 연결 체크 ===
    # 주의: 이 조건은 매우 신중하게 적용해야 함
    # 타이틀이 있는 테이블 뒤에 타이틀 없는 테이블이 와도, 명확한 연결 증거가 있어야 병합
    # 단순히 구조가 유사하다고 병합하면 안 됨 (별개의 테이블일 수 있음)
    # 이 조건은 제거하거나 매우 제한적으로만 사용
    # if title1 and not title2 and page_diff == 1:
    #     if has_similar_structure(table1_info, table2_info):
    #         return True, f"타이틀이 있는 테이블 뒤 계속 (구조 유사, 연속 페이지)"

    # === 헤더 동일 체크 ===
    if results['header_similar']:
        # 두 번째 테이블에만 타이틀이 있으면 분리
        if not title1 and title2:
            return False, f"두 번째 테이블에 새로운 타이틀 시작 ('{title2}')"

        # 키값 중복 체크
        keys1 = set([normalize_text(k) for k in table1_info['key_values'][:10]])
        keys2 = set([normalize_text(k) for k in table2_info['key_values'][:10]])

        if keys1 and keys2:
            common_keys = keys1 & keys2
            key_overlap_ratio = len(common_keys) / min(len(keys1), len(keys2)) if min(len(keys1), len(keys2)) > 0 else 0

            if key_overlap_ratio > 0.3:
                return False, f"헤더는 동일하지만 키값이 겹침 ({len(common_keys)}개 중복)"

        if results['header_common_count'] >= 2:
            return True, f"헤더가 동일함 ({results['header_common_count']}개 일치)"
        else:
            return True, f"헤더가 부분 일치함 ({results['header_partial_count']}개 부분 일치)"

    # === 타이틀 유사도 확인 (헤더 체크 이후) ===
    if title1 and title2:
        title_similarity = results.get('title_similarity', 0)

        if title_similarity >= 1.0:
            return True, f"타이틀이 같음 ('{title1}')"

        if title_similarity < 0.85:
            return False, f"타이틀이 다름 ('{title1}' vs '{title2}', 유사도: {title_similarity:.2f})"

    elif not title1 and title2:
        return False, f"두 번째 테이블에 새로운 타이틀 시작 ('{title2}')"

    # === 헤더 있음 -> 헤더 없음 패턴 ===
    if table1_info['headers'] and not table2_info['headers']:
        if page_diff >= 1 and page_diff <= 2:
            # 텍스트 연결성은 사용하지 않음 (너무 많은 오탐)
            # if results['text_connected']:
            #     return True, f"헤더 테이블 뒤 데이터 테이블 - {results['text_reason']}"
            pass

            # 타이틀 있는 헤더 테이블 뒤 데이터 테이블
            # 조건: 타이틀이 있고, 헤더가 있고, 다음 테이블에 헤더가 없고, 열 개수와 너비가 유사한 경우
            if title1 and page_diff == 1:
                col_diff = abs(table1_info['cols'] - table2_info['cols'])

                # 열 개수와 너비도 함께 체크
                if results['width_available']:
                    width_diff_ratio = results['width_diff_ratio']

                    # 열 개수가 같거나 차이가 1 이하이고, 너비 차이가 20% 이내면 연결
                    if col_diff <= 1 and table1_info['cols'] >= 2 and width_diff_ratio <= 0.2:
                        return True, f"타이틀이 있는 헤더 테이블 뒤 데이터 테이블 (열 {table1_info['cols']}개 vs {table2_info['cols']}개, 너비 유사도 {(1-width_diff_ratio)*100:.0f}%, 연속 페이지)"

            # 열 개수와 너비 체크 (타이틀 없는 경우)
            # 조건: 헤더 있는 테이블 뒤에 헤더 없는 테이블이 연속으로 오는 경우
            if page_diff == 1:
                col_diff = abs(table1_info['cols'] - table2_info['cols'])

                if results['width_available']:
                    width_diff_ratio = results['width_diff_ratio']

                    # 옵션 1: 열 개수 정확히 일치, 너비 차이 10% 이내
                    if col_diff == 0 and table1_info['cols'] >= 2 and width_diff_ratio <= 0.1:
                        return True, f"헤더 테이블 뒤 데이터 테이블 (열 {table1_info['cols']}개 동일, 너비 유사도 {(1-width_diff_ratio)*100:.0f}%, 연속 페이지)"

                    # 옵션 2: 열 개수 차이 1, 너비 거의 동일 (1% 이내)
                    # 같은 테이블이 페이지 넘김으로 인해 열이 약간 다르게 인식될 수 있음
                    if col_diff == 1 and table1_info['cols'] >= 2 and width_diff_ratio <= 0.01:
                        return True, f"헤더 테이블 뒤 데이터 테이블 (열 {table1_info['cols']}개 vs {table2_info['cols']}개, 너비 거의 동일 {(1-width_diff_ratio)*100:.1f}%, 연속 페이지)"

    # === 둘 다 헤더 없음 패턴 ===
    if not table1_info['headers'] and not table2_info['headers']:
        if not title1 and not title2 and page_diff == 1:
            col_diff = abs(table1_info['cols'] - table2_info['cols'])

            if results['width_available']:
                width_diff_ratio = results['width_diff_ratio']

                if width_diff_ratio > 0.3:
                    return False, f"테이블 너비 차이가 큼 (너비 {results['width1']:.0f} vs {results['width2']:.0f}, 차이 {width_diff_ratio*100:.0f}%)"

                # 헤더 없는 데이터 테이블끼리는 열 개수가 정확히 일치해야 함
                if col_diff == 0 and table1_info['cols'] >= 2:
                    # 키값 중복 체크
                    keys1 = set([normalize_text(k) for k in table1_info['key_values'][:10]])
                    keys2 = set([normalize_text(k) for k in table2_info['key_values'][:10]])

                    if keys1 and keys2:
                        common_keys = keys1 & keys2
                        key_overlap_ratio = len(common_keys) / min(len(keys1), len(keys2)) if min(len(keys1), len(keys2)) > 0 else 0

                        if key_overlap_ratio > 0.5:
                            return False, f"같은 형식의 반복 테이블 (키값 {len(common_keys)}개 중복, 중복률 {key_overlap_ratio:.0%})"

                    return True, f"헤더 없는 데이터 테이블 연속 (열 {table1_info['cols']}개 동일, 연속 페이지)"

    return False, "연결 조건 미충족"


def merge_tables(table_group: List[Dict]) -> Dict:
    """연결된 테이블들을 하나로 병합"""
    if not table_group:
        return None

    if len(table_group) == 1:
        return table_group[0]['original_table']

    # 첫 번째 테이블을 기반으로 병합
    merged = table_group[0]['original_table'].copy()
    merged['merged_from_pages'] = [t['page_no'] for t in table_group]

    # 모든 셀 합치기
    all_cells = []
    for table_info in table_group:
        all_cells.extend(table_info['cells'])

    merged['data']['table_cells'] = all_cells

    return merged


def _check_connection_pair(args):
    """병렬 처리를 위한 테이블 쌍 비교 함수"""
    table_info_i, table_info_j, i, j = args
    is_connected, reason = check_table_connection(table_info_i, table_info_j)
    return (i, j, is_connected, reason)


def find_connected_table_groups(tables: List[Dict], all_texts: List[Dict] = None, repeated_patterns: set = None, use_parallel: bool = True, max_workers: int = None) -> List[List[int]]:
    """연결된 테이블 그룹 찾기

    Args:
        tables: 테이블 목록
        all_texts: 전체 텍스트 목록
        repeated_patterns: 반복되는 패턴
        use_parallel: 병렬 처리 사용 여부 (기본값: True)
        max_workers: 병렬 처리 워커 수 (기본값: CPU 코어 수)
    """
    print(f"  [1/4] 테이블 정보 추출 중... ({len(tables)}개)", flush=True)

    # 타이틀 추출은 순차 처리 (ML 모델 공유를 위해)
    # 병렬 처리하면 각 프로세스마다 모델을 로드해서 오히려 느림
    table_infos = []
    for idx, table in enumerate(tables):
        if (idx + 1) % 10 == 0 or (idx + 1) == len(tables):
            print(f"      → 진행: {idx + 1}/{len(tables)} 테이블 처리 중", end='\r', flush=True)
        table_infos.append(extract_table_info(table, all_texts, tables, repeated_patterns))

    print(f"\n  [1/4] 완료 - 타이틀 추출됨", flush=True)

    # 페이지 번호순으로 정렬
    print(f"  [2/4] 페이지 순서대로 정렬 중...", flush=True)
    sorted_indices = sorted(range(len(table_infos)),
                          key=lambda i: table_infos[i]['page_no'])

    visited = set()
    groups = []
    connection_reasons = []
    all_disconnection_reasons = {}  # 모든 비연속 이유 저장 (테이블 인덱스별)

    # 먼저 모든 인접 테이블 쌍에 대해 연속성 체크
    connection_results = {}

    print(f"  [3/4] 테이블 연결성 체크 중... ({len(sorted_indices)-1}개 쌍)", flush=True)
    if use_parallel and len(sorted_indices) > 1:
        # 병렬 처리: 모든 인접 쌍을 동시에 비교
        comparison_tasks = []
        for idx in range(len(sorted_indices) - 1):
            i = sorted_indices[idx]
            j = sorted_indices[idx + 1]
            comparison_tasks.append((table_infos[i], table_infos[j], i, j))

        # ThreadPoolExecutor로 병렬 실행 (ProcessPool보다 오버헤드 적음)
        print(f"      → 병렬 처리 모드 (워커: {max_workers or 'CPU 코어 수'})", flush=True)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_check_connection_pair, task) for task in comparison_tasks]

            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                i, j, is_connected, reason = future.result()
                connection_results[(i, j)] = (is_connected, reason)

                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"      → 진행: {completed}/{total} 쌍 완료", end='\r', flush=True)

                if not is_connected:
                    if i not in all_disconnection_reasons:
                        all_disconnection_reasons[i] = []
                    all_disconnection_reasons[i].append(f"Table {i} -X-> {j}: {reason}")
            print(flush=True)  # 줄바꿈
    else:
        # 순차 처리 (기존 방식)
        print(f"      → 순차 처리 모드")
        for idx in range(len(sorted_indices) - 1):
            i = sorted_indices[idx]
            j = sorted_indices[idx + 1]

            is_connected, reason = check_table_connection(
                table_infos[i],
                table_infos[j]
            )

            connection_results[(i, j)] = (is_connected, reason)

            if not is_connected:
                # 비연속인 경우 저장
                if i not in all_disconnection_reasons:
                    all_disconnection_reasons[i] = []
                all_disconnection_reasons[i].append(f"Table {i} -X-> {j}: {reason}")

    print(f"  [3/4] 완료 - 연결성 체크 완료", flush=True)

    # 연결된 그룹 찾기
    print(f"  [4/4] 테이블 그룹핑 중...", flush=True)
    for i in sorted_indices:
        if i in visited:
            continue

        current_group = [i]
        current_reasons = []
        visited.add(i)

        # 순차적으로 바로 다음 테이블만 확인 (건너뛰기 방지)
        current_idx = i
        current_pos = sorted_indices.index(i)

        for j_pos in range(current_pos + 1, len(sorted_indices)):
            j = sorted_indices[j_pos]

            if j in visited:
                continue

            # 이미 계산된 연결 결과 사용
            is_connected, reason = connection_results.get((current_idx, j), (False, "비교되지 않음"))

            if is_connected:
                current_group.append(j)
                current_reasons.append(f"Table {current_idx} -> {j}: {reason}")
                visited.add(j)
                current_idx = j  # 다음 비교를 위해 현재 인덱스 업데이트
            else:
                # 연결되지 않으면 그룹 종료
                break

        groups.append(current_group)
        connection_reasons.append(current_reasons)

    print(f"  [4/4] 완료 - {len([g for g in groups if len(g) > 1])}개 그룹 생성", flush=True)

    return groups, connection_reasons, all_disconnection_reasons, table_infos


def create_visualization_pdf(json_files: List[str], output_dir: str, font_name: str):
    """병합 결과를 시각화한 PDF 생성"""
    pdf_path = os.path.join(output_dir, "merged_tables_visualization.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=16,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=12,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=12,
        spaceBefore=12
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9,
        leading=12
    )

    # 제목
    story.append(Paragraph("테이블 병합 분석 결과", title_style))
    story.append(Spacer(1, 0.5*cm))

    total_merged = 0

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filename = os.path.basename(json_file)
        story.append(Paragraph(f"파일: {filename}", heading_style))

        tables = data.get('tables', [])
        groups, reasons, all_disconnections, table_infos = find_connected_table_groups(tables)

        # 병합된 그룹과 단일 그룹 분리
        merged_groups = [g for g in groups if len(g) > 1]
        merged_reasons = [r for g, r in zip(groups, reasons) if len(g) > 1]
        single_groups = [g for g in groups if len(g) == 1]

        total_merged += len(merged_groups)

        for group_idx, (group, group_reasons) in enumerate(zip(merged_groups, merged_reasons), 1):
            # 그룹 정보
            pages = [table_infos[i]['page_no'] for i in group]
            story.append(Paragraph(
                f"병합 그룹 {group_idx}: {len(group)}개 테이블 (페이지 {pages})",
                normal_style
            ))

            # 타이틀 정보 표시
            titles = [table_infos[i]['title'] for i in group if table_infos[i]['title']]
            if titles:
                # 첫 번째 타이틀 (대표 타이틀)
                story.append(Paragraph(f"  📋 타이틀: {titles[0]}", normal_style))

            # 연결 이유
            for reason in group_reasons:
                story.append(Paragraph(f"  • {reason}", normal_style))

            # 비연속 이유 (이 그룹의 테이블들과 관련된 것)
            for table_idx in group:
                if table_idx in all_disconnections:
                    for reason in all_disconnections[table_idx]:
                        story.append(Paragraph(f"  • (비연속) {reason}", normal_style))

            # 테이블 미리보기 데이터 준비
            preview_data = [["테이블", "페이지", "행수", "열수", "타이틀", "샘플 텍스트"]]

            for idx in group:
                info = table_infos[idx]
                sample_text = info['texts'][0][:20] + "..." if info['texts'] else ""
                title_text = info['title'][:30] + "..." if info['title'] and len(info['title']) > 30 else (info['title'] or "")
                preview_data.append([
                    f"#{idx}",
                    str(info['page_no']),
                    str(info['rows']),
                    str(info['cols']),
                    title_text,
                    sample_text
                ])

            # 테이블 스타일 (타이틀 컬럼 추가로 너비 조정)
            t = Table(preview_data, colWidths=[1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 5*cm, 5*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))

            story.append(Spacer(1, 0.2*cm))
            story.append(t)
            story.append(Spacer(1, 0.5*cm))

        # 단일 테이블 정보 표시
        if single_groups:
            story.append(Spacer(1, 0.5*cm))
            story.append(Paragraph(f"단일 테이블 ({len(single_groups)}개)", normal_style))
            story.append(Spacer(1, 0.3*cm))

            # 단일 테이블 미리보기
            single_preview_data = [["테이블", "페이지", "행수", "열수", "타이틀", "샘플 텍스트"]]

            for group in single_groups[:20]:  # 최대 20개만 표시
                idx = group[0]
                info = table_infos[idx]
                sample_text = info['texts'][0][:20] + "..." if info['texts'] else ""
                title_text = info['title'][:30] + "..." if info['title'] and len(info['title']) > 30 else (info['title'] or "")
                single_preview_data.append([
                    f"#{idx}",
                    str(info['page_no']),
                    str(info['rows']),
                    str(info['cols']),
                    title_text,
                    sample_text
                ])

            t_single = Table(single_preview_data, colWidths=[1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 5*cm, 5*cm])
            t_single.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95A5A6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))

            story.append(t_single)
            if len(single_groups) > 20:
                story.append(Paragraph(f"  (외 {len(single_groups) - 20}개 생략...)", normal_style))

        story.append(PageBreak())

    # 요약 페이지
    story.append(Paragraph("요약", title_style))
    story.append(Paragraph(f"총 병합된 그룹 수: {total_merged}", normal_style))

    doc.build(story)
    return pdf_path


def process_json_files(input_dir: str, output_dir: str, original_json_dir: str = "output"):
    """JSON 파일들을 처리하여 병합된 테이블 생성"""
    os.makedirs(output_dir, exist_ok=True)

    # 한글 폰트 등록
    font_name = register_korean_font()

    json_files = list(Path(input_dir).glob('*.json'))

    all_results = []

    for file_idx, json_file in enumerate(json_files, 1):
        print(f"\n" + "="*60, flush=True)
        print(f"[{file_idx}/{len(json_files)}] 처리 중: {json_file.name}", flush=True)
        print("="*60, flush=True)

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tables = data.get('tables', [])
        print(f"  총 테이블 수: {len(tables)}")

        # 원본 JSON에서 텍스트 정보 로드
        all_texts = []
        source_file = data.get('source_file', '')
        if source_file:
            original_json_path = os.path.join(original_json_dir, os.path.basename(source_file))
            if os.path.exists(original_json_path):
                try:
                    with open(original_json_path, 'r', encoding='utf-8') as f:
                        original_data = json.load(f)
                        all_texts = original_data.get('texts', [])
                        print(f"  원본 JSON에서 {len(all_texts)}개 텍스트 로드")
                except Exception as e:
                    print(f"  경고: 원본 JSON 로드 실패: {e}")

        # 반복되는 header/footer 패턴 감지
        repeated_patterns = set()
        if all_texts:
            repeated_patterns = detect_repeated_headers_footers(all_texts, min_pages=3)
            if repeated_patterns:
                print(f"  감지된 반복 패턴 {len(repeated_patterns)}개: {list(repeated_patterns)[:5]}")

        # 연결된 테이블 그룹 찾기
        groups, reasons, all_disconnections, table_infos = find_connected_table_groups(tables, all_texts, repeated_patterns)

        # 병합된 그룹만 필터링
        merged_groups = [g for g in groups if len(g) > 1]
        merged_reasons = [r for g, r in zip(groups, reasons) if len(g) > 1]

        print(f"  병합된 그룹 수: {len(merged_groups)}")
        print(f"  단일 테이블 수: {len(tables) - sum(len(g) for g in merged_groups)}")

        # 단일 테이블 그룹 (연결되지 않은 테이블들)
        single_table_groups = []
        for g_idx, (group, group_reasons) in enumerate(zip(groups, reasons)):
            if len(group) == 1:
                table_idx = group[0]
                disconnection_reason = ""
                # 이 테이블의 비연속 이유 찾기
                if table_idx in all_disconnections:
                    disconnection_reason = all_disconnections[table_idx][0] if all_disconnections[table_idx] else ""

                single_table_groups.append({
                    'table_idx': table_idx,
                    'page_no': table_infos[table_idx]['page_no'],
                    'title': table_infos[table_idx]['title'],
                    'disconnection_reason': disconnection_reason
                })

        # 결과 저장
        result = {
            'source_file': str(json_file),
            'original_table_count': len(tables),
            'merged_groups': [],
            'single_tables': single_table_groups,  # 단일 테이블 정보
            'all_disconnections': all_disconnections,  # 모든 비연속 정보 저장
            'table_infos': [  # 모든 테이블 정보 저장 (타이틀 포함)
                {
                    'table_idx': idx,
                    'page_no': info['page_no'],
                    'title': info['title']
                }
                for idx, info in enumerate(table_infos)
            ]
        }

        for group_idx, (group, group_reasons) in enumerate(zip(merged_groups, merged_reasons)):
            # 이 그룹에 속한 테이블들의 비연속 이유 수집
            group_disconnections = []
            for table_idx in group:
                if table_idx in all_disconnections:
                    group_disconnections.extend(all_disconnections[table_idx])

            group_info = {
                'group_id': group_idx,
                'table_indices': group,
                'pages': [table_infos[i]['page_no'] for i in group],
                'connection_reasons': group_reasons,
                'disconnection_reasons': group_disconnections,
                'merged_table': merge_tables([table_infos[i] for i in group])
            }
            result['merged_groups'].append(group_info)

            print(f"\n  그룹 {group_idx + 1}:")
            print(f"    테이블 인덱스: {group}")
            print(f"    페이지: {group_info['pages']}")
            # 타이틀 출력
            for idx in group:
                if table_infos[idx]['title']:
                    print(f"    테이블 {idx} 타이틀: {table_infos[idx]['title']}")
            for reason in group_reasons:
                print(f"    - {reason}")
            if group_disconnections:
                for reason in group_disconnections:
                    print(f"    - (비연속) {reason}")

        all_results.append(result)

        # 개별 파일 결과 저장
        # Windows에서 사용할 수 없는 문자 제거
        safe_filename = json_file.stem.replace('+', '_').replace(':', '_').replace('?', '_').replace('*', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        output_file = os.path.join(
            output_dir,
            f"{safe_filename}_merged.json"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"  결과 저장: {output_file}")

    # 전체 요약 저장
    summary_file = os.path.join(output_dir, "merge_summary.json")
    summary = {
        'total_files': len(json_files),
        'total_merged_groups': sum(len(r['merged_groups']) for r in all_results),
        'files': all_results
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n전체 요약 저장: {summary_file}")

    # PDF 시각화 생성
    print("\nPDF 시각화 생성 중...")
    pdf_path = create_visualization_pdf([str(f) for f in json_files], output_dir, font_name)
    print(f"PDF 생성 완료: {pdf_path}")

    return all_results


if __name__ == "__main__":
    import sys
    import io

    # Windows 콘솔 인코딩 문제 해결
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    input_dir = "table_output"
    output_dir = "merged_tables_output"

    print("=" * 60)
    print("테이블 병합 스크립트 시작")
    print("=" * 60)

    results = process_json_files(input_dir, output_dir)

    print("\n" + "=" * 60)
    print("처리 완료!")
    print("=" * 60)
    print(f"\n결과 파일 위치: {output_dir}/")
    print(f"- merge_summary.json: 전체 요약")
    print(f"- *_merged.json: 각 파일별 병합 결과")
    print(f"- merged_tables_visualization.pdf: 시각화 PDF")
