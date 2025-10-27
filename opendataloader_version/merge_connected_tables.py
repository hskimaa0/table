"""
opendataloader-pdf 기반 테이블 병합 시스템
docling merge_connected_tables.py와 동일한 로직 적용
"""
import opendataloader_pdf
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# UTF-8 출력 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def extract_text_from_element(element):
    """재귀적으로 element에서 텍스트 추출"""
    texts = []
    if isinstance(element, dict):
        if 'content' in element:
            texts.append(str(element['content']))
        if 'kids' in element:
            for kid in element['kids']:
                texts.extend(extract_text_from_element(kid))
    elif isinstance(element, list):
        for item in element:
            texts.extend(extract_text_from_element(item))
    return texts


def is_point_in_bbox(x: float, y: float, bbox: tuple) -> bool:
    """점이 bbox 내부에 있는지 확인 (bottom-left origin)"""
    if not bbox or len(bbox) < 4:
        return False
    left, bottom, right, top = bbox
    return left <= x <= right and bottom <= y <= top


def is_text_in_any_table(text_bbox: tuple, all_tables: List[Dict], exclude_table: Dict) -> bool:
    """텍스트가 다른 테이블 안에 있는지 확인"""
    if not text_bbox or len(text_bbox) < 4:
        return False

    # 텍스트의 중심점 계산
    text_center_x = (text_bbox[0] + text_bbox[2]) / 2
    text_center_y = (text_bbox[1] + text_bbox[3]) / 2

    exclude_bbox = exclude_table.get('bbox')

    for table in all_tables:
        table_bbox = table.get('bbox')

        # 현재 테이블 제외
        if table_bbox == exclude_bbox:
            continue

        # 텍스트 중심점이 테이블 내부에 있는지 확인
        if is_point_in_bbox(text_center_x, text_center_y, table_bbox):
            return True

    return False


def find_table_title(table: Dict, all_text_elements: List[Dict], all_tables: List[Dict] = None) -> str:
    """테이블 위 300px 내의 텍스트를 타이틀로 추출 (다른 테이블 안의 텍스트 제외)"""
    table_bbox = table.get('bbox')
    table_page = table.get('page')

    if not table_bbox or table_page is None:
        return ""

    # opendataloader는 bottom-left origin 사용
    left, bottom, right, top = table_bbox
    table_top_y = top

    # 테이블 위 300px 영역의 텍스트 찾기
    title_candidates = []

    for text_elem in all_text_elements:
        text_page = text_elem.get('page')

        # 같은 페이지의 텍스트만 확인
        if text_page != table_page:
            continue

        text_bbox = text_elem.get('bbox')
        if not text_bbox or len(text_bbox) < 4:
            continue

        text_bottom = text_bbox[1]

        # 텍스트의 하단이 테이블 상단보다 위에 있는지 확인
        if text_bottom > table_top_y:
            distance = text_bottom - table_top_y
            if distance <= 300:
                text_content = text_elem.get('text', '').strip()
                if text_content and len(text_content) > 0:
                    title_candidates.append({
                        'text': text_content,
                        'distance': distance,
                        'y_position': text_bottom,
                        'bbox': text_bbox
                    })

    # 거리가 가장 가까운 텍스트를 타이틀로 선택
    if title_candidates:
        title_candidates.sort(key=lambda x: x['distance'])

        # 유효한 타이틀 찾기 (숫자만 있거나 너무 짧은 것 제외)
        for candidate in title_candidates:
            title_text = candidate['text']
            distance = candidate['distance']

            # 숫자만 있는 경우 제외 (페이지 번호 등)
            if title_text.isdigit():
                continue

            # 너무 짧은 텍스트 제외 (3자 미만)
            if len(title_text.strip()) < 3:
                continue

            # 페이지 헤더/장 제목 패턴 제외
            if '│' in title_text or '|' in title_text:
                if re.search(r'\d+\s*[│|]', title_text) or re.search(r'[│|]\s*\d+', title_text):
                    continue
            if re.match(r'^제\d+장', title_text):
                continue

            # 다른 테이블 안에 있는지 확인
            if distance > 50 and all_tables:
                if is_text_in_any_table(candidate['bbox'], all_tables, table):
                    continue

            return title_text

    return ""


def extract_table_info(table: Dict, all_text_elements: List[Dict] = None, all_tables: List[Dict] = None) -> Dict:
    """테이블에서 필요한 정보 추출"""
    rows = table.get('rows', [])
    page_no = table.get('page', -1)
    bbox = table.get('bbox')

    # 모든 셀 수집
    all_cells = []
    for row in rows:
        # row는 이미 cell의 list임
        if isinstance(row, list):
            all_cells.extend(row)
        else:
            # dict 형식인 경우 (호환성)
            cells = row.get('cells', [])
            all_cells.extend(cells)

    # 텍스트 추출
    texts = []
    for cell in all_cells:
        # cell은 이미 {'text': '...', 'bbox': ...} 형식
        cell_text = cell.get('text', '').strip() if isinstance(cell, dict) else ''
        if cell_text:
            texts.append(cell_text)

    # 헤더 추출 (첫 번째 행만 사용)
    headers = []
    if rows:
        first_row = rows[0]
        cells = first_row if isinstance(first_row, list) else first_row.get('cells', [])

        for cell in cells:
            # cell은 이미 {'text': '...', 'bbox': ...} 형식
            text = cell.get('text', '').strip() if isinstance(cell, dict) else ''
            headers.append(text)

        # 헤더가 1개뿐이거나, 모두 비어있으면 헤더가 아님
        non_empty_headers = [h for h in headers if h]
        if len(non_empty_headers) <= 1:
            headers = []
        else:
            # 첫 번째 행이 실제 헤더인지 판단
            # 헤더는 일반적으로 짧고 명확한 단어들
            # 너무 긴 텍스트 (20자 이상)가 2개 이상 있으면 데이터일 가능성
            long_texts = [h for h in non_empty_headers if len(h) > 20]
            if len(long_texts) >= 2:
                headers = []

    # 키값 추출 (첫 번째 열의 데이터, 헤더 제외)
    key_values = []
    for row_idx, row in enumerate(rows):
        # 헤더 후보 행은 제외
        if row_idx < min(3, len(rows)) and headers:
            continue
        cells = row if isinstance(row, list) else row.get('cells', [])
        if cells:
            # cell은 이미 {'text': '...', 'bbox': ...} 형식
            first_cell = cells[0]
            first_cell_text = first_cell.get('text', '').strip() if isinstance(first_cell, dict) else ''
            if first_cell_text:
                key_values.append(first_cell_text)

    # 테이블 구조 정보
    num_rows = len(rows)
    num_cols = max([len(row) if isinstance(row, list) else len(row.get('cells', [])) for row in rows]) if rows else 0

    # 타이틀 추출
    title = ""
    if all_text_elements:
        title = find_table_title(table, all_text_elements, all_tables)

    return {
        'page_no': page_no,
        'cells': all_cells,
        'texts': texts,
        'headers': headers,
        'key_values': key_values,
        'rows': num_rows,
        'cols': num_cols,
        'bbox': bbox,
        'title': title,
        'original_table': table
    }


def normalize_text(text: str) -> str:
    """텍스트 정규화 (비교용)"""
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[·\-]+$', '', text)
    return text.strip()


def is_continuation_text(prev_text: str, curr_text: str) -> bool:
    """이전 텍스트와 현재 텍스트가 이어지는지 확인"""
    if not prev_text or not curr_text:
        return False

    prev_norm = normalize_text(prev_text)
    curr_norm = normalize_text(curr_text)

    # 너무 짧은 텍스트는 제외
    if len(prev_norm) < 2 or len(curr_norm) < 2:
        return False

    # 1. 숫자 연속성 체크
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

    # 2. 단어의 일부가 잘린 경우
    if len(prev_norm) >= 5:
        for length in range(min(15, len(prev_norm)), 4, -1):
            if curr_norm.startswith(prev_norm[-length:]):
                if len(prev_norm[-length:]) >= 5:
                    return True

    # 3. 같은 단어 반복 (일반적인 단어 제외)
    common_words = {'합계', '소계', '계', '총계', '비고', '기타', '구분', '항목', '내용',
                    '번호', '연번', '순번', '-', '·', '※', '○', '●', '□', '■'}

    if prev_norm == curr_norm and len(prev_norm) >= 3 and prev_norm not in common_words:
        return True

    # 4. 문장이 중간에 끊긴 경우
    incomplete_endings = ['의', '를', '을', '가', '이', '에', '에서', '으로', '로',
                         '와', '과', '하', '되', '한', '된', '및', '등', '또는']
    for ending in incomplete_endings:
        if prev_text.strip().endswith(ending) and len(prev_text.strip()) > len(ending) + 2:
            if len(curr_text.strip()) > 2 and not curr_text.strip()[0].isdigit():
                return True

    return False


def has_similar_structure(table1_info: Dict, table2_info: Dict) -> bool:
    """두 테이블이 비슷한 구조를 가지는지 확인"""
    if table1_info['headers'] and table2_info['headers']:
        headers1 = [normalize_text(h) for h in table1_info['headers']]
        headers2 = [normalize_text(h) for h in table2_info['headers']]

        # 헤더 일치도 확인
        common = set(headers1) & set(headers2)
        if len(common) / max(len(headers1), len(headers2)) > 0.5:
            return True

        # 부분 일치 확인
        match_count = 0
        for h1 in headers1:
            for h2 in headers2:
                if len(h1) >= 2 and len(h2) >= 2:
                    if h1 in h2 or h2 in h1:
                        match_count += 1
                        break

        if match_count >= 2:
            return True

    return True


def calculate_title_similarity(title1: str, title2: str) -> float:
    """두 타이틀의 유사도 계산 (0~1 사이 값)"""
    if not title1 or not title2:
        return 0.0

    t1 = normalize_text(title1)
    t2 = normalize_text(title2)

    if not t1 or not t2:
        return 0.0

    if t1 == t2:
        return 1.0

    if t1 in t2 or t2 in t1:
        return 0.8

    # 공통 단어 기반 유사도 (2글자 이상 단어만)
    words1 = set([w for w in t1.split() if len(w) >= 2])
    words2 = set([w for w in t2.split() if len(w) >= 2])

    if words1 and words2:
        common = words1 & words2
        all_words = words1 | words2
        if common:
            similarity = len(common) / len(all_words)
            return similarity

    return 0.0


def check_table_connection(table1_info: Dict, table2_info: Dict) -> Tuple[bool, str]:
    """두 테이블이 연결되어 있는지 확인"""
    # 페이지 차이 확인
    page_diff = table2_info['page_no'] - table1_info['page_no']

    if page_diff < 0 or page_diff > 2:
        return False, "페이지가 너무 멀리 떨어져 있음"

    # 구조 유사성 확인
    if not has_similar_structure(table1_info, table2_info):
        return False, "테이블 구조가 다름"

    # 타이틀 체크 (텍스트 연결보다 우선)
    title1 = table1_info.get('title', '')
    title2 = table2_info.get('title', '')

    if title1 and title2:
        title_similarity = calculate_title_similarity(title1, title2)

        if page_diff == 0 and title_similarity < 0.8:
            return False, f"같은 페이지의 다른 테이블 ('{title1}' vs '{title2}', 유사도: {title_similarity:.2f})"

        if title_similarity < 0.5:
            return False, f"타이틀이 다름 ('{title1}' vs '{title2}', 유사도: {title_similarity:.2f})"

    # 두 번째 테이블에만 타이틀이 있는 경우
    elif not title1 and title2:
        return False, f"두 번째 테이블에 새로운 타이틀 시작 ('{title2}')"

    # 텍스트 연결성 확인
    if table1_info['texts'] and table2_info['texts']:
        last_texts = table1_info['texts'][-5:]
        first_texts = table2_info['texts'][:5]

        for prev_text in last_texts:
            for curr_text in first_texts:
                if is_continuation_text(prev_text, curr_text):
                    return True, f"텍스트 연결: '{prev_text}' -> '{curr_text}'"

    # 헤더가 동일한 경우
    if table1_info['headers'] and table2_info['headers']:
        headers1 = [normalize_text(h) for h in table1_info['headers'] if len(normalize_text(h)) > 0]
        headers2 = [normalize_text(h) for h in table2_info['headers'] if len(normalize_text(h)) > 0]

        if len(headers1) >= 2 and len(headers2) >= 2:
            headers1_set = set(headers1)
            headers2_set = set(headers2)

            # 공통 헤더 개수와 비율 계산
            common = headers1_set & headers2_set
            similarity = len(common) / max(len(headers1_set), len(headers2_set))

            # 부분 일치 개수 계산
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

            if (similarity >= 0.6 and len(common) >= 2) or (partial_similarity >= 0.5 and partial_match_count >= 2):
                # 타이틀 체크
                if not title1 and title2:
                    return False, f"두 번째 테이블에 새로운 타이틀 시작 ('{title2}')"

                # 키값 체크 (30% 이상 겹치면 다른 테이블)
                keys1 = set([normalize_text(k) for k in table1_info['key_values'][:10]])
                keys2 = set([normalize_text(k) for k in table2_info['key_values'][:10]])

                if keys1 and keys2:
                    common_keys = keys1 & keys2
                    key_overlap_ratio = len(common_keys) / min(len(keys1), len(keys2)) if min(len(keys1), len(keys2)) > 0 else 0

                    if key_overlap_ratio > 0.3:
                        return False, f"헤더는 동일하지만 키값이 겹침 ({len(common_keys)}개 중복)"

                    # 키값이 전혀 겹치지 않으면 같은 형식의 별도 테이블 (템플릿 테이블)
                    if len(keys1) >= 2 and len(keys2) >= 2 and len(common_keys) == 0:
                        return False, f"헤더는 동일하지만 키값이 전혀 겹치지 않음 (템플릿 테이블)"

                if len(common) >= 2:
                    return True, f"헤더가 동일함 ({len(common)}개 일치)"
                else:
                    return True, f"헤더가 부분 일치함 ({partial_match_count}개 부분 일치)"

    # 첫 번째 테이블에만 헤더가 있는 경우
    elif table1_info['headers'] and not table2_info['headers']:
        if page_diff >= 1 and page_diff <= 2:
            # 텍스트 연결성 체크
            if table1_info['texts'] and table2_info['texts']:
                last_texts = table1_info['texts'][-5:]
                first_texts = table2_info['texts'][:5]

                for prev_text in last_texts:
                    for curr_text in first_texts:
                        if is_continuation_text(prev_text, curr_text):
                            return True, f"헤더 테이블 뒤 데이터 테이블 - 텍스트 연결: '{prev_text}' -> '{curr_text}'"

            # 첫 번째 테이블에 타이틀이 있고, 연속 페이지면 연결
            if title1 and page_diff == 1:
                return True, f"타이틀이 있는 헤더 테이블 뒤 데이터 테이블 (연속 페이지)"

            # 타이틀 없어도, 같은 열 개수이고 연속 페이지면 연결
            if page_diff == 1 and table1_info['cols'] == table2_info['cols'] and table1_info['cols'] >= 2:
                return True, f"헤더 테이블 뒤 데이터 테이블 (열 {table1_info['cols']}개, 연속 페이지)"

    # 첫 번째 테이블에 타이틀이 있고 두 번째에 없는 경우
    if title1 and not title2:
        if page_diff == 1:
            if table1_info['headers'] and table2_info['headers']:
                if len(table1_info['headers']) >= 3 and len(table2_info['headers']) >= 3:
                    if has_similar_structure(table1_info, table2_info):
                        return True, f"타이틀이 있는 테이블 뒤 계속 (구조 유사, 연속 페이지)"

            elif not table1_info['headers'] and not table2_info['headers']:
                if table1_info['cols'] == table2_info['cols'] and table1_info['cols'] >= 2:
                    return True, f"타이틀이 있는 데이터 테이블 뒤 계속 (열 {table1_info['cols']}개, 연속 페이지)"

    # 둘 다 헤더가 없고 타이틀도 없는 경우
    if not table1_info['headers'] and not table2_info['headers']:
        if not title1 and not title2:
            if page_diff == 1:
                col_diff = abs(table1_info['cols'] - table2_info['cols'])
                if col_diff <= 1 and table1_info['cols'] >= 2:
                    # 키값 중복 체크 (50% 이상 겹치면 반복 테이블)
                    keys1 = set([normalize_text(k) for k in table1_info['key_values'][:10]])
                    keys2 = set([normalize_text(k) for k in table2_info['key_values'][:10]])

                    if keys1 and keys2:
                        common_keys = keys1 & keys2
                        key_overlap_ratio = len(common_keys) / min(len(keys1), len(keys2)) if min(len(keys1), len(keys2)) > 0 else 0

                        if key_overlap_ratio > 0.5:
                            return False, f"같은 형식의 반복 테이블 (키값 {len(common_keys)}개 중복, 중복률 {key_overlap_ratio:.0%})"

                    return True, f"헤더 없는 데이터 테이블 연속 (열 {table1_info['cols']}개 vs {table2_info['cols']}개, 연속 페이지)"

    # 마지막 체크: 첫 번째 테이블에 헤더 없고, 두 번째 테이블에 헤더가 있는 경우
    # (다른 모든 연결 조건이 실패한 경우에만 이 이유로 차단)
    if not table1_info['headers'] and table2_info['headers']:
        return False, "헤더 재등장 (첫 번째 테이블 헤더 없음, 두 번째 테이블 헤더 있음)"

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

    # 모든 행 합치기
    all_rows = []
    for table_info in table_group:
        table = table_info['original_table']
        rows = table.get('rows', [])
        all_rows.extend(rows)

    merged['rows'] = all_rows

    return merged


def find_connected_table_groups(tables: List[Dict], all_text_elements: List[Dict] = None) -> Tuple:
    """연결된 테이블 그룹 찾기"""
    table_infos = [extract_table_info(table, all_text_elements, tables) for table in tables]

    # 페이지 번호순으로 정렬
    sorted_indices = sorted(range(len(table_infos)),
                          key=lambda i: table_infos[i]['page_no'])

    visited = set()
    groups = []
    connection_reasons = []
    all_disconnection_reasons = {}

    # 먼저 모든 인접 테이블 쌍에 대해 연속성 체크
    for idx in range(len(sorted_indices) - 1):
        i = sorted_indices[idx]
        j = sorted_indices[idx + 1]

        is_connected, reason = check_table_connection(
            table_infos[i],
            table_infos[j]
        )

        if not is_connected:
            if i not in all_disconnection_reasons:
                all_disconnection_reasons[i] = []
            all_disconnection_reasons[i].append(f"Table {i} -X-> {j}: {reason}")

    # 연결된 그룹 찾기
    for i in sorted_indices:
        if i in visited:
            continue

        current_group = [i]
        current_reasons = []
        visited.add(i)

        current_idx = i
        current_pos = sorted_indices.index(i)

        for j_pos in range(current_pos + 1, len(sorted_indices)):
            j = sorted_indices[j_pos]

            if j in visited:
                continue

            is_connected, reason = check_table_connection(
                table_infos[current_idx],
                table_infos[j]
            )

            if is_connected:
                current_group.append(j)
                current_reasons.append(f"Table {current_idx} -> {j}: {reason}")
                visited.add(j)
                current_idx = j
            else:
                break

        groups.append(current_group)
        connection_reasons.append(current_reasons)

    return groups, connection_reasons, all_disconnection_reasons, table_infos


def extract_tables_and_text_from_kids(data: Dict) -> Tuple[List[Dict], List[Dict]]:
    """opendataloader JSON의 kids 구조에서 테이블과 텍스트 추출

    Args:
        data: opendataloader JSON 데이터

    Returns:
        (tables, text_elements) 튜플
    """
    tables = []
    text_elements = []

    def process_element(element, depth=0):
        """재귀적으로 element 처리"""
        if not isinstance(element, dict):
            return

        elem_type = element.get('type', '')

        # 테이블 추출
        if elem_type == 'table':
            # opendataloader bbox: [left, bottom, right, top]
            bbox = element.get('bounding box')
            page_num = element.get('page number', 0)

            # rows 배열에서 셀 데이터 추출
            rows = []
            for row_elem in element.get('rows', []):
                if isinstance(row_elem, dict) and row_elem.get('type') == 'table row':
                    row_cells = []
                    for cell_elem in row_elem.get('cells', []):
                        if isinstance(cell_elem, dict) and cell_elem.get('type') == 'table cell':
                            # 셀 내용 추출 (kids에서 content 가져오기)
                            cell_text = extract_text_from_element(cell_elem)
                            cell_text_str = ' '.join(cell_text).strip()
                            cell_bbox = cell_elem.get('bounding box')

                            row_cells.append({
                                'text': cell_text_str,
                                'bbox': cell_bbox
                            })
                    if row_cells:
                        rows.append(row_cells)

            table_data = {
                'bbox': bbox,
                'page': page_num,
                'rows': rows,
                'num_rows': len(rows),
                'num_cols': len(rows[0]) if rows else 0
            }
            tables.append(table_data)

        # 텍스트 요소 추출 (paragraph, heading, text block 등)
        if elem_type in ['paragraph', 'heading', 'text block', 'caption', 'header', 'footer']:
            bbox = element.get('bounding box')
            page_num = element.get('page number', 0)
            text_content = extract_text_from_element(element)
            text_str = ' '.join(text_content).strip()

            if text_str and bbox:
                text_elements.append({
                    'text': text_str,
                    'bbox': bbox,
                    'page': page_num,
                    'type': elem_type
                })

        # list의 list items 처리
        if elem_type == 'list' and 'list items' in element:
            for item in element.get('list items', []):
                if isinstance(item, dict):
                    item_bbox = item.get('bounding box')
                    item_page = item.get('page number', 0)
                    item_content = extract_text_from_element(item)
                    item_text = ' '.join(item_content).strip()

                    if item_text and item_bbox:
                        text_elements.append({
                            'text': item_text,
                            'bbox': item_bbox,
                            'page': item_page,
                            'type': 'list item'
                        })

                    # list item의 kids도 재귀적으로 처리
                    if 'kids' in item:
                        for kid in item.get('kids', []):
                            process_element(kid, depth + 1)

        # 재귀적으로 kids 처리 (테이블 내부는 이미 처리했으므로 제외)
        if elem_type != 'table' and 'kids' in element:
            for kid in element.get('kids', []):
                process_element(kid, depth + 1)

    # 최상위 kids 처리
    if 'kids' in data:
        for element in data.get('kids', []):
            process_element(element)

    return tables, text_elements


def process_pdf(pdf_path: str, output_dir="merged_tables_output"):
    """PDF 처리 메인 함수

    Args:
        pdf_path: PDF 파일 경로
        output_dir: 출력 디렉토리 (opendataloader_version/merged_tables_output)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"{'='*60}\n")

    # opendataloader로 PDF 변환
    output_temp_dir = "output"  # opendataloader 원본 JSON 저장 위치
    os.makedirs(output_temp_dir, exist_ok=True)

    opendataloader_pdf.convert(
        input_path=[pdf_path],
        output_dir=output_temp_dir,
        format=["json"]
    )

    # 생성된 JSON 파일 찾기
    pdf_name = Path(pdf_path).stem
    json_files = list(Path(output_temp_dir).glob(f"{pdf_name}*.json"))

    if not json_files:
        print(f"[ERROR] No JSON files generated for {pdf_name}")
        return None

    json_file = json_files[0]
    print(f"[OK] JSON generated: {json_file}")

    # JSON 로드
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # opendataloader JSON의 kids 구조에서 테이블과 텍스트 추출
    tables, all_text_elements = extract_tables_and_text_from_kids(data)

    print(f"[OK] Found {len(tables)} tables")
    print(f"[OK] Found {len(all_text_elements)} text elements")

    # 연결된 테이블 그룹 찾기
    groups, reasons, all_disconnections, table_infos = find_connected_table_groups(tables, all_text_elements)

    merged_groups = [g for g in groups if len(g) > 1]
    print(f"[OK] Found {len(merged_groups)} merged groups")

    # 단일 테이블 그룹
    single_table_groups = []
    for g_idx, (group, group_reasons) in enumerate(zip(groups, reasons)):
        if len(group) == 1:
            table_idx = group[0]
            disconnection_reason = ""
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
        'source_file': str(pdf_path),
        'original_table_count': len(tables),
        'merged_groups': [],
        'single_tables': single_table_groups,
        'all_disconnections': all_disconnections,
        'table_infos': [
            {
                'table_idx': idx,
                'page_no': info['page_no'],
                'title': info['title']
            }
            for idx, info in enumerate(table_infos)
        ]
    }

    for group_idx, (group, group_reasons) in enumerate(zip(merged_groups, zip(reasons, groups))):
        if len(group) <= 1:
            continue

        group_disconnections = []
        for table_idx in group:
            if table_idx in all_disconnections:
                group_disconnections.extend(all_disconnections[table_idx])

        actual_reasons = group_reasons[0]
        group_info = {
            'group_id': group_idx,
            'table_indices': group,
            'pages': [table_infos[i]['page_no'] for i in group],
            'connection_reasons': actual_reasons,
            'disconnection_reasons': group_disconnections,
            'merged_table': merge_tables([table_infos[i] for i in group])
        }
        result['merged_groups'].append(group_info)

        print(f"\n  Group {group_idx + 1}:")
        print(f"    Tables: {group}")
        print(f"    Pages: {group_info['pages']}")
        for idx in group:
            if table_infos[idx]['title']:
                print(f"    Table {idx} title: {table_infos[idx]['title']}")
        for reason in actual_reasons:
            print(f"    - {reason}")

    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{pdf_name}_merged.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 추출된 테이블 데이터를 table_output에 저장 (visualize에서 사용)
    table_output_dir = "table_output"
    os.makedirs(table_output_dir, exist_ok=True)

    table_json_data = {
        'tables': tables,
        'text_elements': all_text_elements
    }
    table_output_file = os.path.join(table_output_dir, f"{pdf_name}.json")
    with open(table_output_file, 'w', encoding='utf-8') as f:
        json.dump(table_json_data, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Result saved: {output_file}")
    return output_file


if __name__ == "__main__":
    input_dir = "../input"
    output_dir = "merged_tables_output"

    print("=" * 60)
    print("opendataloader 기반 테이블 병합")
    print("=" * 60)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found!")
    else:
        pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in '{input_dir}'")
        else:
            for pdf_file in pdf_files:
                pdf_path = os.path.join(input_dir, pdf_file)
                try:
                    process_pdf(pdf_path, output_dir)
                except Exception as e:
                    print(f"\n[ERROR] {pdf_file} processing failed: {e}\n")
                    continue

            print("\n" + "=" * 60)
            print("Complete!")
            print("=" * 60)
