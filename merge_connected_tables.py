"""
페이지를 넘어가는 테이블을 찾아서 병합하는 스크립트
텍스트 연결성을 기반으로 같은 테이블로 판단
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER


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


def find_table_title(table: Dict, all_texts: List[Dict], page_height: float = 792.0) -> str:
    """테이블 위 300px 내의 텍스트를 타이틀로 추출"""
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

    # 거리가 가장 가까운 텍스트를 타이틀로 선택
    if title_candidates:
        title_candidates.sort(key=lambda x: x['distance'])
        return title_candidates[0]['text']

    # 타이틀이 없을 수도 있음
    return ""


def extract_table_info(table: Dict, all_texts: List[Dict] = None) -> Dict:
    """테이블에서 필요한 정보 추출"""
    cells = table.get('data', {}).get('table_cells', [])
    page_no = table.get('prov', [{}])[0].get('page_no', -1)

    # 텍스트 추출
    texts = [cell.get('text', '').strip() for cell in cells if cell.get('text', '').strip()]

    # 헤더 셀 찾기
    headers = [cell.get('text', '').strip() for cell in cells
               if cell.get('column_header', False) and cell.get('text', '').strip()]

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
        title = find_table_title(table, all_texts)

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
    # 열 개수가 같거나 비슷한지
    col_diff = abs(table1_info['cols'] - table2_info['cols'])
    if col_diff > 2:  # 열 개수 차이가 2개 이하
        return False

    # 헤더가 비슷한지 확인
    if table1_info['headers'] and table2_info['headers']:
        headers1 = [normalize_text(h) for h in table1_info['headers']]
        headers2 = [normalize_text(h) for h in table2_info['headers']]

        # 헤더 일치도 확인
        common = set(headers1) & set(headers2)
        if len(common) / max(len(headers1), len(headers2)) > 0.5:
            return True

    return True  # 구조 정보만으로는 판단하기 어려우면 True


def check_table_connection(table1_info: Dict, table2_info: Dict) -> Tuple[bool, str]:
    """두 테이블이 연결되어 있는지 확인"""
    # 페이지 차이 확인
    page_diff = table2_info['page_no'] - table1_info['page_no']

    # 페이지가 너무 멀리 떨어져 있으면 연결 안됨
    if page_diff < 0 or page_diff > 2:
        return False, "페이지가 너무 멀리 떨어져 있음"

    # 구조 유사성 확인
    if not has_similar_structure(table1_info, table2_info):
        return False, "테이블 구조가 다름"

    # 텍스트 연결성 확인 (명확한 텍스트 이어짐)
    if table1_info['texts'] and table2_info['texts']:
        # 마지막 몇 개의 셀 텍스트와 첫 몇 개의 셀 텍스트 비교
        last_texts = table1_info['texts'][-5:]
        first_texts = table2_info['texts'][:5]

        for prev_text in last_texts:
            for curr_text in first_texts:
                if is_continuation_text(prev_text, curr_text):
                    return True, f"텍스트 연결: '{prev_text}' -> '{curr_text}'"

    # 헤더가 동일한 경우도 연결된 테이블로 간주
    # 첫 번째 테이블에 헤더가 있고, 두 번째 테이블에 헤더가 없는 경우도 허용
    # (헤더가 있는 테이블 뒤에 데이터만 있는 테이블이 올 수 있음)
    if table1_info['headers'] and table2_info['headers']:
        # 둘 다 헤더가 있는 경우: 헤더 비교
        headers1 = [normalize_text(h) for h in table1_info['headers'] if len(normalize_text(h)) > 0]
        headers2 = [normalize_text(h) for h in table2_info['headers'] if len(normalize_text(h)) > 0]

        # 헤더가 충분히 있어야 함 (최소 2개)
        if len(headers1) >= 2 and len(headers2) >= 2:
            headers1_set = set(headers1)
            headers2_set = set(headers2)

            # 공통 헤더 개수와 비율 계산
            common = headers1_set & headers2_set
            similarity = len(common) / max(len(headers1_set), len(headers2_set))

            # 헤더가 80% 이상 일치하고, 공통 헤더가 최소 2개 이상
            if similarity >= 0.8 and len(common) >= 2:
                # 헤더가 동일한 경우, 키값도 체크
                keys1 = set([normalize_text(k) for k in table1_info['key_values'][:10]])  # 최대 10개만 비교
                keys2 = set([normalize_text(k) for k in table2_info['key_values'][:10]])

                # 키값이 겹치면 다른 테이블로 판단
                if keys1 and keys2:
                    common_keys = keys1 & keys2
                    key_overlap_ratio = len(common_keys) / min(len(keys1), len(keys2)) if min(len(keys1), len(keys2)) > 0 else 0

                    if key_overlap_ratio > 0.3:  # 30% 이상 겹치면 다른 테이블
                        return False, f"헤더는 동일하지만 키값이 겹침 ({len(common_keys)}개 중복)"

                return True, f"헤더가 동일함 ({len(common)}개 일치)"

    # 첫 번째 테이블에만 헤더가 있는 경우
    # 텍스트 연결성이 있으면 연결 (열 개수가 달라도 가능)
    elif table1_info['headers'] and not table2_info['headers']:
        # 페이지가 바로 이어지는 경우 (1-2페이지 차이)
        if page_diff >= 1 and page_diff <= 2:
            # 텍스트 연결성 체크 (이미 위에서 체크했지만 여기서도 확인)
            if table1_info['texts'] and table2_info['texts']:
                last_texts = table1_info['texts'][-5:]
                first_texts = table2_info['texts'][:5]

                for prev_text in last_texts:
                    for curr_text in first_texts:
                        if is_continuation_text(prev_text, curr_text):
                            return True, f"헤더 테이블 뒤 데이터 테이블 - 텍스트 연결: '{prev_text}' -> '{curr_text}'"

            # 텍스트 연결성이 명확하지 않아도, 열 개수가 비슷하고 연속 페이지면 연결
            col_diff = abs(table1_info['cols'] - table2_info['cols'])
            if col_diff <= 2 and page_diff == 1 and table1_info['cols'] >= 2:
                return True, f"헤더 테이블 뒤 데이터 테이블 (열 {table1_info['cols']}개 vs {table2_info['cols']}개, 유사 구조)"

    # 두 번째 테이블에만 헤더가 있는 경우는 일반적으로 새로운 테이블
    # (연결되지 않음)

    # "연속 페이지, 유사 구조" 조건 제거
    # 명확한 연결성(텍스트 이어짐, 헤더 동일, 헤더+데이터)만 인정

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


def find_connected_table_groups(tables: List[Dict], all_texts: List[Dict] = None) -> List[List[int]]:
    """연결된 테이블 그룹 찾기"""
    table_infos = [extract_table_info(table, all_texts) for table in tables]

    # 페이지 번호순으로 정렬
    sorted_indices = sorted(range(len(table_infos)),
                          key=lambda i: table_infos[i]['page_no'])

    visited = set()
    groups = []
    connection_reasons = []
    all_disconnection_reasons = {}  # 모든 비연속 이유 저장 (테이블 인덱스별)

    # 먼저 모든 인접 테이블 쌍에 대해 연속성 체크
    for idx in range(len(sorted_indices) - 1):
        i = sorted_indices[idx]
        j = sorted_indices[idx + 1]

        is_connected, reason = check_table_connection(
            table_infos[i],
            table_infos[j]
        )

        if not is_connected:
            # 비연속인 경우 저장
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

        # 순차적으로 바로 다음 테이블만 확인 (건너뛰기 방지)
        current_idx = i
        for j_pos in range(sorted_indices.index(i) + 1, len(sorted_indices)):
            j = sorted_indices[j_pos]

            if j in visited:
                continue

            # IMPORTANT: 현재 그룹의 마지막 테이블과 바로 다음 테이블만 비교
            # 테이블을 건너뛰면 안됨 (j는 current_idx의 바로 다음 인덱스여야 함)
            if j != current_idx + 1:
                # 다음 테이블이 아니면 더 이상 연결 확인하지 않음
                break

            is_connected, reason = check_table_connection(
                table_infos[current_idx],
                table_infos[j]
            )

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

        # 병합된 그룹만 필터링
        merged_groups = [g for g in groups if len(g) > 1]
        merged_reasons = [r for g, r in zip(groups, reasons) if len(g) > 1]

        if not merged_groups:
            story.append(Paragraph("병합된 테이블 없음", normal_style))
            story.append(Spacer(1, 0.3*cm))
            continue

        total_merged += len(merged_groups)

        for group_idx, (group, group_reasons) in enumerate(zip(merged_groups, merged_reasons), 1):
            # 그룹 정보
            pages = [table_infos[i]['page_no'] for i in group]
            story.append(Paragraph(
                f"병합 그룹 {group_idx}: {len(group)}개 테이블 (페이지 {pages})",
                normal_style
            ))

            # 연결 이유
            for reason in group_reasons:
                story.append(Paragraph(f"  • {reason}", normal_style))

            # 비연속 이유 (이 그룹의 테이블들과 관련된 것)
            for table_idx in group:
                if table_idx in all_disconnections:
                    for reason in all_disconnections[table_idx]:
                        story.append(Paragraph(f"  • (비연속) {reason}", normal_style))

            # 테이블 미리보기 데이터 준비
            preview_data = [["테이블", "페이지", "행수", "열수", "샘플 텍스트"]]

            for idx in group:
                info = table_infos[idx]
                sample_text = info['texts'][0][:30] + "..." if info['texts'] else ""
                preview_data.append([
                    f"#{idx}",
                    str(info['page_no']),
                    str(info['rows']),
                    str(info['cols']),
                    sample_text
                ])

            # 테이블 스타일
            t = Table(preview_data, colWidths=[2*cm, 2*cm, 2*cm, 2*cm, 8*cm])
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

    for json_file in json_files:
        print(f"\n처리 중: {json_file.name}")

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

        # 연결된 테이블 그룹 찾기
        groups, reasons, all_disconnections, table_infos = find_connected_table_groups(tables, all_texts)

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
        output_file = os.path.join(
            output_dir,
            f"{json_file.stem}_merged.json"
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
