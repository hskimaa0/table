"""
원본 PDF에 연결된 테이블들을 색깔별로 오버레이하여 시각화
"""
import json
import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io


def register_korean_font():
    """한글 폰트 등록"""
    try:
        pdfmetrics.registerFont(TTFont('Korean', 'malgun.ttf'))
        return 'Korean'
    except:
        try:
            pdfmetrics.registerFont(TTFont('Korean', 'C:/Windows/Fonts/malgun.ttf'))
            return 'Korean'
        except:
            print("Warning: Could not load Korean font. Using default font.")
            return 'Helvetica'


def get_color_for_group(group_idx: int) -> Color:
    """그룹 인덱스에 따라 서로 다른 색상 반환"""
    colors = [
        HexColor('#FF6B6B'),  # 빨강
        HexColor('#4ECDC4'),  # 청록
        HexColor('#45B7D1'),  # 파랑
        HexColor('#FFA07A'),  # 연어색
        HexColor('#98D8C8'),  # 민트
        HexColor('#F7DC6F'),  # 노랑
        HexColor('#BB8FCE'),  # 보라
        HexColor('#85C1E2'),  # 하늘색
        HexColor('#F8B739'),  # 주황
        HexColor('#52BE80'),  # 초록
        HexColor('#EC7063'),  # 분홍
        HexColor('#5DADE2'),  # 밝은 파랑
        HexColor('#48C9B0'),  # 에메랄드
        HexColor('#F39C12'),  # 오렌지
        HexColor('#9B59B6'),  # 자주
        HexColor('#1ABC9C'),  # 청록2
        HexColor('#E74C3C'),  # 빨강2
        HexColor('#3498DB'),  # 파랑2
        HexColor('#2ECC71'),  # 초록2
        HexColor('#E67E22'),  # 주황2
    ]
    return colors[group_idx % len(colors)]


def convert_bbox_coordinates(bbox: Dict, page_height: float) -> Dict:
    """
    JSON bbox 좌표를 PDF 좌표로 변환
    TOPLEFT origin -> BOTTOMLEFT origin (PDF 좌표계)
    """
    if bbox.get('coord_origin') == 'TOPLEFT':
        return {
            'x1': bbox['l'],
            'y1': page_height - bbox['b'],  # bottom을 변환
            'x2': bbox['r'],
            'y2': page_height - bbox['t'],  # top을 변환
        }
    else:  # BOTTOMLEFT
        return {
            'x1': bbox['l'],
            'y1': bbox['b'],
            'x2': bbox['r'],
            'y2': bbox['t'],
        }


def create_overlay_pdf(pdf_path: str, merged_data: Dict, original_tables: List[Dict], output_path: str, font_name: str):
    """원본 PDF에 연결된 테이블 박스를 오버레이"""

    # 원본 PDF 읽기
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        num_pages = len(pdf_reader.pages)

        # 첫 페이지에서 크기 가져오기
        first_page = pdf_reader.pages[0]
        page_width = float(first_page.mediabox.width)
        page_height = float(first_page.mediabox.height)

    print(f"  PDF 크기: {page_width} x {page_height}")

    # 병합된 그룹 정보 가져오기
    merged_groups = merged_data.get('merged_groups', [])

    if not merged_groups:
        print("  병합된 테이블 그룹이 없습니다.")
        return

    # 페이지별로 그려야 할 박스들 정리
    page_annotations = {}

    for group in merged_groups:
        group_id = group['group_id']
        color = get_color_for_group(group_id)
        table_indices = group.get('table_indices', [])
        pages = group.get('pages', [])

        print(f"    그룹 {group_id + 1}: 테이블 인덱스 {table_indices}, 페이지 {pages}")

        # 각 테이블 인덱스에 대해 원본 테이블의 bbox 가져오기
        for table_idx in table_indices:
            if table_idx >= len(original_tables):
                continue

            original_table = original_tables[table_idx]
            prov_list = original_table.get('prov', [])

            for prov in prov_list:
                page_no = prov.get('page_no', -1)
                bbox = prov.get('bbox', {})

                if page_no > 0 and bbox:
                    if page_no not in page_annotations:
                        page_annotations[page_no] = []

                    page_annotations[page_no].append({
                        'bbox': bbox,
                        'color': color,
                        'group_id': group_id,
                        'table_idx': table_idx,
                        'group_info': f"Group {group_id + 1} ({len(table_indices)} tables, {len(pages)} pages)"
                    })

    # 각 페이지에 대한 오버레이 생성
    overlay_pages = {}

    for page_no, annotations in page_annotations.items():
        # 메모리에 오버레이 PDF 생성
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(page_width, page_height))

        # 각 annotation 그리기
        for ann in annotations:
            bbox = ann['bbox']
            color = ann['color']

            # 좌표 변환
            coords = convert_bbox_coordinates(bbox, page_height)

            # 반투명 박스 그리기
            can.setFillColor(color, alpha=0.15)
            can.setStrokeColor(color, alpha=0.8)
            can.setLineWidth(3)

            # 사각형 그리기
            can.rect(
                coords['x1'],
                coords['y1'],
                coords['x2'] - coords['x1'],
                coords['y2'] - coords['y1'],
                fill=1,
                stroke=1
            )

            # 그룹 번호 표시
            can.setFont(font_name, 10)
            can.setFillColor(color, alpha=1.0)
            can.drawString(
                coords['x1'] + 5,
                coords['y2'] - 15,
                f"그룹 {ann['group_id'] + 1}"
            )

        can.save()
        packet.seek(0)
        overlay_pages[page_no] = PyPDF2.PdfReader(packet).pages[0]

    # 원본 PDF에 오버레이 병합
    pdf_writer = PyPDF2.PdfWriter()

    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            page_no = page_num + 1  # 1-based indexing

            # 해당 페이지에 오버레이가 있으면 병합
            if page_no in overlay_pages:
                page.merge_page(overlay_pages[page_no])

            pdf_writer.add_page(page)

    # 출력 파일 저장
    with open(output_path, 'wb') as f:
        pdf_writer.write(f)

    print(f"  오버레이 PDF 생성 완료: {output_path}")
    print(f"  총 {len(merged_groups)}개 그룹을 {len(page_annotations)}개 페이지에 표시")


def process_all_pdfs(input_pdf_dir: str, merged_json_dir: str, table_json_dir: str, output_dir: str):
    """모든 PDF 파일 처리"""

    os.makedirs(output_dir, exist_ok=True)

    # 한글 폰트 등록
    font_name = register_korean_font()

    # JSON 파일들 찾기
    json_files = list(Path(merged_json_dir).glob('*_merged.json'))

    # merge_summary.json 제외
    json_files = [f for f in json_files if f.name != 'merge_summary.json']

    print(f"\n총 {len(json_files)}개 파일 처리\n")

    for json_file in json_files:
        print(f"처리 중: {json_file.name}")

        # JSON 데이터 로드
        with open(json_file, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)

        # 원본 테이블 JSON 로드
        source_file = merged_data.get('source_file', '')
        table_json_path = os.path.join(table_json_dir, os.path.basename(source_file))

        if not os.path.exists(table_json_path):
            print(f"  경고: 원본 테이블 JSON을 찾을 수 없습니다: {table_json_path}")
            continue

        with open(table_json_path, 'r', encoding='utf-8') as f:
            table_data = json.load(f)
            original_tables = table_data.get('tables', [])

        print(f"  원본 테이블 수: {len(original_tables)}")

        # 해당하는 원본 PDF 찾기
        # JSON 파일명에서 _merged.json 제거하고 _tables도 제거
        base_name = json_file.stem.replace('_merged', '').replace('_tables', '')
        pdf_name = base_name + '.pdf'
        pdf_path = os.path.join(input_pdf_dir, pdf_name)

        if not os.path.exists(pdf_path):
            print(f"  경고: PDF 파일을 찾을 수 없습니다: {pdf_path}")
            continue

        # 출력 파일명
        output_filename = base_name + '_connected_tables.pdf'
        output_path = os.path.join(output_dir, output_filename)

        # 오버레이 생성
        create_overlay_pdf(pdf_path, merged_data, original_tables, output_path, font_name)
        print()


def create_legend_pdf(output_dir: str, summary_data: Dict, table_json_dir: str, font_name: str):
    """범례가 포함된 설명 PDF 생성"""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    pdf_path = os.path.join(output_dir, "연결된_테이블_설명.pdf")
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
        fontSize=10,
        leading=14
    )

    detail_style = ParagraphStyle(
        'Detail',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=8,
        leading=10,
        leftIndent=20
    )

    # 제목
    story.append(Paragraph("연결된 테이블 시각화 가이드", title_style))
    story.append(Spacer(1, 0.5*cm))

    # 설명
    story.append(Paragraph("이 문서는 PDF에서 여러 페이지에 걸쳐 이어지는 테이블들을 색깔별로 표시한 것입니다.", normal_style))
    story.append(Paragraph("같은 색깔의 박스로 표시된 테이블들은 하나의 연결된 테이블입니다.", normal_style))
    story.append(Spacer(1, 0.5*cm))

    # 파일별 상세 정보
    for file_data in summary_data.get('files', []):
        source_file = os.path.basename(file_data['source_file'])
        merged_groups = file_data.get('merged_groups', [])

        if not merged_groups:
            continue

        # 원본 테이블 데이터 로드
        table_json_path = os.path.join(table_json_dir, source_file)
        original_tables = []
        if os.path.exists(table_json_path):
            with open(table_json_path, 'r', encoding='utf-8') as f:
                table_data = json.load(f)
                original_tables = table_data.get('tables', [])

        story.append(Paragraph(f"파일: {source_file}", heading_style))

        # 각 그룹별 상세 정보
        for group in merged_groups:
            group_id = group['group_id']
            color = get_color_for_group(group_id)
            pages = group['pages']
            table_indices = group['table_indices']
            reasons = group['connection_reasons']

            # 그룹 헤더
            story.append(Spacer(1, 0.3*cm))
            color_box = f'<para backColor="{color.hexval()}" textColor="white"><b> 그룹 {group_id + 1} </b></para>'
            story.append(Paragraph(color_box, normal_style))
            story.append(Paragraph(f"테이블 인덱스: {table_indices}", detail_style))
            story.append(Paragraph(f"페이지: {pages}", detail_style))
            story.append(Spacer(1, 0.2*cm))

            # 연결 상세 정보
            if reasons:
                story.append(Paragraph("<b>연결 상세:</b>", detail_style))
                for reason in reasons:
                    # 테이블 인덱스 추출
                    parts = reason.split(': ')
                    if len(parts) >= 2:
                        connection_part = parts[0]  # "Table X -> Y"
                        reason_part = ': '.join(parts[1:])  # 나머지 이유

                        # 테이블 인덱스 추출
                        import re
                        match = re.search(r'Table (\d+) -> (\d+)', connection_part)
                        if match and original_tables:
                            idx1 = int(match.group(1))
                            idx2 = int(match.group(2))

                            # 각 테이블의 샘플 텍스트 가져오기
                            table1_sample = ""
                            table2_sample = ""

                            if idx1 < len(original_tables):
                                cells1 = original_tables[idx1].get('data', {}).get('table_cells', [])
                                if cells1:
                                    # 마지막 몇 개 셀의 텍스트
                                    last_texts = [c.get('text', '').strip() for c in cells1[-3:] if c.get('text', '').strip()]
                                    table1_sample = ' | '.join(last_texts)[:100]

                            if idx2 < len(original_tables):
                                cells2 = original_tables[idx2].get('data', {}).get('table_cells', [])
                                if cells2:
                                    # 첫 몇 개 셀의 텍스트
                                    first_texts = [c.get('text', '').strip() for c in cells2[:3] if c.get('text', '').strip()]
                                    table2_sample = ' | '.join(first_texts)[:100]

                            # 상세 정보 출력
                            story.append(Paragraph(f"• {connection_part}: {reason_part}", detail_style))
                            if table1_sample:
                                story.append(Paragraph(f"  └ Table {idx1} 끝: {table1_sample}...", detail_style))
                            if table2_sample:
                                story.append(Paragraph(f"  └ Table {idx2} 시작: {table2_sample}...", detail_style))
                        else:
                            story.append(Paragraph(f"• {reason}", detail_style))
                    else:
                        story.append(Paragraph(f"• {reason}", detail_style))

            story.append(Spacer(1, 0.3*cm))

        story.append(PageBreak())

    doc.build(story)
    print(f"\n범례 PDF 생성 완료: {pdf_path}")


if __name__ == "__main__":
    import sys
    import io

    # Windows 콘솔 인코딩 문제 해결
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    input_pdf_dir = "input"
    merged_json_dir = "merged_tables_output"
    table_json_dir = "table_output"
    output_dir = "visualized_pdfs"

    print("=" * 60)
    print("연결된 테이블 오버레이 시각화 시작")
    print("=" * 60)

    # 한글 폰트 등록
    font_name = register_korean_font()

    # PDF 처리
    process_all_pdfs(input_pdf_dir, merged_json_dir, table_json_dir, output_dir)

    # 범례 PDF 생성
    summary_file = os.path.join(merged_json_dir, "merge_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        create_legend_pdf(output_dir, summary_data, table_json_dir, font_name)

    print("\n" + "=" * 60)
    print("처리 완료!")
    print("=" * 60)
    print(f"\n결과 파일 위치: {output_dir}/")
    print(f"- *_connected_tables.pdf: 오버레이된 PDF 파일들")
    print(f"- 연결된_테이블_설명.pdf: 범례 및 설명")
