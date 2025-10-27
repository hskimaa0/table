"""
opendataloader 기반 테이블 병합 전체 파이프라인
1. PDF에서 테이블 추출 및 병합
2. 원본 PDF에 오버레이 시각화
"""

import os
import sys
from pathlib import Path

# UTF-8 출력 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from merge_connected_tables import process_pdf
from visualize_connected_tables import create_overlay_pdf, register_korean_font
import json


def process_single_pdf(pdf_path,
                       merged_dir="merged_tables_output",
                       visualized_dir="visualized_pdfs",
                       table_dir="table_output"):
    """단일 PDF 전체 처리"""

    print(f"\n{'='*70}")
    print(f"OPENDATALOADER 기반 테이블 병합 파이프라인")
    print(f"{'='*70}")
    print(f"PDF: {pdf_path}\n")

    pdf_name = Path(pdf_path).stem

    # Step 1: 테이블 추출 및 병합
    print(f"\n{'='*70}")
    print(f"STEP 1: 테이블 추출 및 병합")
    print(f"{'='*70}\n")

    merge_result = process_pdf(pdf_path, merged_dir)

    if not merge_result:
        print("[ERROR] 병합 실패!")
        return False

    merged_json = os.path.join(merged_dir, f"{pdf_name}_merged.json")

    # Step 2: 원본 PDF에 오버레이
    print(f"\n{'='*70}")
    print(f"STEP 2: 원본 PDF에 오버레이")
    print(f"{'='*70}\n")

    # 한글 폰트 등록
    font_name = register_korean_font()

    # 병합 결과 로드
    with open(merged_json, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)

    # 원본 테이블 JSON 로드
    table_json_path = os.path.join(table_dir, f"{pdf_name}.json")

    if not os.path.exists(table_json_path):
        print(f"[ERROR] 원본 테이블 JSON을 찾을 수 없습니다: {table_json_path}")
        return False

    with open(table_json_path, 'r', encoding='utf-8') as f:
        table_data = json.load(f)
        original_tables = table_data.get('tables', [])

    # 출력 파일명
    os.makedirs(visualized_dir, exist_ok=True)
    output_filename = f"{pdf_name}_connected_tables.pdf"
    output_path = os.path.join(visualized_dir, output_filename)

    # 오버레이 생성
    try:
        create_overlay_pdf(pdf_path, merged_data, original_tables, output_path, font_name)
        print(f"[OK] 오버레이 완료: {output_path}")
    except Exception as e:
        print(f"[ERROR] 오버레이 실패: {e}")
        return False

    # 최종 결과 요약
    print(f"\n{'='*70}")
    print(f"완료! 결과 파일:")
    print(f"{'='*70}")
    print(f"1. 병합 JSON:      {merged_json}")
    print(f"2. 오버레이 PDF:   {output_path}")
    print(f"{'='*70}\n")

    return True


def process_all_pdfs_in_folder(input_dir="../input",
                               merged_dir="merged_tables_output",
                               visualized_dir="visualized_pdfs",
                               table_dir="table_output"):
    """input 폴더의 모든 PDF 처리"""

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found!")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'")
        return

    print(f"\n{'='*70}")
    print(f"총 {len(pdf_files)}개의 PDF 파일 처리 시작")
    print(f"{'='*70}\n")

    success_count = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)

        try:
            if process_single_pdf(pdf_path, merged_dir, visualized_dir, table_dir):
                success_count += 1
        except Exception as e:
            print(f"\n[ERROR] {pdf_file} 처리 중 오류: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print(f"전체 처리 완료: {success_count}/{len(pdf_files)} 성공")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='opendataloader 기반 테이블 병합 파이프라인')
    parser.add_argument('--pdf', type=str, help='처리할 PDF 파일 경로 (지정 안 하면 input 폴더의 모든 PDF 처리)')
    parser.add_argument('--input-dir', type=str, default='../input', help='입력 PDF 폴더 (기본값: ../input)')

    args = parser.parse_args()

    if args.pdf:
        # 단일 PDF 처리
        if os.path.exists(args.pdf):
            process_single_pdf(args.pdf)
        else:
            print(f"Error: PDF file not found: {args.pdf}")
    else:
        # 폴더 내 모든 PDF 처리
        process_all_pdfs_in_folder(args.input_dir)
