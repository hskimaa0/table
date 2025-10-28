import os
import json
from pathlib import Path
from docling.document_converter import DocumentConverter

def convert_pdfs_to_docling(input_folder="pdfs", output_folder="output"):
    """
    PDF 파일들을 Docling으로 변환하여 JSON 출력 생성
    """
    # 입력/출력 폴더 경로 설정
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    # DocumentConverter 초기화
    converter = DocumentConverter()

    # PDF 파일 목록 가져오기
    pdf_files = list(input_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    # 각 PDF 파일 처리
    for pdf_file in pdf_files:
        # JSON 출력 파일 경로
        output_file = output_path / f"{pdf_file.stem}.json"

        # 이미 변환된 파일은 스킵 (0KB 파일 제외)
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"⊙ Skipping (already exists): {pdf_file.name}")
            continue

        print(f"\nProcessing: {pdf_file.name}")

        try:
            # PDF 변환
            result = converter.convert(str(pdf_file))

            # Document를 dict로 변환 후 JSON으로 저장
            doc_dict = result.document.export_to_dict()

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, ensure_ascii=False, indent=2)

            file_size = output_file.stat().st_size / 1024  # KB
            print(f"✓ Saved: {output_file.name} ({file_size:.1f} KB)")

        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {str(e)}")
            # 실패 시 0KB 파일 삭제
            if output_file.exists() and output_file.stat().st_size == 0:
                output_file.unlink()
            continue

if __name__ == "__main__":
    # ../input 폴더의 PDF 처리
    convert_pdfs_to_docling(
        input_folder="../input",
        output_folder="output"
    )
    print("\n✓ Conversion complete!")
