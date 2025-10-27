"""
PDF to JSON Converter using Docling
Processes all PDF files in the input folder and outputs JSON files to the output folder.
"""

import os
import json
from pathlib import Path
from docling.document_converter import DocumentConverter


def convert_pdfs_to_json(input_folder="input", output_folder="output"):
    """
    Convert all PDF files from input folder to JSON format in output folder.

    Args:
        input_folder (str): Path to folder containing PDF files
        output_folder (str): Path to folder where JSON files will be saved
    """
    # Create folders if they don't exist
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Get all PDF files from input folder
    pdf_files = list(input_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{input_folder}' folder")
        return

    print(f"Found {len(pdf_files)} PDF file(s)")

    # Check which files already have JSON output
    files_to_process = []
    skipped_files = []

    for pdf_file in pdf_files:
        output_filename = pdf_file.stem + ".json"
        output_file = output_path / output_filename

        if output_file.exists():
            skipped_files.append(pdf_file.name)
        else:
            files_to_process.append(pdf_file)

    if skipped_files:
        print(f"\nSkipping {len(skipped_files)} already processed file(s):")
        for filename in skipped_files:
            print(f"  ✓ {filename}")

    if not files_to_process:
        print(f"\nAll PDF files already processed! Nothing to do.")
        return

    print(f"\nProcessing {len(files_to_process)} new file(s)...")

    # Check GPU availability
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🖥️  Device: {device.upper()}")
        if device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
    except ImportError:
        print("\n⚠️  PyTorch not found - GPU acceleration may not be available")

    # Initialize Docling converter with optimized settings
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

    # Configure pipeline options for better performance
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True  # Enable table detection
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # Accurate mode

    # Create format options
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }

    # Initialize converter with optimized settings
    converter = DocumentConverter(
        format_options=format_options
    )

    print("✓ Converter initialized (GPU will be used automatically if available)")

    # Process each PDF file
    processed_count = 0
    error_files = []

    for pdf_file in files_to_process:
        try:
            print(f"\nProcessing: {pdf_file.name}")

            # Convert PDF using Docling
            result = converter.convert(str(pdf_file))

            # Generate output filename
            output_filename = pdf_file.stem + ".json"
            output_file = output_path / output_filename

            # Export to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                # Convert document to dict and save as JSON
                json.dump(result.document.export_to_dict(), f, ensure_ascii=False, indent=2)

            print(f"✓ Successfully converted: {pdf_file.name} -> {output_filename}")
            processed_count += 1

        except RuntimeError as e:
            error_msg = str(e)
            print(f"✗ Error processing {pdf_file.name}: {error_msg}")
            error_files.append((pdf_file.name, error_msg))
            continue
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Unexpected error processing {pdf_file.name}: {error_msg}")
            error_files.append((pdf_file.name, error_msg))
            continue

    print(f"\nConversion complete!")
    print(f"  ✓ Processed: {processed_count}")
    print(f"  ⊘ Skipped (already done): {len(skipped_files)}")
    print(f"  ✗ Errors: {len(error_files)}")
    print(f"  Total: {len(pdf_files)}")

    if error_files:
        print(f"\n⚠ Files with errors:")
        for filename, error in error_files:
            print(f"  - {filename}")
            print(f"    Reason: {error[:100]}...")

        # Save error log
        error_log_file = output_path / "conversion_errors.log"
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write("PDF to JSON Conversion Errors\n")
            f.write("=" * 80 + "\n\n")
            for filename, error in error_files:
                f.write(f"File: {filename}\n")
                f.write(f"Error: {error}\n")
                f.write("-" * 80 + "\n\n")
        print(f"\n  Error log saved to: {error_log_file}")

    print(f"\nCheck the '{output_folder}' folder for results.")


if __name__ == "__main__":
    # Run the conversion
    convert_pdfs_to_json()
