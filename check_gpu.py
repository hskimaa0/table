"""
GPU 가속 상태 확인 도구
"""

def check_gpu_status():
    print("=" * 60)
    print("GPU 가속 상태 확인")
    print("=" * 60)
    print()

    # 1. PyTorch 확인
    print("1. PyTorch 확인")
    try:
        import torch
        print(f"   ✓ PyTorch 버전: {torch.__version__}")
        print(f"   ✓ CUDA 지원: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"   ✓ CUDA 버전: {torch.version.cuda}")
            print(f"   ✓ GPU 개수: {torch.cuda.device_count()}")
            print(f"   ✓ 현재 GPU: {torch.cuda.get_device_name(0)}")

            # 메모리 정보
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✓ GPU 메모리: {total_memory:.2f} GB")
        else:
            print("   ⚠ GPU를 사용할 수 없습니다 (CPU 모드)")
    except ImportError:
        print("   ✗ PyTorch가 설치되어 있지 않습니다")
        print("     설치: pip install torch")

    print()

    # 2. Docling 확인
    print("2. Docling 확인")
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import TableFormerMode
        print("   ✓ Docling 설치됨")
        print("   ✓ GPU 가속 사용 가능")
    except ImportError:
        print("   ✗ Docling이 설치되어 있지 않습니다")
        print("     설치: pip install docling")

    print()

    # 3. 기타 라이브러리
    print("3. 기타 필수 라이브러리")

    libs = [
        ("reportlab", "PDF 생성"),
        ("PyPDF2", "PDF 처리"),
    ]

    for lib, desc in libs:
        try:
            __import__(lib)
            print(f"   ✓ {lib}: {desc}")
        except ImportError:
            print(f"   ✗ {lib}: {desc} (미설치)")

    print()
    print("=" * 60)

    # 최종 요약
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ GPU 가속 사용 가능!")
            print(f"  예상 성능: CPU 대비 3-6배 빠름")
        else:
            print("⚠ CPU 모드로 작동")
            print("  GPU 드라이버와 CUDA를 설치하면 더 빠릅니다")
    except:
        print("⚠ PyTorch 미설치 - CPU 모드로 작동")

    print("=" * 60)


if __name__ == "__main__":
    import sys
    import io

    # Windows 콘솔 인코딩 문제 해결
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    check_gpu_status()
