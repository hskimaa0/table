"""
input 폴더의 PDF 파일명에서 Windows에서 사용할 수 없는 문자를 제거
"""
import os
from pathlib import Path


def rename_files_in_directory(directory: str):
    """디렉토리 내 파일명의 특수문자를 _로 변경"""

    renamed_count = 0

    # 모든 PDF 파일 확인
    for file_path in Path(directory).glob('*.pdf'):
        original_name = file_path.name

        # 안전한 파일명 생성
        safe_name = original_name.replace('+', '_').replace(':', '_').replace('?', '_').replace('*', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

        # 이름이 변경되는 경우에만 rename
        if original_name != safe_name:
            new_path = file_path.parent / safe_name

            # 이미 존재하는지 확인
            if new_path.exists():
                print(f'⚠ 건너뛰기 (이미 존재): {safe_name}')
                continue

            file_path.rename(new_path)
            print(f'✓ {original_name}')
            print(f'  → {safe_name}\n')
            renamed_count += 1

    if renamed_count == 0:
        print('변경할 파일이 없습니다.')
    else:
        print(f'총 {renamed_count}개 파일명 변경 완료')

    return renamed_count


if __name__ == "__main__":
    import sys
    import io

    # Windows 콘솔 인코딩 문제 해결
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    input_dir = "../input"

    print("=" * 60)
    print("PDF 파일명 정리 시작")
    print("=" * 60)
    print()

    rename_files_in_directory(input_dir)

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
