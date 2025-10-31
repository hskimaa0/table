"""
output 폴더의 JSON 파일에서 테이블 데이터만 추출하여 table_output 폴더에 저장
"""
import json
import os
from pathlib import Path


def extract_tables_from_json(input_dir: str, output_dir: str):
    """JSON 파일에서 테이블 데이터만 추출"""

    os.makedirs(output_dir, exist_ok=True)

    # input 디렉토리의 모든 JSON 파일 처리
    json_files = list(Path(input_dir).glob('*.json'))

    print(f"총 {len(json_files)}개의 JSON 파일 발견\n")

    for json_file in json_files:
        print(f"처리 중: {json_file.name}")

        try:
            # JSON 파일 읽기
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON 파싱 오류: {e}")
            print(f"  → 파일이 비어있거나 손상됨\n")
            continue
        except Exception as e:
            print(f"  ✗ 파일 읽기 오류: {e}\n")
            continue

        # tables 데이터만 추출
        if 'tables' in data:
            tables = data['tables']

            # 새로운 데이터 구조 생성
            table_data = {
                'source_file': str(json_file),
                'num_tables': len(tables),
                'tables': tables
            }

            # 출력 파일명 생성 (원본 이름 + _tables.json)
            # Windows에서 사용할 수 없는 문자 제거
            safe_stem = json_file.stem.replace('+', '_').replace(':', '_').replace('?', '_').replace('*', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            output_filename = safe_stem + '_tables.json'
            output_path = os.path.join(output_dir, output_filename)

            # 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(table_data, f, ensure_ascii=False, indent=2)

            print(f"  ✓ 테이블 {len(tables)}개 추출 완료")
            print(f"  → {output_path}\n")
        else:
            print(f"  ✗ 테이블 데이터 없음\n")


if __name__ == "__main__":
    import sys
    import io

    # Windows 콘솔 인코딩 문제 해결
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    input_dir = "output"
    output_dir = "table_output"

    print("=" * 60)
    print("테이블 데이터 추출 시작")
    print("=" * 60)
    print()

    extract_tables_from_json(input_dir, output_dir)

    print("=" * 60)
    print("추출 완료!")
    print("=" * 60)
    print(f"\n결과 위치: {output_dir}/")
