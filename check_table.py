import json
import sys
import io

# Windows 콘솔 인코딩 문제 해결
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

with open('table_output/파일_tables.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    tables = data['tables']

    for idx in [20, 21]:
        if idx < len(tables):
            t = tables[idx]
            page = t.get('prov', [{}])[0].get('page_no', -1)
            cells = t.get('data', {}).get('table_cells', [])
            headers = [c['text'] for c in cells if c.get('column_header') and c.get('text')]
            all_texts = [c['text'] for c in cells if c.get('text')]
            keys = [c['text'] for c in cells if not c.get('column_header') and c.get('start_col_offset_idx') == 0 and c.get('text')][:5]
            rows = max([c.get('end_row_offset_idx', 0) for c in cells]) if cells else 0
            cols = max([c.get('end_col_offset_idx', 0) for c in cells]) if cells else 0

            print(f'\n테이블 {idx}:')
            print(f'  페이지: {page}')
            print(f'  행/열: {rows}/{cols}')
            print(f'  헤더: {headers[:5]}')
            print(f'  첫 열 데이터: {keys}')
            print(f'  마지막 5개 텍스트: {all_texts[-5:]}')
            print(f'  처음 5개 텍스트: {all_texts[:5]}')
