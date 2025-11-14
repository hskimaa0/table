"""
add_sample.json의 데이터를 train_data_example.json에 자동으로 병합하는 스크립트
"""
import json


def merge_training_data(add_file='add_sample.json', main_file='train_data_example.json', backup=True):
    """
    add_sample.json의 데이터를 train_data_example.json에 병합

    Args:
        add_file: 추가할 데이터가 담긴 JSON 파일
        main_file: 메인 학습 데이터 파일
        backup: True면 백업 파일 생성
    """
    print("=" * 60)
    print("학습 데이터 병합 스크립트")
    print("=" * 60)

    # 1. add_sample.json 로드
    try:
        with open(add_file, 'r', encoding='utf-8') as f:
            new_data = json.load(f)
        print(f"✅ {add_file} 로드 완료: {len(new_data)}개 항목")
    except FileNotFoundError:
        print(f"❌ {add_file} 파일이 없습니다.")
        print(f"\n{add_file} 파일을 먼저 생성하세요:")
        print("""
[
  {
    "text": "추가할 텍스트",
    "label": 0,
    "note": "메모(선택)"
  }
]
        """)
        return
    except json.JSONDecodeError as e:
        print(f"❌ {add_file} JSON 파싱 오류: {e}")
        return

    # 2. train_data_example.json 로드
    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            main_data = json.load(f)
        print(f"✅ {main_file} 로드 완료: {len(main_data['train_data'])}개 항목")
    except FileNotFoundError:
        print(f"❌ {main_file} 파일이 없습니다.")
        return
    except json.JSONDecodeError as e:
        print(f"❌ {main_file} JSON 파싱 오류: {e}")
        return

    # 3. 백업 생성 (옵션)
    if backup:
        backup_file = main_file.replace('.json', '_backup.json')
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(main_data, f, ensure_ascii=False, indent=2)
        print(f"💾 백업 생성: {backup_file}")

    # 4. 데이터 검증
    label_counts = {0: 0, 1: 0, 2: 0}
    valid_items = []

    for idx, item in enumerate(new_data):
        # 필수 필드 확인
        if 'text' not in item or 'label' not in item:
            print(f"⚠️  항목 {idx+1} 건너뜀: 'text' 또는 'label' 필드 없음")
            continue

        # 라벨 검증
        if item['label'] not in [0, 1, 2]:
            print(f"⚠️  항목 {idx+1} 건너뜀: 잘못된 label 값 ({item['label']})")
            continue

        valid_items.append(item)
        label_counts[item['label']] += 1

    if not valid_items:
        print("❌ 추가할 유효한 데이터가 없습니다.")
        return

    print(f"\n📊 추가할 데이터 통계:")
    print(f"  제목(0): {label_counts[0]}개")
    print(f"  설명(1): {label_counts[1]}개")
    print(f"  제목아님(2): {label_counts[2]}개")
    print(f"  총합: {len(valid_items)}개")

    # 5. 데이터 병합
    before_count = len(main_data['train_data'])
    main_data['train_data'].extend(valid_items)
    after_count = len(main_data['train_data'])

    # 6. 저장
    with open(main_file, 'w', encoding='utf-8') as f:
        json.dump(main_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 병합 완료!")
    print(f"  이전: {before_count}개")
    print(f"  추가: {after_count - before_count}개")
    print(f"  현재: {after_count}개")
    print(f"  저장: {main_file}")

    # 7. add_sample.json 비우기 (선택)
    print(f"\n💡 {add_file}을 비우시겠습니까? (y/n): ", end='')
    choice = input().strip().lower()
    if choice == 'y':
        with open(add_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print(f"✅ {add_file} 초기화 완료")


if __name__ == '__main__':
    merge_training_data()
