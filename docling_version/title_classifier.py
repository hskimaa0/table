"""
타이틀 후보 중에서 가장 적절한 타이틀을 선택하는 모델
"""
import re
from typing import List, Dict, Tuple


def score_title_candidate(text: str, distance: float) -> float:
    """
    타이틀 후보의 적합성 점수 계산 (0~1, 높을수록 좋음)

    점수 기준:
    - 표 패턴 ("표 X.X", "Table X.X"): 높은 점수
    - 길이: 적당한 길이 (10~100자)
    - 부가 정보 패턴 ("(단위", "(Unit"): 낮은 점수
    - 거리: 가까울수록 좋지만, 표 패턴이 있으면 멀어도 OK
    """
    score = 0.0

    # 1. 표 번호 패턴 체크 (가장 중요)
    table_patterns = [
        r'^표\s*[\dA-Z]+\.[\dA-Z]+',  # 표 4.21, 표 A.1
        r'^Table\s*[\dA-Z]+\.[\dA-Z]+',  # Table 4.21
        r'^<표\s*[\dA-Z]+\.[\dA-Z]+>',  # <표 4.21>
        r'^\[표\s*[\dA-Z]+\.[\dA-Z]+\]',  # [표 4.21]
    ]

    has_table_pattern = False
    for pattern in table_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 50.0  # 매우 높은 점수
            has_table_pattern = True
            break

    # 2. 길이 점수 (적당한 길이 선호)
    text_len = len(text.strip())
    if 10 <= text_len <= 100:
        score += 20.0
    elif 5 <= text_len < 10:
        score += 10.0
    elif text_len > 100:
        score += 5.0

    # 3. 부가 정보 패턴 체크 (패널티)
    auxiliary_patterns = [
        r'^\s*\(\s*단위',  # (단위, ( 단위
        r'^\s*\(\s*Unit',  # (Unit
        r'^\s*\(\s*mm\s*\)',  # (mm)
        r'^\s*\(\s*cm\s*\)',  # (cm)
        r'^\s*\(\s*m\s*\)',  # (m)
        r'^\s*\(\s*계속\s*\)',  # (계속)
        r'^\s*\(\s*continued\s*\)',  # (continued)
    ]

    for pattern in auxiliary_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score -= 40.0  # 큰 패널티
            break

    # 4. 거리 점수
    # 표 패턴이 있으면 거리 페널티 완화
    if has_table_pattern:
        # 표 패턴이 있으면 300px 이내는 모두 괜찮음
        if distance <= 300:
            score += 10.0
    else:
        # 표 패턴 없으면 가까울수록 좋음
        if distance <= 30:
            score += 15.0
        elif distance <= 60:
            score += 10.0
        elif distance <= 100:
            score += 5.0
        else:
            score -= (distance - 100) * 0.05  # 거리가 멀수록 감점

    # 5. 특수 문자 비율 체크
    special_char_count = len(re.findall(r'[^\w\s가-힣]', text))
    if special_char_count / max(text_len, 1) > 0.5:
        score -= 10.0  # 특수문자가 너무 많으면 감점

    # 6. 숫자만 있는 경우 (페이지 번호)
    if text.strip().isdigit():
        score -= 50.0

    # 7. 설명이 포함된 경우 (긍정)
    descriptive_words = ['의', '에 대한', '관한', '현황', '결과', '분석', '비교', '목록']
    for word in descriptive_words:
        if word in text:
            score += 5.0
            break

    return max(0.0, score)


def select_best_title(candidates: List[Dict]) -> str:
    """
    타이틀 후보 중 가장 적합한 것을 선택

    Args:
        candidates: [{'text': str, 'distance': float, ...}, ...]

    Returns:
        선택된 타이틀 (없으면 빈 문자열)
    """
    if not candidates:
        return ""

    # 각 후보의 점수 계산
    scored_candidates = []
    for candidate in candidates:
        text = candidate['text']
        distance = candidate['distance']
        score = score_title_candidate(text, distance)

        scored_candidates.append({
            'text': text,
            'distance': distance,
            'score': score
        })

    # 점수순으로 정렬
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)

    # 가장 높은 점수의 후보 선택
    best = scored_candidates[0]

    # 최소 점수 기준 (0 이상이어야 선택)
    if best['score'] > 0:
        return best['text']

    return ""


def compare_titles_for_merge(title1: str, title2: str) -> Tuple[bool, float]:
    """
    두 타이틀이 같은 테이블인지 판단

    Returns:
        (is_same_table, similarity)
    """
    if not title1 or not title2:
        return False, 0.0

    # 정규화
    t1 = re.sub(r'\s+', '', title1)
    t2 = re.sub(r'\s+', '', title2)

    # 완전 일치
    if t1 == t2:
        return True, 1.0

    # 표 번호 추출
    pattern = r'표\s*([\dA-Z]+\.[\dA-Z]+)'
    match1 = re.search(pattern, title1)
    match2 = re.search(pattern, title2)

    if match1 and match2:
        num1 = match1.group(1)
        num2 = match2.group(1)

        # 표 번호가 다르면 다른 테이블
        if num1 != num2:
            return False, 0.0

        # 표 번호가 같으면 같은 테이블
        return True, 1.0

    # 일반 유사도 계산
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, t1, t2).ratio()

    if ratio >= 0.85:
        return True, ratio
    else:
        return False, ratio


if __name__ == "__main__":
    # 테스트
    test_candidates = [
        {'text': '( 단위 cm)', 'distance': 20},
        {'text': '표 4.21 목계호안의 채움돌 입경 (1:1.0 의 경우 )', 'distance': 60},
        {'text': '(2) 통나무격자 호안', 'distance': 100},
    ]

    print("타이틀 후보 점수:")
    for c in test_candidates:
        score = score_title_candidate(c['text'], c['distance'])
        print(f"  [{score:6.2f}점] {c['text']}")

    print(f"\n선택된 타이틀: {select_best_title(test_candidates)}")
