"""
사전 학습된 Transformer 모델을 사용한 타이틀 분류기
Lazy loading: 실제 사용 시점에 모델 로드
"""
import re
from typing import List, Dict
from transformers import pipeline
import torch


class TitleClassifier:
    """
    사전 학습된 NLP 모델을 사용하여 타이틀 후보를 평가하는 분류기
    Lazy loading: 실제로 사용될 때 모델 로드
    """

    def __init__(self):
        """
        초기화 (모델은 로드하지 않음, 필요할 때 로드)
        """
        self.classifier = None
        self._loading_attempted = False

    def _ensure_model_loaded(self):
        """
        모델이 로드되지 않았으면 로드 (lazy loading)
        최초 1회만 실행됨
        """
        if self._loading_attempted:
            return

        self._loading_attempted = True

        try:
            print("\n" + "="*60)
            print("🤖 타이틀 분류 AI 모델 로딩 중...")
            print("   모델: MiniLM (22MB, 매우 가벼움)")
            print("   최초 다운로드: 인터넷 필요 (1회만)")
            print("   이후 실행: 로컬 캐시 사용 (빠름)")
            print("="*60)

            # GPU 사용 가능하면 GPU, 아니면 CPU
            device = 0 if torch.cuda.is_available() else -1

            # 매우 가벼운 모델 (22MB)
            # 로컬 캐시 활성화 (기본값이지만 명시)
            import os
            os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')

            self.classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-deberta-v3-xsmall",
                device=device
            )

            print("\n" + "="*60)
            print(f"✅ 모델 로드 완료! (device: {'GPU' if device == 0 else 'CPU'})")
            print("="*60 + "\n")
        except Exception as e:
            print(f"\n⚠️  모델 로드 실패: {e}")
            print("📝 Fallback: 규칙 기반 시스템 사용\n")
            self.classifier = None

    def score_candidate(self, text: str, distance: float) -> float:
        """
        타이틀 후보의 적합성 점수 계산

        Args:
            text: 후보 텍스트
            distance: 테이블로부터의 거리 (픽셀)

        Returns:
            점수 (0~100)
        """
        # Lazy loading: 최초 사용 시 모델 로드
        self._ensure_model_loaded()

        if self.classifier is None:
            # Fallback: 규칙 기반
            return self._score_rule_based(text, distance)

        # 모델 기반 분류
        try:
            # Zero-shot classification으로 "타이틀다움" 평가
            candidate_labels = [
                "This is a formal table title with table number like 'Table 4.21' or '표 4.21' and a descriptive caption",
                "This is auxiliary information such as unit notation like '(단위: cm)' or '(Unit: mm)' or continuation mark like '(계속)' or '(continued)'",
                "This is just a page number like '184' or section number",
                "This is a section heading or subsection title without table number",
            ]

            result = self.classifier(
                text,
                candidate_labels,
                multi_label=False
            )

            # 라벨별 점수 계산
            label_scores = dict(zip(result['labels'], result['scores']))

            # 표 타이틀인 경우
            if result['labels'][0] == candidate_labels[0]:
                score = result['scores'][0] * 100
            # 부가 정보인 경우
            elif result['labels'][0] == candidate_labels[1]:
                score = (1 - result['scores'][0]) * 20
            # 페이지 번호인 경우
            elif result['labels'][0] == candidate_labels[2]:
                score = (1 - result['scores'][0]) * 10
            # 섹션 제목인 경우
            else:
                score = result['scores'][0] * 40

            # 거리 보정 (가까우면 보너스)
            if distance <= 50:
                score += 15
            elif distance <= 100:
                score += 10
            elif distance <= 200:
                score += 5

            return min(100, max(0, score))

        except Exception as e:
            print(f"⚠️  모델 추론 실패: {e}, fallback 사용")
            return self._score_rule_based(text, distance)

    def _score_rule_based(self, text: str, distance: float) -> float:
        """
        Fallback 규칙 기반 점수 계산
        """
        score = 0.0

        # 표 패턴
        if re.search(r'^표\s*[\dA-Z]+\.[\dA-Z]+', text) or re.search(r'^Table\s*[\dA-Z]+', text, re.IGNORECASE):
            score += 60.0

        # 길이
        text_len = len(text.strip())
        if 10 <= text_len <= 100:
            score += 20.0

        # 부가 정보 패턴 (패널티)
        if re.search(r'^\s*\(\s*단위', text) or re.search(r'^\s*\(\s*Unit', text, re.IGNORECASE):
            score -= 50.0

        # 거리
        if distance <= 30:
            score += 15.0
        elif distance <= 60:
            score += 10.0

        return max(0.0, score)

    def select_best_title(self, candidates: List[Dict]) -> str:
        """
        타이틀 후보 중 가장 적합한 것을 선택

        Args:
            candidates: [{'text': str, 'distance': float, ...}, ...]

        Returns:
            선택된 타이틀 (없으면 빈 문자열)
        """
        if not candidates:
            return ""

        # 후보가 1개면 모델 없이 바로 반환 (속도 최적화)
        if len(candidates) == 1:
            return candidates[0]['text']

        # 후보가 2개 이상일 때는 모델 사용 (하드코딩 금지)
        # 각 후보의 점수 계산
        scored = []
        for c in candidates:
            score = self.score_candidate(c['text'], c['distance'])
            scored.append({
                'text': c['text'],
                'score': score,
                'distance': c['distance']
            })

        # 점수순 정렬
        scored.sort(key=lambda x: x['score'], reverse=True)

        # 최소 점수 기준
        if scored[0]['score'] > 30:
            return scored[0]['text']

        return ""


# 전역 인스턴스 (싱글톤 패턴)
_classifier_instance = None


def get_classifier():
    """
    전역 분류기 인스턴스 반환 (싱글톤)
    최초 1회만 인스턴스 생성, 이후에는 재사용
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = TitleClassifier()
    return _classifier_instance


def select_best_title_with_model(candidates: List[Dict]) -> str:
    """
    모델을 사용하여 최적의 타이틀 선택

    Args:
        candidates: [{'text': str, 'distance': float, ...}, ...]

    Returns:
        선택된 타이틀
    """
    classifier = get_classifier()
    return classifier.select_best_title(candidates)


if __name__ == "__main__":
    # 테스트
    print("타이틀 분류기 테스트\n")

    test_candidates = [
        {'text': '( 단위 cm)', 'distance': 20},
        {'text': '표 4.21 목계호안의 채움돌 입경 (1:1.0 의 경우 )', 'distance': 60},
        {'text': '(2) 통나무격자 호안', 'distance': 100},
        {'text': '184', 'distance': 300},
    ]

    classifier = TitleClassifier()

    print("타이틀 후보 점수:")
    for c in test_candidates:
        score = classifier.score_candidate(c['text'], c['distance'])
        print(f"  [{score:6.2f}점] (거리: {c['distance']:3.0f}px) {c['text']}")

    best = classifier.select_best_title(test_candidates)
    print(f"\n선택된 타이틀: '{best}'")
