"""
타이틀 추출 API
Zero-shot 분류 방식: 표 제목 1개 + 설명 여러 개 추출
"""
from flask import Flask, jsonify, request
import copy
import re
import numpy as np

app = Flask(__name__)

# ========== 상수 정의 ==========
# 거리 및 필터링 관련
Y_LINE_TOLERANCE = 100  # 같은 줄로 간주할 y 좌표 허용 오차 (px)
X_GAP_THRESHOLD = 100  # 텍스트 사이 공백 추가 기준 간격 (px)
UP_MULTIPLIER = 1.5  # 표 위쪽 탐색 범위 (표 높이의 배수)
X_TOLERANCE = 800  # 수평 근접 허용 거리 (px)

# ML 모델 관련
ZEROSHOT_MODEL = "joeddav/xlm-roberta-large-xnli"  # Zero-shot 분류 (다국어, 한국어 우수)
ML_DEVICE = 0  # -1: CPU, 0: GPU
MAX_TEXT_INPUT_LENGTH = 512  # ML 모델 입력 최대 길이

# Zero-shot 설정
TITLE_SCORE_THRESHOLD = 0.5  # 제목 판정 최소 점수
DESC_SCORE_THRESHOLD = 0.3   # 설명 판정 최소 점수

# ML 모델 로드
zeroshot_classifier = None
embedding_model = None
mecab = None  # 형태소 분석기

# 디바이스 설정
import torch

def _resolve_device():
    """ML_DEVICE 설정과 CUDA 가용성에 따라 디바이스 결정"""
    use_gpu = (ML_DEVICE == 0 and torch.cuda.is_available())
    return "cuda:0" if use_gpu else "cpu"

DEVICE_STR = _resolve_device()
print(f"▶ Inference device = {DEVICE_STR}")

# GPU 성능 최적화 (A100/RTX40 계열)
if DEVICE_STR.startswith("cuda"):
    torch.set_float32_matmul_precision("high")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# Zero-shot 분류기 로드 (FP16 지원)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import sentencepiece  # noqa: F401

    # SentencePiece 기반 slow tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ZEROSHOT_MODEL, use_fast=False)

    # FP16 로드 (GPU면 FP16, CPU면 FP32)
    dtype = torch.float16 if DEVICE_STR.startswith("cuda") else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        ZEROSHOT_MODEL,
        torch_dtype=dtype
    )

    device_id = 0 if DEVICE_STR.startswith("cuda") else -1
    zeroshot_classifier = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device=device_id
    )
    print(f"✅ Zero-shot 로드 완료 ({ZEROSHOT_MODEL}, device={DEVICE_STR}, dtype={dtype})")
except ImportError as e:
    print(f"⚠️  라이브러리 없음: {e}")
    print("   → pip install sentencepiece transformers 실행 필요")
    zeroshot_classifier = None
except Exception as e:
    print(f"⚠️  Zero-shot 로드 실패: {e}")
    zeroshot_classifier = None

# Sentence Transformer 임베딩 모델 로드
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=DEVICE_STR)
    print(f"✅ Embedding 모델 로드 완료 (paraphrase-multilingual-MiniLM-L12-v2, device={DEVICE_STR})")
except ImportError as e:
    print(f"⚠️  라이브러리 없음: {e}")
    print("   → pip install sentence-transformers 실행 필요")
    embedding_model = None
except Exception as e:
    print(f"⚠️  Embedding 모델 로드 실패: {e}")
    embedding_model = None

# 형태소 분석기 로드 (MeCab-ko)
try:
    # 방법 1: python-mecab-ko (Windows 전용, 사전 포함)
    import mecab as mecab_module
    mecab_tagger = mecab_module.MeCab()

    class MecabWrapper:
        """MeCab wrapper to match konlpy interface"""
        def __init__(self, tagger):
            self.tagger = tagger

        def pos(self, text):
            """형태소 분석 [(단어, 품사), ...]"""
            try:
                result = self.tagger.pos(text)
                return result
            except Exception as e:
                print(f"  형태소 분석 예외: {e}")
                return []

    mecab = MecabWrapper(mecab_tagger)
    # 테스트
    test_result = mecab.pos("테스트")
    print(f"✅ 형태소 분석기 로드 완료 (python-mecab-ko) - 테스트: {len(test_result)}개 형태소")
except Exception as e:
    try:
        # 방법 2: konlpy.tag.Mecab (Linux/Mac 또는 mecab 직접 설치한 경우)
        from konlpy.tag import Mecab
        mecab = Mecab()
        print(f"✅ 형태소 분석기 로드 완료 (konlpy.Mecab)")
    except:
        try:
            # 방법 3: mecab-python (pip install mecab-python)
            import MeCab
            mecab_tagger = MeCab.Tagger()

            class MecabWrapper2:
                """MeCab wrapper to match konlpy interface"""
                def __init__(self, tagger):
                    self.tagger = tagger

                def pos(self, text):
                    """형태소 분석 [(단어, 품사), ...]"""
                    result = []
                    node = self.tagger.parseToNode(text)
                    while node:
                        if node.surface:  # 빈 노드 제외
                            word = node.surface
                            features = node.feature.split(',')
                            pos_tag = features[0] if features else 'UNKNOWN'
                            result.append((word, pos_tag))
                        node = node.next
                    return result

            mecab = MecabWrapper2(mecab_tagger)
            print(f"✅ 형태소 분석기 로드 완료 (mecab-python)")
        except Exception as e:
            print(f"⚠️  형태소 분석기 로드 실패: {e}")
            print(f"   Windows: pip install python-mecab-ko")
            print(f"   Linux/Mac: pip install konlpy && apt-get install mecab-ko mecab-ko-dic")
            mecab = None

# ========== 유틸리티 함수 ==========
def clean_text(s: str) -> str:
    """텍스트 정리"""
    return re.sub(r"\s+", " ", s).strip()

def clamp_text_len(s: str, max_chars=MAX_TEXT_INPUT_LENGTH):
    """텍스트 길이 제한"""
    s = clean_text(s)
    return s[:max_chars] if len(s) > max_chars else s


def is_trivial(text: str) -> bool:
    """무의미한 텍스트 필터링 (페이지 번호, 저작권 등)"""
    s = text.strip()

    # 1~3자리 숫자만
    if re.fullmatch(r"\d{1,3}", s):
        return True

    # 저작권 표시
    if "all rights reserved" in s.lower():
        return True
    if s.startswith("©"):
        return True

    # 너무 짧음
    if len(s) <= 2:
        return True

    # 숫자만 포함
    if re.match(r'^[\d\s\.\-]+$', s):
        return True

    # 특수문자만
    if len(re.sub(r"[\W_]+", "", s)) <= 1:
        return True

    return False

def iou_1d(a: tuple, b: tuple) -> float:
    """1차원 IoU (수평 겹침 계산)"""
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - overlap
    return overlap / union if union > 0 else 0.0

def horizontally_near(table_x: tuple, text_x: tuple, tol: int = X_TOLERANCE) -> bool:
    """수평으로 겹치거나 근접한지 확인"""
    if iou_1d(table_x, text_x) > 0:
        return True
    return (text_x[1] >= table_x[0] - tol and text_x[0] <= table_x[1] + tol)


# ========== 패턴 스코어/페널티 ==========
UNIT_TOKENS = ["단위", "unit", "u.", "℃", "°c", "%", "mm", "kg", "km", "원", "개", "회"]

# 소제목/섹션 패턴
CIRCLED_RX = r"[\u2460-\u2473\u3251-\u325F]"  # ①-⑳, 21-35
BULLET_RX = r"[•∙·\-–—]"  # 불릿/대시
PAREN_NUM_RX = r"\(?\d{1,2}\)?[.)]"  # (1) 1) 1.

def is_subtitle_like(s: str) -> bool:
    """테이블 바로 위에 자주 오는 '소제목' 패턴:
    - ① ② … / (1) 1) 1. / • - 등 불릿 시작
    - 길이 4~40자 권장
    """
    t = s.strip()
    if re.match(rf"^({CIRCLED_RX}|{PAREN_NUM_RX}|{BULLET_RX})\s*\S+", t):
        if 4 <= len(t) <= 40:
            return True
    return False

def is_section_header_like(s: str) -> bool:
    """상위 섹션 헤더(문서 구조용):
    - 1. 1.1 1.1.1 형태
    - 제3장, 제2절 등
    - ㅇ, ※ 등으로 시작
    """
    t = s.strip()
    # 1. 1.1 1.1.1 형태
    if re.match(r"^\d+(\.\d+){0,3}[\.\s]+\S+", t):
        return True
    # 제3장, 제2절 등
    if re.match(r"^제\s*\d+\s*(장|절|항)\b", t):
        return True
    # ㅇ, ※, - 등으로 시작하는 항목
    if re.match(r"^[ㅇ※\-•]\s+", t):
        return True
    return False

def is_datetime_like(s: str) -> bool:
    """날짜/시간 패턴 판별 (제목으로 부적절)

    주의: 날짜/시간이 '주요 내용'인 텍스트만 필터링 (언급만 하는 경우는 허용)
    """
    t = s.strip()

    # 텍스트가 짧고 주로 날짜/시간으로 구성된 경우만 필터
    # 예: "(산업기사) 2025.11.15.(토) [오후] 14:00 ~"

    # 날짜 + 시간이 함께 있고 텍스트가 짧은 경우
    has_date = re.search(r"\d{4}\.\d{1,2}\.\d{1,2}", t)
    has_time = re.search(r"\d{1,2}:\d{2}", t)
    has_ampm = re.search(r"\[(오전|오후|AM|PM)\]", t)
    has_day = re.search(r"\(월|화|수|목|금|토|일\)", t)

    datetime_count = sum([has_date is not None, has_time is not None,
                          has_ampm is not None, has_day is not None])

    # 날짜/시간 요소가 2개 이상이고, 전체 길이가 짧으면 날짜/시간 텍스트로 판단
    if datetime_count >= 2 and len(t) < 80:
        return True

    # 시간 형식으로 시작하거나 끝나는 경우
    if re.match(r"^\d{1,2}:\d{2}", t) or re.search(r"\d{1,2}:\d{2}\s*[~\-]?\s*$", t):
        return True

    return False

def is_unit_like(s: str) -> bool:
    """단위 표기 라인 판별"""
    t = s.strip().lower()
    # [단위: ℃], (단위 : %), <단위: mm> 등
    if re.match(r"^[\[\(\<]?\s*(단위|unit)\s*[:：]?\s*[\w%°℃㎜\-/]+[\]\)\>]?\s*$", t):
        return True
    # 길이 아주 짧고(<=12) 유닛 토큰만 있는 경우
    if len(t) <= 12 and any(tok in t for tok in UNIT_TOKENS):
        letters = re.sub(r"[^가-힣a-zA-Z]", "", t)
        return len(letters) <= 1
    return False

def is_table_title_like(s: str) -> bool:
    """표 제목 패턴 판별"""
    # '표 B.8 월별 기온', '표 A.6 토지이용현황', '표 B .4' (공백 포함), '표 3-2 연간 실적' 등
    if re.search(r"^표\s*[A-Za-z]?\s*[\.\-]?\s*\d+([\-\.]\d+)?", s.strip()):
        return True
    # 섹션/표 제목 형태(숫자.숫자 제목)
    if re.search(r"^\d+(\.\d+){0,2}\s+[^\[\(]{2,}", s.strip()):
        return True
    return False

def is_cross_reference(s: str) -> bool:
    """교차 참조/설명 문장 판별 (강화)"""
    t = s.strip().replace(" ", "")

    # '표 A.20에의하면', '표 B .4에서', '표 3.2에따르면' 등 (표 번호로 시작하는 참조 문장)
    if re.search(r"^표[A-Za-z]?[\.\-]?\d+([\-\.]\d+)?(에의하면|에따르면|에서|을보면|에나타난|에서와같이|과같이)", t):
        return True

    # '상세한 내용은 다음 표 B.12와 같다' 류
    if re.search(r"(다음표|하기표|본표|아래표)[A-Za-z]?\d+.*(같다|나타낸다|보인다|정리|참조)", t):
        return True

    # '상세한내용은다음표B.12와같다' 같이 붙어쓴 OCR도 커버
    if re.search(r"(상세한내용|자세한내용).*(다음표|아래표).*(같다|나타난다|참조)", t):
        return True

    # 명확한 설명 문장만 (동사 어미로 끝나는 긴 문장)
    if len(t) >= 35 and re.search(r"(바랍니다|바란다|협의대로|안내하|요청)", t):
        return True

    return False

# === 일반 '문장' 판별 (설명/서술형) ===
KOREAN_SENT_END_RX = r"(이다|였다|하였다|했다|된다|되었다|나타난다|나타났다|보인다|보였다|이다\.|다\.|다,)$"
CLAUSE_TOKENS = r"(이며|면서|면서도|고서|고 있으며|으로|로써|으로서|에 따라|에 의하면)"

def is_sentence_like(s: str) -> bool:
    """일반 문장(설명/서술형) 판별

    주의: 제목 패턴 체크 후 사용할 것 (표 A.3 ... 같은 제목을 문장으로 오판 방지)
    """
    t = s.strip()

    # 한국어 서술어/종결 어미 (가장 확실한 문장 패턴)
    if re.search(KOREAN_SENT_END_RX, t):
        return True

    # 분사/이어주는 절 표시 & 길이
    if len(t) >= 18 and re.search(CLAUSE_TOKENS, t):
        return True

    # 마침표로 끝나고 길면 문장 (쉼표는 제목에도 많아서 제외)
    if len(t) >= 30 and (t.endswith(".") or "。" in t or "．" in t):
        return True

    # 라인 내 공백 거의 없고 길이 긴 경우(붙어쓴 문장 OCR)
    if len(t) >= 40 and re.search(r"[가-힣]{15,}", t):
        return True

    return False

def is_noun_phrase(s: str) -> bool:
    """명사형 종결인지 판별 (표 제목은 명사형으로 끝나야 함)

    Returns:
        True: 명사형으로 끝남 (표 제목 가능)
        False: 동사/형용사 어미로 끝남 (표 제목 불가)
    """
    t = s.strip()

    # 빈 문자열
    if not t:
        return False

    # 형태소 분석기 사용 가능하면 우선 사용
    if mecab:
        try:
            # 마침표 제거
            text_to_analyze = t.rstrip(".").rstrip("~").strip()

            # 형태소 분석
            pos_tags = mecab.pos(text_to_analyze)

            if not pos_tags:
                # 분석 실패시 regex로 fallback
                pass
            else:
                # 마지막 형태소의 품사 확인
                last_word, last_pos = pos_tags[-1]

                # 명사형 품사: NNG(일반명사), NNP(고유명사), NNB(의존명사),
                #             XSN(명사파생접미사), SN(숫자), SL(외국어), SH(한자)
                noun_tags = ['NNG', 'NNP', 'NNB', 'XSN', 'SN', 'SL', 'SH']

                # 동사/형용사 관련 품사: VV(동사), VA(형용사), VX(보조용언),
                #                      EC(연결어미), EF(종결어미), EP(선어말어미)
                verb_tags = ['VV', 'VA', 'VX', 'EC', 'EF', 'EP']

                # 조사/부사 (제목으로 부적절): JKB(부사격조사), JX(보조사), JC(접속조사)
                bad_tags = ['JKB', 'JX', 'JC']

                # 기호/구두점은 무시하고 그 앞 형태소 확인
                if last_pos in ['SF', 'SP', 'SS', 'SE', 'SO', 'SW']:
                    if len(pos_tags) >= 2:
                        last_word, last_pos = pos_tags[-2]
                    else:
                        return True  # 기호만 있으면 허용

                # 마지막이 명사형이면 OK
                if last_pos in noun_tags:
                    return True

                # 마지막이 동사/형용사/어미이면 제목 불가
                if last_pos in verb_tags or last_pos in bad_tags:
                    return False

                # 불확실한 경우 fallback to regex

        except Exception as e:
            # fallback to regex
            pass

    # Regex 기반 fallback (형태소 분석기 없거나 실패시)
    verb_endings = [
        # 현재형
        r"한다$", r"된다$", r"있다$", r"없다$", r"같다$",
        r"받는다$", r"준다$", r"진다$", r"난다$", r"란다$",
        r"싶다$", r"좋다$", r"크다$", r"작다$", r"높다$", r"낮다$",

        # 과거형
        r"였다$", r"했다$", r"이다$", r"았다$", r"었다$",
        r"하였다$", r"되었다$", r"나타났다$", r"보였다$",
        r"받았다$", r"주었다$", r"졌다$", r"났다$",

        # 현재진행/상태
        r"하고있다$", r"되고있다$", r"하고$", r"하며$", r"되며$",
        r"이며$", r"면서$", r"지만$", r"으나$", r"지$",

        # 요청/명령형
        r"바란다$", r"바랍니다$", r"바라며$", r"요청한다$",
        r"주시기바란다$", r"주시기바랍니다$", r"참고하여주시기바랍니다$",
        r"하여주시기바랍니다$", r"해주시기바랍니다$",
        r"주십시오$", r"하십시오$", r"하시오$",

        # 보조용언
        r"나타난다$", r"보인다$", r"시킨다$", r"시켰다$",
        r"드린다$", r"드렸다$", r"올린다$", r"올렸다$",

        # 부사형/조사 결합
        r"대로$", r"따라$", r"따르면$", r"의하면$", r"통하여$",
        r"위하여$", r"통해$", r"위해$", r"같이$", r"처럼$",

        # 마침표 포함
        r"한다\.$", r"된다\.$", r"있다\.$", r"없다\.$",
        r"였다\.$", r"했다\.$", r"이다\.$", r"바랍니다\.$"
    ]

    for pattern in verb_endings:
        if re.search(pattern, t):
            return False

    # 마침표로 끝나는 문장 (단, "표 X.X ..." 패턴은 제외)
    if t.endswith(".") and not is_table_title_like(t):
        # 마침표 제거 후 다시 체크
        t_no_period = t.rstrip(".")
        for pattern in verb_endings:
            if re.search(pattern, t_no_period):
                return False

    # 명사형으로 간주
    return True


# ========== bbox 추출 ==========
def get_bbox_from_text(text):
    """text 객체에서 bbox 정보 추출 [l, t, r, b] 형식으로 반환"""
    if 'rect' in text and isinstance(text['rect'], list) and len(text['rect']) >= 4:
        return text['rect']
    elif 'bbox' in text and isinstance(text['bbox'], list) and len(text['bbox']) >= 4:
        return text['bbox']
    elif 'bbox' in text and isinstance(text['bbox'], dict):
        bbox = text['bbox']
        return [bbox.get('l', 0), bbox.get('t', 0), bbox.get('r', 0), bbox.get('b', 0)]
    elif 'merged_bbox' in text:
        return text['merged_bbox']
    return None

def get_bbox_from_table(table):
    """table 객체에서 bbox 정보 추출 [l, t, r, b] 형식으로 반환"""
    if 'bbox' in table and isinstance(table['bbox'], list) and len(table['bbox']) >= 4:
        return table['bbox']
    elif 'bbox' in table and isinstance(table['bbox'], dict):
        bbox = table['bbox']
        return [bbox.get('l', 0), bbox.get('t', 0), bbox.get('r', 0), bbox.get('b', 0)]
    return None

# ========== 텍스트 추출 및 병합 ==========
def extract_text_content(text_obj):
    """text 객체에서 실제 텍스트 추출"""
    if 'merged_text' in text_obj:
        return text_obj['merged_text']

    if 't' in text_obj and isinstance(text_obj['t'], list) and len(text_obj['t']) > 0:
        t_items = text_obj['t']
        sorted_items = sorted(t_items, key=lambda item: item.get('tid', 0))
        texts = []
        for t_item in sorted_items:
            if 'text' in t_item:
                texts.append(t_item['text'])
        if texts:
            return ' '.join(texts)  # 공백으로 연결

    if 'text' in text_obj:
        return text_obj['text']
    elif 'v' in text_obj:
        return text_obj['v']
    return ""

def flatten_text_objects(texts):
    """paraIndex로 그룹화된 텍스트를 개별 't' 요소로 분해"""
    flattened = []
    for text_obj in texts:
        if 't' in text_obj and isinstance(text_obj['t'], list):
            for t_item in text_obj['t']:
                if 'bbox' in t_item and 'text' in t_item:
                    flattened.append({
                        'bbox': t_item['bbox'],
                        'text': t_item['text'],
                        'tid': t_item.get('tid', 0)
                    })
        else:
            flattened.append(text_obj)
    return flattened

def group_texts_by_line(texts, y_tolerance=Y_LINE_TOLERANCE):
    """같은 줄의 텍스트들을 그룹화"""
    if not texts:
        return []

    flattened = flatten_text_objects(texts)
    sorted_texts = sorted(flattened, key=lambda t: get_bbox_from_text(t)[1] if get_bbox_from_text(t) else float('inf'))

    grouped = []
    current_group = []
    group_y_min = None

    for text in sorted_texts:
        bbox = get_bbox_from_text(text)
        if not bbox:
            continue

        y_center = (bbox[1] + bbox[3]) / 2

        if group_y_min is None or abs(y_center - group_y_min) <= y_tolerance:
            current_group.append(text)
            if group_y_min is None:
                group_y_min = y_center
        else:
            if current_group:
                merged = merge_text_group(current_group)
                if merged:
                    grouped.append(merged)
            current_group = [text]
            group_y_min = y_center

    if current_group:
        merged = merge_text_group(current_group)
        if merged:
            grouped.append(merged)

    grouped.sort(key=lambda g: g.get('merged_bbox', [0, 0, 0, 0])[1] if g else float('inf'))
    return grouped

def merge_text_group(text_group):
    """같은 줄의 텍스트들을 x 좌표 순서대로 병합"""
    if not text_group:
        return None

    sorted_group = sorted(text_group, key=lambda t: get_bbox_from_text(t)[0] if get_bbox_from_text(t) else 0)

    text_parts = []
    prev_bbox = None

    for t in sorted_group:
        text_content = extract_text_content(t)
        if not text_content:
            continue

        current_bbox = get_bbox_from_text(t)

        # 텍스트 조각 사이에 항상 공백 추가 (첫 조각 제외)
        if prev_bbox and current_bbox:
            text_parts.append(' ')

        text_parts.append(text_content)
        prev_bbox = current_bbox

    merged_text = ''.join(text_parts)

    bboxes = [get_bbox_from_text(t) for t in sorted_group]
    bboxes = [b for b in bboxes if b]

    if not bboxes:
        return None

    merged_bbox = [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes)
    ]

    merged_obj = text_group[0].copy() if text_group else {}
    merged_obj['merged_text'] = merged_text
    merged_obj['merged_bbox'] = merged_bbox

    return merged_obj

# ========== 후보 수집 ==========
def collect_candidates_for_table(table, texts, all_tables=None):
    """표 위쪽 + 아래쪽 텍스트 후보 수집 (규칙 기반 필터링)

    모든 위치 제한 제거: 표 위/아래 거리 제한 없음, 개수 제한 없음, 다른 표 차단 없음
    """
    table_bbox = get_bbox_from_table(table)
    if not table_bbox:
        return []

    tbx1, tby1, tbx2, tby2 = table_bbox

    # 그룹화된 텍스트
    grouped_texts = group_texts_by_line(texts, y_tolerance=Y_LINE_TOLERANCE)

    candidates = []  # 모든 후보

    for text in grouped_texts:
        if not text:
            continue

        text_bbox = text.get('merged_bbox') or get_bbox_from_text(text)
        if not text_bbox:
            continue

        px1, py1, px2, py2 = text_bbox

        # 수평으로 겹치거나 근접한지 확인
        if not horizontally_near((tbx1, tbx2), (px1, px2), tol=X_TOLERANCE):
            continue

        text_content = clean_text(text.get('merged_text') or extract_text_content(text))

        # 무의미한 텍스트 필터링
        if not text_content or is_trivial(text_content):
            continue

        # 표와 겹치지 않는 텍스트만 수집
        if py2 <= tby1 or py1 >= tby2:
            cand = {
                'text': text_content,
                'bbox': text_bbox
            }
            candidates.append(cand)

    # 중복 제거
    unique = {}
    for c in candidates:
        unique.setdefault(c['text'], c)

    return list(unique.values())


# ========== Zero-shot 분류 ==========

def zeroshot_title_score_batch(candidates, table_bbox):
    """Zero-shot 분류: 각 후보가 '표 제목'/'설명'일 확률 반환

    Returns:
        tuple: (title_scores, desc_scores) - 각각 numpy array of scores (0~1)
    """
    import numpy as np
    if not zeroshot_classifier or not candidates:
        return np.ones(len(candidates)) * 0.5, np.zeros(len(candidates))  # 중립 점수

    try:
        # 위치 컨텍스트 추가: 표 위/아래 정보를 텍스트에 명시
        texts = []
        for c in candidates:
            pos_tag = ""
            if table_bbox:
                is_above = c.get("bbox", [0, 0, 0, 0])[3] <= table_bbox[1]
                pos_tag = "[표 위쪽] " if is_above else "[표 아래쪽] "
            texts.append(clamp_text_len(pos_tag + c['text']))

        # 라벨 (제목, 설명, 기타)
        labels = [
            "표 제목",
            "표 설명",
            "본문 텍스트",
            "페이지 번호",
            "단위 표기",
        ]

        # 템플릿
        templates = [
            "이것은 {}이다.",
        ]

        def score_once(tmpl: str) -> tuple:
            """특정 템플릿으로 multi-label 분류 후 제목/설명 점수 반환"""
            try:
                res = zeroshot_classifier(
                    texts, labels,
                    hypothesis_template=tmpl,
                    multi_label=True,
                    truncation=True  # 긴 텍스트 안전 처리
                )
            except Exception as e:
                print(f"  Zero-shot 템플릿 '{tmpl[:20]}...' 예외: {e}")
                return np.ones(len(candidates)) * 0.5, np.zeros(len(candidates))

            title_sc = np.zeros(len(candidates), dtype=float)
            desc_sc = np.zeros(len(candidates), dtype=float)

            for i, r in enumerate(res):
                probs = {lab: float(p) for lab, p in zip(r["labels"], r["scores"])}

                # 제목/설명 확률 분리
                title_sc[i] = probs.get("표 제목", 0.0)
                desc_sc[i] = probs.get("표 설명", 0.0)

                # 디버그: 첫 3개만 출력
                if i < 3:
                    print(f"    [Zero-shot] '{texts[i][:30]}...' -> 제목:{probs.get('표 제목', 0):.3f}, 설명:{probs.get('표 설명', 0):.3f}")

            return title_sc, desc_sc

        # 템플릿 평균
        results = [score_once(t) for t in templates]
        title_scores = np.mean([r[0] for r in results], axis=0)
        desc_scores = np.mean([r[1] for r in results], axis=0)

        return title_scores, desc_scores

    except Exception as e:
        print(f"  Zero-shot 오류: {e}")
        return np.ones(len(candidates)) * 0.5, np.zeros(len(candidates))

def extract_table_text(table):
    """표 안의 모든 row 텍스트를 추출하여 하나의 문자열로 반환"""
    all_texts = []

    if 'rows' not in table:
        return ""

    for row in table['rows']:
        if not isinstance(row, list):
            continue
        for cell in row:
            if 'texts' in cell and isinstance(cell['texts'], list):
                for text_obj in cell['texts']:
                    if 'v' in text_obj:
                        all_texts.append(text_obj['v'])

    return ' '.join(all_texts)

def compute_embedding_similarity(table_text, candidate_text, model):
    """임베딩 코사인 유사도 계산

    Returns:
        float: 유사도 (0~1)
    """
    if not model or not table_text or not candidate_text:
        return 0.5  # 중립 점수

    try:
        # 임베딩 생성
        embeddings = model.encode([table_text, candidate_text])
        table_emb = embeddings[0]
        cand_emb = embeddings[1]

        # 코사인 유사도
        similarity = np.dot(table_emb, cand_emb) / (
            np.linalg.norm(table_emb) * np.linalg.norm(cand_emb)
        )

        return float(similarity)
    except Exception as e:
        print(f"  임베딩 유사도 계산 오류: {e}")
        return 0.5

def score_candidates_zeroshot(candidates, table_bbox, table=None):
    """Zero-shot 분류 + 임베딩 유사도 + 패턴 기반 스코어링"""
    import numpy as np

    # Zero-shot 분류: 각 후보가 '표 제목'/'설명'일 확률
    title_scores, desc_scores = zeroshot_title_score_batch(candidates, table_bbox)

    # 표 내용 추출 (임베딩 유사도용)
    table_text = ""
    if table:
        table_text = extract_table_text(table)

    scored = []
    for i, c in enumerate(candidates):
        txt, bb = c['text'], c['bbox']

        title_score = float(title_scores[i])
        desc_score = float(desc_scores[i])

        # 임베딩 유사도 계산
        embedding_sim = 0.5
        if embedding_model and table_text:
            embedding_sim = compute_embedding_similarity(table_text, txt, embedding_model)

        # 패턴 기반 보너스/페널티
        pattern_title_boost = 0.0
        pattern_desc_boost = 0.0

        # 날짜/시간 패턴은 제목 불가
        if is_datetime_like(txt):
            pattern_title_boost = -0.6  # 매우 강력한 페널티
            pattern_desc_boost = 0.1

        # 명사형이 아니면 제목 불가 (강력한 필터)
        elif not is_noun_phrase(txt):
            pattern_title_boost = -0.5  # 강력한 페널티
            pattern_desc_boost = 0.2    # 설명 가능성 증가

        # "표 X.X ..." 패턴은 강력한 제목 신호
        elif is_table_title_like(txt):
            pattern_title_boost = 0.3
            # 표 제목 패턴이면 설명 가능성 낮춤
            pattern_desc_boost = -0.2

        # 섹션 헤더 (1. 제목, ㅇ 제목 등)
        elif is_section_header_like(txt):
            pattern_desc_boost = 0.15
            pattern_title_boost = -0.15  # 제목보다는 설명에 가까움

        # 문장형 패턴은 설명 신호
        elif is_sentence_like(txt):
            pattern_desc_boost = 0.2
            pattern_title_boost = -0.1

        # 단위 표기는 제목도 설명도 아님
        elif is_unit_like(txt):
            pattern_title_boost = -0.2
            pattern_desc_boost = -0.1

        # 교차 참조는 설명
        elif is_cross_reference(txt):
            pattern_desc_boost = 0.25
            pattern_title_boost = -0.3

        # 표 위쪽 위치 보너스 (제목은 보통 표 위에 있음)
        position_boost = 0.0
        if table_bbox:
            is_above = bb[3] <= table_bbox[1]
            if is_above:
                position_boost = 0.15  # 위쪽 제목 보너스
            else:
                position_boost = -0.1  # 아래쪽 제목 페널티

        # 최종 점수 계산
        # Zero-shot 50% + 임베딩 30% + 패턴 15% + 위치 5%
        combined_title_score = (
            title_score * 0.5 +
            embedding_sim * 0.3 +
            pattern_title_boost +
            position_boost
        )

        combined_desc_score = (
            desc_score * 0.5 +
            embedding_sim * 0.3 +
            pattern_desc_boost
        )

        # 0~1 범위로 클리핑
        combined_title_score = max(0.0, min(1.0, combined_title_score))
        combined_desc_score = max(0.0, min(1.0, combined_desc_score))

        scored.append({
            "text": txt,
            "bbox": bb,
            "title_score": combined_title_score,
            "desc_score": combined_desc_score,
            "embedding_sim": embedding_sim,  # 디버깅용
            "pattern_title": pattern_title_boost,  # 디버깅용
            "pattern_desc": pattern_desc_boost,  # 디버깅용
        })

    return scored

# ========== 메인 로직 ==========
def find_title_for_table(table, texts, all_tables=None, used_titles=None):
    """Zero-shot 분류로 표 제목 1개 및 설명 여러 개 찾기

    Returns:
        tuple: (title, title_bbox, descriptions, desc_bboxes)
            - descriptions: 설명 텍스트 리스트
            - desc_bboxes: 설명 bbox 리스트
    """
    table_bbox = get_bbox_from_table(table)
    if not table_bbox:
        print("  테이블 bbox 없음")
        return "", None, [], []

    if used_titles is None:
        used_titles = set()

    # Step 1: 후보 수집 (규칙 기반 필터링)
    candidates = collect_candidates_for_table(table, texts, all_tables)

    # 이미 사용된 제목 제외
    candidates = [c for c in candidates if c['text'] not in used_titles]

    if not candidates:
        return "", None, [], []

    # Step 2: Zero-shot 분류 + 임베딩 유사도
    scored = score_candidates_zeroshot(candidates, table_bbox, table=table)

    # 디버깅: 후보 점수 출력
    print("\n  [하이브리드 분류 결과 (ZeroShot + 임베딩 + 패턴)]")
    for x in scored:
        t = x["text"][:50] if len(x["text"]) > 50 else x["text"]
        print(f"    '{t}'")
        print(f"      제목: {x['title_score']:.3f} (패턴: {x.get('pattern_title', 0):+.2f}) | "
              f"설명: {x['desc_score']:.3f} (패턴: {x.get('pattern_desc', 0):+.2f}) | "
              f"임베딩: {x.get('embedding_sim', 0):.3f}")

    # 제목 선택 (제목 점수가 가장 높은 것)
    scored.sort(key=lambda x: x['title_score'], reverse=True)
    best = scored[0]

    # 임계값 체크
    if best['title_score'] < TITLE_SCORE_THRESHOLD:
        # 제목 점수가 임계값 미만이면 제목 없음
        # 하지만 설명은 여전히 수집
        print(f"  ⚠️  제목 없음 (최고 점수: {best['title_score']:.3f} < {TITLE_SCORE_THRESHOLD})")
        best_title = ""
        best_bbox = None
    else:
        best_title = best['text']
        best_bbox = best['bbox']
        print(f"  ✅ 제목: '{best_title[:60]}' (제목 점수: {best['title_score']:.3f})")

    # 설명 선택 (제목 제외, 설명 점수가 높은 모든 후보)
    if best_title:
        desc_candidates = [x for x in scored if x['text'] != best_title]
    else:
        desc_candidates = scored  # 제목이 없으면 모든 후보를 설명 후보로

    descriptions = []
    desc_bboxes = []

    if desc_candidates:
        # 설명 점수 기준으로 정렬
        desc_candidates.sort(key=lambda x: x['desc_score'], reverse=True)

        # 설명 점수가 임계값 이상인 모든 후보를 설명으로 선택
        for desc_cand in desc_candidates:
            if desc_cand['desc_score'] >= DESC_SCORE_THRESHOLD:
                descriptions.append(desc_cand['text'])
                desc_bboxes.append(desc_cand['bbox'])
                print(f"  ✅ 설명: '{desc_cand['text'][:60]}' (설명 점수: {desc_cand['desc_score']:.3f})")

    print()  # 빈 줄
    return best_title, best_bbox, descriptions, desc_bboxes

@app.route('/get_title', methods=['POST'])
def get_title():
    """받은 데이터(tables, texts)에 각 테이블마다 title, descriptions 프로퍼티를 추가해서 되돌려주는 API"""
    data = request.get_json()

    if isinstance(data, dict):
        tables = data.get('tables', [])
        texts = data.get('texts', [])

        print(f"받은 테이블 수: {len(tables)}")
        print(f"받은 텍스트 수: {len(texts)}")

        result_tables = []
        used_titles = set()

        for idx, table in enumerate(tables):
            table_with_title = copy.deepcopy(table)
            title, title_bbox, descriptions, desc_bboxes = find_title_for_table(
                table, texts, all_tables=tables, used_titles=used_titles
            )
            table_with_title['title'] = title
            table_with_title['title_bbox'] = title_bbox
            table_with_title['descriptions'] = descriptions  # ★ 설명 리스트
            table_with_title['description_bboxes'] = desc_bboxes  # ★ 설명 bbox 리스트

            if title:
                used_titles.add(title)

            result_tables.append(table_with_title)

        return jsonify(result_tables)

    elif isinstance(data, list):
        result = []
        for idx, table in enumerate(data):
            table_with_title = copy.deepcopy(table)
            table_with_title['title'] = f'테이블 {idx + 1}의 타이틀'
            table_with_title['descriptions'] = []  # ★ 설명 리스트
            result.append(table_with_title)
        return jsonify(result)

    else:
        return jsonify({'error': 'Data must be an object with tables and texts, or an array of tables'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)
