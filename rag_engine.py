import sys
import re
import json
import time
from pathlib import Path
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, date

import logging
logging.getLogger("neo4j").setLevel(logging.ERROR)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from llm_util import ask_llm, ask_llm_messages, ask_llm_stream_iter_messages, LLM
from embedding_util import get_embedding, node_text

# ────────────────────────────────────────────────────────────────────────────
# 코드→물질명 정규화 (ReportsDB 전용)
# ────────────────────────────────────────────────────────────────────────────
# 적재 파이프라인의 A1_prompt 모듈에 있는 preprocess_text / CODE_MAP 을 재사용해
# 사용자 질의의 코드(예: "D1")를 물질명("HfO2")으로 변환한다.
# ReportsDB 의 임베딩/컨텍스트가 물질명(content_norm) 기반이므로, 질의도 같은
# 어휘로 맞춰야 검색이 잘 맞는다.
#
# A1_prompt 는 적재 파이프라인 쪽 파일이라 이 앱 저장소에 없을 수 있다.
# 그럴 때는 정규화 없이(원본 질의 그대로) 동작하도록 안전하게 폴백한다.
# CODE_MAP / preprocess_text 는 pipelines/A0_code_map.py 에 분리되어 있다.
# 1차: 패키지 형태(pipelines.A0_code_map)로 import 시도
# 2차: pipelines 디렉터리를 sys.path 에 추가해 모듈 직접 import
try:
    from pipelines.A0_code_map import preprocess_text as _preprocess_text, CODE_MAP as _CODE_MAP
    _NORMALIZE_AVAILABLE = True
except Exception:
    try:
        sys.path.insert(0, str(ROOT / 'pipelines'))
        from A0_code_map import preprocess_text as _preprocess_text, CODE_MAP as _CODE_MAP
        _NORMALIZE_AVAILABLE = True
    except Exception as _e:
        _preprocess_text = None
        _CODE_MAP = None
        _NORMALIZE_AVAILABLE = False
        print(f"[Debug] A0_code_map 미탑재 → 코드 정규화 비활성화 ({_e})")

if _NORMALIZE_AVAILABLE and _CODE_MAP:
    # ★ 디버그: CODE_MAP 의 실제 키 표기를 확인하기 위해 샘플을 앱 시작 시 출력.
    #   _detect_codes() 는 이 키와 "정확히" 일치(대소문자 무시)하는 문자열만 찾으므로,
    #   실제 질문에 쓰는 코드 표기(하이픈/공백 등)가 이 키들과 같은 형식인지 비교해본다.
    _sample_keys = list(_CODE_MAP.keys())[:10]
    print(f"[Debug] CODE_MAP 키 샘플 ({len(_CODE_MAP)}개 중 {len(_sample_keys)}개): {_sample_keys}")


def _normalize_query(text: str) -> str:
    """질의 문자열의 코드를 물질명으로 변환. 모듈이 없으면 원본 그대로 반환."""
    if not text or not _NORMALIZE_AVAILABLE:
        return text
    try:
        return _preprocess_text(text, _CODE_MAP)
    except Exception as e:
        print(f"[Debug] 질의 정규화 실패 → 원본 사용: {e}")
        return text


def _split_keywords(keywords_str: str) -> list[str]:
    """
    "이창수, Lee Changsoo" 같은 키워드 문자열을 쉼표 기준으로만 나눈다
    (공백으로도 나누면 "Lee Changsoo" 가 "Lee"/"Changsoo" 로 쪼개져 지나치게
    광범위하게 매칭되는 문제가 있다 — 자세한 이유는 이 함수를 쓰는 곳들 참고).

    ★ 안전장치: LLM 이 가끔 "이창수 (Lee Changsoo)" 처럼 괄호로 번역을 덧붙여
    출력하는 경우가 있는데, DB 값은 괄호 없는 순수 텍스트("이창수")만 저장돼
    있어 CONTAINS 매칭이 실패한다. 여기서 각 항목의 괄호+내용을 제거해 방어한다.
    (프롬프트에서도 이런 형식을 쓰지 말라고 지시하지만, 이중 안전장치로 코드에서도 처리)
    """
    if not keywords_str:
        return []
    cleaned = []
    for kw in keywords_str.split(","):
        kw = re.sub(r'\s*\([^)]*\)\s*', ' ', kw).strip()
        if kw:
            cleaned.append(kw)
    return cleaned


def _detect_codes(text: str) -> dict:
    """
    질문 문자열에 CODE_MAP 의 코드(예: "D1", "BD30")가 등장하는지 검사해
    {코드: 물질명} 형태로 돌려준다.

    검색 컨텍스트(트리플/문서)는 이미 물질명(content_norm) 기준으로 되어 있어서,
    LLM 입장에서는 "질문의 코드"와 "컨텍스트의 물질명"이 다른 단어로 보여
    서로 연결 짓지 못하고 답변을 못할 수 있다. 이 매핑을 시스템 프롬프트에
    명시적으로 알려주면 LLM 이 "D1 = HfO2" 라는 걸 알고 답변할 수 있다.
    """
    if not text or not _NORMALIZE_AVAILABLE or not _CODE_MAP:
        return {}
    found = {}
    # ★ A0_code_map.preprocess_text 와 동일한 경계 패턴을 사용한다:
    #   (?<![a-zA-Z0-9_]) / (?![a-zA-Z0-9_]) — 영문/숫자/밑줄이 아니면 경계로 인정.
    #   \b 는 Python 유니코드 정규식에서 한글도 "단어 문자"로 취급해, "XX1에"처럼
    #   코드 뒤에 한글 조사가 공백 없이 붙으면 경계 인식에 실패한다.
    #   이 lookaround 패턴은 애초에 ASCII 여부만 보므로 그 문제가 없다.
    #   (preprocess_text 와 로직을 통일해 두 곳이 서로 다르게 동작할 여지를 없앤다)
    for code, material in _CODE_MAP.items():
        pattern = r'(?<![a-zA-Z0-9_])' + re.escape(code) + r'(?![a-zA-Z0-9_])'
        if re.search(pattern, text, re.IGNORECASE):
            found[code] = material
    return found


def _date_to_week_label(date_str: str) -> str:
    """
    'YYYY-MM-DD' 날짜 문자열을 ISO 주차 표기 'YYYY년 WNN' 로 변환한다.
    (ReportsDB 는 week 속성이 별도로 없어 date 로부터 계산해야 한다)
    파싱 실패 시 원본 문자열을 그대로 반환한다(안전 폴백).
    """
    if not date_str:
        return date_str
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        iso_year, iso_week, _ = d.isocalendar()
        return f"{iso_year}년 W{iso_week:02d}"
    except Exception:
        return date_str


def _stored_week_to_label(week_str: str) -> str:
    """
    Confluence 의 저장된 week 속성('2026-W02' 형식)을 'YYYY년 WNN' 표기로 재포맷한다.
    (계산이 아니라 저장된 값을 그대로 신뢰 — 적재 시점 기준이 Python isocalendar 와
    다를 수 있으므로 재계산하지 않고 표기만 바꾼다)
    형식이 안 맞으면 원본을 그대로 반환한다(안전 폴백).
    """
    if not week_str:
        return week_str
    # ★ DB 실제 값은 소문자 'w' (예: "2026-w26") — 대소문자 무관하게 인식
    m = re.match(r'^(\d{4})-[Ww](\d{1,2})$', week_str.strip())
    if m:
        return f"{m.group(1)}년 W{int(m.group(2)):02d}"
    return week_str

from dotenv import load_dotenv
import os

load_dotenv(ROOT / '.env')

NEO4J_URI      = os.getenv('NEO4J_URI')
NEO4J_USER     = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

REPORTS_DATASET = os.getenv('NEO4J_REPORTS_DATASET', 'ReportsDB')
REPORTS_DESC    = os.getenv('NEO4J_REPORTS_DESC',    '내부 연구 보고서 기반 KG')

PAPERS_DATASET  = os.getenv('NEO4J_PAPERS_DATASET',  'PapersDB')
PAPERS_DESC     = os.getenv('NEO4J_PAPERS_DESC',     '논문 기반 인과관계 KG')

# ★ Confluence 문서 데이터셋 (구 BD_OKR_bf2026 → 동일한 Confluence 주간보고
#   데이터라 하나로 통합됨). 트리플/엔티티가 없는 순수 문서형 데이터셋이라
#   (단일 노드 구조 — 자세한 스키마는 아래 CONFLUENCE_DATASET 설정 주석 참고),
#   DATASETS 설정에서 엔티티 관련 필드(vector_index, hop2_relations 등)는
#   비워두고 문서 채널(doc_vector_index/doc_fulltext_index/doc_label)만 채운다.
CONFLUENCE_DATASET = os.getenv('NEO4J_CONFLUENCE_DATASET', 'Confluence')
CONFLUENCE_DESC    = os.getenv('NEO4J_CONFLUENCE_DESC',    'Confluence 주간보고/문서')

# ★ 데이터셋별 켬/끔 스위치. Neo4j 의 실제 데이터는 그대로 두고, 검색 대상에서만
#   빼고 싶을 때 사용한다 (예: 문서를 아직 안 넣었거나, 잠시 검색에서 제외하고
#   싶을 때). .env 에 아래처럼 추가하면 된다 (기본값은 true, 즉 항상 켜짐):
#     NEO4J_REPORTS_ENABLED=false
#     NEO4J_PAPERS_ENABLED=false
#     NEO4J_CONFLUENCE_ENABLED=false
#   "false"/"0"/"no" (대소문자 무관) 만 꺼짐으로 인식하고, 그 외(빈 값 포함)는 켜짐.
def _env_enabled(key: str) -> bool:
    return os.getenv(key, 'true').strip().lower() not in ('false', '0', 'no')

REPORTS_ENABLED    = _env_enabled('NEO4J_REPORTS_ENABLED')
PAPERS_ENABLED     = _env_enabled('NEO4J_PAPERS_ENABLED')
CONFLUENCE_ENABLED = _env_enabled('NEO4J_CONFLUENCE_ENABLED')

DEFAULT_SEARCH_LIMIT = 50
DEFAULT_SEARCH_HOPS  = 2
DEFAULT_SEARCH_MODE  = None   # None / 'text' / 'vector' / 'hybrid'

RRF_K             = 60
MAX_HISTORY_TURNS = 5

# ★ 디버그 출력 레벨 기본값. .env 의 KMAP_DEBUG_LEVEL 로 앱 전체 기본값을 바꿀 수 있고,
#   세션별로는 사이드바 UI에서 rag.debug_level 을 즉시 조정할 수 있다 (재시작 불필요).
#   0=에러만, 1=요약(기본값), 2=보통, 3=상세(개별 결과·전체 프롬프트까지 전부 출력)
DEFAULT_DEBUG_LEVEL = int(os.getenv('KMAP_DEBUG_LEVEL', '1'))

# 문서 섹션 관련 상수
# ★ 5 → 10: A(트리플 출처)/D(저자)/E(주차)/F(날짜범위) 채널과 달리 B(벡터)/C(키워드)는
#   "관련도 상위 N개"라서 5는 관련 문서가 실제로 더 있는데도 놓치는 경우가 있었다.
#   10으로 늘려 재현율을 높인다 — 컨텍스트 길이/응답 시간이 조금 늘 수 있지만,
#   문서 본문은 어차피 doc_id 조회 단계에서 한 번 더 걸러지므로(관련 없는 문서는
#   대개 트리플/저자/기간 채널과 안 겹쳐 자연히 컨텍스트에서 옅게 반영됨) 부담이 크지 않다.
DOC_SEARCH_LIMIT  = 10    # 문서 단위 벡터/키워드 검색으로 가져올 문서 개수 (B/C 기능)
DOC_SCORE_MIN     = 0.6   # 문서 벡터 검색 최소 유사도 컷 (초록/전문처럼 긴 텍스트끼리 비교)
DOC_BODY_MAXLEN   = 3000  # 답변 컨텍스트에 넣을 문서 본문(abstract/content) 최대 길이
                          # ★ 700자였을 때 수치/데이터가 문서 뒷부분에 있으면 잘려서
                          #   빠지는 문제가 있어 3000자로 늘림 (응답 시간이 늘 수 있음)
# ★ 저자 검색(D 기능)은 author 필드에 대한 "정확한" 매칭이라 B/C(유사도/관련도 기반
#   검색)보다 훨씬 신뢰도가 높다. "그 사람이 쓴 모든 보고서" 같은 질문은 5건으로
#   자르면 최근 문서를 놓칠 수 있으므로 훨씬 넉넉하게 잡는다.
AUTHOR_SEARCH_LIMIT = 30

# ★ 주차 "범위" 열거 검색(E 채널)의 상한. "10~15주차 보고문서 전부" 같은 질문은
#   관련도가 아니라 그 기간의 문서 자체를 원하므로 relevance 컷 없이 전부 담되,
#   너무 넓은 범위에서 컨텍스트가 폭주하지 않도록 안전 상한을 둔다(최신순 우선).
WEEK_RANGE_LIMIT = 60

# 엔티티 벡터 검색(짧은 "name (type)" 텍스트 vs 긴 질문 문장) 최소 유사도 컷.
# ★ 실측 결과, BGE-M3 임베딩은 무관한 쌍끼리도 코사인 유사도가 0.8 근처에서
#   시작하는 baseline 이 높아 절대값 컷오프로는 관련/무관을 구분하기 어렵다
#   (실측: 관련도 무관도 전부 0.79~0.85 사이에 몰려 있음).
#   그래서 절대값 대신 "1등 점수 대비 상대적 격차"로 자른다 (아래 RELATIVE_SCORE_GAP).
#   이 값 자체는 폴백 하한선(너무 낮은 절대 점수는 그냥 제외)으로만 쓴다.
ENTITY_VECTOR_SCORE_MIN = 0.3

# 상대적 컷오프: 1등 점수 대비 이 값 이상 차이 나면 제외.
# BGE-M3 처럼 점수가 좁은 범위(0.03~0.05)에 몰리는 임베딩에서, 절대값 컷 대신
# "얼마나 1등과 벌어지는지"로 관련도를 가른다. 값이 작을수록 더 엄격하게 거른다.
# ★ 잠정값이며, 실제 데이터로 관련/무관 결과의 격차를 보고 계속 튜닝해야 한다.
#   변경 이력:
#     0.02 → 0.03 : PapersDB 실측 범위 0.804~0.848(폭 0.044)에서 0.02는 150건 중
#                   4건만 남겨 다소 과했음.
#     0.03 → 0.05 : 관련 있는 결과가 잘못 잘려나갈 위험을 더 줄이기 위해 여유를 둠.
#                   (운영하면서 debug 로그의 점수 분포를 계속 관찰해 최적값을 찾을 것)
#   참고: ReportsDB 는 실측 폭이 0.006으로 극히 좁아 이 값을 키워도 어차피 다 통과함
#   — 즉 지금 폭 안에서는 gap 값 조정이 ReportsDB 엔티티 벡터 검색에 영향을 주지 않는다.
RELATIVE_SCORE_GAP = 0.05

# 디버그 로그에서 결과를 줄 단위로 출력할 때 보여줄 최대 개수.
# 전체를 다 찍으면(최대 150건) 터미널이 감당 안 되고, 너무 줄이면 튜닝이 어려우니
# "분포 요약(min/max/avg) + 상위 N개 미리보기 + 컷오프 전후 건수"로 절충한다.
_DEBUG_ROW_PREVIEW = 10

# ────────────────────────────────────────────────────────────────────────────
# 데이터셋별 검색 설정
# ────────────────────────────────────────────────────────────────────────────
# 각 데이터셋(ReportsDB / PapersDB)마다 검색 방식이 조금씩 다르므로,
# 쿼리 생성에 필요한 모든 파라미터를 이 dict 하나에 모아둔다.
# 여기 값만 바꾸면 Cypher 쿼리 코드를 건드리지 않고 검색 동작을 조정할 수 있다.
#
# 각 필드 의미:
#   description    : LLM 프롬프트에 넣는 데이터셋 설명
#   node_types     : 해당 데이터셋의 엔티티 타입 목록 (Neo4j 첫 번째 레이블).
#                    LLM 프롬프트에만 쓰이며 검색 로직에는 영향 없음.
#   sort_field/order: text 검색 결과 정렬 기준 (관계 속성 r.confidence 등)
#   return_fields  : 관계(r)에서 꺼내와 답변 컨텍스트에 넣을 속성들.
#                    → 새 스키마에서 ReportsDB 관계에도 source_url/title/author 가
#                      추가되었으므로 여기에 포함시켜 답변에서 출처를 인용할 수 있게 함.
#   search_fields  : text(키워드) 검색 시 CONTAINS 로 훑을 필드들
#   has_date       : r.date 로 날짜 필터를 걸 수 있는지
#   min_confidence : r.confidence 하한 (논문은 노이즈가 많아 0.6 컷)
#   vector_index   : 이 데이터셋 "엔티티 노드"용 벡터 인덱스 이름.
#                    ★ 스키마 변경: ReportsDB 엔티티 인덱스가
#                       'reportsdb_embedding' → 'reportsdb_entity_embedding' 로 바뀜.
#   meta_label     : ★ 신규. 같은 데이터셋 레이블(:ReportsDB)을 공유하지만
#                    엔티티가 아닌 "메타 노드"의 레이블.
#                    ReportsDB 엔티티 벡터 인덱스에는 Report(구 Document) 메타 노드도 섞여
#                    들어오므로, 벡터 검색 결과에서 이 레이블을 가진 노드를
#                    'WHERE NOT node:<meta_label>' 로 걸러내야 한다.
#   default_mode   : 사용자가 모드를 지정하지 않았을 때 기본 검색 모드
#   search_hops    : 그래프 확장 hop 수 (None 이면 전역 기본값 사용)
#   hop2_relations : ★ 2-hop 확장(1차에서 찾은 엔티티를 앵커로 삼아 한 단계 더
#                    나아가는 단계)에서 "따라갈" 관계 타입 화이트리스트.
#                    구조/조성 관계(HAS_PHASE, STACKED_ON 등)까지 다 따라가면
#                    원래 질문과 무관한 잡음이 급격히 늘어나므로,
#                    "인과·성능" 계열 관계로만 한정해 의미 있는 추론 체인만 확장한다.
#                    (1차 검색 자체는 이 제한을 받지 않는다 — 질문 키워드가
#                     직접 매칭되면 관계 타입 무관하게 잡아야 하므로)
DATASETS: dict = {
    REPORTS_DATASET: {
        'description':    REPORTS_DESC,
        # 새 스키마의 ReportsDB 엔티티 타입 11종
        'node_types':     'Dielectric, Electrode, InterfacialLayer, InsertionLayer, '
                          'Stack_MIM, Stack_Top, Stack_Bot, Measurement, '
                          'Performance, Process, Dopant',
        'sort_field':     'confidence',
        'sort_order':     'DESC',
        # ★ ReportsDB 관계에 source_url/title/author 가 추가되어 답변 출처 인용 가능
        'return_fields':  ['evidence', 'confidence', 'source_url', 'title', 'author', 'date'],
        'search_fields':  ['s.name', 'o.name', 'r.evidence', 'r.title', 'r.author'],
        'has_date':       True,
        # ★ 내부 문서(사내 주간보고)는 정확한 날짜보다 "몇 주차"로 얘기하는 게
        #   실무 관례에 맞아서, 날짜 표시 시 date 대신 주차(예: "2026년 W02")로
        #   변환해서 보여준다. (PapersDB 논문은 그대로 정확한 날짜를 유지)
        'date_as_week':   True,
        'min_confidence': None,
        'vector_index':   'reportsdb_entity_embedding',  # ★ 이름 변경됨
        'meta_label':     'Report',                      # ★ 벡터 검색에서 제외할 메타 노드 (구 Document → Report로 레이블 변경됨)
        'default_mode':   'hybrid',
        'search_hops':    None,
        # ReportsDB 인과·성능 계열 관계만 2-hop 확장 대상으로 삼는다.
        # (구조 계열 HAS_BOTTOM_ELECTRODE/HAS_DIELECTRIC/INSERTED_* 등은 제외)
        'hop2_relations': ['DEPOSITED_BY', 'TREATED_BY', 'ACHIEVES',
                           'IMPROVES', 'DEGRADES', 'COMPARED_TO', 'DOPED_WITH'],
        # ── 문서(메타 노드) 관련 설정 (A: doc_id 조인 / B: 문서 벡터 검색) ──
        'doc_vector_index': 'reportsdb_doc_embedding',   # Report 노드 벡터 인덱스
        'doc_label':        'Report',                    # 메타 노드 레이블 (구 Document → Report)
        # ★ content_norm: 코드(D1)를 물질명(HfO2)으로 변환한 정규화 본문.
        #   embedding 도 content_norm 기반이고 LLM 컨텍스트도 물질명으로 주는 게
        #   의미 파악에 유리하므로, 답변 컨텍스트용 본문으로 content_norm 을 쓴다.
        'doc_body_field':   'content_norm',
        'doc_body_label':   '내용',              # LLM 컨텍스트에 표기할 본문 레이블
        'doc_fulltext_index': 'reportsdb_doc_fulltext',   # ★ 이름 변경됨 (구 doc_fulltext)
        # ★ ReportsDB 는 사용자 질의에 코드(D1 등)가 섞일 수 있으므로,
        #   검색 전에 코드→물질명으로 질의를 정규화한다 (아래 _normalize_query).
        'normalize_query':  True,
    },
    PAPERS_DATASET: {
        'description':    PAPERS_DESC,
        # 새 스키마의 PapersDB 엔티티 타입 15종
        'node_types':     'Problem, Cause, Mechanism, Solution, ProcessApproach, '
                          'Phenomenon, Constraint, Material, Dopant, Process, '
                          'ProcessCondition, Phase, Property, Performance, Device',
        'sort_field':     'confidence',
        'sort_order':     'DESC',
        'return_fields':  ['evidence', 'confidence', 'source_url', 'title', 'author', 'date'],
        'search_fields':  ['s.name', 'o.name', 'r.evidence', 'r.title', 'r.author'],
        'has_date':       True,
        'min_confidence': 0.6,
        'vector_index':   'papersdb_entity_embedding',  # ★ 이름 변경됨 (구 papersdb_embedding)
        # papersdb_entity_embedding 은 n.embedding 을 인덱싱하는데, Paper 메타 노드는
        # n.abstract_embedding(다른 속성)을 쓰므로 이 인덱스에 애초에 포함되지 않는다.
        # 그래도 방어적으로 필터를 걸어 안전하게 처리한다.
        'meta_label':     'Paper',
        'default_mode':   'hybrid',
        'search_hops':    None,
        # PapersDB 인과·성능 계열 관계만 2-hop 확장 대상으로 삼는다.
        # (물질·공정 계열 DEPOSITED_BY/STACKED_ON/COMPOSED_OF 등은 제외)
        'hop2_relations': ['CAUSED_BY', 'EXPLAINED_BY', 'SOLVED_BY', 'SUPPRESSES',
                           'INDUCES', 'PREVENTS', 'TRIGGERS',
                           'IMPROVES', 'DEGRADES', 'ACHIEVES', 'AFFECTS',
                           'CORRELATED_WITH', 'TRADEOFF_WITH', 'HAS_PERFORMANCE'],
        # ── 문서(메타 노드) 관련 설정 (A: doc_id 조인 / B: 문서 벡터 검색) ──
        'doc_vector_index': 'papersdb_doc_embedding',    # ★ 이름 변경됨 (구 paper_abstract_embedding)
        'doc_label':        'Paper',                     # 메타 노드 레이블
        'doc_body_field':   'abstract',                  # 본문 속성명 (논문 초록)
        'doc_body_label':   '초록',                       # LLM 컨텍스트에 표기할 본문 레이블
        'doc_fulltext_index': 'papersdb_doc_fulltext',   # ★ 이름 변경됨 (구 paper_fulltext)
        # ★ 코드→물질명 매핑(CODE_MAP)은 사내 코드에 대한 일반적인 번역이라
        #   ReportsDB 뿐 아니라 PapersDB 검색에도 동일하게 적용해야 한다.
        #   (예: "BD30" 으로 논문 검색 시에도 물질명으로 정규화되어야 논문
        #    임베딩/엔티티명과 어휘가 맞는다.) CODE_MAP 에 없는 단어는 그대로 반환되므로
        #   켜둬도 부작용이 없다.
        'normalize_query':  True,
    },
    # ── Confluence: 트리플/엔티티가 없는 순수 문서형 데이터셋 ────────────────────
    # (entity)-[r]->(entity) 트리플 자체가 없으므로, "엔티티 벡터 검색·2-hop 확장"에
    # 해당하는 필드(vector_index, meta_label, hop2_relations)는 아예 넣지 않는다.
    # → retrieve() 가 entity 검색을 text 모드로 자동 폴백하고(빈 결과, 무해),
    #   문서 채널(B: 벡터, C: FULLTEXT, D: 저자)만으로 문서를 찾아 컨텍스트를 구성한다.
    #
    # ★ 데이터 구조 (2026-08 최종): Document/Chunk 분리 구조를 시도했다가,
    #   다른 데이터셋(PapersDB/ReportsDB)과의 단순성/일관성을 위해 폐기하고
    #   단일 노드 구조로 되돌아왔다:
    #     (:Confl_doc:Document:Confluence {
    #        doc_id (유니크, 예: confluence_705713335_chunk_2),
    #        doc_group_id (그룹핑 키, 유니크 아님, 예: confluence_705713335),
    #        chunk_index, total_chunks, token_count,
    #        title, title_norm, author, date, year_week,
    #        source, source_url, space_name, page_title,
    #        title_path, title_path_norm, last_modified,
    #        research_item, research_item_norm, summary, summary_norm,
    #        content, content_norm, embedding, dataset
    #     }) -[:NEXT_CHUNK]-> (같은 doc_group_id 내 다음 청크, total_chunks==1 이면 없음)
    #   청킹 여부와 무관하게 doc_id 는 항상 "{그룹키}_chunk_N" 패턴이고, total_chunks
    #   로만 청킹 여부(1 vs 2 이상)를 판단한다 — ReportsDB/PapersDB 와 동일한
    #   단일 노드 구조라 아래 doc_label/doc_key 설정만으로 기존 코드 경로를 그대로 탄다.
    CONFLUENCE_DATASET: {
        'description':    CONFLUENCE_DESC,
        'node_types':     '해당 없음 (문서 전용 데이터셋 — 트리플/엔티티 없음)',
        'sort_field':     'confidence',
        'sort_order':     'DESC',
        'return_fields':  ['evidence', 'confidence', 'source_url', 'title', 'author', 'date'],
        'search_fields':  ['s.name', 'o.name', 'r.evidence'],
        'has_date':       True,
        # ★ ReportsDB 와 마찬가지로 정확한 날짜보다 "몇 주차" 로 얘기하는 게
        #   실무 관례에 맞아서, 날짜 표시 시 date 대신 주차로 변환해서 보여준다.
        'date_as_week':   True,
        'min_confidence': None,
        'default_mode':   'hybrid',
        'search_hops':    None,
        # ── 문서(메타 노드) 관련 설정 (A: doc_id 조인 / B: 문서 벡터 검색) ──
        'doc_vector_index': 'confluence_doc_embedding',   # embed_nodes.py 가 생성
        'doc_label':        'Confl_doc',                  # 단일 노드 레이블
        'doc_body_field':   'content_norm',           # 본문 속성명 (물질명 정규화본)
        'doc_body_label':   '내용',                    # LLM 컨텍스트에 표기할 본문 레이블
        # FULLTEXT 는 원본/정규화 쌍 (둘 다 검색해 결과를 합친다 — 기존 코드가
        # doc_fulltext_index 를 리스트로 받으면 이미 이렇게 동작한다)
        'doc_fulltext_index': ['confluence_doc_fulltext', 'confluence_doc_fulltext_norm'],
        # ReportsDB 와 마찬가지로 물질 코드가 섞일 수 있으므로 질의 정규화 적용
        'normalize_query':  True,
        # ★ 청킹됨(total_chunks>1)에 관계없이 doc_id 가 항상 청크 단위 유니크 키다.
        #   같은 페이지의 청크들을 순서대로 묶어 보여주기 위해 doc_group_id 로
        #   그룹핑/정렬한다(문자열 doc_id 만으로는 "_chunk_10" 이 "_chunk_2" 보다
        #   사전식으로 앞에 와 순서가 깨진다).
        'chunked':          True,
        'doc_group_field':  'doc_group_id',
    },
}

# ★ 켬/끔 스위치가 꺼진 데이터셋은 DATASETS 에서 완전히 제거한다 — 이후 모든 로직
#   (LLM 의 target_datasets 자동 선택, 탭 목록, 검색 등)이 DATASETS 를 기준으로
#   동작하므로, 여기서 빼두면 실제 Neo4j 데이터는 그대로 두고도 검색에서 완전히
#   제외된다.
_DATASET_ENABLED = {
    REPORTS_DATASET:    REPORTS_ENABLED,
    PAPERS_DATASET:     PAPERS_ENABLED,
    CONFLUENCE_DATASET: CONFLUENCE_ENABLED,
}
_ALL_DATASETS_UNFILTERED = DATASETS
DATASETS = {ds: cfg for ds, cfg in _ALL_DATASETS_UNFILTERED.items() if _DATASET_ENABLED.get(ds, True)}
if not DATASETS:
    # 전부 꺼져 있으면 검색할 데이터가 아예 없어지므로, 완전한 설정을 가진 첫
    # 데이터셋을 그대로(빈 dict 아님) 강제로 살려 앱이 죽지 않게 한다.
    _fallback_ds, _fallback_cfg = next(iter(_ALL_DATASETS_UNFILTERED.items()))
    print(f"[Debug] 경고: 모든 데이터셋이 꺼져 있어 '{_fallback_ds}'를 강제로 켭니다 "
          f"(.env 의 NEO4J_*_ENABLED 를 최소 하나는 true 로 설정하세요).")
    DATASETS = {_fallback_ds: _fallback_cfg}

_NO_RESULT = "관련 트리플을 찾지 못했습니다."

# ★ 답변에서 데이터셋 섹션을 보여주는 순서(사용자 요청): ReportsDB → Confluence → PapersDB.
#   search_targets/results 의 순서는 LLM 자동 선택 배열 순서나 다중 선택 시 set
#   순회 순서 등 제각각이라 이 순서를 보장하지 않으므로, 결과를 조합할 때 항상
#   이 순서로 정렬해서 내보낸다.
_OUTPUT_ORDER = [REPORTS_DATASET, CONFLUENCE_DATASET, PAPERS_DATASET]

# ★ 스키마 변경 대응: 구조(출처) 관계 타입.
# 새 스키마는 엔티티와 메타 노드를 (entity)-[:FROM_PAPER]->(:Paper) /
# (entity)-[:FROM_DOC]->(:Report) 로 연결한다. 그런데 Paper/Report 메타 노드도
# 데이터셋 레이블(:PapersDB / :ReportsDB)을 공유하므로,
# MATCH (s:PapersDB)-[r]->(o:PapersDB) 같은 패턴이 이 구조 관계까지 잡아버린다.
# 이들은 "의미 트리플"이 아니라 출처 연결이므로, 검색 시 관계 타입으로 제외한다.
#   ★ NEXT_CHUNK 도 구조 관계다: Confluence 는 단일 노드 구조라 트리플 자체가
#     없지만, 같은 청크 노드끼리 :Confluence 라벨을 공유하므로 혹시라도
#     (s:Confluence)-[r]->(o:Confluence) 패턴이 이 관계를 "의미 트리플"로 잡지
#     않도록 방어적으로 제외한다.
_STRUCTURAL_RELS = ['FROM_PAPER', 'FROM_DOC', 'NEXT_CHUNK']


class GraphRAG:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.history: list[dict] = []
        self.last_retrieved_nodes: list[str] = []
        self._pending_nodes: set[str] = set()

        # 벡터 검색 컷오프 값들 — 세션(사용자)별로 UI에서 조정 가능하도록
        # 모듈 상수를 인스턴스 속성으로 복사해 둔다. (기본값은 모듈 상수를 따름)
        self.entity_score_min = ENTITY_VECTOR_SCORE_MIN   # 엔티티 벡터 검색 절대 하한
        self.relative_gap     = RELATIVE_SCORE_GAP        # 1등 대비 상대 컷오프
        self.doc_score_min    = DOC_SCORE_MIN             # 문서 벡터 검색 절대 하한

        # ★ 디버그 출력 레벨 — 세션별 조정 가능 (사이드바 UI에서 rag.debug_level 로 변경).
        #   0 = 끔 (에러만)
        #   1 = 요약 (질문/키워드/최종 채널별 개수 등 핵심 정보만) — 기본값
        #   2 = 보통 (모드/폴백/컷오프 적용 전후 등 중간 단계 정보 추가)
        #   3 = 상세 (개별 결과 점수 나열, 전체 컨텍스트/LLM 프롬프트 원문까지 전부)
        self.debug_level = DEFAULT_DEBUG_LEVEL

        # ★ 최종 답변 생성에 쓸 LLM 모델 — 세션별로 UI에서 선택 가능
        #   (llm_util._LLM_CONFIGS 에 등록된 이름과 일치해야 함).
        #   None 이면 llm_util 의 .env 기본값(LLM)을 그대로 사용한다.
        self.answer_llm = None

    def _dbg(self, level: int, *args, **kwargs):
        """level <= self.debug_level 일 때만 출력. 에러(level=0)는 항상 출력된다."""
        if self.debug_level >= level:
            print(*args, **kwargs)

    def close(self):
        self.driver.close()

    def clear_history(self):
        self.history = []
        self._dbg(2, "[Debug] 대화 히스토리 초기화")

    def set_history(self, messages: list[dict]):
        """
        저장된 대화를 다시 불러올 때 LLM 히스토리를 복원한다.

        DB 에서 읽은 메시지에는 dataset/graph_id 등 UI 전용 필드가 섞여 있으므로,
        LLM 이 필요한 role/content 만 추려 담고 최근 MAX_HISTORY_TURNS 턴으로 자른다.
        """
        cleaned = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        max_messages = MAX_HISTORY_TURNS * 2
        self.history = cleaned[-max_messages:]
        self._dbg(2, f"[Debug] 대화 히스토리 복원: {len(self.history)}개 메시지")

    def _extract_keywords(self, query: str) -> dict:
        """
        질문에서 검색 키워드/날짜 범위를 추출한다.

        ★ 이전 대화(self.history)를 함께 프롬프트에 넣어, "그 사람", "그 연구원",
        "그거"처럼 이름을 생략하고 되묻는 후속 질문에서도 실제 키워드(예: 연구원 이름)를
        복원해 추출한다. 이게 없으면 후속 질문이 검색 단계에서 빈손이 되어(원래
        키워드가 현재 질문 텍스트에 없으므로) 검색 결과가 안 나오는 문제가 있었다.
        LLM 답변 자체는 self.history 전체를 이미 참고하지만, "검색용 키워드 추출"은
        지금까지 이 히스토리를 안 보고 있었다.
        """
        # 최근 대화 몇 턴만 넣는다 (전체를 넣으면 프롬프트가 길어지고, 오래된 맥락은
        # 지금 질문의 지칭 대상과 무관할 가능성이 높다).
        recent_turns = self.history[-(MAX_HISTORY_TURNS * 2):] if self.history else []
        if recent_turns:
            history_lines = "\n".join(
                f"{'사용자' if m['role'] == 'user' else 'AI'}: {m['content']}"
                for m in recent_turns
            )
            history_block = f"""
[이전 대화 — 지칭 표현(그 사람/그거/그 연구원 등) 해석에 참고]
{history_lines}
"""
        else:
            history_block = ""

        today     = datetime.today()
        today_str = today.strftime("%Y-%m-%d")
        m3_ago    = (today - timedelta(days=90)).strftime("%Y-%m-%d")
        m6_ago    = (today - timedelta(days=180)).strftime("%Y-%m-%d")
        y1_ago    = (today - timedelta(days=365)).strftime("%Y-%m-%d")
        y2_ago    = (today - timedelta(days=730)).strftime("%Y-%m-%d")

        # ★ 데이터셋 자동 선택용: 각 데이터셋의 설명/엔티티 타입을 프롬프트에 보여주고,
        #   질문 내용상 어떤 데이터셋을 검색해야 하는지 LLM이 함께 판단하게 한다.
        #   (사용자가 특정 탭을 선택한 경우는 이 판단을 쓰지 않고 그 탭만 검색하지만,
        #    "전체" 탭일 때는 지금까지 무조건 데이터셋 3개를 다 검색해서 느리고
        #    관련 없는 데이터셋의 결과가 컨텍스트에 잡음으로 섞이는 문제가 있었다.)
        dataset_choices = "\n".join(
            f"- {ds}: {cfg.get('description', '')} (엔티티/문서: {cfg.get('node_types', '문서 전용')})"
            for ds, cfg in DATASETS.items()
        )
        valid_dataset_keys = list(DATASETS.keys())

        extract_prompt = f"""다음 질문에서 검색 키워드와 날짜 범위를 추출하세요.
{history_block}
오늘 날짜: {today_str}
자주 쓰는 날짜 참고:
- 최근 3개월 이내: {m3_ago} ~ {today_str}
- 최근 6개월 이내: {m6_ago} ~ {today_str}
- 최근 1년 이내:   {y1_ago} ~ {today_str}
- 최근 2년 이내:   {y2_ago} ~ {today_str}

사용 가능한 데이터셋:
{dataset_choices}

출력 형식 (JSON):
{{
  "keywords": "키워드1, 키워드2, ...",
  "author_names": "저자명1, 저자명2, ... (질문이 특정 인물이 작성한 문서를 찾는 게 아니면 빈 문자열)",
  "target_datasets": ["관련된 데이터셋명", ...],
  "date_from": "YYYY-MM-DD 또는 null",
  "date_to": "YYYY-MM-DD 또는 null",
  "recency_focus": true 또는 false,
  "oldest_focus": true 또는 false,
  "year_week": "YYYY-WNN 또는 null",
  "year_week_from": "YYYY-WNN 또는 null",
  "year_week_to": "YYYY-WNN 또는 null",
  "full_content": true 또는 false,
  "prefer_normalized": true 또는 false
}}

full_content 판단 규칙:
- "전체를 보여줘", "전문을 보여줘", "원문 그대로", "요약하지 말고 전부", "생략하지 말고
  다 보여줘" 처럼 문서 본문을 자르거나 요약하지 말고 있는 그대로 전부 보여달라는
  의도면 → true.
- 그 외(일반적인 질문/요약 요청)는 → false.

prefer_normalized 판단 규칙:
- ReportsDB/Confluence 원본 문서에는 사내 코드(예: BD30) 그대로 적힌 버전과, 코드를
  물질명(예: HfO2)으로 변환한 버전이 둘 다 있습니다. 기본은 코드 원본을 우선
  보여주지만, "물질명으로 변환된 내용으로 보여줘", "물질명 기준으로 정리해줘",
  "코드 말고 물질명으로" 처럼 물질명 변환본을 명시적으로 원하는 질문이면 → true.
- 그 외(특별한 언급 없음, 또는 반대로 "코드 그대로/원본으로 보여줘")는 → false.

author_names 판단 규칙 (매우 중요):
- 질문이 "그 사람이 쓴 논문/보고서", "이창수가 작성한 문서", "Tao Li 최근 논문" 처럼
  "특정 인물이 작성한 문서를 찾는" 의도일 때만 그 사람 이름을 채우세요.
- ★ "누가의 연구내용/연구성과/작업내용/실험내용을 정리/요약/표로 정리해줘" 처럼,
  문서를 "작성했다"는 동사가 직접 안 쓰였어도 특정 인물 한 명이 수행/관여한 내용
  전체를 모아달라는 의도면 마찬가지로 그 사람 이름을 채우세요.
  예: "박보은의 2026년 연구내용을 표로 정리해줘" → author_names: "박보은"
      (그 사람이 관련된 문서를 "전부" 찾아야 하는 열거형 질문이므로, 관련도 기반
      검색만으로는 상위 몇 건만 뽑혀 일부가 누락될 수 있음 — author_names 를 채워야
      exact-match 채널이 전부 찾아낸다)
- ★ 그 외의 모든 질문(기술/소재/방법을 묻는 일반 질문 등, 특정 인물이 언급되지 않은
  질문)은 author_names 를 반드시 빈 문자열("")로 두세요. 절대로 일반 기술 용어
  (예: "MIM", "capacitor", "TiO2")를 author_names 에 넣지 마세요 — author 필드는
  부분 문자열(CONTAINS) 매칭이라, 관련 없는 용어가 저자명에 우연히 포함되면 전혀
  관련 없는 문서가 대량으로 섞여 들어갑니다.
- ★ DB에는 순수 이름만 저장돼 있으므로("이창수"), "이창수님", "이창수 연구원",
  "이창수 책임"처럼 존칭/직급이 붙어 질문에 나와도 author_names 에는 존칭/직급을
  떼어낸 순수 이름만 넣으세요. (예: "이창수 책임이 작성한 보고서" → author_names: "이창수")

날짜 변환 규칙:
- "N년 이후", "N년 이상", "N년부터" → date_from: "N-01-01", date_to: null
- "N년 이전", "N년 이하", "N년까지" → date_from: null, date_to: "N-12-31"
- "N년~M년", "N년에서 M년" → date_from: "N-01-01", date_to: "M-12-31"
- ★ "N년"만 단독으로 언급 (이후/이전/부터/까지 등 수식어 없이, 특정 월/일도 없이)
  → 그 해 전체를 의미 → date_from: "N-01-01", date_to: "N-12-31"
  예: "2026년 이창수가 작성한 보고서" → date_from: "2026-01-01", date_to: "2026-12-31"
- "최근 N개월" → 오늘 기준 N개월 전 날짜 계산
- "최근 N년" → 오늘 기준 N년 전 날짜 계산
- "올해" → date_from: "{today.year}-01-01", date_to: "{today_str}"
- 날짜 언급 없음 → date_from: null, date_to: null

target_datasets 판단 규칙 (매우 중요):
- 질문 내용을 보고 위 "사용 가능한 데이터셋" 중 실제로 검색이 필요한 데이터셋만
  골라 배열로 넣으세요.
- ★ ReportsDB 와 Confluence 는 둘 다 사내 내부 문서를 다루지만 성격이 다릅니다
  (아래 각 데이터셋 설명 참고). 질문이 특정 연구원 개인이 작성한 내용을
  찾는 것이면 Confluence 위주로, 종합/정리된 연구보고서 내용을 찾는 것이면
  ReportsDB 위주로 판단하되, 확신이 없으면 둘 다 포함하세요.
- 논문/연구자/저널 관련 질문은 PapersDB를 포함하세요.
- 여러 데이터셋에 걸칠 수 있는 질문이면 관련된 것을 모두 포함하세요.
- ★ 확신이 없거나 질문이 모호하면, 좁히지 말고 관련 있을 수 있는 데이터셋을
  전부 포함하세요 (좁혀서 놓치는 것보다 넓게 잡는 게 안전합니다).
- 이전 대화의 맥락(예: 이전에 특정 데이터셋 관련 대상이 언급됨)도 참고하세요.

year_week 판단 규칙 (내부 문서 ReportsDB/Confluence 는 주차로 관리됨 — 매우 중요):
- "2026년 26주차", "2026-W26", "26주차"(올해로 간주) 처럼 특정 연도+주차가
  언급되면 → year_week: "YYYY-WNN" 형식으로 추출 (예: "2026-W26", 주차는 2자리 0-패딩).
- 연도 없이 "26주차"만 언급되면 오늘 연도({today.year})를 사용하세요.
- 주차 언급이 없으면 → year_week: null
- ★ "10주차에서 15주차 사이", "10주차~15주차", "10주차부터 15주차까지" 처럼 주차
  "범위"가 언급되면, year_week 는 null 로 두고 대신 year_week_from/year_week_to 에
  각각 시작 주차/끝 주차를 "YYYY-WNN" 형식으로 넣으세요 (연도 없으면 오늘 연도 사용).
  예: "2026년 10주차에서 15주차 사이" → year_week_from: "2026-W10", year_week_to: "2026-W15"
- 주차 범위 언급이 없으면 → year_week_from: null, year_week_to: null

recency_focus 판단 규칙 (매우 중요):
- "가장 최근", "제일 최근", "최신", "최근 결과", "요즘" 처럼 구체적인 기간(개월/년) 없이
  "최근/최신"만 언급되어 시간 순으로 정렬해서 보여달라는 의도가 있으면 → true
- 위와 같이 recency_focus 가 true 인 경우, 검색 결과를 confidence(신뢰도) 순이 아니라
  date(날짜) 내림차순으로 정렬해야 하므로 반드시 true 로 표시하세요.
- 구체적 기간이 명시되어 date_from/date_to 가 채워진 경우에도, 그 기간 안에서 역시
  최신순 정렬이 자연스러우므로 recency_focus: true 로 표시하세요.
- 날짜/최근 관련 언급이 전혀 없으면 → false

oldest_focus 판단 규칙 (매우 중요):
- "가장 오래된", "제일 오래된", "최초의", "맨 처음", "가장 먼저 작성된" 처럼 시간상
  가장 앞선(과거) 자료를 찾아달라는 의도가 있으면 → true
- recency_focus 와 정반대 방향이므로 둘 다 true 가 될 수 없습니다 — oldest_focus 가
  true 면 recency_focus 는 반드시 false 로 두세요.
- 위와 같이 oldest_focus 가 true 인 경우, 검색 결과를 confidence(신뢰도) 순이 아니라
  date(날짜) 오름차순(과거 → 현재)으로 정렬해야 하므로 반드시 true 로 표시하세요.
- 구체적 기간이 명시되어 date_from/date_to 가 채워진 경우에도, 그 기간 중 "가장
  오래된 것"을 찾는 의도면 oldest_focus: true 로 표시하세요.
- 날짜/오래된 것 관련 언급이 전혀 없으면 → false

키워드 형식 규칙 (매우 중요 — 이 형식을 안 지키면 검색이 실패합니다):
- keywords 는 반드시 "순수한 단어/구를 쉼표로 나열"한 문자열이어야 합니다.
- ★ 절대 괄호로 번역이나 부가설명을 덧붙이지 마세요. 각 언어 표현은 별도의
  쉼표 항목으로 분리하세요.
- 올바른 예: "이창수, Lee Changsoo, 보고서, Report"
- 잘못된 예: "이창수 (Lee Changsoo), 보고서 (Report)"  ← 괄호가 붙으면 DB 값과
  정확히 일치하지 않아 검색이 실패합니다 (예: DB에는 "이창수"만 저장되어 있는데
  검색어가 "이창수 (Lee Changsoo)"이면 매칭이 안 됨).

키워드 추출 규칙:
1. 핵심 명사(엔티티) 위주로 추출
2. 한국어 키워드는 반드시 영어 번역도 "별도의 쉼표 항목"으로 함께 추출
3. 관련 동의어/유사어도 포함 (역시 별도 쉼표 항목으로)
4. 날짜 관련 표현은 키워드에서 제외
5. 저자/작성자 이름이 언급되면 반드시 키워드에 포함하되, "연구원", "박사", "교수",
   "님", "씨" 같은 직함/호칭은 이름에서 떼어내고 순수한 이름만 넣으세요.
   (DB의 author 필드는 순수 이름만 저장돼 있어, 직함이 붙으면 검색이 안 됩니다)
   예: "이창수 연구원이 쓴 보고서" → "이창수" (❌ "이창수 연구원" 아님)
       "Tao Li가 쓴 논문" → "Tao Li"
6. 질문에 "그 사람", "그거", "그 연구원", "이거" 처럼 대상이 생략되거나 대명사로만
   지칭된 경우, [이전 대화]를 참고해 실제로 무엇/누구를 가리키는지 찾아내고
   그 구체적인 이름/명칭을 키워드에 포함하세요.
   예: 이전 대화에서 "Tao Li" 가 언급됐고, 이번 질문이 "그 사람의 다른 논문도 있어?"
   라면 → keywords 에 "Tao Li" 를 포함해야 함.

질문: {query}
출력:"""

        result = ask_llm(
            USER_MESSAGE=extract_prompt,
            SYSTEM_PROMPT="JSON 형식으로만 출력하세요. 설명 없이 JSON만 출력하세요.",
            temperature=0.0,
            reasoning_effort="low"
        )

        try:
            clean = (result.strip()
                     .removeprefix("```json")
                     .removeprefix("```")
                     .removesuffix("```")
                     .strip())
            parsed = json.loads(clean)

            # ★ year_week 값 검증/정규화 ("YYYY-WNN", 주차 2자리 0-패딩). 형식이 안 맞으면
            #   무시(None) — LLM 이 가끔 다른 포맷으로 줄 수 있어 방어적으로 처리.
            year_week_raw = parsed.get('year_week')
            year_week = None
            if year_week_raw:
                # LLM 이 대/소문자 'w' 어느 쪽으로 주든 허용 (내부적으로는 대문자로 통일;
                # 실제 DB 비교는 toLower() 로 하므로 대소문자 자체는 무관함)
                m = re.match(r'^(\d{4})-W(\d{1,2})$', str(year_week_raw).strip(), re.IGNORECASE)
                if m:
                    year_week = f"{m.group(1)}-W{int(m.group(2)):02d}"

            date_from = parsed.get('date_from')
            date_to   = parsed.get('date_to')

            # ★ year_week 가 감지됐는데 date_from/date_to 가 비어 있으면, 그 주차의
            #   월/일요일을 계산해 채워준다. 이렇게 하면 text 검색의 r.date 필터,
            #   recency_focus 정렬 등 날짜 기반 로직도 자연스럽게 이 주차 범위를 따른다.
            if year_week and not date_from and not date_to:
                wm = re.match(r'^(\d{4})-W(\d{1,2})$', year_week)
                if wm:
                    try:
                        y, w = int(wm.group(1)), int(wm.group(2))
                        date_from = date.fromisocalendar(y, w, 1).isoformat()  # 월요일
                        date_to   = date.fromisocalendar(y, w, 7).isoformat()  # 일요일
                    except Exception as e:
                        self._dbg(0, f"[Debug] 주차→날짜 범위 계산 실패: {e}")

            # ★ 주차 "범위"(year_week_from~year_week_to) 정규화 ("YYYY-WNN", 2자리 0-패딩).
            #   이 값은 내부 문서(date_as_week) 검색 시 year_week 문자열 범위로 직접
            #   필터링하는 데 쓴다 — 내부 문서는 정확한 date 가 없는 경우가 있어 date 범위로
            #   거르면 매칭이 0이 되기 때문(주차로만 관리됨).
            def _norm_yw(v):
                if not v:
                    return None
                mm = re.match(r'^(\d{4})-W(\d{1,2})$', str(v).strip(), re.IGNORECASE)
                return f"{mm.group(1)}-W{int(mm.group(2)):02d}" if mm else None

            year_week_from = _norm_yw(parsed.get('year_week_from'))
            year_week_to   = _norm_yw(parsed.get('year_week_to'))

            # 주차 범위가 감지됐는데 date_from/date_to 가 비어 있으면, 시작 주차의
            # 월요일 ~ 끝 주차의 일요일로 날짜 범위도 함께 계산한다 (date 로 관리되는
            # 데이터셋/채널 및 정렬 로직이 이 범위를 따르도록).
            if not date_from and not date_to and (year_week_from or year_week_to):
                try:
                    if year_week_from:
                        wm = re.match(r'^(\d{4})-W(\d{1,2})$', year_week_from)
                        date_from = date.fromisocalendar(int(wm.group(1)), int(wm.group(2)), 1).isoformat()
                    if year_week_to:
                        wm = re.match(r'^(\d{4})-W(\d{1,2})$', year_week_to)
                        date_to = date.fromisocalendar(int(wm.group(1)), int(wm.group(2)), 7).isoformat()
                except Exception as e:
                    self._dbg(0, f"[Debug] 주차 범위→날짜 범위 계산 실패: {e}")

            # ★ target_datasets 검증: 유효한 데이터셋명만 남기고, 결과가 비었거나
            #   파싱이 이상하면 안전하게 "전체 데이터셋"으로 폴백한다 — 잘못 좁혀서
            #   관련 결과를 놓치는 것보다, 넓게 검색하는 게 훨씬 안전하기 때문.
            target_raw = parsed.get('target_datasets')
            target_datasets = None
            if isinstance(target_raw, list):
                target_datasets = [ds for ds in target_raw if ds in valid_dataset_keys]
            if not target_datasets:
                target_datasets = valid_dataset_keys

            # ★ ReportsDB(종합 보고서)를 검색하면 Confluence(연구원 개별 보고서)도
            #   항상 함께 검색한다 — 두 데이터셋은 사내 내부 보고 문서로 내용이
            #   겹치므로, ReportsDB 만 검색하면 개별 연구원 보고 내용을 놓친다.
            #   (단방향: Confluence 만 선택된 경우는 그대로 둔다.)
            if REPORTS_DATASET in target_datasets and CONFLUENCE_DATASET not in target_datasets \
                    and CONFLUENCE_DATASET in valid_dataset_keys:
                target_datasets = target_datasets + [CONFLUENCE_DATASET]

            # 단일 주차(year_week)만 있고 범위가 없으면, 문서 채널(B/C/D)의 주차 필터가
            # 그 한 주차로 좁혀지도록 범위 양끝을 같은 값으로 채운다.
            if year_week and not year_week_from and not year_week_to:
                year_week_from = year_week
                year_week_to   = year_week

            author_names = (parsed.get('author_names') or '').strip()

            # ★ recency_focus/oldest_focus 는 정반대 방향 정렬이라 둘 다 true 면
            #   모순이다. LLM 이 실수로 둘 다 true 를 내놓는 경우를 방어적으로
            #   처리한다 — recency_focus(최신)를 우선시킨다(더 흔한 요청이므로).
            recency_focus = bool(parsed.get('recency_focus', False))
            oldest_focus  = bool(parsed.get('oldest_focus', False)) and not recency_focus

            return {
                'keywords':        parsed.get('keywords', query),
                'author_names':    author_names,
                'date_from':       date_from,
                'date_to':         date_to,
                'recency_focus':   recency_focus,
                'oldest_focus':    oldest_focus,
                'year_week':       year_week,
                'year_week_from':  year_week_from,
                'year_week_to':    year_week_to,
                'target_datasets': target_datasets,
                'full_content':    bool(parsed.get('full_content', False)),
                'prefer_normalized': bool(parsed.get('prefer_normalized', False)),
                'extraction_failed': False,
            }
        except Exception as e:
            # ★ result 가 None 인 경우(=LLM API 호출 자체가 실패, 예: 게이트웨이 503)와
            #   JSON 파싱/형식 오류를 구분해서 로그를 남긴다. 원인 파악에 유용하다.
            reason = "LLM API 호출 실패 (재시도 후에도 응답 없음)" if result is None else f"{e}"
            self._dbg(0, f"[Debug] 키워드 추출 실패 → 원본 질문으로 폴백 (날짜/저자 등 추출 없이 검색): "
                         f"{reason}" + (f" / 원본: {result}" if result is not None else ""))
            return {'keywords': query, 'author_names': '', 'date_from': None, 'date_to': None,
                    'recency_focus': False, 'oldest_focus': False, 'year_week': None,
                    'year_week_from': None, 'year_week_to': None,
                    'target_datasets': list(DATASETS.keys()),
                    'full_content': False,
                    'prefer_normalized': False,
                    'extraction_failed': True}

    def _build_return_clause(self, return_fields: list) -> str:
        base = [
            "s.name AS sname", "s.type AS stype",
            "type(r) AS rel",
            "o.name AS oname", "o.type AS otype",
            # ★ A 기능: 각 트리플이 어느 문서(doc_id)에서 나왔는지 항상 가져온다.
            #   이 doc_id 로 나중에 Paper/Report 메타 노드를 조인해 원문을 붙인다.
            #   (return_fields 에는 없으므로 트리플 줄에는 출력되지 않고, 내부 수집용으로만 쓰임)
            "CASE WHEN r.doc_id IS NOT NULL THEN r.doc_id ELSE '' END AS doc_id",
            # ★ ReportsDB 는 트리플 생성 시점부터 관계에 year_week 를 직접 저장해
            #   두었으므로 (date 로부터 재계산하지 않고) 저장된 값을 그대로 쓴다.
            #   year_week 가 없는 데이터셋의 관계는 그냥 빈 문자열이 되어
            #   _format_rows 에서 date 로 폴백된다.
            "CASE WHEN r.year_week IS NOT NULL THEN r.year_week ELSE '' END AS year_week",
        ]
        fields = [
            f"CASE WHEN r.{f} IS NOT NULL THEN r.{f} ELSE '' END AS {f}"
            for f in return_fields
        ]
        return ", ".join(base + fields)

    def _build_where(self, cfg: dict, keywords: list,
                     date_from: str, date_to: str) -> tuple[list, dict]:
        search_fields  = cfg.get('search_fields', ['s.name', 'o.name', 'r.evidence'])
        min_confidence = cfg.get('min_confidence', None)
        has_date       = cfg.get('has_date', False)

        field_conditions = []
        for field in search_fields:
            if field.startswith('r.'):
                field_conditions.append(
                    f"({field} IS NOT NULL AND toLower({field}) CONTAINS toLower(kw))"
                )
            else:
                field_conditions.append(f"toLower({field}) CONTAINS toLower(kw)")

        where_parts = [f"any(kw IN $keywords WHERE {' OR '.join(field_conditions)})"]
        params: dict = {'keywords': keywords}

        if min_confidence is not None:
            where_parts.append("r.confidence >= $min_confidence")
            params['min_confidence'] = min_confidence
        if has_date and date_from:
            where_parts.append("r.date >= $date_from")
            params['date_from'] = date_from
        if has_date and date_to:
            where_parts.append("r.date <= $date_to")
            params['date_to'] = date_to

        return where_parts, params

    # ★ 트리플 컨텍스트에 필드명을 그대로(source_url, author 등 영어) 노출하면
    #   LLM이 그 영어 라벨을 답변에 그대로 베껴 쓰는 경우가 있었다.
    #   (예: "source_url: https://..." 처럼 라벨이 섞여 나옴)
    #   한글 라벨로 바꿔서 LLM이 자연스러운 한국어 문장으로 답하도록 유도한다.
    _FIELD_LABELS = {
        'evidence':    '근거',
        'source_url':  '출처',
        'title':       '제목',
        'author':      '저자',
        'journal':     '저널',
        'date':        '날짜',
    }

    def _format_rows(self, rows: list, return_fields: list, date_as_week: bool = False) -> str:
        if not rows:
            return _NO_RESULT

        for r in rows:
            if r.get('sname'): self._pending_nodes.add(r['sname'])
            if r.get('oname'): self._pending_nodes.add(r['oname'])

        lines = []
        for r in rows:
            line = (f"- ({r.get('sname')} :{r.get('stype')}) "
                    f"--[{r.get('rel')}]--> "
                    f"({r.get('oname')} :{r.get('otype')})")
            for f in return_fields:
                if f != 'confidence' and r.get(f):
                    label = self._FIELD_LABELS.get(f, f)
                    value = r[f]
                    # ★ ReportsDB/Confluence 는 정확한 날짜 대신 "몇 주차"로 표시
                    #   (date_as_week=True 인 데이터셋만; PapersDB 는 그대로 날짜 유지)
                    #   관계에 저장된 r.year_week 가 있으면 그 값을 그대로 신뢰하고
                    #   (재계산 없음), 없으면 date 로부터 계산한다 (doc_id 처럼 year_week
                    #   도 base 필드로 항상 조회됨).
                    if f == 'date' and date_as_week:
                        value = (_stored_week_to_label(r['year_week']) if r.get('year_week')
                                 else _date_to_week_label(value))
                    line += f"\n  {label}: {value}"
            lines.append(line)
        return "\n".join(lines)

    def _text_retrieve_raw(self, keywords_str: str, dataset: str, cfg: dict,
                           limit: int, date_from: str, date_to: str,
                           allowed_rels: list[str] = None,
                           recency_focus: bool = False,
                           oldest_focus: bool = False) -> list[dict]:
        """
        allowed_rels 가 주어지면 그 관계 타입으로만 결과를 제한한다.
        (1차 검색에서는 None 으로 호출해 관계 타입 무관하게 키워드 매칭하고,
         2-hop 확장에서는 cfg['hop2_relations'] 를 넘겨 인과·성능 계열로만 제한한다.)

        recency_focus/oldest_focus 가 True 이고 이 데이터셋이 날짜(date)를 지원하면,
        정렬 기준을 cfg 의 기본값(confidence DESC) 대신 date DESC(최신)/ASC(가장 오래된)로
        바꾼다.
        ★ 이건 Cypher ORDER BY + LIMIT 단계에서 바로 적용되어야 의미가 있다.
          결과를 다 가져온 뒤 Python 에서 재정렬하면, 애초에 confidence 기준으로
          LIMIT 에 걸려 짤린 "진짜로 최신(또는 가장 오래된)인데 confidence 가 낮은"
          트리플을 영영 놓치게 된다.
        """
        keywords      = _split_keywords(keywords_str)
        return_fields = cfg['return_fields']
        return_clause = self._build_return_clause(return_fields)
        where_parts, params = self._build_where(cfg, keywords, date_from, date_to)
        params['limit'] = limit
        # ★ 출처(구조) 관계 제외: FROM_PAPER / FROM_DOC 는 의미 트리플이 아님
        params['struct_rels'] = _STRUCTURAL_RELS

        rel_filter = ""
        if allowed_rels:
            params['allowed_rels'] = allowed_rels
            rel_filter = "AND type(r) IN $allowed_rels"

        if recency_focus and cfg.get('has_date'):
            sort_field, sort_order = 'date', 'DESC'   # RETURN 절의 별칭(alias) 참조
        elif oldest_focus and cfg.get('has_date'):
            sort_field, sort_order = 'date', 'ASC'
        else:
            sort_field, sort_order = cfg['sort_field'], cfg['sort_order']

        query_str = f"""
            MATCH (s:{dataset})-[r]->(o:{dataset})
            WHERE NOT type(r) IN $struct_rels
              {rel_filter}
              AND {" AND ".join(where_parts)}
            RETURN {return_clause}
            ORDER BY {sort_field} {sort_order}
            LIMIT $limit
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query_str, **params)]

    def _vector_retrieve_raw(self, query_text: str, dataset: str, cfg: dict,
                             limit: int, date_from: str, date_to: str) -> list[dict]:
        vector_index  = cfg.get('vector_index')
        if not vector_index:
            return []

        return_fields  = cfg['return_fields']
        min_confidence = cfg.get('min_confidence', None)
        has_date       = cfg.get('has_date', False)
        meta_label     = cfg.get('meta_label')   # ★ 벡터 인덱스에 섞인 메타 노드 레이블

        # 1) 질문 텍스트를 임베딩(1024차원 벡터)으로 변환
        q_emb = get_embedding(query_text)
        if q_emb is None:
            # ★ 임베딩 API 호출 실패(예: 게이트웨이 502). 0벡터로 검색을 강행하면
            #   Neo4j 가 "노름이 0인 벡터" 라며 에러를 던지므로, 아예 벡터 검색을
            #   건너뛴다 — text 검색 등 다른 채널은 영향받지 않고 계속 동작한다.
            self._dbg(0, "[Debug] 임베딩 실패로 엔티티 벡터 검색 건너뜀")
            return []

        # 2) 벡터 검색 후 추가로 적용할 필터 조건들을 모은다.
        #    filter_parts 는 관계(r) 속성에 대한 조건이고,
        #    node_filter 는 벡터 검색으로 나온 노드(s) 자체에 대한 조건이다.
        filter_parts = []
        params: dict = {'q_emb': q_emb, 'limit': limit}

        if min_confidence is not None:
            filter_parts.append("r.confidence >= $min_confidence")
            params['min_confidence'] = min_confidence
        if has_date and date_from:
            filter_parts.append("r.date >= $date_from")
            params['date_from'] = date_from
        if has_date and date_to:
            filter_parts.append("r.date <= $date_to")
            params['date_to'] = date_to

        filter_clause = f"AND {' AND '.join(filter_parts)}" if filter_parts else ""

        # ★ 스키마 변경 대응:
        #   reportsdb_entity_embedding 인덱스는 FOR (n:ReportsDB) 로 생성되는데,
        #   Report 메타 노드(구 Document)도 :ReportsDB 레이블 + embedding 속성을
        #   가지므로 이 인덱스에 함께 포함된다. 따라서 벡터 검색 결과에서 메타
        #   노드를 'WHERE NOT s:Report' 로 제외해야 엔티티만 남는다.
        #   (PapersDB 는 메타 노드가 다른 속성명을 써서 애초에 안 섞이지만,
        #    일관성/안전을 위해 동일하게 필터를 건다.)
        node_filter = f"AND NOT s:{meta_label}" if meta_label else ""

        # ★ 출처(구조) 관계 제외 (text 검색과 동일한 이유)
        params['struct_rels'] = _STRUCTURAL_RELS

        field_returns = ", ".join([
            f"CASE WHEN r.{f} IS NOT NULL THEN r.{f} ELSE '' END AS {f}"
            for f in return_fields
        ])

        # ★ 최소 유사도 컷: 이게 없으면 질문과 무관해도 "그나마 가장 가까운"
        #   상위 limit 개가 그냥 다 반환되어 컨텍스트에 노이즈가 섞인다.
        params['score_min'] = self.entity_score_min

        # 3) 벡터 검색 → 나온 노드(s)를 트리플의 주어로 삼아 관계까지 확장
        #    - queryNodes 로 상위 (limit*3) 개 후보를 넉넉히 뽑고,
        #      메타 노드 제외 + 관계 필터 + 최소 유사도 컷을 적용한 뒤
        #      vec_score 순으로 limit 개만 사용.
        query_str = f"""
            CALL db.index.vector.queryNodes('{vector_index}', $limit * 3, $q_emb)
            YIELD node AS s, score AS vec_score
            WHERE vec_score > $score_min
            MATCH (s:{dataset})-[r]->(o:{dataset})
            WHERE NOT type(r) IN $struct_rels {node_filter} {filter_clause}
            RETURN s.name AS sname, s.type AS stype,
                   type(r) AS rel,
                   o.name AS oname, o.type AS otype,
                   vec_score,
                   CASE WHEN r.doc_id IS NOT NULL THEN r.doc_id ELSE '' END AS doc_id,
                   {field_returns}
            ORDER BY vec_score DESC
            LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                rows = [dict(r) for r in session.run(query_str, **params)]
            if rows:
                scores = [r['vec_score'] for r in rows if r.get('vec_score') is not None]
                if scores:
                    self._dbg(2, f"[Debug] {dataset} 엔티티 벡터 점수 분포: "
                          f"min={min(scores):.3f} max={max(scores):.3f} "
                          f"avg={sum(scores)/len(scores):.3f} (컷={self.entity_score_min}, {len(scores)}건)")
                    # 결과 하나하나의 점수를 다 찍으면 (최대 limit*3=150건) 터미널이
                    # 감당 안 되므로, level 3(상세)에서만 상위 _DEBUG_ROW_PREVIEW 개를 보여준다.
                    for r in rows[:_DEBUG_ROW_PREVIEW]:
                        self._dbg(3, f"  [Debug]   score={r['vec_score']:.3f}  "
                              f"({r.get('sname')}) --[{r.get('rel')}]--> ({r.get('oname')})")
                    if len(rows) > _DEBUG_ROW_PREVIEW:
                        self._dbg(3, f"  [Debug]   ... 외 {len(rows) - _DEBUG_ROW_PREVIEW}건 생략")

                    # ★ 상대 컷오프: BGE-M3 는 무관한 쌍도 baseline 유사도가 높아
                    #   절대값 컷(entity_score_min)만으로는 옥석이 안 걸러진다.
                    #   1등 점수 대비 relative_gap 이상 뒤처지는 결과는 제외한다.
                    top_score = max(scores)
                    before = len(rows)
                    rows = [r for r in rows
                            if r.get('vec_score') is not None
                            and top_score - r['vec_score'] <= self.relative_gap]
                    if len(rows) != before:
                        self._dbg(2, f"[Debug] {dataset} 상대 컷오프 적용(gap≤{self.relative_gap}): "
                              f"{before}건 → {len(rows)}건")
            return rows
        except Exception as e:
            self._dbg(0, f"[Debug] {dataset} vector 검색 실패 → text 결과만 사용: {e}")
            return []

    def _vector_retrieve_2hop(self, query_text: str, dataset: str, cfg: dict,
                              limit: int, date_from: str, date_to: str) -> list[dict]:
        """
        벡터 검색으로 찾은 엔티티(1차, 의미상 가장 관련 있는 진입점)를 앵커 삼아,
        순수 그래프 탐색으로 한 단계 더 확장한다 (텍스트 2-hop과 동일한 철학).

        임베딩을 다시 계산하지 않고 그래프 구조만 따라가며, hop2_relations
        화이트리스트(인과·성능 계열)로 확장 범위를 제한해 잡음을 줄인다.
        """
        hop1_rows = self._vector_retrieve_raw(
            query_text, dataset, cfg, limit, date_from, date_to
        )

        hop1_nodes = set()
        for r in hop1_rows:
            if r.get('sname'): hop1_nodes.add(r['sname'])
            if r.get('oname'): hop1_nodes.add(r['oname'])

        hop2_relations = cfg.get('hop2_relations')
        if not hop1_nodes or not hop2_relations:
            return hop1_rows[:limit]

        return_fields = cfg['return_fields']
        return_clause = self._build_return_clause(return_fields)
        min_confidence = cfg.get('min_confidence', None)
        has_date       = cfg.get('has_date', False)

        filter_parts = []
        params: dict = {
            'anchors':      list(hop1_nodes),
            'allowed_rels': hop2_relations,
            'struct_rels':  _STRUCTURAL_RELS,
            'limit':        limit,
        }
        if min_confidence is not None:
            filter_parts.append("r.confidence >= $min_confidence")
            params['min_confidence'] = min_confidence
        if has_date and date_from:
            filter_parts.append("r.date >= $date_from")
            params['date_from'] = date_from
        if has_date and date_to:
            filter_parts.append("r.date <= $date_to")
            params['date_to'] = date_to
        filter_clause = f"AND {' AND '.join(filter_parts)}" if filter_parts else ""

        # 앵커 노드(1차 결과)가 주어 또는 목적어로 등장하는, 인과·성능 계열
        # 관계만 그래프에서 직접 탐색한다 (임베딩 재계산 없음, 순수 그래프 확장).
        query_str = f"""
            MATCH (s:{dataset})-[r]->(o:{dataset})
            WHERE NOT type(r) IN $struct_rels
              AND type(r) IN $allowed_rels
              AND (s.name IN $anchors OR o.name IN $anchors)
              {filter_clause}
            RETURN {return_clause}
            LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                hop2_rows = [dict(r) for r in session.run(query_str, **params)]
        except Exception as e:
            self._dbg(0, f"[Debug] {dataset} 벡터 2-hop 확장 실패: {e}")
            hop2_rows = []

        seen: dict = {}
        for r in hop1_rows + hop2_rows:
            key = f"{r.get('sname')}|{r.get('rel')}|{r.get('oname')}"
            seen[key] = r

        self._dbg(2, f"[Debug] {dataset} 벡터 2-hop: 1차 {len(hop1_rows)}개 + "
              f"2차(인과·성능 확장) {len(hop2_rows)}개 → 중복 제거 후 {min(len(seen), limit)}개")
        return list(seen.values())[:limit]

    def _text_retrieve_2hop(self, keywords_str: str, dataset: str, cfg: dict,
                            limit: int, date_from: str, date_to: str,
                            recency_focus: bool = False,
                            oldest_focus: bool = False) -> list[dict]:
        hop_limit = limit * 2

        # 1차 검색: 질문 키워드로 매칭 — 관계 타입 무관 (원 키워드가 직접 맞은 것이므로)
        hop1_rows = self._text_retrieve_raw(
            keywords_str, dataset, cfg, hop_limit, date_from, date_to,
            recency_focus=recency_focus, oldest_focus=oldest_focus
        )

        hop1_nodes = set()
        for r in hop1_rows:
            if r.get('sname'): hop1_nodes.add(r['sname'])
            if r.get('oname'): hop1_nodes.add(r['oname'])

        if not hop1_nodes:
            return hop1_rows[:limit]

        # 2차 확장: 1차 엔티티를 앵커로 삼아 한 단계 더 나아가되,
        # ★ hop2_relations 화이트리스트로 "인과·성능 계열" 관계만 따라간다.
        #   (구조/조성 관계까지 다 따라가면 원 질문과 무관한 잡음이 급증하기 때문)
        hop2_relations = cfg.get('hop2_relations')
        extended_keywords = keywords_str + ", " + ", ".join(list(hop1_nodes)[:10])
        hop2_rows = self._text_retrieve_raw(
            extended_keywords, dataset, cfg, hop_limit, date_from, date_to,
            allowed_rels=hop2_relations, recency_focus=recency_focus, oldest_focus=oldest_focus
        )

        seen: dict = {}
        for r in hop1_rows + hop2_rows:
            key = f"{r.get('sname')}|{r.get('rel')}|{r.get('oname')}"
            seen[key] = r

        results = list(seen.values())
        if recency_focus and cfg.get('has_date'):
            # hop1/hop2 는 각각 날짜순으로 정렬돼 있지만, 두 결과를 합치면 전체 순서가
            # 깨지므로 병합 후 다시 날짜 내림차순으로 재정렬한다. (날짜 없는 항목은 뒤로)
            results.sort(key=lambda r: r.get('date') or '', reverse=True)
        elif oldest_focus and cfg.get('has_date'):
            # ★ 오름차순(과거 → 현재)으로 재정렬. 날짜 없는 항목은 아주 먼 미래 값으로
            #   취급해 뒤로 보낸다(그래야 "가장 오래된" 결과 맨 앞에 진짜 오래된 것만 옴).
            results.sort(key=lambda r: r.get('date') or '9999-12-31')

        self._dbg(2, f"[Debug] {dataset} 2-hop: 1차 {len(hop1_rows)}개 + "
              f"2차 {len(hop2_rows)}개 → 중복 제거 후 {min(len(results), limit)}개")
        return results[:limit]

    # ── 공통: 메타 노드 날짜/주차 범위 필터 ──────────────────────────────────────
    def _meta_date_filter(self, cfg: dict, alias: str,
                          date_from: str = None, date_to: str = None,
                          yw_from: str = None, yw_to: str = None) -> tuple[str, dict]:
        """
        문서 메타 노드(alias)에 걸 날짜/주차 범위 필터절과 파라미터를 만든다.

        ★ ReportsDB/Confluence 같은 date_as_week 데이터셋은 정확한 date 속성이
          없거나(주차로만 관리) 신뢰할 수 없어서, date 범위로 거르면 매칭이 0이 되는
          문제가 있었다. 이 경우 year_week 문자열 범위("2026-W10"~"2026-W15")로
          거른다. "YYYY-WNN"(주차 2자리 0-패딩) 포맷은 사전식 비교가 연대순과
          일치하므로 문자열 부등호 비교로 안전하게 범위 필터가 된다. DB에는 소문자
          'w'(예: "2026-w10")로 저장돼 있어 toLower() 로 양쪽을 맞춘다.

        그 외(PapersDB 등) 데이터셋은 기존대로 date 범위로 거른다.
        """
        parts: list[str] = []
        params: dict = {}
        if cfg.get('date_as_week') and (yw_from or yw_to):
            # ★ Confluence 는 이제 두 소스가 섞여 있다:
            #     - 주간보고   : year_week 값 있음 (주차로 판단)
            #     - 서브페이지 : year_week 가 빈 값이고 date(last_modified)만 있음
            #   예전처럼 year_week 만으로 거르면 서브페이지가 전부 탈락하므로,
            #   "year_week 이 있으면 주차 범위로, 없으면 date 범위로" 판단한다.
            wk: list[str] = []
            if yw_from:
                wk.append(f"toLower({alias}.year_week) >= toLower($yw_from)")
                params['yw_from'] = yw_from
            if yw_to:
                wk.append(f"toLower({alias}.year_week) <= toLower($yw_to)")
                params['yw_to'] = yw_to
            week_cond = (f"(COALESCE({alias}.year_week, '') <> '' AND "
                         f"{' AND '.join(wk)})")

            dt: list[str] = []
            if date_from:
                dt.append(f"{alias}.date >= $date_from")
                params['date_from'] = date_from
            if date_to:
                dt.append(f"{alias}.date <= $date_to")
                params['date_to'] = date_to
            if dt:
                date_cond = (f"(COALESCE({alias}.year_week, '') = '' AND "
                             f"{' AND '.join(dt)})")
                parts.append(f"({week_cond} OR {date_cond})")
            else:
                parts.append(week_cond)
        else:
            if date_from:
                parts.append(f"{alias}.date >= $date_from")
                params['date_from'] = date_from
            if date_to:
                parts.append(f"{alias}.date <= $date_to")
                params['date_to'] = date_to
        clause = "".join(f" AND {p}" for p in parts)
        return clause, params

    # ── B 기능: 문서 단위 벡터 검색 ────────────────────────────────────────────
    def _doc_vector_retrieve(self, query_text: str, cfg: dict,
                             limit: int = DOC_SEARCH_LIMIT,
                             date_from: str = None, date_to: str = None,
                             yw_from: str = None, yw_to: str = None) -> list[str]:
        """
        Paper/Report 메타 노드를 대상으로 벡터 검색을 수행해 관련 문서의
        doc_id 목록을 돌려준다.

        엔티티 벡터 검색(_vector_retrieve_raw)이 "관계(트리플)"를 찾는 것과 달리,
        이건 "문서 자체"(논문 초록 / 보고서 전문)를 질문과의 유사도로 찾는다.
        → "이 주제 관련 논문/보고서 찾아줘" 류의 질의에 강하다.

        papersdb_doc_embedding / reportsdb_doc_embedding / confluence_doc_embedding
        인덱스는 각각 해당 데이터셋의 단일 노드 레이블만 포함하므로 메타 노드
        제외 필터가 필요 없다.

        ★ 청킹된 데이터셋(Confluence)은 doc_id 자체가 청크 단위 유니크 키이므로
          그대로 doc_id 를 돌려주면 된다(별도 조인/분기 불필요).
        """
        doc_index = cfg.get('doc_vector_index')
        if not doc_index:
            return []

        q_emb = get_embedding(query_text)
        if q_emb is None:
            self._dbg(0, "[Debug] 임베딩 실패로 문서 벡터 검색 건너뜀")
            return []
        # ★ 기간(date/주차) 필터: 없으면 "2026년 관련 보고서" 처럼 기간이 지정된
        #   질문에서도 벡터 유사도만 보고 연도 제한이 전혀 안 걸려 다른 연도 문서까지
        #   섞여 나오는 문제가 있었다. date_as_week 데이터셋은 year_week 범위로 거른다.
        doc_key = cfg.get('doc_key', 'doc_id')
        date_filter, date_params = self._meta_date_filter(
            cfg, 'node', date_from, date_to, yw_from, yw_to
        )
        query_str = f"""
            CALL db.index.vector.queryNodes($index, $limit, $q_emb)
            YIELD node, score
            WHERE score > $score_min {date_filter}
            RETURN node.{doc_key} AS doc_id, score
            ORDER BY score DESC
        """
        params = {
            'index':     doc_index,
            'limit':     limit,
            'q_emb':     q_emb,
            'score_min': self.doc_score_min,
            **date_params,
        }
        try:
            with self.driver.session() as session:
                rows = [dict(r) for r in session.run(query_str, **params)]
            if rows:
                self._dbg(2, f"[Debug] 문서 벡터 검색 결과 (컷={self.doc_score_min}, {len(rows)}건):")
                for r in rows:
                    self._dbg(3, f"  [Debug]   score={r['score']:.3f}  doc_id={r.get('doc_id')}")

                # ★ 상대 컷오프 (엔티티 벡터 검색과 동일한 이유)
                top_score = max(r['score'] for r in rows)
                before = len(rows)
                rows = [r for r in rows if top_score - r['score'] <= self.relative_gap]
                if len(rows) != before:
                    self._dbg(2, f"[Debug] 문서 벡터 상대 컷오프 적용(gap≤{self.relative_gap}): "
                          f"{before}건 → {len(rows)}건")
            return [r['doc_id'] for r in rows if r.get('doc_id')]
        except Exception as e:
            self._dbg(0, f"[Debug] 문서 벡터 검색 실패: {e}")
            return []

    # ── C 기능: 문서 본문 키워드(FULLTEXT) 검색 ─────────────────────────────────
    def _doc_fulltext_retrieve(self, keywords_str: str, cfg: dict,
                               limit: int = DOC_SEARCH_LIMIT,
                               date_from: str = None, date_to: str = None,
                               yw_from: str = None, yw_to: str = None) -> list[str]:
        """
        doc_fulltext_index 로 지정된 FULLTEXT 인덱스에서 문서 본문 키워드를 검색해
        관련 문서의 doc_id 목록을 돌려준다.

        벡터 검색(_doc_vector_retrieve)이 "의미 유사도"로 찾는다면, 이건 "단어 일치"로
        찾는다. 코드(D1)·모델명·수치처럼 정확한 표기가 중요한 검색에 강하다.

        ★ doc_fulltext_index 는 문자열 하나 또는 문자열 리스트를 받는다.
          Confluence 처럼 원본(코드)용/정규화(물질명)용 FULLTEXT 인덱스가
          두 개로 분리된 데이터셋은 리스트로 [원본 인덱스, 정규화 인덱스] 를 주면
          둘 다 검색해서 결과를 합친다 (ReportsDB 는 인덱스 하나에 두 필드가
          함께 들어있어 문자열 하나로 충분).
        """
        ft_indexes = cfg.get('doc_fulltext_index')
        if isinstance(ft_indexes, str):
            ft_indexes = [ft_indexes]
        if not keywords_str or not ft_indexes:
            return []

        # Lucene 질의 문자열 구성:
        # 키워드에 '/'·'-' 등 Lucene 특수문자가 있으면 파싱 오류가 나므로,
        # 각 키워드를 큰따옴표로 감싼 구(phrase)로 만들고 내부 특수문자는 이스케이프한다.
        # 따옴표로 감싼 구들을 공백으로 이으면 Lucene 기본 OR(should) 매칭이 된다.
        keywords = _split_keywords(keywords_str)
        if not keywords:
            return []

        def _escape(kw: str) -> str:
            return kw.replace('\\', '\\\\').replace('"', '\\"')

        lucene_query = " ".join(f'"{_escape(kw)}"' for kw in keywords)

        doc_ids: list[str] = []

        # ★ 기간(date/주차) 필터 (다른 문서 채널과 동일한 이유)
        date_filter, date_params = self._meta_date_filter(
            cfg, 'node', date_from, date_to, yw_from, yw_to
        )

        doc_key = cfg.get('doc_key', 'doc_id')
        query_str = f"""
            CALL db.index.fulltext.queryNodes($index, $q, {{limit: $limit}})
            YIELD node, score
            WHERE true {date_filter}
            RETURN node.{doc_key} AS doc_id, score
            ORDER BY score DESC
        """
        for ft_index in ft_indexes:
            params = {'index': ft_index, 'q': lucene_query, 'limit': limit,
                      **date_params}
            try:
                with self.driver.session() as session:
                    rows = [dict(r) for r in session.run(query_str, **params)]
                doc_ids.extend(r['doc_id'] for r in rows if r.get('doc_id'))
            except Exception as e:
                self._dbg(0, f"[Debug] 문서 FULLTEXT 검색 실패 ({ft_index}): {e}")
        return doc_ids

    # ── D 기능: 저자명으로 문서 직접 검색 ────────────────────────────────────────
    def _doc_author_retrieve(self, keywords_str: str, cfg: dict,
                             limit: int = AUTHOR_SEARCH_LIMIT,
                             date_from: str = None, date_to: str = None,
                             yw_from: str = None, yw_to: str = None) -> list[str]:
        """
        Paper/Report 메타 노드의 author 속성을 키워드로 직접 검색해 doc_id 를 돌려준다.

        text 검색(A)은 트리플의 r.author 를 보므로 "그 저자의 트리플이 있어야"만
        찾아지고, 저자만 언급되고 트리플에 안 걸린 논문은 놓친다. 이 채널은
        메타 노드를 author 기준으로 직접 조회하므로 "이 사람이 쓴 논문/보고서
        찾아줘" 류의 질문에 안정적으로 대응한다.

        ★ author 필드에 대한 정확한 매칭이라 B(벡터)/C(FULLTEXT) 보다 훨씬 신뢰도가
        높다. 그런데 기존에 DOC_SEARCH_LIMIT(5)를 그대로 썼더니, "그 사람이 쓴
        보고서 최근 걸 보여줘" 처럼 한 사람이 문서를 많이 쓴 경우 5건으로 잘려서
        정작 최신 문서가 후보에서 누락되는 문제가 있었다. AUTHOR_SEARCH_LIMIT(30)로
        넉넉히 잡고, 잘리더라도 최신 문서가 먼저 담기도록 date 내림차순으로 정렬한다.
        """
        doc_label = cfg.get('doc_label')
        if not doc_label or not keywords_str:
            return []

        keywords = _split_keywords(keywords_str)
        if not keywords:
            return []

        # ★ 저자 조건 (author/researcher 둘 다 지원 — 데이터셋마다 필드명이 다를 수 있음)
        #
        # ★ 양방향 CONTAINS: DB에는 순수 이름만 저장돼 있는데(예: "이창수"), LLM이
        #   추출하는 author_names 는 질문 문구에 따라 "이창수님", "이창수 연구원",
        #   "이창수 책임"처럼 존칭/직급이 붙어 나올 수 있다. 한쪽 방향(author CONTAINS kw)
        #   만 보면 kw 가 author 보다 길어지는 순간(존칭이 붙는 순간) 항상 실패한다
        #   (짧은 kw 가 긴 author 안에 있는지는 맞지만, 긴 kw 가 짧은 author 안에
        #   있을 리 없기 때문). 한국어는 존칭/직급이 이름 뒤에 붙으므로 kw 가 author 를
        #   prefix 로 포함하는 반대 방향도 함께 확인해야 "이창수님" 같은 표현이 걸린다.
        author_cond = (
            "(%(a)s.author IS NOT NULL OR %(a)s.researcher IS NOT NULL) "
            "AND any(kw IN $keywords WHERE "
            "        (COALESCE(%(a)s.author, '') <> '' AND "
            "         (toLower(%(a)s.author) CONTAINS toLower(kw) "
            "          OR toLower(kw) CONTAINS toLower(%(a)s.author))) "
            "     OR (COALESCE(%(a)s.researcher, '') <> '' AND "
            "         (toLower(%(a)s.researcher) CONTAINS toLower(kw) "
            "          OR toLower(kw) CONTAINS toLower(%(a)s.researcher))))"
        )

        # ★ 기간(date/주차) 필터: 이게 없으면 "2026년에 이창수가 쓴 보고서"처럼
        #   기간이 지정된 질문에서도 저자 매칭만 되고 연도 제한이 전혀 적용되지 않아,
        #   다른 연도의 문서까지 다 섞여 나오는 문제가 있었다.
        date_filter, date_params = self._meta_date_filter(
            cfg, 'm', date_from, date_to, yw_from, yw_to
        )
        params = {'keywords': keywords, 'limit': limit, **date_params}

        doc_key = cfg.get('doc_key', 'doc_id')
        query_str = f"""
            MATCH (m:{doc_label})
            WHERE {author_cond % {'a': 'm'}}
              {date_filter}
            RETURN m.{doc_key} AS doc_id
            ORDER BY m.date DESC
            LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                rows = [dict(r) for r in session.run(query_str, **params)]
            return [r['doc_id'] for r in rows if r.get('doc_id')]
        except Exception as e:
            self._dbg(0, f"[Debug] 저자 검색 실패: {e}")
            return []

    # ── E 기능: 주차(week)로 문서 정확 매칭 / 범위 열거 검색 ──────────────────────
    def _doc_week_retrieve(self, year_week: str, cfg: dict,
                           yw_from: str = None, yw_to: str = None) -> list[str]:
        """
        Report/Document(Confluence) 메타 노드의 year_week 속성으로 문서를 찾는다.

        두 가지 모드:
          (1) 단일 주차(year_week): 그 주차 정확 매칭
          (2) 주차 범위(yw_from~yw_to): 그 범위에 속하는 문서 열거

        "2026년 26주차 보고내용" 또는 "10~15주차 보고문서 전부" 같은 질문은
        "관련도 상위 몇 개"가 아니라 "그 기간에 해당하는 문서 전부"를 원하는
        열거형(enumeration) 질문이다. B(벡터)/C(FULLTEXT)는 관련도 기준이라
        일반적인 키워드("보고문서/요약")로는 대부분 컷오프에 걸려 누락되므로,
        기간이 명시된 경우 이 채널이 relevance 컷 없이 문서를 그대로 담아준다.

        ★ Cypher 문자열 비교는 대소문자를 구분하고, "YYYY-WNN"(2자리 0-패딩) 포맷은
          사전식 비교가 연대순과 일치하므로 toLower() 로 맞춘 뒤 부등호로 범위를 건다.
        """
        doc_label = cfg.get('doc_label')
        if not doc_label:
            return []

        a = 'm'
        if year_week:
            where = f"toLower({a}.year_week) = toLower($year_week)"
            params = {'year_week': year_week}
        elif yw_from or yw_to:
            conds = []
            params = {}
            if yw_from:
                conds.append(f"toLower({a}.year_week) >= toLower($yw_from)")
                params['yw_from'] = yw_from
            if yw_to:
                conds.append(f"toLower({a}.year_week) <= toLower($yw_to)")
                params['yw_to'] = yw_to
            where = f"{a}.year_week IS NOT NULL AND " + " AND ".join(conds)
        else:
            return []

        params['limit'] = WEEK_RANGE_LIMIT
        doc_key = cfg.get('doc_key', 'doc_id')
        query_str = f"""
            MATCH (m:{doc_label})
            WHERE {where}
            RETURN m.{doc_key} AS doc_id
            ORDER BY m.date DESC
            LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                rows = [dict(r) for r in session.run(query_str, **params)]
            return [r['doc_id'] for r in rows if r.get('doc_id')]
        except Exception as e:
            self._dbg(0, f"[Debug] 주차 검색 실패: {e}")
            return []

    # ── F 기능: 날짜 범위로 문서 열거 (주차 관리가 아닌 데이터셋용, 예: PapersDB) ──────
    def _doc_daterange_retrieve(self, cfg: dict,
                                date_from: str = None, date_to: str = None) -> list[str]:
        """
        m.date 로 관리되는 데이터셋(PapersDB 등)에서, 날짜 범위에 속하는 문서를
        relevance 컷 없이 전부 열거한다. _doc_week_retrieve 의 "날짜(date) 버전".

        "2026년 6월 등록된 논문을 보여줘" 같은 질문은 열거형인데, B(벡터)/C(FULLTEXT)는
        관련도 상위 5건으로 제한되어 있어 "6월"처럼 흔한 키워드로는 대부분 걸러지고
        소수만 나오는 문제가 있었다. 이 채널은 그 기간에 해당하는 문서를 관련도와
        무관하게 모두 담아준다(컨텍스트 폭주 방지를 위해 상한만 둠).

        ★ date_as_week 데이터셋(ReportsDB/Confluence)에서 year_week 를 가진 문서는
          _doc_week_retrieve(E채널)가 주차 기준으로 이미 담당하므로 중복을 피해 제외한다.
          단 Confluence 서브페이지처럼 year_week 가 비어 있고 date 만 있는 레코드는
          E채널이 잡을 수 없으므로, 이 채널이 date 기준으로 채워준다.
        """
        doc_label = cfg.get('doc_label')
        if not doc_label or not (date_from or date_to):
            return []

        a = 'm'
        conds = [f"{a}.date IS NOT NULL"]
        params: dict = {}
        if date_from:
            conds.append(f"{a}.date >= $date_from")
            params['date_from'] = date_from
        if date_to:
            conds.append(f"{a}.date <= $date_to")
            params['date_to'] = date_to
        if cfg.get('date_as_week'):
            # year_week 가 있는 문서는 E채널 담당 → 여기서는 없는 것만
            conds.append(f"COALESCE({a}.year_week, '') = ''")

        params['limit'] = WEEK_RANGE_LIMIT
        doc_key = cfg.get('doc_key', 'doc_id')
        query_str = f"""
            MATCH (m:{doc_label})
            WHERE {" AND ".join(conds)}
            RETURN m.{doc_key} AS doc_id
            ORDER BY m.date DESC
            LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                rows = [dict(r) for r in session.run(query_str, **params)]
            return [r['doc_id'] for r in rows if r.get('doc_id')]
        except Exception as e:
            self._dbg(0, f"[Debug] 날짜 범위 열거 검색 실패: {e}")
            return []

    # ── A 기능: doc_id 로 메타 노드 원문(abstract/content) 조회 ──────────────────
    def _fetch_documents(self, doc_ids: set[str], cfg: dict,
                         recency_focus: bool = False,
                         oldest_focus: bool = False,
                         full_content: bool = False,
                         prefer_normalized: bool = False) -> str:
        """
        doc_id 집합을 받아 Paper/Report 메타 노드에서 제목·본문·출처를 조회하고,
        LLM 컨텍스트에 넣을 "관련 문서" 섹션 문자열로 만든다.

        트리플은 (주어)-[관계]->(목적어) 형태라 근거 문장(evidence) 정도만 담지만,
        여기서 원문(논문 초록 / 보고서 전문)을 붙여 주면 답변 근거가 훨씬 풍부해진다.
        본문은 DOC_BODY_MAXLEN 로 잘라 컨텍스트 폭주를 막는다.

        recency_focus 가 True 면 문서를 날짜 내림차순("가장 최근" 질문),
        oldest_focus 가 True 면 날짜 오름차순("가장 오래된" 질문)으로 정렬해 반환한다.
        (둘 다 True 일 수는 없다 — _extract_keywords 에서 recency_focus 를 우선시켜
        방어적으로 배타 처리한다.)
        """
        doc_ids = {d for d in doc_ids if d}
        if not doc_ids or not self.driver:
            return ""

        doc_label  = cfg.get('doc_label')
        body_field = cfg.get('doc_body_field')
        body_label = cfg.get('doc_body_label', '내용')   # 논문='초록' / 보고서='내용'
        if not doc_label or not body_field:
            return ""

        # doc_label 로 메타 노드를 특정하고, 본문 속성명은 데이터셋마다 다르므로
        # 쿼리 문자열에 직접 끼워넣는다(값이 아니라 스키마라 파라미터화 불가).
        # body_field(content_norm 등)가 비어 있는 노드를 대비해 원본 content 로 폴백.
        # (Paper 노드엔 content 가 없으므로 COALESCE 는 자연히 body_field 값만 남긴다)
        # ★ full_content(원문/전체를 그대로 보여달라는 요청)일 때는 우선순위를 뒤집어
        #   원본 content(사내 코드 그대로, 예: BD30)를 먼저 쓰고 없으면 정규화본
        #   (content_norm, 물질명 변환본)으로 폴백한다 — "내부 문서를 그대로 보여줘"
        #   요청에서 코드가 물질명으로 바뀐 채 나오는 문제를 막기 위함. 평소(검색/요약
        #   목적)에는 반대로 content_norm 을 우선해 어휘를 통일한다.
        # ★ journal 은 Paper 노드 전용, year_week 는 내부 문서 노드 전용 필드
        #   (다른 데이터셋엔 없으면 null 반환되어 meta_bits 에서 자연히 제외된다).
        # ★ prefer_normalized(물질명 변환본을 명시적으로 요청)가 true 면 full_content
        #   여부와 무관하게 항상 content_norm(물질명)을 우선한다 — "물질명으로 변환된
        #   내용으로 보여줘" 요청은 원문 전체 요청(full_content)과 함께 와도 물질명이
        #   우선이어야 하기 때문.
        if prefer_normalized:
            body_expr = f"COALESCE(m.{body_field}, m.content)"
        elif full_content:
            body_expr = f"COALESCE(m.content, m.{body_field})"
        else:
            body_expr = f"COALESCE(m.{body_field}, m.content)"
        # ★ 청킹 대응 (단일 노드 구조 — Confluence 도 ReportsDB/PapersDB 와 동일):
        #   - 조회 키: doc_id 가 항상 유니크 키(청킹된 경우 "{그룹키}_chunk_N" 형태).
        #   - 정렬: 같은 문서(그룹)의 청크들이 흩어지지 않고 순서대로 붙어 나오도록
        #     doc_group_field(없으면 doc_id), chunk_index 로 정렬한다
        #     (최신순 요청이면 date 를 앞에 둔다). doc_id 문자열만으로 정렬하면
        #     "_chunk_10"이 "_chunk_2"보다 사전식으로 앞에 와 순서가 깨질 수 있어,
        #     그룹 키가 지정된 데이터셋은 반드시 그 필드로 그룹핑한다.
        #   - title/author 는 새 스키마의 page_title/researcher 도 함께 받는다.
        doc_key = cfg.get('doc_key', 'doc_id')
        is_chunked = bool(cfg.get('chunked'))
        group_field = cfg.get('doc_group_field')
        group_expr = f"m.{group_field}" if group_field else "m.doc_id"

        if doc_key != 'doc_id':
            where_clause = f"m.{doc_key} IN $ids OR m.doc_id IN $ids"
        else:
            where_clause = "m.doc_id IN $ids"
        if recency_focus:
            date_order = "m.date DESC"
        elif oldest_focus:
            date_order = "m.date ASC"
        else:
            date_order = None
        if is_chunked:
            order_clause = (f"ORDER BY {date_order}, {group_expr}, m.chunk_index"
                            if date_order else f"ORDER BY {group_expr}, m.chunk_index")
        else:
            order_clause = f"ORDER BY {date_order}" if date_order else ""
        query_str = f"""
            MATCH (m:{doc_label})
            WHERE {where_clause}
            RETURN m.{doc_key}  AS uid,
                   m.doc_id     AS doc_id,
                   {group_expr} AS doc_group_id,
                   COALESCE(m.title, m.page_title)   AS title,
                   COALESCE(m.author, m.researcher)  AS author,
                   m.date       AS date,
                   m.source_url AS source_url,
                   m.journal    AS journal,
                   m.year_week  AS year_week,
                   m.title_path   AS title_path,
                   m.chunk_index  AS chunk_index,
                   m.total_chunks AS total_chunks,
                   {body_expr} AS body
            {order_clause}
        """
        try:
            with self.driver.session() as session:
                docs = [dict(r) for r in session.run(query_str, ids=list(doc_ids))]
        except Exception as e:
            self._dbg(0, f"[Debug] 문서 조회 실패: {e}")
            return ""

        if not docs:
            return ""

        # ★ 중복 제거: 같은 노드가 여러 채널에서 중복으로 잡히거나(DB 중복), 제목+출처가
        #   동일한 사실상 같은 문서가 여러 건 저장돼 있으면 같은 내용이 두 번 나온다.
        #   유니크 키(doc_id)로 먼저 걸러내고, 그 키가 달라도
        #   제목+출처가 완전히 같으면 같은 문서로 보고 한 번만 남긴다(순서 유지).
        #   ★★ 청킹 데이터셋에서는 "제목+출처" 기준을 쓰면 한 페이지의 모든 청크가
        #   같은 제목·같은 URL 이라 전부 하나로 뭉개져 내용 대부분이 사라진다.
        #   그래서 청킹된 경우 이 2차 기준에 청크 식별자를 포함시킨다.
        deduped = []
        seen_ids: set = set()
        seen_keys: set = set()
        for d in docs:
            did = d.get('uid') or d.get('doc_id')
            if did and did in seen_ids:
                continue
            # 제목이 같아도 주차/출처가 다르면 다른 문서로 취급(오검출 방지)
            key = (d.get('title'), d.get('source_url'), d.get('year_week'))
            if is_chunked:
                key = key + (d.get('doc_group_id'), d.get('chunk_index'))
            has_key = any(k is not None for k in key)
            if has_key and key in seen_keys:
                continue
            if did:
                seen_ids.add(did)
            if has_key:
                seen_keys.add(key)
            deduped.append(d)
        if len(deduped) != len(docs):
            self._dbg(2, f"[Debug] {doc_label} 문서 중복 제거: {len(docs)}건 → {len(deduped)}건")
        docs = deduped

        lines = ["[관련 문서 원문]"]
        for d in docs:
            # ★ 예전엔 본문의 모든 줄바꿈을 공백으로 치환해 "한 줄"로 뭉갰는데,
            #   원본 문서(특히 Confluence 주간보고)에 마크다운 표가 포함된 경우
            #   그 표의 줄 구조(헤더/구분선/데이터 행)까지 통째로 파괴되어 LLM 에게
            #   이미 망가진 입력이 전달되고, LLM 은 그걸 최대한 표로 복원하려다
            #   형식이 어긋난 표를 내놓는 문제가 있었다(실제 원인으로 확인됨).
            #   이제 줄바꿈은 보존하고, 과도한 연속 빈 줄만 1개로 줄인다.
            body = re.sub(r'\n{3,}', '\n\n', (d.get('body') or '').strip())
            # ★ "전체/전문/원문 그대로 보여줘" 처럼 전체 내용을 원하는 질문이면
            #   DOC_BODY_MAXLEN(기본 700자) 로 자르지 않고 본문 전체를 그대로 넣는다.
            #   (평소엔 컨텍스트 폭주를 막기 위해 잘라서 보여준다.)
            if not full_content and len(body) > DOC_BODY_MAXLEN:
                body = body[:DOC_BODY_MAXLEN] + " …(생략)"

            header = f"- {d.get('title') or d.get('doc_id')}"
            # ★ 내부 문서(ReportsDB/Confluence)는 정확한 날짜 대신 주차로 표시.
            #   - year_week 속성이 있으면 재포맷 ("2026-W02" → "2026년 W02")
            #   - 없으면 date 로부터 계산
            #   - PapersDB 등 date_as_week 가 아닌 데이터셋: 날짜 그대로 표시
            if cfg.get('date_as_week'):
                date_bit = (_stored_week_to_label(d['year_week']) if d.get('year_week')
                           else _date_to_week_label(d.get('date')))
            else:
                date_bit = d.get('date')
            meta_bits = [b for b in (d.get('author'), d.get('journal'), date_bit) if b]
            # ★ 청킹된 문서는 "이 내용이 문서의 몇 번째 조각인지"를 알려준다 —
            #   그래야 LLM 이 잘린 문맥을 문서 전체로 오해하지 않는다.
            #   total_chunks == 1 이면 청킹되지 않은 문서이므로 표기하지 않는다.
            total_chunks = d.get('total_chunks')
            if total_chunks and total_chunks > 1:
                meta_bits.append(f"{(d.get('chunk_index') or 0) + 1}/{total_chunks} 부분")
            if meta_bits:
                header += f" ({', '.join(meta_bits)})"
            lines.append(header)
            if d.get('title_path'):
                lines.append(f"  경로: {d['title_path']}")
            if d.get('source_url'):
                lines.append(f"  출처: {d['source_url']}")
            if body:
                # ★ body 가 여러 줄(표 등 구조 포함)이면 "내용:" 라벨과 같은 줄에
                #   억지로 붙이지 않고 다음 줄부터 별도 블록으로 넣어 구조가
                #   깨지지 않게 한다. 한 줄짜리 본문은 기존처럼 한 줄로 표기.
                if '\n' in body:
                    lines.append(f"  {body_label}:\n{body}")
                else:
                    lines.append(f"  {body_label}: {body}")
        n_with_body = sum(1 for d in docs if (d.get('body') or '').strip())
        self._dbg(2, f"[Debug] {doc_label} 문서 원문 컨텍스트: {len(docs)}건 구성 "
                     f"(본문 있음 {n_with_body}건 / 본문 없음 {len(docs) - n_with_body}건)")
        return "\n".join(lines)

    def _collect_doc_ids(self, rows: list[dict]) -> set[str]:
        """트리플 검색 결과 rows 에서 출처 doc_id 들을 모은다 (A 기능용)."""
        return {r['doc_id'] for r in rows if r.get('doc_id')}

    def retrieve(self, keywords_str: str, query_text: str,
                 dataset: str, mode: str = None,
                 limit: int = DEFAULT_SEARCH_LIMIT,
                 date_from: str = None,
                 date_to: str = None,
                 recency_focus: bool = False,
                 oldest_focus: bool = False,
                 year_week: str = None,
                 year_week_from: str = None,
                 year_week_to: str = None,
                 author_names: str = None,
                 full_content: bool = False,
                 prefer_normalized: bool = False) -> str:
        cfg          = DATASETS.get(dataset, list(DATASETS.values())[0])
        search_mode  = mode or DEFAULT_SEARCH_MODE or cfg.get('default_mode', 'text')
        hops         = cfg.get('search_hops') or DEFAULT_SEARCH_HOPS
        vector_index = cfg.get('vector_index')

        # ★ ReportsDB 등 정규화 대상 데이터셋은 검색 전에 질의의 코드를 물질명으로 변환.
        #   키워드 검색(keywords_str)과 벡터 검색(query_text) 양쪽 모두 정규화해
        #   content_norm/정규화된 엔티티명과 어휘를 맞춘다.
        if cfg.get('normalize_query'):
            norm_kw    = _normalize_query(keywords_str)
            norm_query = _normalize_query(query_text)
            if norm_kw != keywords_str or norm_query != query_text:
                self._dbg(2, f"[Debug] {dataset} 질의 정규화: '{query_text}' → '{norm_query}'")

            # 키워드/FULLTEXT 검색: 원본 + 정규화 키워드를 모두 사용한다.
            #   - 원본 키워드(BD30 등) → content(원본 코드) / 원본 엔티티명 매칭
            #   - 정규화 키워드(HfO2 등) → content_norm(물질명) / 정규화 엔티티명 매칭
            #   한 쿼리에서 두 어휘를 동시에 훑으므로 어느 쪽에 저장돼 있든 잡힌다.
            if norm_kw and norm_kw != keywords_str:
                keywords_str = keywords_str + ", " + norm_kw

            # 벡터 검색: 인덱스가 content_norm(물질명) 기반이므로 정규화 질문만 사용한다.
            query_text = norm_query

        # ★ requested_mode: 문서 채널(B/C) 게이팅 전용으로 "사용자가 원래 요청한 모드"를
        #   보존한다. OKR처럼 엔티티 트리플이 아예 없는(엔티티 벡터 인덱스 없음) 데이터셋도
        #   문서 자체의 벡터 인덱스(doc_vector_index)는 가질 수 있는데, 아래에서
        #   entity 벡터 인덱스 부재로 search_mode 를 'text' 로 강제 폴백해버리면
        #   문서 벡터 검색(B 채널)까지 같이 꺼지는 문제가 있었다.
        requested_mode = search_mode

        if search_mode in ('vector', 'hybrid') and not vector_index:
            self._dbg(2, f"[Debug] {dataset}: 엔티티 벡터 index 없음 → 트리플 검색만 text 모드로 폴백")
            search_mode = 'text'

        self._dbg(2, f"[Debug] {dataset} | mode: {search_mode} | hop: {hops}")
        # ★ 실제 검색에 쓰이는 최종 키워드/질의 문자열 (정규화 반영 후)
        self._dbg(2, f"[Debug] {dataset} 실제 검색 키워드(text/fulltext): '{keywords_str}'")
        self._dbg(2, f"[Debug] {dataset} 실제 검색 질의(vector): '{query_text}'")

        if search_mode == 'text':
            rows = (self._text_retrieve_2hop(keywords_str, dataset, cfg, limit, date_from, date_to,
                                             recency_focus=recency_focus, oldest_focus=oldest_focus)
                    if hops == 2
                    else self._text_retrieve_raw(keywords_str, dataset, cfg, limit, date_from, date_to,
                                                 recency_focus=recency_focus, oldest_focus=oldest_focus))

        elif search_mode == 'vector':
            rows = (self._vector_retrieve_2hop(query_text, dataset, cfg, limit, date_from, date_to)
                    if hops == 2
                    else self._vector_retrieve_raw(query_text, dataset, cfg, limit, date_from, date_to))

        else:  # hybrid
            fetch_limit = limit * 3
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_text = executor.submit(
                    self._text_retrieve_2hop if hops == 2 else self._text_retrieve_raw,
                    keywords_str, dataset, cfg, fetch_limit, date_from, date_to,
                    recency_focus=recency_focus, oldest_focus=oldest_focus
                )
                f_vec = executor.submit(
                    self._vector_retrieve_2hop if hops == 2 else self._vector_retrieve_raw,
                    query_text, dataset, cfg, fetch_limit, date_from, date_to
                )
                text_rows = f_text.result()
                vec_rows  = f_vec.result()

            def row_key(r: dict) -> str:
                return f"{r.get('sname')}|{r.get('rel')}|{r.get('oname')}"

            scores: dict[str, float]  = {}
            all_rows: dict[str, dict] = {}

            for rank, r in enumerate(text_rows):
                key = row_key(r)
                scores[key]   = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
                all_rows[key] = r

            for rank, r in enumerate(vec_rows):
                key = row_key(r)
                scores[key]   = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
                all_rows[key] = r

            sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:limit]
            rows = [all_rows[k] for k in sorted_keys]

        # ★ "가장 최근"/"가장 오래된" 의도가 감지되면, 검색 방식(text/vector/hybrid)과
        #   무관하게 최종 결과를 날짜순으로 다시 정렬한다. text 모드는 이미 Cypher
        #   단계에서 date 로 정렬돼 있어 사실상 no-op 이지만, vector/hybrid 는
        #   confidence·유사도·RRF 순으로 뽑힌 후보라서 여기서 최종적으로
        #   사용자 의도(최신순/오래된순)에 맞게 다시 정렬해야 한다.
        if recency_focus and cfg.get('has_date') and rows:
            rows.sort(key=lambda r: r.get('date') or '', reverse=True)
        elif oldest_focus and cfg.get('has_date') and rows:
            rows.sort(key=lambda r: r.get('date') or '9999-12-31')

        # ── 트리플 컨텍스트 ─────────────────────────────────────────────────────
        triples_ctx = self._format_rows(rows, cfg['return_fields'], date_as_week=cfg.get('date_as_week', False))

        # ── 문서 컨텍스트 (A + B) ───────────────────────────────────────────────
        # 문서 doc_id 후보 = 트리플에서 나온 출처 doc_id (A)
        #                  ∪ 문서 벡터 검색으로 찾은 관련 문서 doc_id (B, vector/hybrid 모드)
        ids_a = self._collect_doc_ids(rows)                              # A: 트리플 출처 문서
        # ★ B/C 채널은 requested_mode(트리플용 폴백 이전 값) 기준으로 게이팅한다.
        #   그래야 OKR처럼 엔티티 벡터 인덱스가 없어 search_mode 가 'text'로
        #   폴백된 경우에도, 문서 자체의 벡터 검색(B)은 정상적으로 동작한다.
        ids_b = set(self._doc_vector_retrieve(query_text, cfg, date_from=date_from, date_to=date_to, yw_from=year_week_from, yw_to=year_week_to)) if requested_mode in ('vector', 'hybrid') else set()   # B
        ids_c = set(self._doc_fulltext_retrieve(keywords_str, cfg, date_from=date_from, date_to=date_to, yw_from=year_week_from, yw_to=year_week_to)) if requested_mode in ('text', 'hybrid') else set()  # C
        # ★ D(저자) 채널은 author_names 가 명시적으로 있을 때만 실행한다. 예전엔
        #   일반 keywords_str 전체로 author 필드를 CONTAINS 검색했는데, "MIM",
        #   "capacitor" 같은 일반 기술 용어가 저자명에 우연히 부분 일치해 관련 없는
        #   문서가 대량으로 섞여 들어가는 문제(컨텍스트 폭주 → 응답 지연)가 있었다.
        ids_d = set(self._doc_author_retrieve(author_names, cfg, date_from=date_from, date_to=date_to, yw_from=year_week_from, yw_to=year_week_to)) if author_names else set()  # D: 저자명 직접 검색 (모든 모드)
        ids_e = set(self._doc_week_retrieve(year_week, cfg, yw_from=year_week_from, yw_to=year_week_to))  # E: 주차 정확/범위 매칭 (모든 모드)
        ids_f = set(self._doc_daterange_retrieve(cfg, date_from=date_from, date_to=date_to))  # F: 날짜 범위 열거 (date_as_week 아닌 데이터셋, 모든 모드)
        doc_ids = ids_a | ids_b | ids_c | ids_d | ids_e | ids_f
        self._dbg(1, f"[Debug] {dataset} 문서 doc_id: A(트리플)={len(ids_a)} "
              f"B(벡터)={len(ids_b)} C(키워드)={len(ids_c)} D(저자)={len(ids_d)} "
              f"E(주차)={len(ids_e)} F(날짜범위)={len(ids_f)} → 중복 제거 후 총 {len(doc_ids)}건")

        docs_ctx = self._fetch_documents(doc_ids, cfg, recency_focus=recency_focus,
                                          oldest_focus=oldest_focus,
                                          full_content=full_content, prefer_normalized=prefer_normalized)
        if doc_ids and not docs_ctx:
            self._dbg(0, f"[Debug] {dataset} 경고: doc_id {len(doc_ids)}개인데 메타 노드 조회 결과 0개 "
                  f"(doc_id 불일치 또는 doc_label/속성 확인 필요). 예시 id: {list(doc_ids)[:3]}")

        # ── 트리플/문서 컨텍스트 결합 ────────────────────────────────────────────
        # 둘 다 비면 _NO_RESULT. 하나라도 있으면 해당 섹션만 이어붙인다.
        parts = []
        if triples_ctx != _NO_RESULT:
            parts.append("[지식 트리플]\n" + triples_ctx)
        if docs_ctx:
            parts.append(docs_ctx)

        if not parts:
            return _NO_RESULT
        return "\n\n".join(parts)

    def answer_stream(self, query: str, dataset: str | list[str] | None = None, mode: str = None):
        """
        답변을 스트리밍으로 생성하는 제너레이터.

        UI 에 진행 단계를 표시하기 위해, 두 종류의 이벤트를 dict 형태로 내보낸다:
          {'type': 'status',  'text': '...'}  → 진행 상태 (검색 중/답변 생성 중 등)
          {'type': 'content', 'text': '...'}  → 실제 답변 텍스트 청크
        UI(chat_page.py)는 type 을 보고 상태줄을 갱신하거나 답변을 이어붙인다.
        """
        self._pending_nodes = set()
        self.last_retrieved_nodes = []

        self._dbg(1, f"\n[Debug] ========================================")
        self._dbg(1, f"[Debug] 질문: {query}")
        self._dbg(1, f"[Debug] 데이터셋: {dataset}")

        # [단계 1] 질문 분석 (키워드/날짜 추출)
        yield {'type': 'status', 'text': '🔍 질문 분석 중…'}

        _t_extract_start = time.monotonic()
        extracted          = self._extract_keywords(query)
        _t_extract = time.monotonic() - _t_extract_start
        extracted_keywords = extracted['keywords']
        author_names       = extracted['author_names']
        date_from          = extracted['date_from']
        date_to            = extracted['date_to']
        recency_focus      = extracted['recency_focus']
        oldest_focus       = extracted['oldest_focus']
        year_week          = extracted['year_week']
        year_week_from     = extracted['year_week_from']
        year_week_to       = extracted['year_week_to']
        target_datasets    = extracted['target_datasets']
        full_content       = extracted['full_content']
        prefer_normalized  = extracted['prefer_normalized']

        # ★ 질문 분석(키워드/날짜/저자 추출) 자체가 실패했으면(LLM API 오류 등),
        #   원본 질문 전체를 키워드로 그냥 쓰는 저품질 폴백으로 넘어간다 — 날짜/저자
        #   필터, 데이터셋 자동 선택이 전혀 안 되므로 검색 품질이 떨어진다. 사용자가
        #   이유 모르게 "검색이 이상하다"고 느끼지 않도록 화면에 짧게 알려준다.
        if extracted.get('extraction_failed'):
            yield {'type': 'status', 'text': '⚠️ 질문 분석 중 일시적 오류 — 원본 질문으로 검색합니다'}

        self._dbg(1, f"[Debug] 추출된 키워드: {extracted_keywords}")
        # ★ 코드→물질명 변환은 각 데이터셋 검색 단계(retrieve) 안에서 실제로 일어나고
        #   그 상세 로그는 레벨 2 에서만 보이는데, "코드로 검색해도 물질명으로 정상
        #   변환되는지"를 빠르게 확인할 수 있도록 레벨 1 에서도 변환 결과를 보여준다.
        _norm_kw_preview = _normalize_query(extracted_keywords)
        if _norm_kw_preview != extracted_keywords:
            self._dbg(1, f"[Debug] 코드→물질명 변환: {extracted_keywords} → {_norm_kw_preview}")
        self._dbg(1, f"[Debug] 저자명(D채널 전용): {author_names or '(없음)'}")
        if full_content:
            self._dbg(1, "[Debug] 전체 내용 요청 감지 → 문서 본문 잘라내지 않음")
        if prefer_normalized:
            self._dbg(1, "[Debug] 물질명 변환본 우선 요청 감지 → content_norm 우선 사용")
        week_disp = (f"{year_week_from}~{year_week_to}"
                     if (year_week_from or year_week_to) else year_week)
        self._dbg(1, f"[Debug] 날짜 범위: {date_from} ~ {date_to} | 최신순: {recency_focus} | "
              f"가장오래된순: {oldest_focus} | 주차: {week_disp} | 선택된 데이터셋: {target_datasets}")

        # ★ "안녕" 같은 인사말/잡담은 키워드 추출 결과가 비어 있다 — 이런 경우 검색
        #   자체가 무의미하다(빈 텍스트로 벡터 임베딩을 만들면 임의의 최근접 이웃이
        #   뽑혀 나오는데, 이게 실제로는 아무 의미 없는 결과라 시간만 낭비된다).
        #   키워드가 하나도 없으면 검색을 통째로 건너뛴다.
        skip_search = not extracted_keywords.strip()
        if skip_search:
            self._dbg(1, "[Debug] 추출된 키워드가 없어 검색을 건너뜁니다 (일반 대화로 처리)")
            search_targets = []
            results = {}
            _t_retrieve = 0.0
        else:
            # ★ dataset 은 "All"(또는 None), 단일 데이터셋명 문자열, 또는 여러 개를
            #   동시에 고른 경우 리스트/집합일 수 있다(사이드바 다중 선택 지원).
            if isinstance(dataset, (list, tuple, set)):
                selected = [d for d in dataset if d in DATASETS]
                # 유효한 선택이 있으면 그것만, 없으면 안전하게 LLM 자동 선택으로 폴백
                search_targets = selected if selected else target_datasets
            elif dataset and dataset != 'All' and dataset in DATASETS:
                # 사용자가 특정 탭 하나만 명시적으로 선택한 경우 — LLM 판단과 무관하게 그 탭만 검색
                search_targets = [dataset]
            else:
                # "전체" 탭일 때만 LLM이 판단한 관련 데이터셋으로 검색 범위를 좁힌다.
                # (판단이 애매하면 target_datasets 자체가 전체 목록으로 안전하게 폴백됨)
                search_targets = target_datasets

            # [단계 2] 지식 그래프 검색 + N-hop 확장
            #   대상 데이터셋들의 hop 수를 모아 표시 (보통 2-hop). retrieve() 내부에서
            #   text 검색 시 실제 N-hop 확장이 수행된다.
            hop_set = {(DATASETS[ds].get('search_hops') or DEFAULT_SEARCH_HOPS)
                       for ds in search_targets}
            hop_label = f"{max(hop_set)}-hop " if hop_set else ""
            yield {'type': 'status', 'text': f'📚 지식 그래프 검색 중… ({hop_label}확장)'}

            _t_retrieve_start = time.monotonic()
            with ThreadPoolExecutor(max_workers=len(search_targets)) as executor:
                futures = {
                    ds: executor.submit(
                        self.retrieve,
                        extracted_keywords, query, ds, mode,
                        DEFAULT_SEARCH_LIMIT, date_from, date_to, recency_focus, oldest_focus,
                        year_week, year_week_from, year_week_to, author_names, full_content,
                        prefer_normalized
                    )
                    for ds in search_targets
                }
                results = {ds: f.result() for ds, f in futures.items()}
            _t_retrieve = time.monotonic() - _t_retrieve_start

        self.last_retrieved_nodes = list(self._pending_nodes)
        self._dbg(1, f"[Debug] 검색된 노드 수: {len(self.last_retrieved_nodes)}")
        self._dbg(1, f"[Debug] ⏱ 검색 소요 시간: {_t_retrieve:.2f}초 "
                     f"(데이터셋 {len(search_targets)}개 병렬, 키워드 추출 {_t_extract:.2f}초 별도)")

        sections        = []
        active_datasets = []

        # ReportsDB → Confluence → PapersDB 고정 순서로 정렬. _OUTPUT_ORDER 에
        # 없는(향후 추가될) 데이터셋은 뒤에 원래 순서대로 붙인다.
        ordered_ds = sorted(
            results.keys(),
            key=lambda d: _OUTPUT_ORDER.index(d) if d in _OUTPUT_ORDER else len(_OUTPUT_ORDER)
        )
        for ds in ordered_ds:
            context = results[ds]
            desc = DATASETS[ds]['description']
            # ★ 데이터셋별 전체 컨텍스트 원문(트리플+문서 본문 전부)은 분량이 매우 커서
            #   가장 터미널을 어지럽히는 항목 중 하나 — level 3(상세)에서만 출력.
            self._dbg(3, f"[Debug] {ds} 검색 결과:\n{context}\n")
            if context != _NO_RESULT:
                sections.append(f"=== {ds} ({desc}) ===\n{context}")
                active_datasets.append(ds)
                self._dbg(1, f"[Debug] {ds} 컨텍스트 길이: {len(context):,}자 → LLM 프롬프트에 포함됨")
            else:
                self._dbg(1, f"[Debug] {ds} 컨텍스트 없음(_NO_RESULT) → LLM 프롬프트에서 제외됨")

        combined_context = "\n\n".join(sections) if sections else _NO_RESULT

        if active_datasets:
            prompt_sections = []
            for ds in active_datasets:
                cfg = DATASETS[ds]
                prompt_sections.append(
                    f"[{ds}] — {cfg['description']}\n"
                    f"- 노드 타입: {cfg['node_types']}"
                )
            dataset_info = "\n\n".join(prompt_sections)
        else:
            dataset_info = "검색된 데이터셋 없음"

        # ★ "전체/전문을 보여줘" 요청이면, 컨텍스트의 문서 본문(이미 안 잘림)을
        #   LLM 이 또 요약/축약하지 말고 있는 그대로 전부 옮기도록 명시한다.
        #   (검색 단계에서 DOC_BODY_MAXLEN 절단을 이미 껐어도, LLM 이 답변을 짧게
        #   쓰려고 스스로 요약해버리면 "생략" 문제가 똑같이 재발할 수 있다.)
        full_content_rule = ""
        if full_content:
            full_content_rule = """

[전체 내용 요청 — 반드시 지킬 것]
사용자가 문서 내용을 요약하지 말고 전체/전문을 그대로 보여달라고 요청했습니다.
컨텍스트의 "[관련 문서 원문]" 본문을 절대 요약하거나 임의로 줄이지 말고, 있는
그대로 전부 옮겨서 답하세요. "…(생략)", "요약하면" 같은 표현으로 내용을 줄이지
마세요."""

        # ★ 여러 데이터셋이 검색된 경우, 답변에서 데이터셋별로 소제목을 강제해
        #   LLM 이 한쪽(주로 첫 번째)만 요약하고 나머지를 빠뜨리는 문제를 막는다.
        multi_dataset_rule = ""
        if len(active_datasets) > 1:
            heading_list = "\n".join(f"## {ds}" for ds in active_datasets)
            ds_names = ", ".join(active_datasets)
            multi_dataset_rule = f"""

[여러 데이터셋 답변 형식 — 반드시 지킬 것]
이번 검색은 {len(active_datasets)}개 데이터셋({ds_names})에서 결과가 나왔습니다.
답변은 반드시 아래처럼 데이터셋마다 별도의 마크다운 소제목(##)으로 나누고,
각 소제목 아래에 그 데이터셋의 내용을 정리하세요. 한 데이터셋도 빠뜨리지 마세요.
{heading_list}
- 표로 요약하라는 요청이면, 각 소제목 아래에 그 데이터셋의 표를 각각 만드세요.
- 어떤 데이터셋의 컨텍스트에 문서가 있으면, 그 소제목을 절대 생략하지 마세요.
  (해당 섹션에 결과가 실제로 없을 때만 "관련 내용 없음"이라고 적으세요.)"""

        date_info = ""
        if date_from or date_to:
            date_info = (f"\n검색 적용 날짜 범위: "
                         f"{date_from or '제한없음'} ~ {date_to or '제한없음'}")
        if recency_focus:
            date_info += ("\n★ 아래 지식 그래프 컨텍스트의 트리플/문서는 날짜 내림차순"
                          "(최신이 맨 위)으로 정렬되어 있습니다. 사용자가 '가장 최근'을 물었다면"
                          " 맨 위(가장 먼저 나오는) 항목을 기준으로 답하세요.")

        # ★ 답변에서는 가능하면 물질명 대신 사내 코드명을 쓰도록 한다 (사용자 요청).
        #   검색 컨텍스트(트리플/문서)는 물질명(content_norm) 기준으로 저장돼 있어서,
        #   질문에 코드가 없어도 컨텍스트 안의 물질명 각각에 대응하는 코드가 있으면
        #   그걸 알려줘야 LLM 이 답변을 코드로 바꿔 쓸 수 있다. 그래서 "질문에 등장한
        #   코드"뿐 아니라 "컨텍스트에 실제로 등장하는 물질명 중 CODE_MAP 에 있는 것"도
        #   함께 찾아서 매핑에 포함시킨다.
        code_map_found = _detect_codes(query)
        if _NORMALIZE_AVAILABLE and _CODE_MAP:
            for _code, _material in _CODE_MAP.items():
                if _material and _material in combined_context and _code not in code_map_found:
                    code_map_found[_code] = _material
        code_info = ""
        if code_map_found:
            mapping_lines = "\n".join(
                f"- '{code}' ↔ '{material}'"
                for code, material in code_map_found.items()
            )
            example_code, example_material = next(iter(code_map_found.items()))
            code_info = f"""

[사내 코드명 ↔ 실제 물질명 매핑]
검색 컨텍스트는 물질명 기준으로 되어 있지만, 사내에서는 아래처럼 코드명으로 부르는
경우가 많습니다. 코드명과 물질명이 같은 대상을 가리킨다는 것을 확실히 인지하세요.
{mapping_lines}

답변 규칙 (코드명 관련 — 매우 중요):
- ★ 답변 본문에서는 물질명 대신 위 매핑의 사내 코드명만 사용하세요.
  예: 컨텍스트에 '{example_material}'가 나와도 답변에는 '{example_code}'로만 쓰세요.
- ★ "{example_code}({example_material})"처럼 물질명을 괄호로 함께 표기하지 마세요.
  처음 언급할 때도, 이후에도 코드명만 쓰세요 (물질명 병기 금지).
- 위 매핑에 없는 물질은 코드가 없으므로 그대로 물질명을 쓰세요."""

        # ★ 디버그: LLM 프롬프트에 실제로 들어가는 "코드↔물질명 매핑" 섹션만 따로 출력.
        #   이 섹션이 비어 있으면(아래 (없음)) LLM 은 코드와 물질명을 연결하지 못한다.
        self._dbg(2, "[Debug] ===== 코드↔물질명 매핑 (LLM 프롬프트에 삽입될 내용) =====")
        self._dbg(2, f"  모듈 로드 여부: {_NORMALIZE_AVAILABLE} "
              f"(CODE_MAP 항목 수: {len(_CODE_MAP) if _CODE_MAP else 0})")
        self._dbg(2, f"  원본 질문: '{query}'")
        self._dbg(2, f"  감지된 매핑: {code_map_found if code_map_found else '(없음 — 프롬프트에 매핑 섹션 미삽입)'}")
        self._dbg(2, "[Debug] ===========================================================")

        system_prompt = f"""당신은 DRAM MIM 커패시터 소재 연구 전문가입니다.
다음 지식 그래프 컨텍스트가 제공됩니다.

{dataset_info}{date_info}{code_info}{multi_dataset_rule}{full_content_rule}

답변 규칙:
- 제공된 컨텍스트와 이전 대화 내용을 적극적으로 활용하여 답하세요.
- 컨텍스트에 저자, 제목, 저널, 출처, 날짜 등의 메타데이터가 있으면 반드시 활용하세요.
- 저자를 묻는 경우 컨텍스트의 "저자" 값을 답하세요.
- 게재 저널을 묻는 경우 컨텍스트의 "저널" 값을 답하세요.
- 링크, 출처, URL, DOI를 묻는 경우 컨텍스트의 "출처" 값을 답하세요.
- ★★ 데이터셋(ReportsDB/PapersDB/Confluence) 구분 없이, 질문이 링크를 요청했는지와
  무관하게 답변에서 언급하는 각 문서/논문마다 컨텍스트에 "출처" 값이 있으면 항상
  함께 표기하세요. 여러 문서를 나열/표로 답할 때도 각 행/항목에 그 문서의 링크를
  빠짐없이 포함하세요. 다만 컨텍스트에 "출처" 값이 없는 문서는 링크를 지어내지
  말고 그냥 생략하세요.
- ★★ 저자/담당자(연구원) 이름도 마찬가지로, 컨텍스트에 "저자" 값이 있으면 질문이
  묻지 않았어도 문서를 언급할 때 함께 표기하세요(예: "OOO 보고서(작성자: 이창수)").
  값이 없으면 지어내지 말고 생략하세요.
- ★ 링크/URL/DOI 는 절대로 지어내거나 추측하지 마세요. 오직 컨텍스트에 실제로 적힌
  "출처" 값을 글자 그대로 복사해서 답하세요. 알고 있는 것 같아도 학습 데이터의
  기억에 의존해 URL/DOI를 생성하지 마세요 — 틀린 링크일 수 있습니다.
- 질문한 대상(논문/문서)에 대해 컨텍스트에 "출처" 값이 없으면, 절대로 링크를
  만들어내지 말고 "제공된 정보에는 해당 문서의 출처 링크가 없습니다" 라고 답하세요.
- 여러 문서가 컨텍스트에 있을 경우, 어떤 문서의 링크인지 제목/doc_id 로 정확히
  확인한 뒤 그 문서의 "출처" 값만 답하세요. 다른 문서의 링크를 섞어 쓰지 마세요.
- 논문 제목을 묻는 경우 컨텍스트의 "제목" 값을 답하세요.
- 컨텍스트의 필드 라벨(근거, 출처, 제목, 저자, 저널, 날짜 등)은 답변 문장에 그대로
  베껴 쓰지 말고, 자연스러운 한국어 문장으로 풀어서 답하세요.
- ReportsDB/Confluence(내부 문서)는 정확한 날짜 대신 "2026년 W02"처럼 주차로
  컨텍스트에 표시됩니다. 이 값을 그대로 "2026년 2주차" 같은 자연스러운 표현으로
  답하세요 (내부 문서에 대해 실제 날짜(YYYY-MM-DD)를 추측해서 답하지 마세요).
  반면 PapersDB(논문)는 정확한 날짜 그대로 컨텍스트에 있으니 날짜로 답하세요.
- 이전 대화에서 언급된 메타데이터도 참고하세요.
- ★ 컨텍스트에 여러 데이터셋(예: ReportsDB, Confluence)의 "=== 데이터셋명 ===" 섹션이
  함께 있으면, 특정 데이터셋만 쓰지 말고 모든 섹션의 내용을 빠짐없이 반영해 답하세요.
- ★★ 열거형 질문("기간 안의 보고문서를 표로 요약", "그 사람의 연구내용을 정리해줘"처럼
  특정 기간/인물/주제에 해당하는 것을 "전부" 모아달라는 질문)은, 컨텍스트의
  "[관련 문서 원문]" 섹션에 있는 문서를 하나도 빠뜨리지 말고 전부 표/목록에
  포함하세요. 컨텍스트에 문서가 10건 넘게 있어도 답변을 짧게 하려고 일부만
  골라 요약하지 말고, 있는 문서 수만큼 전부 행/항목으로 나열하세요. 문서가 너무
  많아 전부 나열하기 부담스러워도 절대 임의로 생략하지 말고, 모든 문서를
  포함한 뒤 필요하면 표 아래에 짧은 총평만 덧붙이세요.
- 컨텍스트와 이전 대화 모두에 없는 내용만 모른다고 답하세요.
- 답변은 한국어로 작성하세요.
- 수식/화학식은 LaTeX 문법으로 쓸 수 있습니다 (화면에서 MathJax 로 렌더링됨).
  인라인 수식은 $...$, 블록 수식은 $$...$$ 로 감싸세요.
  예: "$\\text{{ZrO}}_2$ 두께", "$J(\\text{{TiO}}_2) \\gg J(\\text{{ZrO}}_2)$".
  간단한 화학식은 그냥 "ZrO2" 처럼 일반 텍스트로 써도 됩니다.

링크 작성 규칙 (중요):
- URL/링크는 마크다운 굵게(**) 표시를 절대 사용하지 마세요. "**https://...**" 같은
  형태는 링크 인식이 깨져 클릭할 수 없게 됩니다.
- 링크 앞뒤에는 반드시 공백이나 줄바꿈을 두세요. 단어나 문장부호를 링크에 바로
  붙여 쓰지 마세요 (예: "자세한내용은https://example.com참고" ❌).
- 짧은 링크는 그냥 URL 그대로 쓰거나 마크다운 링크 형식 [설명](URL) 으로 쓰세요.
- ★ 링크(URL)가 길어서 답변 문장이나 표 안에서 지저분해 보이면, 마크다운 링크
  대신 HTML 앵커 태그로 줄여서 표기하세요: `<a href="실제 긴 URL 그대로">link</a>`.
  href 안의 URL은 컨텍스트의 "출처" 값을 한 글자도 바꾸지 말고 그대로 넣고,
  화면에 보이는 텍스트만 "link"처럼 짧게 쓰세요. 표의 "출처/링크" 칸에는
  특히 이 형태를 우선 사용하세요(긴 URL이 그대로 들어가면 표가 깨져 보입니다).

표 작성 규칙 (중요 — 이 형식을 안 지키면 표 대신 깨진 텍스트로 보입니다):
- 모든 행(헤더/구분선/데이터 행)은 반드시 맨 앞과 맨 뒤에 "|" 를 붙이세요.
  예: "| 항목 | 값 |" (❌ "항목 | 값 |" 처럼 앞의 "|" 를 빼지 마세요)
- 헤더 행과 그 바로 아래 구분선 행("| --- | --- |")의 열(칸) 개수는 반드시
  정확히 같아야 합니다. 열 개수가 하나라도 다르면 표 전체가 표로 인식되지 않고
  일반 텍스트로 깨져서 보입니다.
- 빈 칸이 있어도 칸 자체는 반드시 유지하세요 (예: "| week26 |  |  |" 처럼
  칸을 비우더라도 "|"는 다 채우세요. 칸을 아예 생략하면 안 됩니다)."""

        user_message_content = f"""[추출된 키워드]
{extracted_keywords}

[지식 그래프 컨텍스트]
{combined_context}

[원본 질문]
{query}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message_content})

        # ★ 디버그: LLM 에 실제로 전달되는 프롬프트 전문 — 분량이 가장 크므로
        #   level 3(상세)에서만 출력한다.
        self._dbg(3, "[Debug] ===== LLM 시스템 프롬프트 =====")
        self._dbg(3, system_prompt)
        self._dbg(3, "[Debug] ===== LLM 사용자 메시지 =====")
        self._dbg(3, user_message_content)
        self._dbg(3, "[Debug] ================================")

        # [단계 3] LLM 답변 생성 (여기서부터 content 청크가 스트리밍됨)
        answer_model_name = self.answer_llm or LLM  # None 이면 llm_util 의 .env 기본값(LLM)
        self._dbg(1, f"[Debug] 답변 생성 모델: {answer_model_name}")
        yield {'type': 'status', 'text': '✍️ 답변 생성 중…'}

        _t_answer_start = time.monotonic()
        _t_first_chunk = None
        full_result = []
        for chunk in ask_llm_stream_iter_messages(
            messages=messages,
            temperature=0.05,
            reasoning_effort="medium",
            llm=self.answer_llm,
        ):
            if _t_first_chunk is None:
                _t_first_chunk = time.monotonic() - _t_answer_start
            full_result.append(chunk)
            yield {'type': 'content', 'text': chunk}
        _t_answer = time.monotonic() - _t_answer_start

        _t_total = time.monotonic() - _t_extract_start
        # ★ _t_first_chunk 는 첫 청크가 도착해야만 채워지는데, LLM 호출 자체가
        #   실패해 청크가 하나도 안 온 경우(예: 게이트웨이 오류) None 으로 남아
        #   ":.2f" 포맷팅이 TypeError 로 죽는 문제가 있었다. 방어적으로 처리한다.
        first_chunk_disp = f"{_t_first_chunk:.2f}초" if _t_first_chunk is not None else "응답 없음"
        self._dbg(1, f"[Debug] ⏱ 답변 생성 소요 시간: {_t_answer:.2f}초 "
                     f"(첫 응답까지 {first_chunk_disp}, 모델: {answer_model_name})")
        self._dbg(1, f"[Debug] ⏱ 전체 소요 시간: {_t_total:.2f}초 "
                     f"(질문 분석 {_t_extract:.2f}초 + 검색 {_t_retrieve:.2f}초 + 답변 생성 {_t_answer:.2f}초)")

        result = "".join(full_result)
        if result:
            self.history.append({"role": "user",      "content": query})
            self.history.append({"role": "assistant",  "content": result})

            max_messages = MAX_HISTORY_TURNS * 2
            if len(self.history) > max_messages:
                self.history = self.history[-max_messages:]
                self._dbg(2, f"[Debug] 히스토리 트리밍: 최근 {MAX_HISTORY_TURNS}턴 유지")
