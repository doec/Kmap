import sys
import re
import json
from pathlib import Path
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, date

import logging
logging.getLogger("neo4j").setLevel(logging.ERROR)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from llm_util import ask_llm, ask_llm_messages, ask_llm_stream_iter_messages
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
#   (research_item + summary + content 를 결합 임베딩한 Confl_doc 노드만 존재),
#   DATASETS 설정에서 엔티티 관련 필드(vector_index, hop2_relations 등)는
#   비워두고 문서 채널(doc_vector_index/doc_fulltext_index/doc_label)만 채운다.
CONFLUENCE_DATASET = os.getenv('NEO4J_CONFLUENCE_DATASET', 'Confluence')
CONFLUENCE_DESC    = os.getenv('NEO4J_CONFLUENCE_DESC',    'Confluence 주간보고/문서')

DEFAULT_SEARCH_LIMIT = 50
DEFAULT_SEARCH_HOPS  = 2
DEFAULT_SEARCH_MODE  = None   # None / 'text' / 'vector' / 'hybrid'

RRF_K             = 60
MAX_HISTORY_TURNS = 5

# 문서 섹션 관련 상수
DOC_SEARCH_LIMIT  = 5     # 문서 단위 벡터/키워드 검색으로 가져올 문서 개수 (B/C 기능)
DOC_SCORE_MIN     = 0.6   # 문서 벡터 검색 최소 유사도 컷 (초록/전문처럼 긴 텍스트끼리 비교)
DOC_BODY_MAXLEN   = 700   # 답변 컨텍스트에 넣을 문서 본문(abstract/content) 최대 길이
# ★ 저자 검색(D 기능)은 author 필드에 대한 "정확한" 매칭이라 B/C(유사도/관련도 기반
#   검색)보다 훨씬 신뢰도가 높다. "그 사람이 쓴 모든 보고서" 같은 질문은 5건으로
#   자르면 최근 문서를 놓칠 수 있으므로 훨씬 넉넉하게 잡는다.
AUTHOR_SEARCH_LIMIT = 30

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
        # ── 문서(Confl_doc 노드) 관련 설정 ──
        'doc_vector_index': 'confluence_doc_embedding',   # 결합 임베딩(research_item_norm+summary_norm+content_norm)
        'doc_label':        'Confl_doc',                  # 메타 노드 레이블
        # ★ Confluence 는 원본(코드)/정규화(물질명) FULLTEXT 인덱스가 분리되어 있어
        #   리스트로 둘 다 지정 → _doc_fulltext_retrieve 가 두 인덱스를 모두 검색해 합친다.
        'doc_fulltext_index': ['confluence_doc_fulltext', 'confluence_doc_fulltext_norm'],
        'doc_body_field':   'content_norm',           # 본문 속성명 (물질명 정규화본)
        'doc_body_label':   '내용',                    # LLM 컨텍스트에 표기할 본문 레이블
        # ReportsDB 와 마찬가지로 물질 코드가 섞일 수 있으므로 질의 정규화 적용
        'normalize_query':  True,
    },
}

_NO_RESULT = "관련 트리플을 찾지 못했습니다."

# ★ 스키마 변경 대응: 구조(출처) 관계 타입.
# 새 스키마는 엔티티와 메타 노드를 (entity)-[:FROM_PAPER]->(:Paper) /
# (entity)-[:FROM_DOC]->(:Report) 로 연결한다. 그런데 Paper/Report 메타 노드도
# 데이터셋 레이블(:PapersDB / :ReportsDB)을 공유하므로,
# MATCH (s:PapersDB)-[r]->(o:PapersDB) 같은 패턴이 이 구조 관계까지 잡아버린다.
# 이들은 "의미 트리플"이 아니라 출처 연결이므로, 검색 시 관계 타입으로 제외한다.
_STRUCTURAL_RELS = ['FROM_PAPER', 'FROM_DOC']


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

    def close(self):
        self.driver.close()

    def clear_history(self):
        self.history = []
        print("[Debug] 대화 히스토리 초기화")

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
  "target_datasets": ["관련된 데이터셋명", ...],
  "date_from": "YYYY-MM-DD 또는 null",
  "date_to": "YYYY-MM-DD 또는 null",
  "recency_focus": true 또는 false,
  "year_week": "YYYY-WNN 또는 null"
}}

날짜 변환 규칙:
- "N년 이후", "N년 이상", "N년부터" → date_from: "N-01-01", date_to: null
- "N년 이전", "N년 이하", "N년까지" → date_from: null, date_to: "N-12-31"
- "N년~M년", "N년에서 M년" → date_from: "N-01-01", date_to: "M-12-31"
- "최근 N개월" → 오늘 기준 N개월 전 날짜 계산
- "최근 N년" → 오늘 기준 N년 전 날짜 계산
- "올해" → date_from: "{today.year}-01-01", date_to: "{today_str}"
- 날짜 언급 없음 → date_from: null, date_to: null

target_datasets 판단 규칙 (매우 중요):
- 질문 내용을 보고 위 "사용 가능한 데이터셋" 중 실제로 검색이 필요한 데이터셋만
  골라 배열로 넣으세요. (예: 논문/연구자 관련 질문 → PapersDB만, 사내 보고서/주차
  관련 질문 → ReportsDB나 Confluence만)
- 여러 데이터셋에 걸칠 수 있는 질문이면 관련된 것을 모두 포함하세요.
- ★ 확신이 없거나 질문이 모호하면, 좁히지 말고 관련 있을 수 있는 데이터셋을
  전부 포함하세요 (좁혀서 놓치는 것보다 넓게 잡는 게 안전합니다).
- 이전 대화의 맥락(예: 이전에 특정 데이터셋 관련 대상이 언급됨)도 참고하세요.

year_week 판단 규칙 (내부 문서 ReportsDB/Confluence 는 주차로 관리됨 — 매우 중요):
- "2026년 26주차", "2026-W26", "26주차"(올해로 간주) 처럼 특정 연도+주차가
  언급되면 → year_week: "YYYY-WNN" 형식으로 추출 (예: "2026-W26", 주차는 2자리 0-패딩).
- 연도 없이 "26주차"만 언급되면 오늘 연도({today.year})를 사용하세요.
- 주차 언급이 없으면 → year_week: null

recency_focus 판단 규칙 (매우 중요):
- "가장 최근", "제일 최근", "최신", "최근 결과", "요즘" 처럼 구체적인 기간(개월/년) 없이
  "최근/최신"만 언급되어 시간 순으로 정렬해서 보여달라는 의도가 있으면 → true
- 위와 같이 recency_focus 가 true 인 경우, 검색 결과를 confidence(신뢰도) 순이 아니라
  date(날짜) 내림차순으로 정렬해야 하므로 반드시 true 로 표시하세요.
- 구체적 기간이 명시되어 date_from/date_to 가 채워진 경우에도, 그 기간 안에서 역시
  최신순 정렬이 자연스러우므로 recency_focus: true 로 표시하세요.
- 날짜/최근 관련 언급이 전혀 없으면 → false

키워드 추출 규칙:
1. 핵심 명사(엔티티) 위주로 추출
2. 한국어 키워드는 반드시 영어 번역도 함께 추출
3. 관련 동의어/유사어도 포함
4. 날짜 관련 표현은 키워드에서 제외
5. 저자/작성자 이름이 언급되면 반드시 키워드에 그대로 포함 (예: "Tao Li가 쓴 논문" → "Tao Li")
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
                        print(f"[Debug] 주차→날짜 범위 계산 실패: {e}")

            # ★ target_datasets 검증: 유효한 데이터셋명만 남기고, 결과가 비었거나
            #   파싱이 이상하면 안전하게 "전체 데이터셋"으로 폴백한다 — 잘못 좁혀서
            #   관련 결과를 놓치는 것보다, 넓게 검색하는 게 훨씬 안전하기 때문.
            target_raw = parsed.get('target_datasets')
            target_datasets = None
            if isinstance(target_raw, list):
                target_datasets = [ds for ds in target_raw if ds in valid_dataset_keys]
            if not target_datasets:
                target_datasets = valid_dataset_keys

            return {
                'keywords':        parsed.get('keywords', query),
                'date_from':       date_from,
                'date_to':         date_to,
                'recency_focus':   bool(parsed.get('recency_focus', False)),
                'year_week':       year_week,
                'target_datasets': target_datasets,
            }
        except Exception as e:
            print(f"[Debug] 키워드 파싱 오류: {e} / 원본: {result}")
            return {'keywords': query, 'date_from': None, 'date_to': None,
                    'recency_focus': False, 'year_week': None,
                    'target_datasets': list(DATASETS.keys())}

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
                           recency_focus: bool = False) -> list[dict]:
        """
        allowed_rels 가 주어지면 그 관계 타입으로만 결과를 제한한다.
        (1차 검색에서는 None 으로 호출해 관계 타입 무관하게 키워드 매칭하고,
         2-hop 확장에서는 cfg['hop2_relations'] 를 넘겨 인과·성능 계열로만 제한한다.)

        recency_focus 가 True 이고 이 데이터셋이 날짜(date)를 지원하면, 정렬 기준을
        cfg 의 기본값(confidence DESC) 대신 date DESC 로 바꾼다.
        ★ 이건 Cypher ORDER BY + LIMIT 단계에서 바로 적용되어야 의미가 있다.
          결과를 다 가져온 뒤 Python 에서 재정렬하면, 애초에 confidence 기준으로
          LIMIT 에 걸려 짤린 "진짜로 최신인데 confidence 가 낮은" 트리플을
          영영 놓치게 된다.
        """
        # ★ 쉼표로만 나눈다 (공백 분리 X). "Lee Changsoo" 처럼 여러 단어로 된 키워드가
        #   "Lee" / "Changsoo" 로 쪼개지면, CONTAINS 매칭이 "Lee"라는 흔한 성씨 하나만
        #   으로도 걸려서 전혀 다른 사람(예: "Lee Jaeho")까지 잘못 매칭되는 문제가 있었다.
        #   (마찬가지로 "La doping" 같은 두 단어 물질명도 "La"/"doping" 각각으로 쪼개지면
        #   너무 광범위하게 매칭되는 문제가 있었음 — 이 문제도 함께 해결됨)
        keywords      = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
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
                    print(f"[Debug] {dataset} 엔티티 벡터 점수 분포: "
                          f"min={min(scores):.3f} max={max(scores):.3f} "
                          f"avg={sum(scores)/len(scores):.3f} (컷={self.entity_score_min}, {len(scores)}건)")
                    # 결과 하나하나의 점수를 다 찍으면 (최대 limit*3=150건) 터미널이
                    # 감당 안 되므로, 튜닝에 필요한 상위 _DEBUG_ROW_PREVIEW 개만 보여준다.
                    for r in rows[:_DEBUG_ROW_PREVIEW]:
                        print(f"  [Debug]   score={r['vec_score']:.3f}  "
                              f"({r.get('sname')}) --[{r.get('rel')}]--> ({r.get('oname')})")
                    if len(rows) > _DEBUG_ROW_PREVIEW:
                        print(f"  [Debug]   ... 외 {len(rows) - _DEBUG_ROW_PREVIEW}건 생략")

                    # ★ 상대 컷오프: BGE-M3 는 무관한 쌍도 baseline 유사도가 높아
                    #   절대값 컷(entity_score_min)만으로는 옥석이 안 걸러진다.
                    #   1등 점수 대비 relative_gap 이상 뒤처지는 결과는 제외한다.
                    top_score = max(scores)
                    before = len(rows)
                    rows = [r for r in rows
                            if r.get('vec_score') is not None
                            and top_score - r['vec_score'] <= self.relative_gap]
                    if len(rows) != before:
                        print(f"[Debug] {dataset} 상대 컷오프 적용(gap≤{self.relative_gap}): "
                              f"{before}건 → {len(rows)}건")
            return rows
        except Exception as e:
            print(f"[Debug] {dataset} vector 검색 실패 → text 결과만 사용: {e}")
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
            print(f"[Debug] {dataset} 벡터 2-hop 확장 실패: {e}")
            hop2_rows = []

        seen: dict = {}
        for r in hop1_rows + hop2_rows:
            key = f"{r.get('sname')}|{r.get('rel')}|{r.get('oname')}"
            seen[key] = r

        print(f"[Debug] {dataset} 벡터 2-hop: 1차 {len(hop1_rows)}개 + "
              f"2차(인과·성능 확장) {len(hop2_rows)}개 → 중복 제거 후 {min(len(seen), limit)}개")
        return list(seen.values())[:limit]

    def _text_retrieve_2hop(self, keywords_str: str, dataset: str, cfg: dict,
                            limit: int, date_from: str, date_to: str,
                            recency_focus: bool = False) -> list[dict]:
        hop_limit = limit * 2

        # 1차 검색: 질문 키워드로 매칭 — 관계 타입 무관 (원 키워드가 직접 맞은 것이므로)
        hop1_rows = self._text_retrieve_raw(
            keywords_str, dataset, cfg, hop_limit, date_from, date_to,
            recency_focus=recency_focus
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
            allowed_rels=hop2_relations, recency_focus=recency_focus
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

        print(f"[Debug] {dataset} 2-hop: 1차 {len(hop1_rows)}개 + "
              f"2차 {len(hop2_rows)}개 → 중복 제거 후 {min(len(results), limit)}개")
        return results[:limit]

    # ── B 기능: 문서 단위 벡터 검색 ────────────────────────────────────────────
    def _doc_vector_retrieve(self, query_text: str, cfg: dict,
                             limit: int = DOC_SEARCH_LIMIT) -> list[str]:
        """
        Paper/Report 메타 노드를 대상으로 벡터 검색을 수행해 관련 문서의
        doc_id 목록을 돌려준다.

        엔티티 벡터 검색(_vector_retrieve_raw)이 "관계(트리플)"를 찾는 것과 달리,
        이건 "문서 자체"(논문 초록 / 보고서 전문)를 질문과의 유사도로 찾는다.
        → "이 주제 관련 논문/보고서 찾아줘" 류의 질의에 강하다.

        papersdb_doc_embedding / reportsdb_doc_embedding 인덱스는 각각
        Paper / Report 노드만 포함하므로 메타 노드 제외 필터가 필요 없다.
        """
        doc_index = cfg.get('doc_vector_index')
        if not doc_index:
            return []

        q_emb = get_embedding(query_text)
        query_str = """
            CALL db.index.vector.queryNodes($index, $limit, $q_emb)
            YIELD node, score
            WHERE score > $score_min
            RETURN node.doc_id AS doc_id, score
            ORDER BY score DESC
        """
        params = {
            'index':     doc_index,
            'limit':     limit,
            'q_emb':     q_emb,
            'score_min': self.doc_score_min,
        }
        try:
            with self.driver.session() as session:
                rows = [dict(r) for r in session.run(query_str, **params)]
            if rows:
                print(f"[Debug] 문서 벡터 검색 결과 (컷={self.doc_score_min}, {len(rows)}건):")
                for r in rows:
                    print(f"  [Debug]   score={r['score']:.3f}  doc_id={r.get('doc_id')}")

                # ★ 상대 컷오프 (엔티티 벡터 검색과 동일한 이유)
                top_score = max(r['score'] for r in rows)
                before = len(rows)
                rows = [r for r in rows if top_score - r['score'] <= self.relative_gap]
                if len(rows) != before:
                    print(f"[Debug] 문서 벡터 상대 컷오프 적용(gap≤{self.relative_gap}): "
                          f"{before}건 → {len(rows)}건")
            return [r['doc_id'] for r in rows if r.get('doc_id')]
        except Exception as e:
            print(f"[Debug] 문서 벡터 검색 실패: {e}")
            return []

    # ── C 기능: 문서 본문 키워드(FULLTEXT) 검색 ─────────────────────────────────
    def _doc_fulltext_retrieve(self, keywords_str: str, cfg: dict,
                               limit: int = DOC_SEARCH_LIMIT) -> list[str]:
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
        if not ft_indexes or not keywords_str:
            return []
        if isinstance(ft_indexes, str):
            ft_indexes = [ft_indexes]

        # ★ 쉼표로만 나눈다 (공백 분리 X) — "Lee Changsoo" 같은 여러 단어 키워드가
        #   쪼개지지 않도록 (자세한 이유는 _text_retrieve_raw 주석 참고).
        # Lucene 질의 문자열 구성:
        # 키워드에 '/'·'-' 등 Lucene 특수문자가 있으면 파싱 오류가 나므로,
        # 각 키워드를 큰따옴표로 감싼 구(phrase)로 만들고 내부 특수문자는 이스케이프한다.
        # 따옴표로 감싼 구들을 공백으로 이으면 Lucene 기본 OR(should) 매칭이 된다.
        keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
        if not keywords:
            return []

        def _escape(kw: str) -> str:
            return kw.replace('\\', '\\\\').replace('"', '\\"')

        lucene_query = " ".join(f'"{_escape(kw)}"' for kw in keywords)

        query_str = """
            CALL db.index.fulltext.queryNodes($index, $q, {limit: $limit})
            YIELD node, score
            RETURN node.doc_id AS doc_id, score
            ORDER BY score DESC
        """
        doc_ids: list[str] = []
        for ft_index in ft_indexes:
            params = {'index': ft_index, 'q': lucene_query, 'limit': limit}
            try:
                with self.driver.session() as session:
                    rows = [dict(r) for r in session.run(query_str, **params)]
                doc_ids.extend(r['doc_id'] for r in rows if r.get('doc_id'))
            except Exception as e:
                print(f"[Debug] 문서 FULLTEXT 검색 실패 ({ft_index}): {e}")
        return doc_ids

    # ── D 기능: 저자명으로 문서 직접 검색 ────────────────────────────────────────
    def _doc_author_retrieve(self, keywords_str: str, cfg: dict,
                             limit: int = AUTHOR_SEARCH_LIMIT) -> list[str]:
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

        # ★ 쉼표로만 나눈다 (공백 분리 X) — "Lee Changsoo" 가 "Lee"/"Changsoo" 로
        #   쪼개지면 "Lee"라는 흔한 성씨 하나만으로도 매칭돼 전혀 다른 사람의
        #   문서까지 걸리는 문제가 있었다 (자세한 이유는 _text_retrieve_raw 주석 참고).
        keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
        if not keywords:
            return []

        query_str = f"""
            MATCH (m:{doc_label})
            WHERE m.author IS NOT NULL
              AND any(kw IN $keywords WHERE toLower(m.author) CONTAINS toLower(kw))
            RETURN m.doc_id AS doc_id
            ORDER BY m.date DESC
            LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                rows = [dict(r) for r in session.run(query_str, keywords=keywords, limit=limit)]
            return [r['doc_id'] for r in rows if r.get('doc_id')]
        except Exception as e:
            print(f"[Debug] 저자 검색 실패: {e}")
            return []

    # ── E 기능: 주차(week)로 문서 정확 매칭 검색 ─────────────────────────────────
    def _doc_week_retrieve(self, year_week: str, cfg: dict) -> list[str]:
        """
        Report/Confl_doc 메타 노드의 year_week 속성을 정확히(exact match) 검색한다.

        "2026년 26주차 보고내용 보여줘" 같은 질문은 "관련도 상위 몇 개"가 아니라
        "그 주차에 해당하는 문서 전부"를 원하는 열거형(enumeration) 질문이다.
        B(벡터)/C(FULLTEXT)는 관련도 기준이라 한 문서가 순위에서 밀려 누락될 수
        있으므로, D(저자)와 마찬가지로 필드를 정확히 매칭하는 전용 채널을 둔다.
        LIMIT 을 두지 않는다 — 특정 주차의 문서 수는 원래 적으므로 전부 반환해도
        컨텍스트 폭주 위험이 낮다.
        """
        doc_label = cfg.get('doc_label')
        if not doc_label or not year_week:
            return []

        # ★ Cypher 문자열 비교는 대소문자를 구분한다. DB에는 소문자 'w'로 저장돼
        #   있는데(예: "2026-w26") 우리 쪽 정규화는 대문자 'W'를 쓰므로,
        #   toLower() 로 양쪽을 맞춰 대소문자와 무관하게 매칭한다.
        query_str = f"""
            MATCH (m:{doc_label})
            WHERE toLower(m.year_week) = toLower($year_week)
            RETURN m.doc_id AS doc_id
            ORDER BY m.date DESC
        """
        try:
            with self.driver.session() as session:
                rows = [dict(r) for r in session.run(query_str, year_week=year_week)]
            return [r['doc_id'] for r in rows if r.get('doc_id')]
        except Exception as e:
            print(f"[Debug] 주차 검색 실패: {e}")
            return []

    # ── A 기능: doc_id 로 메타 노드 원문(abstract/content) 조회 ──────────────────
    def _fetch_documents(self, doc_ids: set[str], cfg: dict,
                         recency_focus: bool = False) -> str:
        """
        doc_id 집합을 받아 Paper/Report 메타 노드에서 제목·본문·출처를 조회하고,
        LLM 컨텍스트에 넣을 "관련 문서" 섹션 문자열로 만든다.

        트리플은 (주어)-[관계]->(목적어) 형태라 근거 문장(evidence) 정도만 담지만,
        여기서 원문(논문 초록 / 보고서 전문)을 붙여 주면 답변 근거가 훨씬 풍부해진다.
        본문은 DOC_BODY_MAXLEN 로 잘라 컨텍스트 폭주를 막는다.

        recency_focus 가 True 면 문서를 날짜 내림차순으로 정렬해 반환한다.
        ("가장 최근 문서 보여줘" 류의 질문에서 최신 문서가 먼저 나오게 함)
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
        # ★ journal 은 Paper 노드 전용, year_week 는 Report/Confl_doc 노드 전용 필드
        #   (다른 데이터셋엔 없으면 null 반환되어 meta_bits 에서 자연히 제외된다).
        order_clause = "ORDER BY m.date DESC" if recency_focus else ""
        query_str = f"""
            MATCH (m:{doc_label})
            WHERE m.doc_id IN $ids
            RETURN m.doc_id     AS doc_id,
                   m.title      AS title,
                   m.author     AS author,
                   m.date       AS date,
                   m.source_url AS source_url,
                   m.journal    AS journal,
                   m.year_week  AS year_week,
                   COALESCE(m.{body_field}, m.content) AS body
            {order_clause}
        """
        try:
            with self.driver.session() as session:
                docs = [dict(r) for r in session.run(query_str, ids=list(doc_ids))]
        except Exception as e:
            print(f"[Debug] 문서 조회 실패: {e}")
            return ""

        if not docs:
            return ""

        lines = ["[관련 문서 원문]"]
        for d in docs:
            body = (d.get('body') or '').strip().replace('\n', ' ')
            if len(body) > DOC_BODY_MAXLEN:
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
            if meta_bits:
                header += f" ({', '.join(meta_bits)})"
            lines.append(header)
            if d.get('source_url'):
                lines.append(f"  출처: {d['source_url']}")
            if body:
                lines.append(f"  {body_label}: {body}")
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
                 year_week: str = None) -> str:
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
                print(f"[Debug] {dataset} 질의 정규화: '{query_text}' → '{norm_query}'")

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
            print(f"[Debug] {dataset}: 엔티티 벡터 index 없음 → 트리플 검색만 text 모드로 폴백")
            search_mode = 'text'

        print(f"[Debug] {dataset} | mode: {search_mode} | hop: {hops}")
        # ★ 실제 검색에 쓰이는 최종 키워드/질의 문자열 (정규화 반영 후)
        print(f"[Debug] {dataset} 실제 검색 키워드(text/fulltext): '{keywords_str}'")
        print(f"[Debug] {dataset} 실제 검색 질의(vector): '{query_text}'")

        if search_mode == 'text':
            rows = (self._text_retrieve_2hop(keywords_str, dataset, cfg, limit, date_from, date_to,
                                             recency_focus=recency_focus)
                    if hops == 2
                    else self._text_retrieve_raw(keywords_str, dataset, cfg, limit, date_from, date_to,
                                                 recency_focus=recency_focus))

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
                    recency_focus=recency_focus
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

        # ★ "가장 최근" 의도가 감지되면, 검색 방식(text/vector/hybrid)과 무관하게
        #   최종 결과를 날짜 내림차순으로 다시 정렬한다. text 모드는 이미 Cypher
        #   단계에서 date 로 정렬돼 있어 사실상 no-op 이지만, vector/hybrid 는
        #   confidence·유사도·RRF 순으로 뽑힌 후보라서 여기서 최종적으로
        #   "최신순으로 보여달라"는 사용자 의도에 맞게 다시 정렬해야 한다.
        if recency_focus and cfg.get('has_date') and rows:
            rows.sort(key=lambda r: r.get('date') or '', reverse=True)

        # ── 트리플 컨텍스트 ─────────────────────────────────────────────────────
        triples_ctx = self._format_rows(rows, cfg['return_fields'], date_as_week=cfg.get('date_as_week', False))

        # ── 문서 컨텍스트 (A + B) ───────────────────────────────────────────────
        # 문서 doc_id 후보 = 트리플에서 나온 출처 doc_id (A)
        #                  ∪ 문서 벡터 검색으로 찾은 관련 문서 doc_id (B, vector/hybrid 모드)
        ids_a = self._collect_doc_ids(rows)                              # A: 트리플 출처 문서
        # ★ B/C 채널은 requested_mode(트리플용 폴백 이전 값) 기준으로 게이팅한다.
        #   그래야 OKR처럼 엔티티 벡터 인덱스가 없어 search_mode 가 'text'로
        #   폴백된 경우에도, 문서 자체의 벡터 검색(B)은 정상적으로 동작한다.
        ids_b = set(self._doc_vector_retrieve(query_text, cfg)) if requested_mode in ('vector', 'hybrid') else set()   # B
        ids_c = set(self._doc_fulltext_retrieve(keywords_str, cfg)) if requested_mode in ('text', 'hybrid') else set()  # C
        ids_d = set(self._doc_author_retrieve(keywords_str, cfg))       # D: 저자명 직접 검색 (모든 모드)
        ids_e = set(self._doc_week_retrieve(year_week, cfg))             # E: 주차 정확 매칭 (모든 모드)
        doc_ids = ids_a | ids_b | ids_c | ids_d | ids_e
        print(f"[Debug] {dataset} 문서 doc_id: A(트리플)={len(ids_a)} "
              f"B(벡터)={len(ids_b)} C(키워드)={len(ids_c)} D(저자)={len(ids_d)} "
              f"E(주차)={len(ids_e)} → 합집합 {len(doc_ids)}")

        docs_ctx = self._fetch_documents(doc_ids, cfg, recency_focus=recency_focus)
        if doc_ids and not docs_ctx:
            print(f"[Debug] {dataset} 경고: doc_id {len(doc_ids)}개인데 메타 노드 조회 결과 0개 "
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

    def answer_stream(self, query: str, dataset: str = None, mode: str = None):
        """
        답변을 스트리밍으로 생성하는 제너레이터.

        UI 에 진행 단계를 표시하기 위해, 두 종류의 이벤트를 dict 형태로 내보낸다:
          {'type': 'status',  'text': '...'}  → 진행 상태 (검색 중/답변 생성 중 등)
          {'type': 'content', 'text': '...'}  → 실제 답변 텍스트 청크
        UI(chat_page.py)는 type 을 보고 상태줄을 갱신하거나 답변을 이어붙인다.
        """
        self._pending_nodes = set()
        self.last_retrieved_nodes = []

        print(f"\n[Debug] ========================================")
        print(f"[Debug] 질문: {query}")
        print(f"[Debug] 데이터셋: {dataset}")

        # [단계 1] 질문 분석 (키워드/날짜 추출)
        yield {'type': 'status', 'text': '🔍 질문 분석 중…'}

        extracted          = self._extract_keywords(query)
        extracted_keywords = extracted['keywords']
        date_from          = extracted['date_from']
        date_to            = extracted['date_to']
        recency_focus      = extracted['recency_focus']
        year_week          = extracted['year_week']
        target_datasets    = extracted['target_datasets']

        print(f"[Debug] 추출된 키워드: {extracted_keywords}")
        print(f"[Debug] 날짜 범위: {date_from} ~ {date_to}")
        print(f"[Debug] 최신순 정렬 필요: {recency_focus}")
        print(f"[Debug] 감지된 주차: {year_week}")
        print(f"[Debug] LLM 선택 데이터셋: {target_datasets}")

        if dataset and dataset != 'All' and dataset in DATASETS:
            # 사용자가 특정 탭을 명시적으로 선택한 경우 — LLM 판단과 무관하게 그 탭만 검색
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

        with ThreadPoolExecutor(max_workers=len(search_targets)) as executor:
            futures = {
                ds: executor.submit(
                    self.retrieve,
                    extracted_keywords, query, ds, mode,
                    DEFAULT_SEARCH_LIMIT, date_from, date_to, recency_focus, year_week
                )
                for ds in search_targets
            }
            results = {ds: f.result() for ds, f in futures.items()}

        self.last_retrieved_nodes = list(self._pending_nodes)
        print(f"[Debug] 검색된 노드 수: {len(self.last_retrieved_nodes)}")

        sections        = []
        active_datasets = []

        for ds, context in results.items():
            desc = DATASETS[ds]['description']
            print(f"[Debug] {ds} 검색 결과:\n{context}\n")
            if context != _NO_RESULT:
                sections.append(f"=== {ds} ({desc}) ===\n{context}")
                active_datasets.append(ds)

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

        date_info = ""
        if date_from or date_to:
            date_info = (f"\n검색 적용 날짜 범위: "
                         f"{date_from or '제한없음'} ~ {date_to or '제한없음'}")
        if recency_focus:
            date_info += ("\n★ 아래 지식 그래프 컨텍스트의 트리플/문서는 날짜 내림차순"
                          "(최신이 맨 위)으로 정렬되어 있습니다. 사용자가 '가장 최근'을 물었다면"
                          " 맨 위(가장 먼저 나오는) 항목을 기준으로 답하세요.")

        # ★ 질문에 사내 코드(D1, BD30 등)가 있으면 물질명 매핑을 프롬프트에 명시한다.
        #   검색 컨텍스트(트리플/문서)는 물질명(content_norm) 기준으로 되어 있어서,
        #   이 매핑이 없으면 LLM 이 "질문의 코드"와 "컨텍스트의 물질명"을 별개로 보고
        #   답변을 못 하거나 엉뚱하게 답할 수 있다.
        code_map_found = _detect_codes(query)
        code_info = ""
        if code_map_found:
            mapping_lines = "\n".join(
                f"- '{code}' 는 사내 코드명이며, 실제 물질명은 '{material}' 입니다."
                for code, material in code_map_found.items()
            )
            example_code, example_material = next(iter(code_map_found.items()))
            code_info = f"""

[사내 코드명 ↔ 실제 물질명 매핑 — 반드시 확인]
아래 사내 코드는 질문에 사용되었지만, 검색 컨텍스트(트리플/문서)는 물질명 기준으로
제공됩니다. 코드명과 물질명이 동일한 대상을 가리킨다는 것을 확실히 인지하고,
둘을 별개의 것으로 혼동하지 마세요.
{mapping_lines}

답변 규칙 (코드명 관련):
- 답변 시작 부분에서 각 코드명과 실제 물질명의 관계를 사용자에게 명확히 알려주세요.
  예: "'{example_code}'는 사내 코드로, 실제 물질명은 '{example_material}'입니다."
- 이후 본문에서는 컨텍스트의 물질명을 기준으로 설명하되, 필요하면
  "{example_code}({example_material})"처럼 코드와 물질명을 함께 표기해 혼동을 줄이세요."""

        # ★ 디버그: LLM 프롬프트에 실제로 들어가는 "코드↔물질명 매핑" 섹션만 따로 출력.
        #   이 섹션이 비어 있으면(아래 (없음)) LLM 은 코드와 물질명을 연결하지 못한다.
        print("[Debug] ===== 코드↔물질명 매핑 (LLM 프롬프트에 삽입될 내용) =====")
        print(f"  모듈 로드 여부: {_NORMALIZE_AVAILABLE} "
              f"(CODE_MAP 항목 수: {len(_CODE_MAP) if _CODE_MAP else 0})")
        print(f"  원본 질문: '{query}'")
        print(f"  감지된 매핑: {code_map_found if code_map_found else '(없음 — 프롬프트에 매핑 섹션 미삽입)'}")
        print("[Debug] ===========================================================")

        system_prompt = f"""당신은 DRAM MIM 커패시터 소재 연구 전문가입니다.
다음 지식 그래프 컨텍스트가 제공됩니다.

{dataset_info}{date_info}{code_info}

답변 규칙:
- 제공된 컨텍스트와 이전 대화 내용을 적극적으로 활용하여 답하세요.
- 컨텍스트에 저자, 제목, 저널, 출처, 날짜 등의 메타데이터가 있으면 반드시 활용하세요.
- 저자를 묻는 경우 컨텍스트의 "저자" 값을 답하세요.
- 게재 저널을 묻는 경우 컨텍스트의 "저널" 값을 답하세요.
- 링크, 출처, URL, DOI를 묻는 경우 컨텍스트의 "출처" 값을 답하세요.
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
- 컨텍스트와 이전 대화 모두에 없는 내용만 모른다고 답하세요.
- 답변은 한국어로 작성하세요.

링크 작성 규칙 (중요):
- URL/링크는 마크다운 굵게(**) 표시를 절대 사용하지 마세요. "**https://...**" 같은
  형태는 링크 인식이 깨져 클릭할 수 없게 됩니다.
- 링크 앞뒤에는 반드시 공백이나 줄바꿈을 두세요. 단어나 문장부호를 링크에 바로
  붙여 쓰지 마세요 (예: "자세한내용은https://example.com참고" ❌).
- 링크는 그냥 URL 그대로 쓰거나 마크다운 링크 형식 [설명](URL) 으로 쓰세요."""

        user_message_content = f"""[추출된 키워드]
{extracted_keywords}

[지식 그래프 컨텍스트]
{combined_context}

[원본 질문]
{query}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message_content})

        # ★ 디버그: LLM 에 실제로 전달되는 프롬프트 전문을 그대로 출력한다.
        print("[Debug] ===== LLM 시스템 프롬프트 =====")
        print(system_prompt)
        print("[Debug] ===== LLM 사용자 메시지 =====")
        print(user_message_content)
        print("[Debug] ================================")

        # [단계 3] LLM 답변 생성 (여기서부터 content 청크가 스트리밍됨)
        yield {'type': 'status', 'text': '✍️ 답변 생성 중…'}

        full_result = []
        for chunk in ask_llm_stream_iter_messages(
            messages=messages,
            temperature=0.05,
            reasoning_effort="medium"
        ):
            full_result.append(chunk)
            yield {'type': 'content', 'text': chunk}

        result = "".join(full_result)
        if result:
            self.history.append({"role": "user",      "content": query})
            self.history.append({"role": "assistant",  "content": result})

            max_messages = MAX_HISTORY_TURNS * 2
            if len(self.history) > max_messages:
                self.history = self.history[-max_messages:]
                print(f"[Debug] 히스토리 트리밍: 최근 {MAX_HISTORY_TURNS}턴 유지")
