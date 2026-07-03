import sys
import re
import json
from pathlib import Path
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

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

DEFAULT_SEARCH_LIMIT = 50
DEFAULT_SEARCH_HOPS  = 2
DEFAULT_SEARCH_MODE  = None   # None / 'text' / 'vector' / 'hybrid'

RRF_K             = 60
MAX_HISTORY_TURNS = 5

# 문서 섹션 관련 상수
DOC_SEARCH_LIMIT  = 5     # 문서 단위 벡터 검색으로 가져올 문서 개수 (B 기능)
DOC_SCORE_MIN     = 0.6   # 문서 벡터 검색 최소 유사도 컷
DOC_BODY_MAXLEN   = 700   # 답변 컨텍스트에 넣을 문서 본문(abstract/content) 최대 길이

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
#                    ReportsDB 엔티티 벡터 인덱스에는 Document 메타 노드도 섞여
#                    들어오므로, 벡터 검색 결과에서 이 레이블을 가진 노드를
#                    'WHERE NOT node:<meta_label>' 로 걸러내야 한다.
#   default_mode   : 사용자가 모드를 지정하지 않았을 때 기본 검색 모드
#   search_hops    : 그래프 확장 hop 수 (None 이면 전역 기본값 사용)
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
        'min_confidence': None,
        'vector_index':   'reportsdb_entity_embedding',  # ★ 이름 변경됨
        'meta_label':     'Document',                    # ★ 벡터 검색에서 제외할 메타 노드
        'default_mode':   'hybrid',
        'search_hops':    None,
        # ── 문서(메타 노드) 관련 설정 (A: doc_id 조인 / B: 문서 벡터 검색) ──
        'doc_vector_index': 'reportsdb_doc_embedding',   # Document 노드 벡터 인덱스
        'doc_label':        'Document',                  # 메타 노드 레이블
        # ★ content_norm: 코드(D1)를 물질명(HfO2)으로 변환한 정규화 본문.
        #   embedding 도 content_norm 기반이고 LLM 컨텍스트도 물질명으로 주는 게
        #   의미 파악에 유리하므로, 답변 컨텍스트용 본문으로 content_norm 을 쓴다.
        'doc_body_field':   'content_norm',
        'doc_body_label':   '내용',              # LLM 컨텍스트에 표기할 본문 레이블
        'doc_fulltext_index': 'doc_fulltext',   # ★ C: 문서 본문 키워드(FULLTEXT) 검색
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
        'vector_index':   'papersdb_embedding',
        # papersdb_embedding 은 n.embedding 을 인덱싱하는데, Paper 메타 노드는
        # n.abstract_embedding(다른 속성)을 쓰므로 이 인덱스에 애초에 포함되지 않는다.
        # 그래도 방어적으로 필터를 걸어 안전하게 처리한다.
        'meta_label':     'Paper',
        'default_mode':   'hybrid',
        'search_hops':    None,
        # ── 문서(메타 노드) 관련 설정 (A: doc_id 조인 / B: 문서 벡터 검색) ──
        'doc_vector_index': 'paper_abstract_embedding',  # Paper 노드(abstract 기반) 벡터 인덱스
        'doc_label':        'Paper',                     # 메타 노드 레이블
        'doc_body_field':   'abstract',                  # 본문 속성명 (논문 초록)
        'doc_body_label':   '초록',                       # LLM 컨텍스트에 표기할 본문 레이블
        'doc_fulltext_index': 'paper_fulltext',          # ★ C: 초록 키워드(FULLTEXT) 검색
        # ★ 코드→물질명 매핑(CODE_MAP)은 사내 코드에 대한 일반적인 번역이라
        #   ReportsDB 뿐 아니라 PapersDB 검색에도 동일하게 적용해야 한다.
        #   (예: "BD30" 으로 논문 검색 시에도 물질명으로 정규화되어야 논문
        #    임베딩/엔티티명과 어휘가 맞는다.) CODE_MAP 에 없는 단어는 그대로 반환되므로
        #   켜둬도 부작용이 없다.
        'normalize_query':  True,
    },
}

_NO_RESULT = "관련 트리플을 찾지 못했습니다."

# ★ 스키마 변경 대응: 구조(출처) 관계 타입.
# 새 스키마는 엔티티와 메타 노드를 (entity)-[:FROM_PAPER]->(:Paper) /
# (entity)-[:FROM_DOC]->(:Document) 로 연결한다. 그런데 Paper/Document 메타 노드도
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

    def close(self):
        self.driver.close()

    def clear_history(self):
        self.history = []
        print("[Debug] 대화 히스토리 초기화")

    def _extract_keywords(self, query: str) -> dict:
        today     = datetime.today()
        today_str = today.strftime("%Y-%m-%d")
        m3_ago    = (today - timedelta(days=90)).strftime("%Y-%m-%d")
        m6_ago    = (today - timedelta(days=180)).strftime("%Y-%m-%d")
        y1_ago    = (today - timedelta(days=365)).strftime("%Y-%m-%d")
        y2_ago    = (today - timedelta(days=730)).strftime("%Y-%m-%d")

        extract_prompt = f"""다음 질문에서 검색 키워드와 날짜 범위를 추출하세요.

오늘 날짜: {today_str}
자주 쓰는 날짜 참고:
- 최근 3개월 이내: {m3_ago} ~ {today_str}
- 최근 6개월 이내: {m6_ago} ~ {today_str}
- 최근 1년 이내:   {y1_ago} ~ {today_str}
- 최근 2년 이내:   {y2_ago} ~ {today_str}

출력 형식 (JSON):
{{
  "keywords": "키워드1, 키워드2, ...",
  "date_from": "YYYY-MM-DD 또는 null",
  "date_to": "YYYY-MM-DD 또는 null"
}}

날짜 변환 규칙:
- "N년 이후", "N년 이상", "N년부터" → date_from: "N-01-01", date_to: null
- "N년 이전", "N년 이하", "N년까지" → date_from: null, date_to: "N-12-31"
- "N년~M년", "N년에서 M년" → date_from: "N-01-01", date_to: "M-12-31"
- "최근 N개월" → 오늘 기준 N개월 전 날짜 계산
- "최근 N년" → 오늘 기준 N년 전 날짜 계산
- "올해" → date_from: "{today.year}-01-01", date_to: "{today_str}"
- 날짜 언급 없음 → date_from: null, date_to: null

키워드 추출 규칙:
1. 핵심 명사(엔티티) 위주로 추출
2. 한국어 키워드는 반드시 영어 번역도 함께 추출
3. 관련 동의어/유사어도 포함
4. 날짜 관련 표현은 키워드에서 제외

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
            return {
                'keywords':  parsed.get('keywords', query),
                'date_from': parsed.get('date_from'),
                'date_to':   parsed.get('date_to'),
            }
        except Exception as e:
            print(f"[Debug] 키워드 파싱 오류: {e} / 원본: {result}")
            return {'keywords': query, 'date_from': None, 'date_to': None}

    def _build_return_clause(self, return_fields: list) -> str:
        base = [
            "s.name AS sname", "s.type AS stype",
            "type(r) AS rel",
            "o.name AS oname", "o.type AS otype",
            # ★ A 기능: 각 트리플이 어느 문서(doc_id)에서 나왔는지 항상 가져온다.
            #   이 doc_id 로 나중에 Paper/Document 메타 노드를 조인해 원문을 붙인다.
            #   (return_fields 에는 없으므로 트리플 줄에는 출력되지 않고, 내부 수집용으로만 쓰임)
            "CASE WHEN r.doc_id IS NOT NULL THEN r.doc_id ELSE '' END AS doc_id",
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

    def _format_rows(self, rows: list, return_fields: list) -> str:
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
                    line += f"\n  {f}: {r[f]}"
            lines.append(line)
        return "\n".join(lines)

    def _text_retrieve_raw(self, keywords_str: str, dataset: str, cfg: dict,
                           limit: int, date_from: str, date_to: str) -> list[dict]:
        keywords      = keywords_str.replace(",", " ").split()
        return_fields = cfg['return_fields']
        return_clause = self._build_return_clause(return_fields)
        where_parts, params = self._build_where(cfg, keywords, date_from, date_to)
        params['limit'] = limit
        # ★ 출처(구조) 관계 제외: FROM_PAPER / FROM_DOC 는 의미 트리플이 아님
        params['struct_rels'] = _STRUCTURAL_RELS

        query_str = f"""
            MATCH (s:{dataset})-[r]->(o:{dataset})
            WHERE NOT type(r) IN $struct_rels
              AND {" AND ".join(where_parts)}
            RETURN {return_clause}
            ORDER BY {cfg['sort_field']} {cfg['sort_order']}
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
        #   Document 메타 노드도 :ReportsDB 레이블 + embedding 속성을 가지므로
        #   이 인덱스에 함께 포함된다. 따라서 벡터 검색 결과에서 메타 노드를
        #   'WHERE NOT s:Document' 로 제외해야 엔티티만 남는다.
        #   (PapersDB 는 메타 노드가 다른 속성명을 써서 애초에 안 섞이지만,
        #    일관성/안전을 위해 동일하게 필터를 건다.)
        node_filter = f"AND NOT s:{meta_label}" if meta_label else ""

        # ★ 출처(구조) 관계 제외 (text 검색과 동일한 이유)
        params['struct_rels'] = _STRUCTURAL_RELS

        field_returns = ", ".join([
            f"CASE WHEN r.{f} IS NOT NULL THEN r.{f} ELSE '' END AS {f}"
            for f in return_fields
        ])

        # 3) 벡터 검색 → 나온 노드(s)를 트리플의 주어로 삼아 관계까지 확장
        #    - queryNodes 로 상위 (limit*3) 개 후보를 넉넉히 뽑고,
        #      메타 노드 제외 + 관계 필터를 적용한 뒤 vec_score 순으로 limit 개만 사용.
        query_str = f"""
            CALL db.index.vector.queryNodes('{vector_index}', $limit * 3, $q_emb)
            YIELD node AS s, score AS vec_score
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
                return [dict(r) for r in session.run(query_str, **params)]
        except Exception as e:
            print(f"[Debug] {dataset} vector 검색 실패 → text 결과만 사용: {e}")
            return []

    def _text_retrieve_2hop(self, keywords_str: str, dataset: str, cfg: dict,
                            limit: int, date_from: str, date_to: str) -> list[dict]:
        hop_limit = limit * 2

        hop1_rows = self._text_retrieve_raw(
            keywords_str, dataset, cfg, hop_limit, date_from, date_to
        )

        hop1_nodes = set()
        for r in hop1_rows:
            if r.get('sname'): hop1_nodes.add(r['sname'])
            if r.get('oname'): hop1_nodes.add(r['oname'])

        if not hop1_nodes:
            return hop1_rows[:limit]

        extended_keywords = keywords_str + ", " + ", ".join(list(hop1_nodes)[:10])
        hop2_rows = self._text_retrieve_raw(
            extended_keywords, dataset, cfg, hop_limit, date_from, date_to
        )

        seen: dict = {}
        for r in hop1_rows + hop2_rows:
            key = f"{r.get('sname')}|{r.get('rel')}|{r.get('oname')}"
            seen[key] = r

        print(f"[Debug] {dataset} 2-hop: 1차 {len(hop1_rows)}개 + "
              f"2차 {len(hop2_rows)}개 → 중복 제거 후 {min(len(seen), limit)}개")
        return list(seen.values())[:limit]

    # ── B 기능: 문서 단위 벡터 검색 ────────────────────────────────────────────
    def _doc_vector_retrieve(self, query_text: str, cfg: dict,
                             limit: int = DOC_SEARCH_LIMIT) -> list[str]:
        """
        Paper/Document 메타 노드를 대상으로 벡터 검색을 수행해 관련 문서의
        doc_id 목록을 돌려준다.

        엔티티 벡터 검색(_vector_retrieve_raw)이 "관계(트리플)"를 찾는 것과 달리,
        이건 "문서 자체"(논문 초록 / 보고서 전문)를 질문과의 유사도로 찾는다.
        → "이 주제 관련 논문/보고서 찾아줘" 류의 질의에 강하다.

        paper_abstract_embedding / reportsdb_doc_embedding 인덱스는 각각
        Paper / Document 노드만 포함하므로 메타 노드 제외 필터가 필요 없다.
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
            'score_min': DOC_SCORE_MIN,
        }
        try:
            with self.driver.session() as session:
                rows = [dict(r) for r in session.run(query_str, **params)]
            return [r['doc_id'] for r in rows if r.get('doc_id')]
        except Exception as e:
            print(f"[Debug] 문서 벡터 검색 실패: {e}")
            return []

    # ── C 기능: 문서 본문 키워드(FULLTEXT) 검색 ─────────────────────────────────
    def _doc_fulltext_retrieve(self, keywords_str: str, cfg: dict,
                               limit: int = DOC_SEARCH_LIMIT) -> list[str]:
        """
        doc_fulltext / paper_fulltext 인덱스로 문서 본문(title+content(_norm)/abstract)에서
        키워드를 검색해 관련 문서의 doc_id 목록을 돌려준다.

        벡터 검색(_doc_vector_retrieve)이 "의미 유사도"로 찾는다면, 이건 "단어 일치"로
        찾는다. 코드(D1)·모델명·수치처럼 정확한 표기가 중요한 검색에 강하다.
        (ReportsDB 는 content 원문(코드)까지 인덱싱돼 있어 정규화 전 코드로도 매칭됨)
        """
        ft_index = cfg.get('doc_fulltext_index')
        if not ft_index or not keywords_str:
            return []

        # Lucene 질의 문자열 구성:
        # 키워드에 '/'·'-' 등 Lucene 특수문자가 있으면 파싱 오류가 나므로,
        # 각 키워드를 큰따옴표로 감싼 구(phrase)로 만들고 내부 특수문자는 이스케이프한다.
        # 따옴표로 감싼 구들을 공백으로 이으면 Lucene 기본 OR(should) 매칭이 된다.
        keywords = [kw.strip() for kw in keywords_str.replace(",", " ").split() if kw.strip()]
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
        params = {'index': ft_index, 'q': lucene_query, 'limit': limit}
        try:
            with self.driver.session() as session:
                rows = [dict(r) for r in session.run(query_str, **params)]
            return [r['doc_id'] for r in rows if r.get('doc_id')]
        except Exception as e:
            print(f"[Debug] 문서 FULLTEXT 검색 실패: {e}")
            return []

    # ── A 기능: doc_id 로 메타 노드 원문(abstract/content) 조회 ──────────────────
    def _fetch_documents(self, doc_ids: set[str], cfg: dict) -> str:
        """
        doc_id 집합을 받아 Paper/Document 메타 노드에서 제목·본문·출처를 조회하고,
        LLM 컨텍스트에 넣을 "관련 문서" 섹션 문자열로 만든다.

        트리플은 (주어)-[관계]->(목적어) 형태라 근거 문장(evidence) 정도만 담지만,
        여기서 원문(논문 초록 / 보고서 전문)을 붙여 주면 답변 근거가 훨씬 풍부해진다.
        본문은 DOC_BODY_MAXLEN 로 잘라 컨텍스트 폭주를 막는다.
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
        query_str = f"""
            MATCH (m:{doc_label})
            WHERE m.doc_id IN $ids
            RETURN m.doc_id     AS doc_id,
                   m.title      AS title,
                   m.author     AS author,
                   m.date       AS date,
                   m.source_url AS source_url,
                   COALESCE(m.{body_field}, m.content) AS body
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
            meta_bits = [b for b in (d.get('author'), d.get('date')) if b]
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
                 date_to: str = None) -> str:
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

        if search_mode in ('vector', 'hybrid') and not vector_index:
            print(f"[Debug] {dataset}: vector index 없음 → text 모드로 폴백")
            search_mode = 'text'

        print(f"[Debug] {dataset} | mode: {search_mode} | hop: {hops}")
        # ★ 실제 검색에 쓰이는 최종 키워드/질의 문자열 (정규화 반영 후)
        print(f"[Debug] {dataset} 실제 검색 키워드(text/fulltext): '{keywords_str}'")
        print(f"[Debug] {dataset} 실제 검색 질의(vector): '{query_text}'")

        if search_mode == 'text':
            rows = (self._text_retrieve_2hop(keywords_str, dataset, cfg, limit, date_from, date_to)
                    if hops == 2
                    else self._text_retrieve_raw(keywords_str, dataset, cfg, limit, date_from, date_to))

        elif search_mode == 'vector':
            rows = self._vector_retrieve_raw(query_text, dataset, cfg, limit, date_from, date_to)

        else:  # hybrid
            fetch_limit = limit * 3
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_text = executor.submit(
                    self._text_retrieve_2hop if hops == 2 else self._text_retrieve_raw,
                    keywords_str, dataset, cfg, fetch_limit, date_from, date_to
                )
                f_vec = executor.submit(
                    self._vector_retrieve_raw,
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

        # ── 트리플 컨텍스트 ─────────────────────────────────────────────────────
        triples_ctx = self._format_rows(rows, cfg['return_fields'])

        # ── 문서 컨텍스트 (A + B) ───────────────────────────────────────────────
        # 문서 doc_id 후보 = 트리플에서 나온 출처 doc_id (A)
        #                  ∪ 문서 벡터 검색으로 찾은 관련 문서 doc_id (B, vector/hybrid 모드)
        ids_a = self._collect_doc_ids(rows)                              # A: 트리플 출처 문서
        ids_b = set(self._doc_vector_retrieve(query_text, cfg)) if search_mode in ('vector', 'hybrid') else set()   # B
        ids_c = set(self._doc_fulltext_retrieve(keywords_str, cfg)) if search_mode in ('text', 'hybrid') else set()  # C
        doc_ids = ids_a | ids_b | ids_c
        print(f"[Debug] {dataset} 문서 doc_id: A(트리플)={len(ids_a)} "
              f"B(벡터)={len(ids_b)} C(키워드)={len(ids_c)} → 합집합 {len(doc_ids)}")

        docs_ctx = self._fetch_documents(doc_ids, cfg)
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

        print(f"[Debug] 추출된 키워드: {extracted_keywords}")
        print(f"[Debug] 날짜 범위: {date_from} ~ {date_to}")

        if dataset and dataset != 'All' and dataset in DATASETS:
            search_targets = [dataset]
        else:
            search_targets = list(DATASETS.keys())

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
                    DEFAULT_SEARCH_LIMIT, date_from, date_to
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

        # ★ 질문에 사내 코드(D1, BD30 등)가 있으면 물질명 매핑을 프롬프트에 명시한다.
        #   검색 컨텍스트(트리플/문서)는 물질명(content_norm) 기준으로 되어 있어서,
        #   이 매핑이 없으면 LLM 이 "질문의 코드"와 "컨텍스트의 물질명"을 별개로 보고
        #   답변을 못 하거나 엉뚱하게 답할 수 있다.
        code_map_found = _detect_codes(query)
        code_info = ""
        if code_map_found:
            mapping_lines = "\n".join(f"- {code} = {material}" for code, material in code_map_found.items())
            code_info = f"""

질문에 사용된 사내 코드명과 실제 물질명 매핑 (컨텍스트는 물질명 기준으로 제공됨):
{mapping_lines}
→ 질문의 코드명이 컨텍스트의 물질명과 같은 대상을 가리킨다는 것을 인지하고 답변하세요."""

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
- 컨텍스트에 author, title, source_url, date 등의 메타데이터가 있으면 반드시 활용하세요.
- 저자를 묻는 경우 컨텍스트의 author 값을 답하세요.
- 링크, 출처, URL, DOI를 묻는 경우 컨텍스트의 source_url 값을 답하세요.
- 논문 제목을 묻는 경우 컨텍스트의 title 값을 답하세요.
- 이전 대화에서 언급된 메타데이터도 참고하세요.
- 컨텍스트와 이전 대화 모두에 없는 내용만 모른다고 답하세요.
- 답변은 한국어로 작성하세요."""

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
