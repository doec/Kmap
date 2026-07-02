import sys
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

    def retrieve(self, keywords_str: str, query_text: str,
                 dataset: str, mode: str = None,
                 limit: int = DEFAULT_SEARCH_LIMIT,
                 date_from: str = None,
                 date_to: str = None) -> str:
        cfg          = DATASETS.get(dataset, list(DATASETS.values())[0])
        search_mode  = mode or DEFAULT_SEARCH_MODE or cfg.get('default_mode', 'text')
        hops         = cfg.get('search_hops') or DEFAULT_SEARCH_HOPS
        vector_index = cfg.get('vector_index')

        if search_mode in ('vector', 'hybrid') and not vector_index:
            print(f"[Debug] {dataset}: vector index 없음 → text 모드로 폴백")
            search_mode = 'text'

        print(f"[Debug] {dataset} | mode: {search_mode} | hop: {hops}")

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

        return self._format_rows(rows, cfg['return_fields'])

    def answer_stream(self, query: str, dataset: str = None, mode: str = None):
        self._pending_nodes = set()
        self.last_retrieved_nodes = []

        print(f"\n[Debug] ========================================")
        print(f"[Debug] 질문: {query}")
        print(f"[Debug] 데이터셋: {dataset}")

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

        system_prompt = f"""당신은 DRAM MIM 커패시터 소재 연구 전문가입니다.
다음 지식 그래프 컨텍스트가 제공됩니다.

{dataset_info}{date_info}

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

        full_result = []
        for chunk in ask_llm_stream_iter_messages(
            messages=messages,
            temperature=0.05,
            reasoning_effort="medium"
        ):
            full_result.append(chunk)
            yield chunk

        result = "".join(full_result)
        if result:
            self.history.append({"role": "user",      "content": query})
            self.history.append({"role": "assistant",  "content": result})

            max_messages = MAX_HISTORY_TURNS * 2
            if len(self.history) > max_messages:
                self.history = self.history[-max_messages:]
                print(f"[Debug] 히스토리 트리밍: 최근 {MAX_HISTORY_TURNS}턴 유지")
