"""
diag_confluence.py
-----------------------------------------------------------------------------
Confluence(단일 노드 구조, 2026-08 최종) 검색이 왜 안 되는지 원인을 확정하기
위한 읽기 전용 진단 스크립트.

    python diag_confluence.py
    python diag_confluence.py SrTiO3   # 본문에 확실히 있는 검색어를 주면 더 정확

rag_engine.py 의 CONFLUENCE_DATASET 설정이 세우는 가정들을 실제 DB 에 물어본다.
쓰기 작업은 전혀 하지 않는다 (MATCH / SHOW 만 사용).

각 항목 앞의 표시:
    [OK]    가정이 맞음
    [FAIL]  검색이 0건이 되는 직접적 원인
    [WARN]  0건까지는 아니지만 결과가 희석/누락될 수 있음
"""

import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI  = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
USER = os.getenv('NEO4J_USER', 'neo4j')
PW   = os.getenv('NEO4J_PASSWORD', '')

# rag_engine.py 의 CONFLUENCE_DATASET 설정과 반드시 일치시켜 둔다.
# ★ 2026-08: Confl_doc/Confluence → ConfluenceDoc/ConfluenceDB 로 라벨명 변경
#   (ReportsDB/PapersDB 와 동일한 "{이름}DB" 패턴에 맞춤). Neo4j 쪽 라벨/제약도
#   함께 마이그레이션했다는 전제 — 아직 안 했다면 옛 라벨('Confl_doc'/'Confluence')로
#   되돌려서 실행하세요.
DOC_LABEL      = 'ConfluenceDoc'      # 단일 노드 레이블
DS_LABEL       = 'ConfluenceDB'       # 데이터셋 공통 레이블
VECTOR_INDEX   = 'confluence_doc_embedding'
FULLTEXT_INDEXES = ['confluence_doc_fulltext', 'confluence_doc_fulltext_norm']
GROUP_FIELD    = 'doc_group_id'
NEXT_CHUNK_REL = 'NEXT_CHUNK'

_problems: list[str] = []


def ok(msg):
    print(f'  [OK]   {msg}')


def fail(msg):
    print(f'  [FAIL] {msg}')
    _problems.append(msg)


def warn(msg):
    print(f'  [WARN] {msg}')


def head(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


def q(session, cypher, **params):
    return [dict(r) for r in session.run(cypher, **params)]


# ── 1. 노드 개수 · 라벨 실측 ─────────────────────────────────────────────────
def check_counts(session):
    head('1. 노드 개수 · 실제 라벨 조합')

    total = q(session, f'MATCH (n:{DOC_LABEL}) RETURN count(n) AS c')[0]['c']
    print(f'  (:{DOC_LABEL}) 노드 = {total}')
    if total == 0:
        fail(f':{DOC_LABEL} 라벨을 가진 노드가 0개 — 적재가 안 됐거나 doc_label 이름이 다르다.')

    print('\n  -- :Confluence 라벨을 가진 노드들의 실제 라벨 조합 --')
    rows = q(session, f'MATCH (n:{DS_LABEL}) '
                      'RETURN labels(n) AS labels, count(*) AS c '
                      'ORDER BY c DESC LIMIT 10')
    for r in rows:
        print(f'     {sorted(r["labels"])}  = {r["c"]}')
    if not rows:
        fail(f':{DS_LABEL} 라벨을 가진 노드가 전혀 없다 — dataset_label 이름이 다르다.')
    elif not any(DOC_LABEL in r['labels'] for r in rows):
        fail(f':{DS_LABEL} 노드는 있지만 :{DOC_LABEL} 라벨을 가진 것이 없다 — '
             f'실제 노드 레이블이 스펙({DOC_LABEL})과 다르게 적재됐을 수 있다.')
    else:
        ok(f':{DOC_LABEL}:{DS_LABEL} 조합 확인됨')

    return total


# ── 2. 코드가 읽는 속성이 실제로 존재하는가 ──────────────────────────────────
def check_properties(session):
    head('2. 코드가 읽는 속성이 실제로 존재하는가')

    rows = q(session, f'''
        MATCH (n:{DOC_LABEL})
        RETURN count(*)                    AS total,
               count(n.doc_id)              AS doc_id,
               count(n.{GROUP_FIELD})       AS doc_group_id,
               count(n.chunk_index)         AS chunk_index,
               count(n.total_chunks)        AS total_chunks,
               count(n.content)             AS content,
               count(n.content_norm)        AS content_norm,
               count(n.embedding)           AS embedding,
               count(n.title)               AS title,
               count(n.page_title)          AS page_title,
               count(n.author)              AS author,
               count(n.researcher)          AS researcher,
               count(n.date)                AS date,
               count(n.year_week)           AS year_week,
               count(n.source_url)          AS source_url,
               count(n.title_path)          AS title_path
    ''')
    if not rows or not rows[0]['total']:
        warn('노드가 없어 속성 확인을 건너뜀 (1번 항목 참고).')
        return
    r = rows[0]
    t = r['total']
    print(f'  {DOC_LABEL} 총 {t}개 중:')
    for k in ('doc_id', 'doc_group_id', 'chunk_index', 'total_chunks',
              'content', 'content_norm', 'embedding', 'title', 'page_title',
              'author', 'researcher', 'date', 'year_week', 'source_url', 'title_path'):
        print(f'     {k:<14} = {r[k]}')

    if r['doc_id'] < t:
        fail(f'doc_id 가 없는 노드가 {t - r["doc_id"]}개 — 유니크 키가 없으면 '
             f'해당 문서는 조회/중복제거 단계에서 조용히 누락된다.')
    else:
        ok('doc_id 전부 존재')

    if r['doc_group_id'] == 0:
        warn(f'{GROUP_FIELD} 가 하나도 없다 — 청킹된 문서의 청크들이 순서대로 묶여 '
             f'보이지 않고, doc_id 문자열 정렬 순서(오정렬 위험)로 표시된다.')

    if r['embedding'] == 0:
        fail('embedding 속성이 하나도 없다 — 벡터 검색(B채널)이 항상 0건.')
    elif r['embedding'] < t:
        warn(f'embedding 이 없는 노드가 {t - r["embedding"]}개 — 그만큼 벡터 검색에서 누락.')
    else:
        ok('embedding 전부 존재')

    if r['content_norm'] == 0:
        fail("doc_body_field='content_norm' 인데 그 속성이 하나도 없다 — "
             "본문이 빈 문자열로 나가서 LLM 이 아무 내용도 못 본다 "
             "(content 로 COALESCE 폴백은 되지만 물질명 정규화가 안 됨).")
    elif r['content_norm'] < t:
        warn(f'content_norm 이 없는 노드가 {t - r["content_norm"]}개 (content 로 폴백됨).')

    if r['date'] == 0 and r['year_week'] == 0:
        fail('date 와 year_week 가 둘 다 없다 — 기간/주차 질문(D·E·F 채널)이 항상 0건.')
    if r['title'] == 0 and r['page_title'] == 0:
        warn('title/page_title 이 둘 다 없다 — 결과 제목이 비어 보인다.')
    if r['author'] == 0 and r['researcher'] == 0:
        warn('author/researcher 가 둘 다 없다 — 저자 검색(D채널)이 이 노드들에서는 항상 0건.')

    print('\n  -- year_week 값 샘플 (표기 확인용) --')
    for r2 in q(session, f'MATCH (n:{DOC_LABEL}) '
                        "WHERE COALESCE(n.year_week,'') <> '' "
                        'RETURN n.year_week AS yw, count(*) AS c '
                        'ORDER BY yw DESC LIMIT 8'):
        print(f'     {r2["yw"]!r} = {r2["c"]}')

    print('\n  -- total_chunks 분포 (1=청킹 안 됨, 2 이상=청킹됨) --')
    for r2 in q(session, f'MATCH (n:{DOC_LABEL}) '
                        'RETURN n.total_chunks AS tc, count(*) AS c '
                        'ORDER BY tc'):
        print(f'     total_chunks={r2["tc"]!r} = {r2["c"]}')


# ── 3. 인덱스 ────────────────────────────────────────────────────────────────
def check_indexes(session):
    head('3. 인덱스 존재 여부 · 대상 라벨')

    try:
        idx = q(session, 'SHOW INDEXES YIELD name, type, entityType, '
                         'labelsOrTypes, properties, state '
                         'RETURN name, type, entityType, labelsOrTypes, properties, state')
    except Exception as e:
        fail(f'SHOW INDEXES 실패: {e}')
        return

    by_name = {r['name']: r for r in idx}

    print('  -- 전체 인덱스 목록 --')
    for r in idx:
        print(f'     {r["name"]:<32} {r["type"]:<10} '
              f'{r["labelsOrTypes"]} {r["properties"]} [{r["state"]}]')

    print()
    v = by_name.get(VECTOR_INDEX)
    if not v:
        fail(f'벡터 인덱스 {VECTOR_INDEX} 가 없다 — B채널(문서 벡터 검색) 전면 0건. '
             f'embed_nodes.py 를 {DOC_LABEL}:{DS_LABEL} 대상으로 다시 실행해야 한다.')
    else:
        labels = v['labelsOrTypes'] or []
        if v['state'] != 'ONLINE':
            fail(f'{VECTOR_INDEX} 상태가 {v["state"]} — populating 중이면 결과가 비거나 부분적이다.')
        if DOC_LABEL in labels:
            ok(f'{VECTOR_INDEX} → {labels} {v["properties"]}')
        else:
            fail(f'{VECTOR_INDEX} 대상이 {labels} — 스펙(FOR (n:{DOC_LABEL}))과 다르다.')

    print()
    for name in FULLTEXT_INDEXES:
        r = by_name.get(name)
        if not r:
            fail(f'FULLTEXT 인덱스 {name} 가 없다 — C채널에서 이 인덱스 몫이 0건.')
            continue
        if r['type'] != 'FULLTEXT':
            fail(f'{name} 의 type 이 {r["type"]} (FULLTEXT 아님).')
            continue
        if r['state'] != 'ONLINE':
            fail(f'{name} 상태가 {r["state"]}.')
        labels = r['labelsOrTypes'] or []
        if DOC_LABEL not in labels:
            fail(f'{name} 대상이 {labels} — 기대: {DOC_LABEL}.')
        else:
            ok(f'{name} → {labels} {r["properties"]}')

    # 유니크 제약 확인
    print()
    try:
        cons = q(session, 'SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties '
                          'RETURN name, labelsOrTypes, properties')
        has_uniq = any(DS_LABEL in (c['labelsOrTypes'] or []) and 'doc_id' in (c['properties'] or [])
                       for c in cons)
        if has_uniq:
            ok(f'doc_id UNIQUE 제약 확인됨 (:{DS_LABEL})')
        else:
            warn(f'doc_id 에 대한 UNIQUE 제약을 못 찾음 — 스펙대로면 '
                 f'CREATE CONSTRAINT FOR (n:{DS_LABEL}) REQUIRE n.doc_id IS UNIQUE 가 있어야 한다.')
    except Exception as e:
        warn(f'SHOW CONSTRAINTS 실패: {e}')


# ── 4. 실제 채널 재현 ────────────────────────────────────────────────────────
def check_channels(session, probe: str):
    head(f'4. 실제 쿼리 재현 (검색어: {probe!r})')

    # C채널 — FULLTEXT
    for name in FULLTEXT_INDEXES:
        try:
            rows = q(session, f'''
                CALL db.index.fulltext.queryNodes($index, $q, {{limit: 5}})
                YIELD node, score
                WHERE node:{DS_LABEL}
                RETURN node.doc_id AS doc_id, score,
                       COALESCE(node.title, node.page_title) AS title
                ORDER BY score DESC
            ''', index=name, q=f'"{probe}"')
            if rows:
                ok(f'{name}: {len(rows)}건')
                for r in rows[:3]:
                    print(f'          score={r["score"]:.2f} {r["doc_id"]} / {r["title"]}')
            else:
                raw = q(session, 'CALL db.index.fulltext.queryNodes($index, $q, '
                                 '{limit: 5}) YIELD node RETURN count(*) AS c',
                        index=name, q=f'"{probe}"')[0]['c']
                if raw == 0:
                    warn(f'{name}: 인덱스 자체가 0건 (검색어가 본문에 없을 수도 있다)')
                else:
                    fail(f'{name}: 인덱스는 {raw}건 맞췄지만 {DS_LABEL} 가드에서 전부 탈락 '
                         f'— 노드에 :{DS_LABEL} 라벨이 없다는 뜻.')
        except Exception as e:
            fail(f'{name} 쿼리 실패: {e}')

    # B채널 — 벡터 (실제 노드의 임베딩을 재사용해 인덱스가 살아있는지 확인)
    try:
        emb = q(session, f'MATCH (n:{DOC_LABEL}) WHERE n.embedding IS NOT NULL '
                         'RETURN n.embedding AS e LIMIT 1')
        if not emb:
            warn('임베딩을 가진 노드가 없어 벡터 검색 재현을 건너뜀 (2번 항목 참고).')
        else:
            rows = q(session, f'''
                CALL db.index.vector.queryNodes($index, 5, $e)
                YIELD node, score
                WHERE node:{DS_LABEL}
                RETURN node.doc_id AS doc_id, score
                ORDER BY score DESC
            ''', index=VECTOR_INDEX, e=emb[0]['e'])
            if rows:
                ok(f'{VECTOR_INDEX}: {len(rows)}건 (자기 임베딩으로 재현 — '
                   f'top score={rows[0]["score"]:.3f})')
            else:
                fail(f'{VECTOR_INDEX}: 실제 임베딩을 넣어도 0건 — 인덱스가 이 노드를 '
                     f'대상으로 하지 않거나 아직 populating.')
    except Exception as e:
        fail(f'{VECTOR_INDEX} 쿼리 실패: {e}')

    # D채널 — 저자 (실제 author 값 하나를 뽑아 재현)
    try:
        a = q(session, f'MATCH (n:{DOC_LABEL}) '
                       "WHERE COALESCE(n.author, n.researcher, '') <> '' "
                       'RETURN COALESCE(n.author, n.researcher) AS a LIMIT 1')
        if not a:
            warn('author/researcher 값을 가진 노드가 없어 저자 검색 재현을 건너뜀.')
        else:
            sample_author = a[0]['a']
            rows = q(session, f'''
                MATCH (n:{DOC_LABEL})
                WHERE toLower(COALESCE(n.author,'')) CONTAINS toLower($kw)
                   OR toLower(COALESCE(n.researcher,'')) CONTAINS toLower($kw)
                RETURN n.doc_id AS doc_id
                ORDER BY n.date DESC LIMIT 5
            ''', kw=sample_author)
            (ok if rows else fail)(
                f'저자 "{sample_author}" 로 재현: {len(rows)}건'
                + ('' if rows else ' — CONTAINS 매칭 자체가 실패, author 필드 값/형식을 확인'))
    except Exception as e:
        fail(f'저자 검색 재현 실패: {e}')

    # NEXT_CHUNK 관계 확인 (Sentence Window 용, 아직 미사용이지만 존재 여부만 체크)
    try:
        n_rel = q(session, f'MATCH (:{DOC_LABEL})-[r:{NEXT_CHUNK_REL}]->(:{DOC_LABEL}) '
                           'RETURN count(r) AS c')[0]['c']
        n_multi = q(session, f'MATCH (n:{DOC_LABEL}) WHERE n.total_chunks > 1 '
                             'RETURN count(n) AS c')[0]['c']
        print(f'\n  {NEXT_CHUNK_REL} 관계 수 = {n_rel} (total_chunks>1 인 노드 {n_multi}개)')
        if n_multi and not n_rel:
            warn(f'total_chunks>1 인 청크가 있는데 {NEXT_CHUNK_REL} 관계가 0개 — '
                 f'앞/뒤 청크 확장(Sentence Window)을 나중에 구현해도 쓸 데이터가 없다.')
    except Exception as e:
        warn(f'{NEXT_CHUNK_REL} 확인 실패: {e}')


# ── 5. 샘플 덤프 ─────────────────────────────────────────────────────────────
def dump_sample(session):
    head('5. 노드 샘플 (속성 이름 눈으로 확인)')
    rows = q(session, f'MATCH (n:{DOC_LABEL}) RETURN n LIMIT 1')
    if not rows:
        warn('샘플을 뽑을 노드가 없다.')
        return
    node = rows[0]['n']
    print(f'\n  -- 샘플 노드 ({sorted(node.labels)}) --')
    for k, v in sorted(dict(node).items()):
        if isinstance(v, list) and len(v) > 8:
            print(f'     {k:<18} = <list len={len(v)}>')
        else:
            s = str(v).replace('\n', '\\n')
            print(f'     {k:<18} = {s[:120]}{"..." if len(s) > 120 else ""}')


def main():
    probe = sys.argv[1] if len(sys.argv) > 1 else '보고'
    print(f'Neo4j: {URI}')
    driver = GraphDatabase.driver(URI, auth=(USER, PW))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f'[FAIL] Neo4j 접속 실패: {e}')
        return 1
    with driver.session() as session:
        total = check_counts(session)
        if total:
            check_properties(session)
        check_indexes(session)
        check_channels(session, probe)
        dump_sample(session)
    driver.close()

    head('요약')
    if _problems:
        print(f'  0건의 직접 원인으로 판단되는 항목 {len(_problems)}개:')
        for i, p in enumerate(_problems, 1):
            print(f'    {i}. {p}')
    else:
        print('  [FAIL] 항목 없음 — 스키마/인덱스는 정상이다.')
        print('  이 경우 원인은 DB 쪽이 아니라 (a) .env 의 '
              'NEO4J_CONFLUENCE_ENABLED 가 false, (b) 데이터셋 탭 선택,')
        print('  (c) 키워드 추출 단계에서 target_datasets/author_names 에 '
              '값이 안 들어간 것 중 하나다.')
        print('  앱을 디버그 레벨 1 이상으로 띄워 채널별 건수(A~F)를 확인해 주세요.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
