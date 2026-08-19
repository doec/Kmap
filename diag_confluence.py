"""
diag_confluence.py
-----------------------------------------------------------------------------
Confluence(Document/Chunk 분리 구조) 검색이 왜 0건인지 원인을 확정하기 위한
읽기 전용 진단 스크립트.

    python diag_confluence.py

rag_engine.py 가 세우고 있는 가정들을 하나씩 실제 DB 에 물어본다.
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
DOC_LABEL        = 'Document'
CHUNK_LABEL      = 'Chunk'
DS_LABEL         = 'Confluence'
CHUNK_REL        = 'HAS_CHUNK'
VECTOR_INDEX     = 'confluence_doc_embedding'
CHUNK_FT_INDEXES = ['confluence_chunk_fulltext', 'confluence_chunk_fulltext_norm']
DOC_FT_INDEXES   = ['confluence_doc_fulltext', 'confluence_doc_fulltext_norm']

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


# ── 1. 노드/관계 개수 ────────────────────────────────────────────────────────
def check_counts(session):
    head('1. 노드 · 관계 개수')

    docs = q(session, f'MATCH (n:{DOC_LABEL}:{DS_LABEL}) RETURN count(n) AS c')[0]['c']
    chunks = q(session, f'MATCH (n:{CHUNK_LABEL}:{DS_LABEL}) RETURN count(n) AS c')[0]['c']
    rels = q(session, f'MATCH (:{DOC_LABEL}:{DS_LABEL})-[r:{CHUNK_REL}]->'
                      f'(:{CHUNK_LABEL}:{DS_LABEL}) RETURN count(r) AS c')[0]['c']

    print(f'  (:{DOC_LABEL}:{DS_LABEL})            = {docs}')
    print(f'  (:{CHUNK_LABEL}:{DS_LABEL})               = {chunks}')
    print(f'  Document-[:{CHUNK_REL}]->Chunk = {rels}')

    if docs == 0:
        fail(f'Document:{DS_LABEL} 노드가 0개 — 적재가 안 됐거나 라벨 이름이 다르다.')
    if chunks == 0:
        fail(f'Chunk:{DS_LABEL} 노드가 0개 — 본문/임베딩이 어디에도 없다.')
    if docs and chunks and rels == 0:
        fail(f'{CHUNK_REL} 관계가 0개 — 코드가 Document 와 Chunk 를 조인하지 못해 '
             f'모든 채널이 0건이 된다. (관계 방향/이름 확인)')
    elif rels:
        ok(f'{CHUNK_REL} 조인 가능 (문서당 평균 {rels / max(docs, 1):.1f} 청크)')

    # 라벨 조합 실측: 가정한 이름이 틀렸을 때 힌트가 되도록 전체 라벨 분포를 보여준다.
    print('\n  -- Confluence 라벨을 가진 노드들의 실제 라벨 조합 --')
    for r in q(session, f'MATCH (n:{DS_LABEL}) '
                        'RETURN labels(n) AS labels, count(*) AS c '
                        'ORDER BY c DESC LIMIT 10'):
        print(f'     {sorted(r["labels"])}  = {r["c"]}')

    if not q(session, f'MATCH (n:{DS_LABEL}) RETURN n LIMIT 1'):
        fail(f':{DS_LABEL} 라벨을 가진 노드가 전혀 없다 — dataset_label 이름이 다르다.')

    return docs, chunks


# ── 2. 속성 확인 ─────────────────────────────────────────────────────────────
def check_properties(session):
    head('2. 코드가 읽는 속성이 실제로 존재하는가')

    # Chunk: chunk_id / chunk_index / content_norm(또는 content) / embedding
    rows = q(session, f'''
        MATCH (c:{CHUNK_LABEL}:{DS_LABEL})
        RETURN count(*)                                            AS total,
               count(c.chunk_id)                                   AS chunk_id,
               count(c.chunk_index)                                AS chunk_index,
               count(c.total_chunks)                               AS total_chunks,
               count(c.content)                                    AS content,
               count(c.content_norm)                               AS content_norm,
               count(c.embedding)                                  AS embedding
    ''')
    if rows and rows[0]['total']:
        r = rows[0]
        t = r['total']
        print(f'  Chunk 총 {t}개 중:')
        for k in ('chunk_id', 'chunk_index', 'total_chunks',
                  'content', 'content_norm', 'embedding'):
            print(f'     {k:<14} = {r[k]}')
        if r['chunk_id'] < t:
            fail(f'chunk_id 가 없는 Chunk 가 {t - r["chunk_id"]}개 — '
                 f'검색 결과의 유니크 키라서 없으면 그 청크는 조용히 버려진다.')
        else:
            ok('chunk_id 전부 존재')
        if r['embedding'] == 0:
            fail('Chunk 에 embedding 속성이 하나도 없다 — 벡터 검색(B채널)이 0건.')
        elif r['embedding'] < t:
            warn(f'embedding 이 없는 Chunk 가 {t - r["embedding"]}개 — 그 청크는 '
                 f'벡터 검색에 안 걸린다(FULLTEXT 로는 걸림).')
        else:
            ok('embedding 전부 존재')
        if r['content_norm'] == 0:
            fail("doc_body_field='content_norm' 인데 그 속성이 하나도 없다 — "
                 "본문이 빈 문자열로 나가서 LLM 이 아무 내용도 못 본다.")
        elif r['content_norm'] < t:
            warn(f'content_norm 이 없는 Chunk 가 {t - r["content_norm"]}개.')
        else:
            ok('content_norm 전부 존재')

    # Document: doc_id / title / date / year_week / source_url
    rows = q(session, f'''
        MATCH (d:{DOC_LABEL}:{DS_LABEL})
        RETURN count(*)                AS total,
               count(d.doc_id)         AS doc_id,
               count(d.title)          AS title,
               count(d.page_title)     AS page_title,
               count(d.author)         AS author,
               count(d.date)           AS date,
               count(d.year_week)      AS year_week,
               count(d.source_url)     AS source_url,
               count(d.title_path)     AS title_path
    ''')
    if rows and rows[0]['total']:
        r = rows[0]
        t = r['total']
        print(f'\n  Document 총 {t}개 중:')
        for k in ('doc_id', 'title', 'page_title', 'author', 'date',
                  'year_week', 'source_url', 'title_path'):
            print(f'     {k:<14} = {r[k]}')
        if r['doc_id'] < t:
            fail(f'doc_id 가 없는 Document 가 {t - r["doc_id"]}개.')
        if r['date'] == 0 and r['year_week'] == 0:
            fail('date 와 year_week 가 둘 다 없다 — 기간/주차 질문(D·E·F 채널)이 항상 0건.')
        if r['title'] == 0 and r['page_title'] == 0:
            warn('title/page_title 이 둘 다 없다 — 결과 제목이 비어 보인다.')

        # year_week 표기 실측 (대문자 W 문제 확인)
        print('\n  -- year_week 값 샘플 --')
        for r2 in q(session, f'MATCH (d:{DOC_LABEL}:{DS_LABEL}) '
                             "WHERE COALESCE(d.year_week,'') <> '' "
                             'RETURN d.year_week AS yw, count(*) AS c '
                             'ORDER BY yw DESC LIMIT 8'):
            print(f'     {r2["yw"]!r} = {r2["c"]}')
        empty = q(session, f'MATCH (d:{DOC_LABEL}:{DS_LABEL}) '
                           "WHERE COALESCE(d.year_week,'') = '' "
                           'RETURN count(*) AS c')[0]['c']
        if empty:
            warn(f'year_week 가 빈 Document 가 {empty}개 — 주차 질문에서는 date 로 '
                 f'대체 판정된다(하이브리드 필터). date 도 없으면 영구 누락.')


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
        print(f'     {r["name"]:<38} {r["type"]:<10} '
              f'{r["labelsOrTypes"]} {r["properties"]} [{r["state"]}]')

    # 3-1. 벡터 인덱스가 Chunk 를 대상으로 하는가 — 가장 의심스러운 지점
    print()
    v = by_name.get(VECTOR_INDEX)
    if not v:
        fail(f'벡터 인덱스 {VECTOR_INDEX} 가 없다 — B채널(문서 벡터 검색) 전면 0건. '
             f'분리 구조로 옮길 때 재생성이 빠진 것으로 보인다.')
    else:
        labels = v['labelsOrTypes'] or []
        if v['state'] != 'ONLINE':
            fail(f'{VECTOR_INDEX} 상태가 {v["state"]} — 아직 populating 이면 결과가 비거나 부분적이다.')
        if CHUNK_LABEL in labels:
            ok(f'{VECTOR_INDEX} → {labels} {v["properties"]} (Chunk 대상, 코드 가정과 일치)')
        elif DOC_LABEL in labels:
            fail(f'{VECTOR_INDEX} 가 아직 {labels} 를 대상으로 한다. 임베딩은 Chunk 로 '
                 f'옮겨갔으므로 이 인덱스는 빈 결과만 돌려준다 → Chunk 대상으로 재생성 필요.')
        else:
            fail(f'{VECTOR_INDEX} 대상 라벨이 예상 밖: {labels}')

    # 3-2. FULLTEXT 인덱스들
    print()
    for name in CHUNK_FT_INDEXES + DOC_FT_INDEXES:
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
        expected = CHUNK_LABEL if name in CHUNK_FT_INDEXES else DOC_LABEL
        if expected not in labels:
            fail(f'{name} 대상이 {labels} — 기대: {expected} 포함.')
        else:
            ok(f'{name} → {labels} {r["properties"]}')
        # ★ 데이터셋 범위 문제: 인덱스가 :Document / :Chunk 전체를 대상으로 하면
        #   다른 데이터셋 노드가 limit 을 먼저 소비해 Confluence 결과가 밀려난다.
        if DS_LABEL not in labels:
            other = q(session, f'MATCH (n:{expected}) WHERE NOT n:{DS_LABEL} '
                               'RETURN count(n) AS c')[0]['c']
            if other:
                warn(f'{name} 은 :{expected} 전체를 대상으로 하는데, '
                     f':{DS_LABEL} 이 아닌 :{expected} 노드가 {other}개 있다. '
                     f'이들이 limit 을 먼저 소비하면 Confluence 결과가 희석된다 '
                     f'(node:{DS_LABEL} 가드는 인덱스 조회 *후* 필터라서 못 막는다).')

    # 3-3. 다른 데이터셋이 :Document 라벨을 공유하는지 (위 WARN 의 근거)
    print()
    for r in q(session, f'MATCH (n:{DOC_LABEL}) RETURN labels(n) AS labels, '
                        'count(*) AS c ORDER BY c DESC LIMIT 10'):
        print(f'     :{DOC_LABEL} 라벨 보유 → {sorted(r["labels"])} = {r["c"]}')


# ── 4. 실제 채널 재현 ────────────────────────────────────────────────────────
def check_channels(session, probe: str):
    head(f'4. 실제 쿼리 재현 (검색어: {probe!r})')

    chunk_rel = CHUNK_REL

    # C채널 — 청크 본문 FULLTEXT (기간 필터 없이)
    for name in CHUNK_FT_INDEXES:
        try:
            rows = q(session, f'''
                CALL db.index.fulltext.queryNodes($index, $q, {{limit: 5}})
                YIELD node, score
                WHERE node:{DS_LABEL}
                MATCH (d:{DOC_LABEL}:{DS_LABEL})-[:{chunk_rel}]->(node)
                RETURN node.chunk_id AS chunk_id, score,
                       COALESCE(d.title, d.page_title) AS title
                ORDER BY score DESC
            ''', index=name, q=f'"{probe}"')
            if rows:
                ok(f'{name}: {len(rows)}건')
                for r in rows[:3]:
                    print(f'          score={r["score"]:.2f} {r["chunk_id"]} / {r["title"]}')
            else:
                # 가드/조인 중 어디서 죽는지 분리해서 본다
                raw = q(session, 'CALL db.index.fulltext.queryNodes($index, $q, '
                                 '{limit: 5}) YIELD node RETURN count(*) AS c',
                        index=name, q=f'"{probe}"')[0]['c']
                if raw == 0:
                    warn(f'{name}: 인덱스 자체가 0건 (검색어가 본문에 없을 수도 있다)')
                else:
                    fail(f'{name}: 인덱스는 {raw}건 맞췄지만 '
                         f'{DS_LABEL} 가드 또는 {chunk_rel} 조인에서 전부 탈락.')
        except Exception as e:
            fail(f'{name} 쿼리 실패: {e}')

    # C채널 — 문서 제목 FULLTEXT → 청크 확장
    for name in DOC_FT_INDEXES:
        try:
            rows = q(session, f'''
                CALL db.index.fulltext.queryNodes($index, $q, {{limit: 5}})
                YIELD node, score
                WHERE node:{DS_LABEL}
                WITH node AS d, score
                MATCH (d)-[:{chunk_rel}]->(c:{CHUNK_LABEL}:{DS_LABEL})
                RETURN c.chunk_id AS chunk_id, score
                ORDER BY score DESC, c.chunk_index LIMIT 5
            ''', index=name, q=f'"{probe}"')
            (ok if rows else warn)(f'{name}: {len(rows)}건')
        except Exception as e:
            fail(f'{name} 쿼리 실패: {e}')

    # B채널 — 벡터 (임베딩 없이 인덱스 살아있는지만: 실제 Chunk 임베딩 하나를 재사용)
    try:
        emb = q(session, f'MATCH (c:{CHUNK_LABEL}:{DS_LABEL}) '
                         'WHERE c.embedding IS NOT NULL '
                         'RETURN c.embedding AS e LIMIT 1')
        if not emb:
            warn('임베딩을 가진 Chunk 가 없어 벡터 검색 재현을 건너뜀 (2번 항목 참고).')
        else:
            rows = q(session, f'''
                CALL db.index.vector.queryNodes($index, 5, $e)
                YIELD node, score
                WHERE node:{DS_LABEL}
                MATCH (d:{DOC_LABEL}:{DS_LABEL})-[:{chunk_rel}]->(node)
                RETURN node.chunk_id AS chunk_id, score
                ORDER BY score DESC
            ''', index=VECTOR_INDEX, e=emb[0]['e'])
            if rows:
                ok(f'{VECTOR_INDEX}: {len(rows)}건 (자기 임베딩으로 재현 — '
                   f'top score={rows[0]["score"]:.3f})')
            else:
                fail(f'{VECTOR_INDEX}: 실제 Chunk 임베딩을 넣어도 0건 — '
                     f'인덱스가 Chunk 를 대상으로 하지 않거나 아직 populating.')
    except Exception as e:
        fail(f'{VECTOR_INDEX} 쿼리 실패: {e}')


# ── 5. 샘플 덤프 ─────────────────────────────────────────────────────────────
def dump_sample(session):
    head('5. Document / Chunk 샘플 (속성 이름 눈으로 확인)')
    rows = q(session, f'''
        MATCH (d:{DOC_LABEL}:{DS_LABEL})-[:{CHUNK_REL}]->(c:{CHUNK_LABEL}:{DS_LABEL})
        RETURN d, c LIMIT 1
    ''')
    if not rows:
        warn('조인되는 Document-Chunk 쌍이 없어 샘플을 뽑을 수 없다.')
        return
    for alias in ('d', 'c'):
        node = rows[0][alias]
        print(f'\n  -- {alias} ({sorted(node.labels)}) --')
        for k, v in sorted(dict(node).items()):
            if isinstance(v, list) and len(v) > 8:
                print(f'     {k:<16} = <list len={len(v)}>')
            else:
                s = str(v).replace('\n', '\\n')
                print(f'     {k:<16} = {s[:120]}{"..." if len(s) > 120 else ""}')


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
        docs, chunks = check_counts(session)
        if docs or chunks:
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
        print('  (c) 키워드 추출 단계에서 target_datasets 에 Confluence 가 '
              '안 들어간 것 중 하나다.')
        print('  앱을 디버그 레벨 1 이상으로 띄워 채널별 건수(A~F)를 확인해 주세요.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
