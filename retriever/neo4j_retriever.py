from __future__ import annotations
from config import NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD


class Neo4jRetriever:
    def __init__(self):
        # TODO: initialize neo4j.GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        pass

    def search(self, query: str, hops: int = 1) -> list[dict]:
        # TODO: BGE-M3 dense + rank_bm25 fulltext + RRF, optionally 2-hop expansion
        return []
