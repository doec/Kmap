from __future__ import annotations
from config import QDRANT_URL


class QdrantRetriever:
    def __init__(self):
        # TODO: initialize qdrant_client.QdrantClient(url=QDRANT_URL)
        pass

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        collections: list[str] | None = None,
    ) -> list[dict]:
        # TODO: BGE-M3 dense + FastEmbed BM25 sparse + RRF
        return []
