# UI defaults — actual search config lives in rag_engine.py (DATASETS dict)
# These are read by the UI layer only.

DEFAULT_SEARCH_MODE = "hybrid"   # "hybrid" | "vector" | "text"
DEFAULT_SEARCH_HOPS = 2          # 1 or 2

# TODO: load allow_2hop per-relation from pipelines/papers/ontology.json at runtime
# ONTOLOGY_PATH = "pipelines/papers/ontology.json"
