import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv(Path(__file__).parent / '.env')

EMBED_API_URL    = "https://apigw.samsungds.net:8443/embedding/1/v1/embeddings"
EMBED_TICKET     = os.getenv("EMBEDDING_DEP_TICKET", "")
EMBED_DIM        = 1024
EMBED_BATCH_SIZE = 64
EMBED_MODEL      = "BGE-M3"

_ssl_env = os.getenv("SSL_CERT_FILE", "False")
SSL_CERT = False if _ssl_env.lower() in ("false", "0", "") else _ssl_env


def node_text(name: str, node_type: str) -> str:
    return f"{name} ({node_type})"


def get_embeddings_batch(texts: list[str]) -> list[list[float] | None]:
    """
    각 텍스트에 대응하는 임베딩 벡터 리스트를 반환한다.

    ★ API 호출이 실패한 배치는 예전엔 [0.0]*EMBED_DIM(0벡터)로 채워 넣었는데,
    이 0벡터를 그대로 Neo4j 벡터 검색(db.index.vector.queryNodes)에 넘기면
    "Vector must only contain finite values, and have positive and finite
    l2-norm" 에러가 발생한다(0벡터는 노름이 0이라 코사인 유사도 계산이 불가능).
    즉 "API 실패를 조용히 넘기려던" 폴백이 오히려 더 알아보기 힘든 2차 에러를
    유발했다. 이제 실패한 항목은 None 으로 표시해 호출부가 "임베딩을 못 구했다"는
    걸 명확히 알고 그 항목의 벡터 검색을 건너뛸 수 있게 한다.
    """
    if not texts:
        return []
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        try:
            response = requests.post(
                EMBED_API_URL,
                headers={'Content-Type': 'application/json', 'x-dep-ticket': EMBED_TICKET},
                json={'model': EMBED_MODEL, 'input': batch},
                timeout=60,
                proxies={'http': None, 'https': None},
                verify=False,
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.extend([item['embedding'] for item in data['data']])
        except Exception as e:
            print(f"임베딩 오류 (batch {i//EMBED_BATCH_SIZE + 1}): {e}")
            all_embeddings.extend([None] * len(batch))
    return all_embeddings


def get_embedding(text: str) -> list[float] | None:
    """임베딩 벡터를 반환한다. API 호출 실패 시 None (호출부가 검색을 건너뛰어야 함)."""
    result = get_embeddings_batch([text])
    return result[0] if result else None
