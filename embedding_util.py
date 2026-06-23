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


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
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
            all_embeddings.extend([[0.0] * EMBED_DIM] * len(batch))
    return all_embeddings


def get_embedding(text: str) -> list[float]:
    result = get_embeddings_batch([text])
    return result[0] if result else [0.0] * EMBED_DIM
