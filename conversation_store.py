"""
conversation_store.py
-----------------------------------------------------------------------------
대화 히스토리를 SQLite 에 영구 저장한다 (새로고침/재접속/서버 재시작에도 유지).

[왜 SQLite 인가]
  - 파이썬 내장(sqlite3)이라 별도 설치·서버 운영이 필요 없다.
  - 파일 하나로 관리되어 백업/이관이 쉽다.
  - 연구팀 규모(동시 사용자 수십 명, 대화 수만 건)에는 성능이 충분하다.
  - Neo4j 가 이미 있지만 그건 "그래프" 저장소다. append-only 대화 로그는
    관계형/문서형이 훨씬 자연스럽고, 지식그래프와 성격이 달라 섞지 않는 게 좋다.
  - 나중에 동시성이 커지면 Postgres 로 옮기기 쉬운 구조로 유지한다.

[WSL 주의]
  DB 파일은 반드시 리눅스 파일시스템(예: 프로젝트 폴더)에 두어야 한다.
  /mnt/c/... (윈도우 드라이브 마운트)에 두면 SQLite 파일 락이 제대로 동작하지
  않아 "database is locked" 오류나 파일 손상이 발생할 수 있다.

[사용자 식별]
  user_key 는 IP 가 아니라 "브라우저 쿠키 기반 식별자"를 쓴다 (chat_page.py 참고).
  IP 를 키로 쓰면 안 되는 이유:
    - WSL: 윈도우 호스트 브라우저에서 접속하면 전부 게이트웨이 IP 하나로 보여
      사용자 구분이 불가능하다(실측 확인됨: 172.31.176.1).
    - 사내망 NAT/프록시: 여러 사람이 같은 IP 로 나타난다.
    - DHCP 로 IP 가 바뀌면 그동안의 대화 기록을 잃는다.
  IP 는 감사(audit) 목적으로 last_ip 컬럼에 함께 기록만 해둔다.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

_DB_PATH = Path(__file__).parent / 'data' / 'conversations.db'
_DB_PATH.parent.mkdir(exist_ok=True)

# NiceGUI 는 비동기 단일 프로세스이지만 핸들러가 여러 스레드에서 호출될 수 있어
# (run_in_executor 등), 커넥션 하나를 락으로 감싸 직렬화한다.
_lock = threading.Lock()
_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row


def _now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def init_db() -> None:
    with _lock:
        # WAL: 읽기/쓰기 동시성이 좋아지고 쓰기 중 읽기가 막히지 않는다.
        _conn.execute('PRAGMA journal_mode=WAL')
        _conn.execute('PRAGMA foreign_keys=ON')
        _conn.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_key   TEXT NOT NULL,
                title      TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_ip    TEXT
            )
        ''')
        _conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id    INTEGER NOT NULL
                           REFERENCES conversations(id) ON DELETE CASCADE,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                dataset    TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        # 사용자별 최근 대화 목록 조회를 빠르게
        _conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_conv_user
            ON conversations(user_key, updated_at DESC)
        ''')
        _conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_msg_conv
            ON messages(conv_id, id)
        ''')
        _conn.commit()


def create_conversation(user_key: str, title: str, ip: str = None) -> int:
    ts = _now()
    with _lock:
        cur = _conn.execute(
            'INSERT INTO conversations (user_key, title, created_at, updated_at, last_ip) '
            'VALUES (?, ?, ?, ?, ?)',
            (user_key, title, ts, ts, ip),
        )
        _conn.commit()
        return cur.lastrowid


def add_message(conv_id: int, role: str, content: str, dataset=None) -> None:
    """
    메시지 한 건을 저장한다.

    dataset 은 "All" 문자열이거나 선택된 데이터셋 key 들의 리스트일 수 있어
    JSON 으로 직렬화해 보관한다(불러올 때 그대로 복원됨).

    ★ graph_id(서브그래프)는 일부러 저장하지 않는다 — 그래프 HTML 은 임시
      디렉터리에 생성되어 서버 재시작 시 사라지므로, 저장해두면 나중에 열었을 때
      깨진 링크가 된다. 대화를 다시 불러오면 텍스트만 복원된다.
    """
    with _lock:
        _conn.execute(
            'INSERT INTO messages (conv_id, role, content, dataset, created_at) '
            'VALUES (?, ?, ?, ?, ?)',
            (conv_id, role, content, json.dumps(dataset, ensure_ascii=False), _now()),
        )
        _conn.execute(
            'UPDATE conversations SET updated_at = ? WHERE id = ?', (_now(), conv_id)
        )
        _conn.commit()


def list_conversations(user_key: str, limit: int = 50) -> list[dict]:
    """해당 사용자의 최근 대화 목록 (최신순)."""
    with _lock:
        # ★ id DESC 타이브레이커: updated_at 이 초 단위라 같은 초에 만들어진 대화들이
        #   동률이 되어 순서가 뒤집히는 문제가 있었다. id 가 큰 쪽이 항상 더 최근이므로
        #   2차 정렬 기준으로 넣어 순서를 확정한다.
        rows = _conn.execute(
            'SELECT id, title, updated_at FROM conversations '
            'WHERE user_key = ? ORDER BY updated_at DESC, id DESC LIMIT ?',
            (user_key, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def load_messages(conv_id: int) -> list[dict]:
    """대화의 메시지들을 시간순으로 복원한다."""
    with _lock:
        rows = _conn.execute(
            'SELECT role, content, dataset FROM messages WHERE conv_id = ? ORDER BY id',
            (conv_id,),
        ).fetchall()
    msgs = []
    for r in rows:
        try:
            dataset = json.loads(r['dataset']) if r['dataset'] else 'All'
        except Exception:
            dataset = 'All'
        msgs.append({'role': r['role'], 'content': r['content'], 'dataset': dataset})
    return msgs


def delete_conversation(conv_id: int, user_key: str) -> None:
    """본인 소유 대화만 삭제 (user_key 를 함께 조건에 넣어 타인 대화 삭제를 막는다)."""
    with _lock:
        _conn.execute(
            'DELETE FROM conversations WHERE id = ? AND user_key = ?', (conv_id, user_key)
        )
        _conn.commit()


def touch_ip(conv_id: int, ip: str) -> None:
    """접속 IP 갱신 (감사 목적)."""
    with _lock:
        _conn.execute('UPDATE conversations SET last_ip = ? WHERE id = ?', (ip, conv_id))
        _conn.commit()
