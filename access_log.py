"""
access_log.py
-----------------------------------------------------------------------------
접속(IP/시간)과 검색(질문/데이터셋/모드) 기록을 파일에 남기는 간단한 로거.

- logs/access.log : HTTP 요청(페이지 접속) 로그 — 시간, IP, 메서드, 경로
- logs/search.log : 검색(질문) 로그 — 시간, IP, 데이터셋, 모드, 질문 내용

회전(rotating) 파일 핸들러를 써서 파일이 무한정 커지지 않게 한다
(파일당 5MB, 최근 5개까지 보관).
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).parent / 'logs'
_LOG_DIR.mkdir(exist_ok=True)


def _make_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False   # 루트 로거(터미널)로 전파해 중복 출력되지 않게
    if not logger.handlers:    # 모듈이 여러 번 import 돼도 핸들러 중복 추가 방지
        handler = RotatingFileHandler(
            _LOG_DIR / filename, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter('%(asctime)s\t%(message)s'))
        logger.addHandler(handler)
    return logger


access_logger = _make_logger('kmap.access', 'access.log')
search_logger = _make_logger('kmap.search', 'search.log')


def log_access(ip: str, method: str, path: str) -> None:
    access_logger.info(f"{ip}\t{method}\t{path}")


def log_search(ip: str, dataset: str, mode: str, query: str) -> None:
    # 탭(\t)이나 줄바꿈이 질문에 섞여 로그 한 줄 형식이 깨지지 않도록 정리
    safe_query = query.replace('\t', ' ').replace('\n', ' ')
    search_logger.info(f"{ip}\t{dataset}\t{mode}\t{safe_query}")


def client_ip(request) -> str:
    """
    요청의 실제 클라이언트 IP를 뽑는다.

    ★ 리버스 프록시(nginx 등) 뒤에서 실행되면 request.client.host 는 프록시 자신의
    IP만 보여준다. X-Forwarded-For 헤더가 있으면(프록시가 세팅) 그 안의 "첫 번째"
    IP(원 클라이언트)를 우선 사용한다.
    """
    forwarded = request.headers.get('x-forwarded-for')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.client.host if request.client else 'unknown'
