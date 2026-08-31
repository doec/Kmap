# =============================================================================
# llm_util.py
# -----------------------------------------------------------------------------
# Samsung DS 내부 LLM API 호출 유틸리티
#
# [설계 개요]
#   - 모듈 import 시 _LLM_CONFIGS에 정의된 모든 모델의 OpenAI client를 미리 생성
#   - 이후 ask_llm() 등 호출마다 client를 새로 만들지 않고 캐시에서 재사용
#   - Samsung 내부망 특성 (x-dep-ticket 인증, SSL 검증 비활성화, 프록시 우회) 반영
#
# [지원 함수]
#   ask_llm()                    : 일반 텍스트 호출 (non-streaming)
#   ask_llm_img()                : 이미지 포함 호출
#   ask_llm_stream()             : streaming 호출, 완성된 문자열 반환 (504 timeout 방지용)
#   ask_llm_stream_iter()        : streaming 호출, chunk 단위 yield (실시간 UI 출력용)
#   ask_llm_messages()           : messages 리스트를 직접 받는 호출 (대화 히스토리 포함용)
#   ask_llm_stream_iter_messages(): messages 리스트 + streaming, chunk 단위 yield
#   ask_llm_async()              : 비동기 호출 (async/await 지원)
#   ask_llm_stream_iter_async()  : 비동기 streaming 호출
#   get_client()                 : 캐시된 (OpenAI client, cfg) 반환 (외부 직접 사용용)
#   get_async_client()           : 캐시된 (AsyncOpenAI client, cfg) 반환
#   cleanup_clients()            : 모든 client 정리 (메모리 해제)
#
# [.env 설정 예시]
#   LLM=gpt-prod                          # 기본 모델 (생략 시 gpt-prod)
#   USER_ID=KNOX_ID
#   SEND_SYSTEM_NAME=my_system
#
#   LLM_MODEL=openai/gpt-oss-120b
#   LLM_API_URL_prod=http://apigw.samsungds.net:.../v1
#   CREDENTIAL_KEY_prod=credential:TICKET-...
#
#   LLM_API_URL_stg=http://apigw-stg.samsungds.net:.../v1
#   CREDENTIAL_KEY_stg=credential:TICKET-...
#
#   LLM_MODEL_GaussO41=GaussO4.1-260330
#   LLM_API_URL_GaussO41=http://...
#   CREDENTIAL_KEY_GaussO41=credential:TICKET-...
#   SEND_SYSTEM_NAME_GaussO41=my_system_gauss
#
#   LLM_MODEL_Gemma4=Gemma4-260430
#   LLM_API_URL_Gemma4=http://...
#   CREDENTIAL_KEY_Gemma4=credential:TICKET-...
#   SEND_SYSTEM_NAME_Gemma4=my_system_gemma
#
# [기본 사용법]
#   from llm_util import ask_llm, ask_llm_stream, ask_llm_stream_iter
#
#   result = ask_llm("요약해줘")                    # 기본 모델 (.env의 LLM)
#   result = ask_llm("요약해줘", llm="Gemma4")      # 경량 모델로 빠른 요약
#   result = ask_llm("트리플 추출", llm="GaussO4.1") # 고성능 모델로 정밀 추출
# =============================================================================

import os
import re
import uuid
import base64
import logging
import asyncio
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv(override=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# httpx 가 매 요청마다 "HTTP Request: POST ... 200 OK" 를 INFO 레벨로 찍는데,
# basicConfig 의 level=INFO 가 httpx 로거에도 그대로 적용되어 매 LLM 호출마다
# 출력된다. 요청 성공 로그는 불필요한 잡음이므로 httpx 로거만 WARNING 이상으로 올려
# 에러(연결 실패 등)만 보이게 한다.
logging.getLogger("httpx").setLevel(logging.WARNING)

# .env의 LLM 값이 ask_llm(..., llm=None) 호출 시 기본값으로 사용됨
LLM = os.getenv("LLM", "gpt-prod")
logger.info(f"LLM model (default): {LLM}")

UserID           = os.getenv('USER_ID')
Send_System_Name = os.getenv('SEND_SYSTEM_NAME')


def _short_err(e: Exception, maxlen: int = 300) -> str:
    """
    예외 메시지를 로그에 찍기 좋게 잘라준다.

    ★ 사내 API 게이트웨이가 503 등 오류 시 HTML 에러 페이지 전체(CSS/SVG 포함,
    수십~수백 줄)를 응답 본문으로 돌려주는 경우가 있는데, openai 라이브러리가
    그 본문을 예외 메시지에 그대로 담아서 터미널 로그가 심하게 어지러워진다.
    핵심(상태/원인)만 보이도록 앞부분만 자른다.
    """
    msg = str(e)
    return msg if len(msg) <= maxlen else msg[:maxlen] + f"... (총 {len(msg)}자, 이하 생략)"


def _validate_env_vars():
    """
    필수 환경 변수 검증

    Raises:
        ValueError: 필수 환경 변수가 누락된 경우
    """
    required_vars = ["USER_ID", "SEND_SYSTEM_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(
            f"필수 환경 변수 누락: {', '.join(missing_vars)}. "
            f".env 파일에 해당 변수를 설정해주세요."
        )

    logger.info("필수 환경 변수 검증 완료")

# 모듈 로드 시 환경 변수 검증
_validate_env_vars()


# =============================================================================
# 지원 LLM 목록
# -----------------------------------------------------------------------------
# 새 모델 추가 시 여기에만 항목을 추가하면 됨.
# 코드 다른 곳은 수정 불필요.
#
# 각 항목의 의미:
#   model_env    : 모델명을 담은 .env 키
#   model_default: .env에 해당 키가 없을 때 사용할 기본 모델명
#   url_env      : API base URL을 담은 .env 키
#   cred_env     : credential key를 담은 .env 키 (x-dep-ticket 헤더값)
#   sysname_env  : Send-System-Name 헤더값을 담은 .env 키
#                  None이면 공통값 SEND_SYSTEM_NAME 사용
# =============================================================================
_LLM_CONFIGS = {
    "gpt-stg": {
        "model_env"    : "LLM_MODEL",
        "model_default": "openai/gpt-oss-120b",
        "url_env"      : "LLM_API_URL_stg",
        "cred_env"     : "CREDENTIAL_KEY_stg",
        "sysname_env"  : None,                   # 공통 SEND_SYSTEM_NAME 사용
    },
    "gpt-prod": {
        "model_env"    : "LLM_MODEL",
        "model_default": "openai/gpt-oss-120b",
        "url_env"      : "LLM_API_URL_prod",
        "cred_env"     : "CREDENTIAL_KEY_prod",
        "sysname_env"  : None,
    },
    "GaussO4.1": {
        "model_env"    : "LLM_MODEL_GaussO41",
        "model_default": "GaussO4.1-260330",
        "url_env"      : "LLM_API_URL_GaussO41",
        "cred_env"     : "CREDENTIAL_KEY_GaussO41",
        "sysname_env"  : "SEND_SYSTEM_NAME_GaussO41",  # 모델 전용 system name
    },
    "Gemma4": {
        "model_env"    : "LLM_MODEL_Gemma4",
        "model_default": "Gemma4-260430",
        "url_env"      : "LLM_API_URL_Gemma4",
        "cred_env"     : "CREDENTIAL_KEY_Gemma4",
        "sysname_env"  : "SEND_SYSTEM_NAME_Gemma4",
    },
    # ★ 로컬(사내망) vLLM 서버로 띄운 Qwen — 삼성 게이트웨이(gpt-*/GaussO4.1/Gemma4)와
    #   달리 x-dep-ticket 인증이 필요 없는 순수 OpenAI 호환 서버다. cred_env 는
    #   .env 에 없으면 빈 문자열이 되는데, _build_client() 가 그 값을 그대로
    #   x-dep-ticket 헤더에 넣어 보내도 vLLM 은 모르는 헤더라 그냥 무시하므로 무해하다.
    "Qwen3.8": {
        "model_env"    : "LLM_MODEL_Qwen38",
        "model_default": "Qwen/Qwen3.8-27B",
        "url_env"      : "LLM_API_URL_Qwen38",
        "cred_env"     : "CREDENTIAL_KEY_Qwen38",
        "sysname_env"  : None,
    },
    # ★ 로컬 llama.cpp 서버(GGUF)로 띄운 MiniMax-M3 — Qwen3.8과 마찬가지로 순수
    #   OpenAI 호환 서버이지만, 이쪽은 실제 API 키 인증이 필요하다(Qwen3.8은
    #   무인증). cred_env 값은 _build_client()에서 이제 실제 Authorization: Bearer
    #   헤더로 쓰인다(아래 _build_client 변경 참고).
    "MiniMax-M3": {
        "model_env"    : "LLM_MODEL_MiniMaxM3",
        "model_default": r".\MiniMax-M3-IQ4_XS\MiniMax-M3-IQ4_MS-00001-of-00006.gguf",
        "url_env"      : "LLM_API_URL_MiniMaxM3",
        "cred_env"     : "CREDENTIAL_KEY_MiniMaxM3",
        "sysname_env"  : None,
    },
}


def _resolve_llm_config(llm: str | None) -> dict:
    """
    llm 이름을 받아 실제 설정값(model명, base_url, credential 등)을 반환.
    llm=None이면 .env의 LLM 기본값 사용.
    """
    target = llm if llm is not None else LLM

    if target not in _LLM_CONFIGS:
        raise ValueError(
            f"Unknown LLM: '{target}'. "
            f"지원 목록: {list(_LLM_CONFIGS.keys())}"
        )

    cfg = _LLM_CONFIGS[target]

    model_name     = os.getenv(cfg["model_env"], cfg["model_default"])
    api_base       = os.getenv(cfg["url_env"], "")
    credential_key = os.getenv(cfg["cred_env"], "")
    sysname        = (
        os.getenv(cfg["sysname_env"], Send_System_Name)
        if cfg["sysname_env"]
        else Send_System_Name
    )

    # openai 라이브러리는 base_url에 /chat/completions를 자동으로 붙이므로
    # .env에 full URL이 들어있는 경우 /chat/completions 이전까지만 잘라냄
    base_url = api_base
    if "chat/completions" in api_base:
        base_url = api_base.split("chat/completions")[0].rstrip("/")

    return {
        "model"           : model_name,
        "base_url"        : base_url,
        "credential_key"  : credential_key,
        "send_system_name": sysname,
    }


# =============================================================================
# 모델별 client 캐시
# -----------------------------------------------------------------------------
# _CLIENT_CACHE 구조:
#   { "gpt-prod": (OpenAI_client, cfg_dict),
#     "GaussO4.1": (OpenAI_client, cfg_dict), ... }
#
# _ASYNC_CLIENT_CACHE 구조:
#   { "gpt-prod": (AsyncOpenAI_client, cfg_dict), ... }
#
# 모듈 import 시 _init_clients()가 자동 실행되어 모든 모델의 client를 미리 생성.
# 이후 ask_llm() 등은 매번 새 client를 만들지 않고 캐시에서 꺼내 재사용.
#
# 장점:
#   - client 객체 생성 오버헤드 제거 (B01처럼 수백 건 배치 처리 시 유리)
#   - get_client()로 외부에서도 client를 직접 꺼내 쓸 수 있어 확장성 좋음
# =============================================================================
_CLIENT_CACHE: dict[str, tuple[OpenAI, dict]] = {}
_ASYNC_CLIENT_CACHE: dict[str, tuple[AsyncOpenAI, dict]] = {}


def _build_client(cfg: dict) -> OpenAI:
    """
    OpenAI client 생성 (내부용).

    Samsung 내부망 설정:
      - api_key   : openai 라이브러리 필수값. Samsung 게이트웨이는 실제 인증을
                    x-dep-ticket 헤더로 처리하므로 credential_key 가 없으면
                    더미값 "api_key" 를 쓴다(DS API HUB 예제 동일 방식).
                    ★ 반면 MiniMax-M3 처럼 표준 OpenAI 호환 서버는 Authorization:
                    Bearer 헤더로 인증하는데, openai 라이브러리는 여기 넘긴
                    api_key 로 그 헤더를 자동 생성해준다 — 그래서 credential_key
                    가 있으면(=.env 에 실제 키가 설정돼 있으면) 그대로 api_key 로
                    쓴다. Samsung 게이트웨이 쪽은 Authorization 헤더를 안 보므로
                    이렇게 바꿔도 기존 동작에 영향 없다.
      - verify    : False → Samsung 내부망 self-signed 인증서 검증 비활성화
      - mounts    : None  → 삼성 사내 프록시 우회, 직접 연결
                    (httpx 0.28+ 문법. 이전 버전은 proxies= 사용)
      - timeout   : 최대한 긴 타임아웃 설정 (연결 60초, 읽기 600초)
    """
    headers = {
        'x-dep-ticket'    : cfg["credential_key"],   # DS API HUB 인증 티켓
        'Send-System-Name': cfg["send_system_name"],  # 시스템 식별자
        'User-Id'         : UserID,                   # Knox ID
        'User-Type'       : "AD_ID",                  # DS API HUB 고정값
        'Prompt-Msg-Id'   : str(uuid.uuid4()),         # 요청 추적용 UUID
        'Completion-Msg-Id': str(uuid.uuid4()),        # 응답 추적용 UUID
    }
    http_client = httpx.Client(
        verify=False,                                  # 내부망 SSL 검증 비활성화
        mounts={"http://": None, "https://": None},    # 프록시 우회
        timeout=httpx.Timeout(600.0, connect=60.0),    # 최대한 긴 타임아웃 설정
    )
    return OpenAI(
        api_key=cfg["credential_key"] or "api_key",
        base_url=cfg["base_url"],
        default_headers=headers,
        http_client=http_client,
    )


def _build_async_client(cfg: dict) -> AsyncOpenAI:
    """
    비동기 OpenAI client 생성 (내부용).

    Parameters:
        cfg: LLM 설정 딕셔너리

    Returns:
        AsyncOpenAI: 비동기 클라이언트 인스턴스
    """
    headers = {
        'x-dep-ticket'    : cfg["credential_key"],
        'Send-System-Name': cfg["send_system_name"],
        'User-Id'         : UserID,
        'User-Type'       : "AD_ID",
        'Prompt-Msg-Id'   : str(uuid.uuid4()),
        'Completion-Msg-Id': str(uuid.uuid4()),
    }
    http_client = httpx.AsyncClient(
        verify=False,
        mounts={"http://": None, "https://": None},
        timeout=httpx.Timeout(600.0, connect=60.0),
    )
    return AsyncOpenAI(
        api_key=cfg["credential_key"] or "api_key",
        base_url=cfg["base_url"],
        default_headers=headers,
        http_client=http_client,
    )


def _init_clients():
    """
    모듈 로드 시 _LLM_CONFIGS의 모든 모델 client를 미리 생성해 캐시에 저장.
    .env에 해당 환경변수가 없는 모델은 경고만 출력하고 건너뜀.
    """
    for llm_name in _LLM_CONFIGS:
        try:
            cfg = _resolve_llm_config(llm_name)
            _CLIENT_CACHE[llm_name] = (_build_client(cfg), cfg)
            _ASYNC_CLIENT_CACHE[llm_name] = (_build_async_client(cfg), cfg)
            logger.info(f"{llm_name} client 초기화 완료")
        except Exception as e:
            logger.warning(f"{llm_name} client 초기화 실패: {e}")

_init_clients()  # import 시 자동 실행


def cleanup_clients():
    """
    모든 client 정리 및 메모리 해제.
    프로그램 종료 시 호출하거나, client 재생성이 필요할 때 사용.
    """
    for llm_name, (client, _) in _CLIENT_CACHE.items():
        try:
            client.close()
            logger.info(f"{llm_name} client 정리 완료")
        except Exception as e:
            logger.error(f"{llm_name} client 정리 실패: {e}")

    _CLIENT_CACHE.clear()
    _ASYNC_CLIENT_CACHE.clear()
    logger.info("모든 client 캐시 정리 완료")


def get_client(llm: str | None = None) -> tuple[OpenAI, dict]:
    """
    캐시된 (OpenAI client, cfg) 튜플을 반환.
    llm=None이면 .env의 LLM 기본값 사용.

    ask_llm() 같은 내부 함수 외에, tool calling이나 LangGraph 연동처럼
    client를 직접 다뤄야 할 때 외부에서 호출하면 됨.

    사용 예시:
        # 기본 모델 client 가져오기
        client, cfg = get_client()
        client.chat.completions.create(model=cfg["model"], messages=[...])

        # 특정 모델 client 가져오기
        client, cfg = get_client("GaussO4.1")
        client.chat.completions.create(
            model=cfg["model"],
            messages=[...],
            tools=[...],       # tool calling
        )

        # LangChain ChatOpenAI와 함께 쓸 때
        from langchain_openai import ChatOpenAI
        client, cfg = get_client("Gemma4")
        llm = ChatOpenAI(model=cfg["model"], ...)

    Raises:
        ValueError: 지원하지 않는 LLM 이름인 경우
    """
    target = llm if llm is not None else LLM
    if target not in _CLIENT_CACHE:
        raise ValueError(
            f"Unknown LLM: '{target}'. "
            f"지원 목록: {list(_CLIENT_CACHE.keys())}"
        )
    return _CLIENT_CACHE[target]


def get_async_client(llm: str | None = None) -> tuple[AsyncOpenAI, dict]:
    """
    캐시된 (AsyncOpenAI client, cfg) 튜플을 반환.
    llm=None이면 .env의 LLM 기본값 사용.

    비동기 함수에서 사용합니다.

    Raises:
        ValueError: 지원하지 않는 LLM 이름인 경우
    """
    target = llm if llm is not None else LLM
    if target not in _ASYNC_CLIENT_CACHE:
        raise ValueError(
            f"Unknown LLM: '{target}'. "
            f"지원 목록: {list(_ASYNC_CLIENT_CACHE.keys())}"
        )
    return _ASYNC_CLIENT_CACHE[target]


# =============================================================================
# ask_llm — 일반 텍스트 호출 (non-streaming)
# -----------------------------------------------------------------------------
# 가장 기본적인 호출 방식. 응답이 완전히 완성된 후 문자열로 반환.
# 응답 시간이 길면 게이트웨이 504 timeout이 발생할 수 있음
# → 긴 작업(트리플 추출 등)은 ask_llm_stream() 사용 권장
#
# 파라미터:
#   USER_MESSAGE     : 사용자 입력 메시지
#   SYSTEM_PROMPT    : 시스템 프롬프트 (기본값: "Semiconductor related workers")
#   temperature      : 생성 다양성 (0.0~1.0, 기본값 0.05 → 결정적 출력)
#   reasoning_effort : 추론 깊이 ("low" / "medium" / "high", 기본값 "medium")
#   llm              : 사용할 모델명. None이면 .env의 LLM 기본값
#                      ex) "gpt-prod", "gpt-stg", "GaussO4.1", "Gemma4"
#
# 반환값:
#   str  : LLM 응답 문자열 (```json 코드블록 자동 제거)
#   None : 호출 실패 시
#
# 사용 예시:
#   # 기본 모델로 호출
#   result = ask_llm("MIM 커패시터의 주요 특성을 설명해줘")
#
#   # 경량 모델로 빠른 요약
#   result = ask_llm("이 논문 한 줄 요약해줘", llm="Gemma4")
#
#   # 고성능 모델로 JSON 추출
#   result = ask_llm(USER_MESSAGE, SYSTEM_PROMPT, llm="GaussO4.1")
#   data   = json.loads(result)
# =============================================================================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
def ask_llm(USER_MESSAGE, SYSTEM_PROMPT="Semiconductor related workers",
            temperature=0.05, reasoning_effort="medium", llm=None) -> Optional[str]:
    """
    일반 텍스트 호출 (non-streaming).

    Parameters:
        USER_MESSAGE: 사용자 입력 메시지
        SYSTEM_PROMPT: 시스템 프롬프트 (기본값: "Semiconductor related workers")
        temperature: 생성 다양성 (0.0~1.0, 기본값 0.05)
        reasoning_effort: 추론 깊이 ("low" / "medium" / "high", 기본값 "medium")
        llm: 사용할 모델명. None이면 .env의 LLM 기본값

    Returns:
        str: LLM 응답 문자열 (```json 코드블록 자동 제거)
        None: 호출 실패 시

    Note:
        응답 시간이 길면 게이트웨이 504 timeout이 발생할 수 있음.
        긴 작업(트리플 추출 등)은 ask_llm_stream() 사용 권장.
    """
    client, cfg = get_client(llm)

    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_MESSAGE},
            ],
            temperature=temperature,
            extra_body={"reasoning_effort": reasoning_effort},
        )
    except httpx.TimeoutException:
        logger.error(f"LLM 호출 타임아웃: {llm or LLM}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP 에러 {e.response.status_code}: {e}")
        return None
    except (AttributeError, IndexError) as e:
        logger.error(f"응답 파싱 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM 호출 실패 ({type(e).__name__}): {_short_err(e)}")
        return None

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        logger.error(f"응답에서 content 추출 실패: {e}")
        return None

    return strip_json_codeblock(content)


# =============================================================================
# ask_llm_img — 이미지 포함 호출
# -----------------------------------------------------------------------------
# 이미지를 base64로 인코딩해 LLM에 함께 전달.
# 지원 포맷: jpg, jpeg, png, gif, webp
#
# 사용 예시:
#   # TEM 이미지 분석
#   result = ask_llm_img("이 TEM 이미지에서 계면층 두께를 추정해줘", "./tem.png")
#
#   # 특정 모델로 호출
#   result = ask_llm_img("그래프의 x축 값을 읽어줘", "./graph.jpg", llm="gpt-prod")
# =============================================================================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
def ask_llm_img(USER_MESSAGE, IMAGE_PATH,
                SYSTEM_PROMPT="Semiconductor related workers",
                temperature=0.05, reasoning_effort="medium", llm=None) -> Optional[str]:
    """
    이미지 포함 호출.

    Parameters:
        USER_MESSAGE: 사용자 입력 메시지
        IMAGE_PATH: 이미지 파일 경로
        SYSTEM_PROMPT: 시스템 프롬프트 (기본값: "Semiconductor related workers")
        temperature: 생성 다양성 (0.0~1.0, 기본값 0.05)
        reasoning_effort: 추론 깊이 ("low" / "medium" / "high", 기본값 "medium")
        llm: 사용할 모델명. None이면 .env의 LLM 기본값

    Returns:
        str: LLM 응답 문자열 (```json 코드블록 자동 제거)
        None: 호출 실패 시

    Note:
        지원 포맷: jpg, jpeg, png, gif, webp
    """
    client, cfg = get_client(llm)

    try:
        with open(IMAGE_PATH, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"이미지 파일을 찾을 수 없음: {IMAGE_PATH}")
        return None
    except Exception as e:
        logger.error(f"이미지 파일 읽기 실패: {e}")
        return None

    ext = str(IMAGE_PATH).rsplit(".", 1)[-1].lower()
    mime_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png",  "gif":  "image/gif",
        "webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/jpeg")

    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": USER_MESSAGE},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }},
                    ],
                },
            ],
            temperature=temperature,
            extra_body={"reasoning_effort": reasoning_effort},
        )
    except httpx.TimeoutException:
        logger.error(f"LLM 호출 타임아웃: {llm or LLM}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP 에러 {e.response.status_code}: {e}")
        return None
    except (AttributeError, IndexError) as e:
        logger.error(f"응답 파싱 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM 호출 실패 ({type(e).__name__}): {_short_err(e)}")
        return None

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        logger.error(f"응답에서 content 추출 실패: {e}")
        return None

    return strip_json_codeblock(content)


# =============================================================================
# ask_llm_stream — streaming 호출, 완성된 문자열 반환
# -----------------------------------------------------------------------------
# 내부적으로는 streaming으로 응답을 받되, 전체를 모아 문자열로 반환.
# ask_llm()과 사용법이 동일하지만 504 gateway timeout에 강함.
#
# 왜 필요한가:
#   non-streaming(ask_llm)은 LLM이 응답을 완성할 때까지 연결을 유지하는데,
#   응답이 오래 걸리면 중간 게이트웨이에서 504 timeout을 내보낼 수 있음.
#   streaming은 첫 chunk가 도착하는 순간부터 연결이 "활성" 상태로 유지되어
#   timeout을 피할 수 있음.
#
# 사용 예시:
#   # 트리플 추출처럼 응답이 긴 작업
#   result = ask_llm_stream(USER_MESSAGE, SYSTEM_PROMPT, llm="GaussO4.1")
#   triples = json.loads(result)
#
#   # B01 배치 처리에서
#   for paper in papers:
#       result = ask_llm_stream(make_prompt(paper), llm="GaussO4.1")
# =============================================================================
def ask_llm_stream(USER_MESSAGE, SYSTEM_PROMPT="Semiconductor related workers",
                   temperature=0.05, reasoning_effort="medium", llm=None) -> Optional[str]:
    """
    Streaming 호출, 완성된 문자열 반환.

    내부적으로는 streaming으로 응답을 받되, 전체를 모아 문자열로 반환.
    ask_llm()과 사용법이 동일하지만 504 gateway timeout에 강함.

    Parameters:
        USER_MESSAGE: 사용자 입력 메시지
        SYSTEM_PROMPT: 시스템 프롬프트 (기본값: "Semiconductor related workers")
        temperature: 생성 다양성 (0.0~1.0, 기본값 0.05)
        reasoning_effort: 추론 깊이 ("low" / "medium" / "high", 기본값 "medium")
        llm: 사용할 모델명. None이면 .env의 LLM 기본값

    Returns:
        str: LLM 응답 문자열 (```json 코드블록 자동 제거)
        None: 호출 실패 시

    Note:
        트리플 추출처럼 응답이 긴 작업에 사용 권장.
    """
    # ask_llm_stream_iter()로 chunk를 모두 수집한 후 합쳐서 반환
    chunks = list(ask_llm_stream_iter(
        USER_MESSAGE, SYSTEM_PROMPT, temperature, reasoning_effort, llm
    ))

    if not chunks:
        logger.warning("비정상 응답: 빈 스트림")
        return None

    return strip_json_codeblock("".join(chunks))


# =============================================================================
# ask_llm_stream_iter — streaming 호출, chunk 단위 yield (제너레이터)
# -----------------------------------------------------------------------------
# chunk가 도착할 때마다 즉시 yield → 실시간 UI 출력에 사용.
# ask_llm_stream()은 이 함수를 내부적으로 사용해 결과를 모음.
#
# ask_llm_stream  vs  ask_llm_stream_iter:
#   ask_llm_stream      : 내부 streaming + 완성된 문자열 반환  → 기존 파이프라인용
#   ask_llm_stream_iter : chunk 즉시 yield                    → 실시간 UI 출력용
#
# 에러 처리:
#   제너레이터 특성상 실패 시 return None 대신 조기 종료(return).
#   호출부에서 빈 결과 여부로 실패를 감지:
#     chunks = list(ask_llm_stream_iter(...))
#     if not chunks:
#         print("LLM 호출 실패")
#
# 사용 예시:
#   # [1] 터미널 실시간 출력
#   for chunk in ask_llm_stream_iter("설명해줘", llm="GaussO4.1"):
#       print(chunk, end="", flush=True)
#   print()
#
#   # [2] 수집 + 실시간 출력 동시에
#   chunks = []
#   for chunk in ask_llm_stream_iter("요약해줘", llm="Gemma4"):
#       print(chunk, end="", flush=True)
#       chunks.append(chunk)
#   full_text = "".join(chunks)
#
#   # [3] NiceGUI 라벨 실시간 갱신
#   async def on_ask():
#       label.text = ""
#       for chunk in ask_llm_stream_iter(USER_MESSAGE):
#           label.text += chunk
#           await asyncio.sleep(0)   # UI 갱신 양보
#
#   # [4] Streamlit 실시간 갱신
#   placeholder = st.empty()
#   result = ""
#   for chunk in ask_llm_stream_iter(USER_MESSAGE):
#       result += chunk
#       placeholder.markdown(result)
# =============================================================================
def ask_llm_stream_iter(USER_MESSAGE, SYSTEM_PROMPT="Semiconductor related workers",
                        temperature=0.05, reasoning_effort="medium", llm=None):
    """
    Streaming 호출, chunk 단위 yield (제너레이터).

    chunk가 도착할 때마다 즉시 yield → 실시간 UI 출력에 사용.
    ask_llm_stream()은 이 함수를 내부적으로 사용해 결과를 모음.

    Parameters:
        USER_MESSAGE: 사용자 입력 메시지
        SYSTEM_PROMPT: 시스템 프롬프트 (기본값: "Semiconductor related workers")
        temperature: 생성 다양성 (0.0~1.0, 기본값 0.05)
        reasoning_effort: 추론 깊이 ("low" / "medium" / "high", 기본값 "medium")
        llm: 사용할 모델명. None이면 .env의 LLM 기본값

    Yields:
        str: 각 chunk의 내용

    Note:
        에러 발생 시 조기 종료(return). 호출부에서 빈 결과 여부로 실패 감지.
    """
    client, cfg = get_client(llm)

    try:
        stream = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_MESSAGE},
            ],
            temperature=temperature,
            extra_body={"reasoning_effort": reasoning_effort},
            stream=True,
        )
    except httpx.TimeoutException:
        logger.error(f"스트림 연결 타임아웃: {llm or LLM}")
        return  # 제너레이터이므로 return (None 반환 불가)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP 에러 {e.response.status_code}: {e}")
        return
    except Exception as e:
        logger.error(f"스트림 연결 실패 ({type(e).__name__}): {e}")
        return

    # openai 라이브러리가 SSE 파싱을 대신 처리 → delta만 꺼내면 됨
    # (requests 버전에서는 iter_lines()로 직접 "data: " 파싱이 필요했음)
    try:
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta   # chunk 도착 즉시 반환
            except (AttributeError, IndexError):
                continue
    except httpx.ReadTimeout:
        logger.error("스트림 읽기 타임아웃")
        yield "[ERROR: 읽기 타임아웃]"
        return
    except Exception as e:
        logger.error(f"스트림 파싱 오류 ({type(e).__name__}): {e}")
        return


# =============================================================================
# ask_llm_messages / ask_llm_stream_iter_messages
# -----------------------------------------------------------------------------
# ask_llm* 계열은 SYSTEM_PROMPT + USER_MESSAGE 딱 한 쌍만 받지만,
# 대화 히스토리(멀티턴)를 포함해 호출해야 할 때는 messages 리스트를
# [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...]
# 형태로 그대로 넘길 수 있어야 한다. rag_engine.py의 답변 생성이 이전 대화
# 맥락(self.history)을 유지하기 위해 이 두 함수를 사용한다.
#
# ask_llm_messages           : messages 리스트, non-streaming
# ask_llm_stream_iter_messages: messages 리스트, streaming (chunk 단위 yield)
#
# 사용 예시:
#   messages = [
#       {"role": "system", "content": "당신은 반도체 소재 전문가입니다."},
#       {"role": "user",   "content": "이전 질문 1"},
#       {"role": "assistant", "content": "이전 답변 1"},
#       {"role": "user",   "content": "이어지는 질문"},
#   ]
#   for chunk in ask_llm_stream_iter_messages(messages, llm="GaussO4.1"):
#       print(chunk, end="", flush=True)
# =============================================================================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
def ask_llm_messages(messages: list[dict],
                     temperature=0.05, reasoning_effort="medium", llm=None) -> Optional[str]:
    """
    messages 리스트를 직접 받아 호출 (non-streaming). 대화 히스토리 포함 호출용.

    Parameters:
        messages: [{"role": ..., "content": ...}, ...] 형태의 대화 메시지 목록
        temperature: 생성 다양성 (0.0~1.0, 기본값 0.05)
        reasoning_effort: 추론 깊이 ("low" / "medium" / "high", 기본값 "medium")
        llm: 사용할 모델명. None이면 .env의 LLM 기본값

    Returns:
        str: LLM 응답 문자열 (```json 코드블록 자동 제거)
        None: 호출 실패 시
    """
    client, cfg = get_client(llm)

    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            temperature=temperature,
            extra_body={"reasoning_effort": reasoning_effort},
        )
    except httpx.TimeoutException:
        logger.error(f"LLM 호출 타임아웃: {llm or LLM}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP 에러 {e.response.status_code}: {e}")
        return None
    except (AttributeError, IndexError) as e:
        logger.error(f"응답 파싱 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM 호출 실패 ({type(e).__name__}): {_short_err(e)}")
        return None

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        logger.error(f"응답에서 content 추출 실패: {e}")
        return None

    return strip_json_codeblock(content)


def ask_llm_stream_iter_messages(messages: list[dict],
                                  temperature=0.05, reasoning_effort="medium", llm=None):
    """
    messages 리스트를 직접 받아 스트리밍으로 chunk 단위 yield (제너레이터).
    대화 히스토리 포함 호출용 (rag_engine.py의 답변 생성이 이 함수를 사용).

    Parameters:
        messages: [{"role": ..., "content": ...}, ...] 형태의 대화 메시지 목록
        temperature: 생성 다양성 (0.0~1.0, 기본값 0.05)
        reasoning_effort: 추론 깊이 ("low" / "medium" / "high", 기본값 "medium")
        llm: 사용할 모델명. None이면 .env의 LLM 기본값

    Yields:
        str: 각 chunk의 내용
    """
    client, cfg = get_client(llm)

    try:
        stream = client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            temperature=temperature,
            extra_body={"reasoning_effort": reasoning_effort},
            stream=True,
        )
    except httpx.TimeoutException:
        logger.error(f"스트림 연결 타임아웃: {llm or LLM}")
        yield f"[ERROR: {llm or LLM} 연결 타임아웃 — API 서버가 응답하지 않습니다. URL/네트워크를 확인하세요.]"
        return
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP 에러 {e.response.status_code}: {e}")
        yield f"[ERROR: {llm or LLM} HTTP {e.response.status_code} — {e}]"
        return
    except Exception as e:
        # ★ 이 예외는 API 호출 자체(연결/인증/요청 형식 등)가 즉시 실패한 경우다.
        #   아래쪽의 "reasoning_effort 소진" 진단과는 성격이 다르므로(그쪽은 연결은
        #   성공하고 정상적으로 스트림이 끝났는데 콘텐츠가 없는 경우), 호출부가
        #   구분할 수 있도록 예외 메시지를 그대로 [ERROR: ...] 콘텐츠로 넘긴다
        #   (예전엔 여기서 그냥 return 만 해서 콘텐츠가 0개인 것과 구분이 안 됐다 —
        #   그래서 실제로는 즉시 연결 실패한 건데도 "reasoning_effort 가 너무 높아서"
        #   라는 엉뚱한 설명이 화면에 뜨는 문제가 있었다).
        logger.error(f"스트림 연결 실패 ({type(e).__name__}): {e}")
        yield f"[ERROR: {llm or LLM} 연결 실패 ({type(e).__name__}): {e}]"
        return

    # ★ reasoning_effort 가 높은 모델(gpt-oss 등)은 최종 답변 전에 "추론" 토큰을
    #   먼저 스트리밍하는데, 서버 구현에 따라 이게 content 필드가 아니라 별도
    #   필드(예: reasoning_content)로 온다. 이 경우 최종 답변 콘텐츠가 하나도
    #   없이(=이 함수가 아무것도 yield 못한 채) 스트림이 "정상적으로" 끝날 수 있다
    #   — 예외가 전혀 안 나서 호출부(rag_engine.py)에도 에러 로그가 안 남고,
    #   화면엔 그냥 "(응답을 받지 못했습니다)"만 뜬다. 이유를 알 수 있도록
    #   content_yielded/reasoning_chunks/finish_reason 을 추적해 진단 로그를 남긴다.
    content_yielded = False
    reasoning_chunks = 0
    last_finish_reason = None

    # ★ Qwen3 계열(및 llama.cpp로 서빙되는 여러 오픈모델)은 "생각 중" 내용을
    #   reasoning_content 같은 별도 필드가 아니라 content 안에 <think>...</think>
    #   태그로 직접 섞어 보낸다 — 그래서 답변 화면에 내부 추론 과정이 그대로
    #   노출된다. 태그가 여러 chunk 에 걸쳐 쪼개져 도착할 수 있어(예: "<th" 다음
    #   chunk 에 "ink>") 작은 버퍼를 두고 태그 밖의 텍스트만 걸러서 내보낸다.
    think_buf = ''
    in_think = False
    THINK_OPEN, THINK_CLOSE = '<think>', '</think>'

    def _filter_think(piece: str) -> str:
        """content 조각에서 <think>...</think> 블록을 제거하고 나머지만 반환.
        열린 태그를 아직 못 닫은 상태는 in_think 로, 태그가 청크 경계에서
        잘린 경우는 think_buf(꼬리 보류분)로 다음 호출까지 이어서 처리한다."""
        nonlocal think_buf, in_think
        think_buf += piece
        out = []
        while True:
            if not in_think:
                idx = think_buf.find(THINK_OPEN)
                if idx == -1:
                    # '<think>' 의 접두사로 끝날 수 있는 꼬리만 다음 조각과 합치기 위해 보류
                    safe_len = len(think_buf)
                    for k in range(min(len(THINK_OPEN) - 1, len(think_buf)), 0, -1):
                        if think_buf.endswith(THINK_OPEN[:k]):
                            safe_len = len(think_buf) - k
                            break
                    out.append(think_buf[:safe_len])
                    think_buf = think_buf[safe_len:]
                    break
                out.append(think_buf[:idx])
                think_buf = think_buf[idx + len(THINK_OPEN):]
                in_think = True
            else:
                idx = think_buf.find(THINK_CLOSE)
                if idx == -1:
                    # 닫는 태그를 아직 못 찾음: 추론 내용 자체는 버리고, 닫는 태그의
                    # 접두사가 될 수 있는 꼬리만 남겨 버퍼가 무한정 커지지 않게 한다.
                    keep = 0
                    for k in range(min(len(THINK_CLOSE) - 1, len(think_buf)), 0, -1):
                        if think_buf.endswith(THINK_CLOSE[:k]):
                            keep = k
                            break
                    think_buf = think_buf[-keep:] if keep else ''
                    break
                think_buf = think_buf[idx + len(THINK_CLOSE):]
                in_think = False
        return ''.join(out)

    try:
        for chunk in stream:
            try:
                choice = chunk.choices[0]
                if getattr(choice, 'finish_reason', None):
                    last_finish_reason = choice.finish_reason
                delta = choice.delta
                content = getattr(delta, 'content', None)
                if content:
                    visible = _filter_think(content)
                    if visible:
                        content_yielded = True
                        yield visible
                    continue
                if getattr(delta, 'reasoning_content', None):
                    reasoning_chunks += 1
            except (AttributeError, IndexError):
                continue
    except httpx.ReadTimeout:
        logger.error("스트림 읽기 타임아웃")
        yield "[ERROR: 읽기 타임아웃]"
        return
    except Exception as e:
        logger.error(f"스트림 파싱 오류 ({type(e).__name__}): {e}")
        return

    if not content_yielded:
        logger.warning(
            f"스트림이 답변 콘텐츠 없이 종료됨 (모델={llm or LLM}, "
            f"reasoning_effort={reasoning_effort}, finish_reason={last_finish_reason}, "
            f"추론(reasoning) 청크 수={reasoning_chunks}) — reasoning_effort 가 너무 높아 "
            f"추론만 하다가 답변 토큰에 도달하기 전에 응답 한도(max_tokens 등)에 걸렸을 "
            f"가능성이 큽니다. reasoning_effort 를 낮춰(medium/low) 다시 시도해 보세요."
        )


# =============================================================================
# strip_json_codeblock — 마크다운 코드블록 제거 유틸
# -----------------------------------------------------------------------------
# LLM이 JSON을 ```json ... ``` 또는 ``` ... ```로 감싸서 반환하는 경우 제거.
# ask_llm / ask_llm_stream 반환 전에 자동 적용됨.
#
# 사용 예시 (직접 호출이 필요한 경우):
#   raw = '```json\n{"key": "value"}\n```'
#   clean = strip_json_codeblock(raw)  # '{"key": "value"}'
#   data  = json.loads(clean)
# =============================================================================
def strip_json_codeblock(text: Optional[str]) -> Optional[str]:
    """
    마크다운 코드블록 제거 유틸.

    LLM이 JSON을 ```json ... ``` 또는 ``` ... ```로 감싸서 반환하는 경우 제거.
    ask_llm / ask_llm_stream 반환 전에 자동 적용됨.

    Parameters:
        text: 처리할 텍스트

    Returns:
        str: 코드블록이 제거된 텍스트
        None: 입력이 None인 경우

    사용 예시 (직접 호출이 필요한 경우):
        raw = '```json\n{"key": "value"}\n```'
        clean = strip_json_codeblock(raw)  # '{"key": "value"}'
        data  = json.loads(clean)
    """
    if text is None:
        return text
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    return text.strip()


# =============================================================================
# 비동기 함수들 (async/await 지원)
# -----------------------------------------------------------------------------
# 고성능 애플리케이션에서 비동기 처리가 필요할 때 사용합니다.
# =============================================================================

async def ask_llm_async(USER_MESSAGE, SYSTEM_PROMPT="Semiconductor related workers",
                       temperature=0.05, reasoning_effort="medium", llm=None) -> Optional[str]:
    """
    비동기 LLM 호출 (async/await 지원).

    Parameters:
        USER_MESSAGE: 사용자 입력 메시지
        SYSTEM_PROMPT: 시스템 프롬프트 (기본값: "Semiconductor related workers")
        temperature: 생성 다양성 (0.0~1.0, 기본값 0.05)
        reasoning_effort: 추론 깊이 ("low" / "medium" / "high", 기본값 "medium")
        llm: 사용할 모델명. None이면 .env의 LLM 기본값

    Returns:
        str: LLM 응답 문자열 (```json 코드블록 자동 제거)
        None: 호출 실패 시

    사용 예시:
        import asyncio
        result = await ask_llm_async("요약해줘", llm="Gemma4")
    """
    client, cfg = get_async_client(llm)

    try:
        response = await client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_MESSAGE},
            ],
            temperature=temperature,
            extra_body={"reasoning_effort": reasoning_effort},
        )
    except httpx.TimeoutException:
        logger.error(f"비동기 LLM 호출 타임아웃: {llm or LLM}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP 에러 {e.response.status_code}: {e}")
        return None
    except (AttributeError, IndexError) as e:
        logger.error(f"응답 파싱 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"비동기 LLM 호출 실패 ({type(e).__name__}): {_short_err(e)}")
        return None

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        logger.error(f"응답에서 content 추출 실패: {e}")
        return None

    return strip_json_codeblock(content)


async def ask_llm_stream_iter_async(USER_MESSAGE, SYSTEM_PROMPT="Semiconductor related workers",
                                   temperature=0.05, reasoning_effort="medium", llm=None):
    """
    비동기 streaming 호출, chunk 단위 yield (비동기 제너레이터).

    chunk가 도착할 때마다 즉시 yield → 실시간 UI 출력에 사용.

    Parameters:
        USER_MESSAGE: 사용자 입력 메시지
        SYSTEM_PROMPT: 시스템 프롬프트 (기본값: "Semiconductor related workers")
        temperature: 생성 다양성 (0.0~1.0, 기본값 0.05)
        reasoning_effort: 추론 깊이 ("low" / "medium" / "high", 기본값 "medium")
        llm: 사용할 모델명. None이면 .env의 LLM 기본값

    Yields:
        str: 각 chunk의 내용

    사용 예시:
        async for chunk in ask_llm_stream_iter_async("설명해줘", llm="GaussO4.1"):
            print(chunk, end="", flush=True)
    """
    client, cfg = get_async_client(llm)

    try:
        stream = await client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_MESSAGE},
            ],
            temperature=temperature,
            extra_body={"reasoning_effort": reasoning_effort},
            stream=True,
        )
    except httpx.TimeoutException:
        logger.error(f"비동기 스트림 연결 타임아웃: {llm or LLM}")
        return
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP 에러 {e.response.status_code}: {e}")
        return
    except Exception as e:
        logger.error(f"비동기 스트림 연결 실패 ({type(e).__name__}): {e}")
        return

    try:
        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            except (AttributeError, IndexError):
                continue
    except httpx.ReadTimeout:
        logger.error("비동기 스트림 읽기 타임아웃")
        yield "[ERROR: 읽기 타임아웃]"
        return
    except Exception as e:
        logger.error(f"비동기 스트림 파싱 오류 ({type(e).__name__}): {e}")
        return
