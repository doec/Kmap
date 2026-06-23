from pathlib import Path
import requests
import json
import uuid
import os
import re
import json_repair
from datetime import datetime
from dotenv import load_dotenv
import base64

load_dotenv(override=True)

LLM = os.getenv("LLM", "gpt-prod")
print("LLM model (default):", LLM)

UserID           = os.getenv('USER_ID')
Send_System_Name = os.getenv('SEND_SYSTEM_NAME')

_LLM_CONFIGS = {
    "gpt-stg": {
        "model_env"    : "LLM_MODEL",
        "model_default": "openai/gpt-oss-120b",
        "url_env"      : "LLM_API_URL_stg",
        "cred_env"     : "CREDENTIAL_KEY_stg",
        "sysname_env"  : None,
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
        "sysname_env"  : "SEND_SYSTEM_NAME_GaussO41",
    },
    "Gemma4": {
        "model_env"    : "LLM_MODEL_Gemma4",
        "model_default": "Gemma4-260430",
        "url_env"      : "LLM_API_URL_Gemma4",
        "cred_env"     : "CREDENTIAL_KEY_Gemma4",
        "sysname_env"  : "SEND_SYSTEM_NAME_Gemma4",
    },
}


def _resolve_llm_config(llm: str | None) -> dict:
    target = llm if llm is not None else LLM
    if target not in _LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: '{target}'. 지원 목록: {list(_LLM_CONFIGS.keys())}")
    cfg = _LLM_CONFIGS[target]
    model_name     = os.getenv(cfg["model_env"], cfg["model_default"])
    api_base       = os.getenv(cfg["url_env"], "")
    credential_key = os.getenv(cfg["cred_env"], "")
    sysname = (
        os.getenv(cfg["sysname_env"], Send_System_Name)
        if cfg["sysname_env"]
        else Send_System_Name
    )
    endpoint = (
        api_base
        if "chat/completions" in api_base
        else api_base.rstrip("/") + "/chat/completions"
    )
    return {
        "model"          : model_name,
        "endpoint"       : endpoint,
        "credential_key" : credential_key,
        "send_system_name": sysname,
    }


def _build_headers(cfg: dict) -> dict:
    return {
        'x-dep-ticket'     : cfg["credential_key"],
        'Send-System-Name' : cfg["send_system_name"],
        'User-Id'          : UserID,
        'User-Type'        : UserID,
        'Prompt-Msg-Id'    : str(uuid.uuid4()),
        'Completion-Msg-Id': str(uuid.uuid4()),
        'Accept'           : 'text/event-stream; charset=utf-8',
        'Content-Type'     : 'application/json',
    }


def ask_llm(USER_MESSAGE, SYSTEM_PROMPT="Semiconductor related workers",
            temperature=0.05, reasoning_effort="medium", llm=None):
    cfg = _resolve_llm_config(llm)
    payload = json.dumps({
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_MESSAGE},
        ],
        "temperature"    : temperature,
        "stream"         : False,
        "reasoning_effort": reasoning_effort,
    })
    try:
        response = requests.post(
            url=cfg["endpoint"], headers=_build_headers(cfg),
            data=payload, timeout=None, proxies={'http': None, 'https': None},
        )
    except Exception as e:
        print(e)
        return None
    recv_json = response.json()
    if "choices" in recv_json:
        return strip_json_codeblock(recv_json["choices"][0]["message"]["content"])
    print("비정상 응답 (Json):", recv_json)
    return None


def ask_llm_img(USER_MESSAGE, IMAGE_PATH,
                SYSTEM_PROMPT="Semiconductor related workers",
                temperature=0.05, reasoning_effort="medium", llm=None):
    cfg = _resolve_llm_config(llm)
    with open(IMAGE_PATH, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    ext = str(IMAGE_PATH).rsplit(".", 1)[-1].lower()
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")
    payload = json.dumps({
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text",      "text": USER_MESSAGE},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                }},
            ]},
        ],
        "temperature"    : temperature,
        "stream"         : False,
        "reasoning_effort": reasoning_effort,
    })
    try:
        response = requests.post(
            url=cfg["endpoint"], headers=_build_headers(cfg),
            data=payload, timeout=None, proxies={'http': None, 'https': None},
        )
    except Exception as e:
        print(e)
        return None
    recv_json = response.json()
    if "choices" in recv_json:
        return strip_json_codeblock(recv_json["choices"][0]["message"]["content"])
    print("비정상 응답 (Json):", recv_json)
    return None


def ask_llm_stream(USER_MESSAGE, SYSTEM_PROMPT="Semiconductor related workers",
                   temperature=0.05, reasoning_effort="medium", llm=None):
    cfg = _resolve_llm_config(llm)
    payload = json.dumps({
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_MESSAGE},
        ],
        "temperature"    : temperature,
        "stream"         : True,
        "reasoning_effort": reasoning_effort,
    })
    try:
        response = requests.post(
            url=cfg["endpoint"], headers=_build_headers(cfg),
            data=payload, stream=True, timeout=(10, 300),
            proxies={'http': None, 'https': None},
        )
        response.raise_for_status()
    except Exception as e:
        print(e)
        return None
    chunks = []
    data = ""
    try:
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            chunk_json = json.loads(data)
            choices = chunk_json.get("choices", [])
            if not choices:
                continue
            delta = choices[0]["delta"].get("content", "")
            if delta:
                chunks.append(delta)
    except Exception as e:
        print(f"스트림 파싱 오류: {e}")
        print(f"문제 청크: {data}")
        return None
    result = "".join(chunks)
    if not result:
        print("비정상 응답: 빈 스트림")
        return ""
    return strip_json_codeblock(result)


def ask_llm_stream_iter(USER_MESSAGE, SYSTEM_PROMPT="Semiconductor related workers",
                        temperature=0.05, reasoning_effort="medium", llm=None):
    cfg = _resolve_llm_config(llm)
    payload = json.dumps({
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_MESSAGE},
        ],
        "temperature"    : temperature,
        "stream"         : True,
        "reasoning_effort": reasoning_effort,
    })
    try:
        response = requests.post(
            url=cfg["endpoint"], headers=_build_headers(cfg),
            data=payload, stream=True, timeout=(10, 300),
            proxies={'http': None, 'https': None},
        )
        response.raise_for_status()
    except Exception as e:
        print(e)
        return
    data = ""
    try:
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            chunk_json = json.loads(data)
            choices = chunk_json.get("choices", [])
            if not choices:
                continue
            delta = choices[0]["delta"].get("content", "")
            if delta:
                yield delta
    except Exception as e:
        print(f"스트림 파싱 오류: {e}")
        print(f"문제 청크: {data}")
        return


def ask_llm_messages(messages: list[dict],
                     temperature=0.05, reasoning_effort="medium", llm=None):
    cfg = _resolve_llm_config(llm)
    payload = json.dumps({
        "model"           : cfg["model"],
        "messages"        : messages,
        "temperature"     : temperature,
        "stream"          : False,
        "reasoning_effort": reasoning_effort,
    })
    try:
        response = requests.post(
            url=cfg["endpoint"], headers=_build_headers(cfg),
            data=payload, timeout=None, proxies={'http': None, 'https': None},
        )
    except Exception as e:
        print(e)
        return None
    recv_json = response.json()
    if "choices" in recv_json:
        return strip_json_codeblock(recv_json["choices"][0]["message"]["content"])
    print("비정상 응답 (Json):", recv_json)
    return None


def ask_llm_stream_iter_messages(messages: list[dict],
                                  temperature=0.05, reasoning_effort="medium", llm=None):
    """messages 리스트를 받아 스트리밍으로 chunk yield (동기 제너레이터)"""
    cfg = _resolve_llm_config(llm)
    payload = json.dumps({
        "model"           : cfg["model"],
        "messages"        : messages,
        "temperature"     : temperature,
        "stream"          : True,
        "reasoning_effort": reasoning_effort,
    })
    try:
        response = requests.post(
            url=cfg["endpoint"], headers=_build_headers(cfg),
            data=payload, stream=True, timeout=(10, 300),
            proxies={'http': None, 'https': None},
        )
        response.raise_for_status()
    except Exception as e:
        print(e)
        return
    data = ""
    try:
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            chunk_json = json.loads(data)
            choices = chunk_json.get("choices", [])
            if not choices:
                continue
            delta = choices[0]["delta"].get("content", "")
            if delta:
                yield delta
    except Exception as e:
        print(f"스트림 파싱 오류: {e}")
        print(f"문제 청크: {data}")
        return


def strip_json_codeblock(text: str) -> str:
    if text is None:
        return text
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    return text.strip()
