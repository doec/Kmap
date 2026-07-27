from __future__ import annotations

import asyncio
import os
import re

from nicegui import ui
from starlette.requests import Request

from rag_engine import GraphRAG, REPORTS_DATASET, PAPERS_DATASET, CONFLUENCE_DATASET, DATASETS
from retriever.neo4j_retriever import Neo4jRetriever
from access_log import client_ip as _client_ip, log_search

# ── 서브그래프 표시 여부 판단용: 답변에 엔티티 이름이 "언급됐는지" 확인할 때 LaTeX
#   표기(\text{SrTiO}_3 등)와 공백/대소문자 차이 때문에 완전히 같은 문자열로는
#   못 찾는 문제가 있다. 이 문자들을 지우고 비교하면 "SrTiO3"와 "\text{SrTiO}_3"
#   가 같은 것으로 인식된다.
_MENTION_NORMALIZE_RE = re.compile(r'[\\{}_$()\[\]\s]+')


def _normalize_for_mention_check(s: str) -> str:
    return _MENTION_NORMALIZE_RE.sub('', s).lower()


# ── linkify: 답변 텍스트의 맨 URL / DOI 를 마크다운 링크로 변환 ───────────────────────────────
# 이미 마크다운 링크 형태( ](url) , <url> )인 것은 건드리지 않는다.
#
# ★ LLM이 "**https://...**"처럼 URL을 마크다운 굵게(**)로 감싸거나, 뒤에 다른 글자를
#   공백 없이 붙여 쓰는 경우가 있었다. 이러면 (a) URL 문자열 자체에 '**'가 섞여
#   링크 target 이 깨지거나, (b) 렌더링이 이상해져 클릭이 안 됐다.
#   대응:
#     1. URL 앞뒤에 붙은 '**'는 (?:\*\*)? 로 함께 소비해서 결과에서 제거한다.
#     2. URL 문자 집합에서 '*' 를 제외해, 중간에 낀 '*' 도 URL로 안 삼켜지게 한다
#        (그 지점에서 URL 매칭이 끝나 나머지는 링크 밖 텍스트로 남는다).
_BARE_URL_RE = re.compile(
    r'(?:\*\*)?(?<![\(\[<"\'=/])(https?://[^\s<>\)\]*]+[^\s<>\)\].,;:!?\'"*])(?:\*\*)?'
)
_DOI_RE = re.compile(
    r'(?:\*\*)?(?<![\w/.])((?:doi:\s*)?(10\.\d{4,9}/[^\s<>\)\]*]+[^\s<>\)\].,;:!?\'"*]))(?:\*\*)?',
    re.IGNORECASE
)


# 수식 구분자 변환용 정규식.
# ★ LLM 은 수식을 $...$ / $$...$$ 로 감싸는데, 마크다운이 그 사이의 '_', '*' 를
#   이탤릭으로 처리하면서 '$' 를 일부만 소비해 잔여 '$' 가 화면에 남는 문제가 있었다.
#   그래서 마크다운 처리 전에 '$' 구분자를 MathJax 기본 구분자인 \(...\) / \[...\] 로
#   바꿔 리터럴 '$' 자체를 없앤다. (display($$) 를 inline($) 보다 먼저 변환)
_MATH_DISPLAY_RE = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
_MATH_INLINE_RE  = re.compile(r'\$(?!\s)([^\$\n]+?)(?<!\s)\$')

# ★ Gemma4 는 종종 \( / \) / \[ / \] 를 짝이 안 맞게(한쪽만, 또는 중첩되게) 흘려
#   놓는다(예: "(\text{TiO}2\)" — 여는 쪽엔 backslash 가 없고 닫는 쪽에만 있음).
#   이걸 그대로 믿고 그 위에 우리가 또 \(...\) 로 감싸면, 짝이 안 맞는 delimiter가
#   겹쳐서 수식 경계가 뒤엉키고 그 사이의 한글 본문까지 수식 폭에 잘못 포함되어
#   MathJax 수식 폰트로 렌더링되는(그래서 "이상하게 굵게/다르게 보인다") 문제가
#   생긴다. 그래서 모델이 미리 흘린 \(/\)/\[/\] 는 신뢰하지 않고 backslash 만 제거해
#   "그냥 괄호"로 되돌린 뒤, 아래 로직이 완전한 수식 구간만 우리 판단으로 새로
#   감싸도록 한다.
_STRAY_DELIM_RE = re.compile(r'\\([()\[\]])')


def _neutralize_stray_delims(text: str) -> str:
    return _STRAY_DELIM_RE.sub(lambda m: m.group(1), text)


# ★ 모델에 따라 $...$ 구분자 없이 LaTeX 명령어를 맨 텍스트로 내놓는 경우가 있다
#   (Gemma4에서 관찰됨: "\text{Al}_2\text{O}_3", "\text{J}(\text{TiO}_2) \gg ..." 등).
#   구분자가 없으면 MathJax 가 인식을 못 해 그대로 노출되므로, "공백 없이 이어지는
#   구간(run)" 중 백슬래시 명령어가 하나라도 포함된 run 전체를 통째로 \(...\) 로
#   감싼다. 함수 표기처럼 일반 괄호 "(", ")" 가 명령어 사이에 끼어 있어도(예: 위의
#   "J(...)") 같은 run 으로 취급해 하나의 수식으로 감싼다 — 모델이 구분자를 쓰든
#   안 쓰든, 얼마나 복잡하게 섞어 쓰든 결과를 통일한다.
#   문자 집합에 '.', '/', ':' 등을 포함하지 않아 URL/DOI(백슬래시가 없음)는 애초에
#   대상이 되지 않는다 — _linkify 가 이후 별도로 처리한다.
_RUN_RE = re.compile(r'[A-Za-z0-9(){}_^\\]+')
# 이미 \(...\) / \[...\] 로 감싸진 구간은 건드리지 않도록 분리해서 처리한다.
_ALREADY_DELIM_RE = re.compile(r'(\\\(.*?\\\)|\\\[.*?\\\])', re.DOTALL)


def _wrap_bare_latex(text: str) -> str:
    def _wrap_if_math(m: re.Match) -> str:
        run = m.group(0)
        return f'\\({run}\\)' if '\\' in run else run   # 백슬래시 명령어가 없는 run 은 그대로 둔다

    parts = _ALREADY_DELIM_RE.split(text)
    for i, part in enumerate(parts):
        if i % 2 == 0:   # 홀수 인덱스는 이미 감싸진 구간(그대로 유지)
            parts[i] = _RUN_RE.sub(_wrap_if_math, part)
    return ''.join(parts)


def _protect_emphasis_chars_in_math(text: str) -> str:
    """
    \(...\) / \[...\] 수식 구간 안의 '_' 와 '*' 를 HTML 숫자 문자 참조로 바꾼다.

    ★ \(...\) 로 감싸는 것만으로는 마크다운으로부터 내용을 보호하지 못한다 —
    마크다운은 \( \) 를 그냥 의미 없는 글자로 보고, 그 안의 '_'/'*' 를 여전히
    강조(이탤릭/볼드) 마커로 해석할 수 있다. 그러면 수식 안의 아래첨자용 '_'가
    문서 뒤쪽 어딘가의 다른 '_'와 짝지어져 그 사이 전체가 굵게/기울임으로
    바뀌어버리는 문제가 생긴다(실제 관찰된 증상).

    '_' -> '&#95;', '*' -> '&#42;' 로 바꿔두면 마크다운에는 그냥 무해한 문자열로
    보여 건드리지 않고, 브라우저가 HTML 을 파싱할 때 이 문자 참조를 자동으로
    원래 문자로 복원하므로 MathJax 는 최종적으로 정상적인 '_'(아래첨자)를 보게
    된다 — 마크다운과 MathJax 양쪽의 요구사항을 동시에 만족시키는 방법이다.
    """
    parts = _ALREADY_DELIM_RE.split(text)
    for i, part in enumerate(parts):
        if i % 2 == 1:   # 홀수 인덱스 = 수식 구간
            parts[i] = part.replace('_', '&#95;').replace('*', '&#42;')
    return ''.join(parts)


# ★ 수식과 무관하게, "글자/숫자 사이에 낀 언더스코어"(예: 코드/치수 표기
#   "600_320_280_320")도 markdown2 가 강조(이탤릭) 마커로 잘못 해석해 언더스코어가
#   사라지고 글자가 붙어버리는 문제가 있다(CommonMark 표준은 이런 "intraword"
#   언더스코어를 강조로 보지 않아야 하는데, markdown2 는 이 규칙을 지키지 않음).
#   수식 구간 밖의 일반 텍스트에도 같은 HTML 문자 참조 보호를 적용한다.
_INTRAWORD_USCORE_RE = re.compile(r'(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])')


def _protect_intraword_underscore(text: str) -> str:
    parts = _ALREADY_DELIM_RE.split(text)
    for i, part in enumerate(parts):
        if i % 2 == 0:   # 짝수 인덱스 = 수식이 아닌 일반 텍스트 구간
            parts[i] = _INTRAWORD_USCORE_RE.sub('&#95;', part)
    return ''.join(parts)


def _convert_math_delims(text: str) -> str:
    text = _neutralize_stray_delims(text)
    text = _MATH_DISPLAY_RE.sub(lambda m: f'\\[{m.group(1)}\\]', text)
    text = _MATH_INLINE_RE.sub(lambda m: f'\\({m.group(1)}\\)', text)
    text = _wrap_bare_latex(text)
    text = _protect_emphasis_chars_in_math(text)
    text = _protect_intraword_underscore(text)
    return text


# CommonMark 가 백슬래시-이스케이프로 인식하는 ASCII 문장부호 전체 집합.
# (백슬래시 뒤에 이 중 하나가 오면 마크다운이 백슬래시를 소비해버린다.)
_MD_ESCAPABLE_PUNCT_RE = re.compile(r'\\(?=[!-/:-@\[-`{-~])')


# ★ 코드 블록(```...```)과 인라인 코드 스팬(`...`)은 마크다운이 원래 내용 그대로
#   (리터럴) 보여준다 — 이 안의 '_'/'\' 는 애초에 강조/수식으로 해석될 위험이 없다.
#   오히려 우리가 여기에 &#95; 같은 보호 처리를 하면, 마크다운이 코드 영역 안의
#   '&' 를 다시 '&amp;' 로 이스케이프해버려서 "&#95;" 가 "&amp;#95;" 로 이중
#   인코딩되고, 브라우저가 이를 실제 문자로 복원하지 못해 화면에 "&#95;" 라는
#   글자 그대로 노출되는 문제가 생긴다. 그래서 코드 영역은 파이프라인 전체에서
#   완전히 제외하고 원문 그대로 통과시킨다.
_CODE_REGION_RE = re.compile(r'(```.*?```|`[^`\n]+`)', re.DOTALL)


def _apply_outside_code(text: str, fn) -> str:
    parts = _CODE_REGION_RE.split(text)
    for i, part in enumerate(parts):
        if i % 2 == 0:   # 짝수 인덱스 = 코드 영역이 아닌 일반 구간
            parts[i] = fn(part)
    return ''.join(parts)


# ★ LLM 이 답변 어딘가에 짝이 안 맞는(홀수 개) ``` 를 흘리면, 마크다운은 그 지점부터
#   (다음 ``` 또는 문서 끝까지) 전부 "코드 블록"으로 인식해버린다 — 그 뒤에 표가
#   있으면 표까지 통째로 그냥 텍스트로 렌더링된다. 스트리밍 중에는 문제의 ```가
#   아직 도착하기 전이라 표가 정상으로 보이다가, 스트림이 끝나고 전체 텍스트를
#   다시 그릴 때 비로소 이 증상이 나타난다. ``` 개수가 홀수면 마지막 하나를
#   무력화해(짝을 맞춰) 이후 내용이 통째로 코드 블록에 먹히지 않게 방지한다.
def _fix_unbalanced_code_fence(text: str) -> str:
    if text.count('```') % 2 == 1:
        idx = text.rindex('```')
        text = text[:idx] + '​```' + text[idx + 3:]  # 폭 없는 문자를 끼워 펜스 무력화
    return text


def _linkify(text: str) -> str:
    """맨 URL과 DOI 문자열을 클릭 가능한 마크다운 링크로 변환한다. (URL에 붙은 ** 제거)"""
    if not text:
        return text

    text = _fix_unbalanced_code_fence(text)

    def _process(t: str) -> str:
        t = _convert_math_delims(t)
        t = _BARE_URL_RE.sub(lambda m: f'[{m.group(1)}]({m.group(1)})', t)
        t = _DOI_RE.sub(
            lambda m: f'[{m.group(1)}](https://doi.org/{m.group(2)})', t
        )
        # ★ 마크다운(CommonMark)은 "백슬래시+ASCII 문장부호"(\(, \), \[, \] 등)를
        #   이스케이프로 해석해 렌더링 시 백슬래시를 소비해버린다. 방금 위에서
        #   MathJax용으로 넣은 \(...\) / \[...\] 구분자가 정확히 이 패턴이라,
        #   ui.markdown() 을 거치면 백슬래시가 사라지고 MathJax 는 구분자를 전혀
        #   못 보게 된다(반면 \text 처럼 backslash+글자는 마크다운 이스케이프
        #   대상이 아니라서 그대로 살아남는다 — 그래서 "\text{...}는 보이는데
        #   감싸는 \( \) 만 사라지는" 증상이 나타났다). "백슬래시+문장부호" 조합만
        #   두 배로 늘리면, 마크다운의 "\\"(백슬래시 자체의 이스케이프) 규칙에
        #   의해 정확히 하나만 남아 원래 의도한 단일 백슬래시가 보존된다.
        t = _MD_ESCAPABLE_PUNCT_RE.sub(r'\\\\', t)
        return t

    return _apply_outside_code(text, _process)

# ── module-level singleton for graph visualization ────────────────────────────────────────────
_neo4j_viz = Neo4jRetriever()

# ── source chip colours ────────────────────────────────────────────────────────────────────────────────
_DATASET_CHIP_COLOR = {
    REPORTS_DATASET:    "blue",
    PAPERS_DATASET:     "teal",
    CONFLUENCE_DATASET: "purple",
    "ExperimentsDB":    "amber",
}

# ── dataset tab labels ──────────────────────────────────────────────────────────────────────────────────
# ★ .env 의 NEO4J_*_ENABLED 로 꺼진 데이터셋은 DATASETS 에서 이미 빠져 있으므로,
#   그 탭도 자동으로 숨긴다 (꺼진 데이터셋 탭을 눌러도 검색될 게 없어 혼란만 준다).
_DATASET_TABS = [
    ("전체",    "All"),
    ("논문",    PAPERS_DATASET),
    ("ReportsDB", REPORTS_DATASET),
    ("Confluence", CONFLUENCE_DATASET),
    ("실험",    "ExperimentsDB"),   # future — disabled
]
_DATASET_TABS = [
    (label, key) for label, key in _DATASET_TABS
    if key in ("All", "ExperimentsDB") or key in DATASETS
]
_DATASET_LABEL_BY_KEY = {key: label for label, key in _DATASET_TABS}


def _dataset_display(ds) -> str:
    """뱃지에 표시할 텍스트. ds 는 "All" 문자열이거나 선택된 데이터셋 key들의 set/list."""
    if ds == "All" or not ds:
        return "전체"
    keys = [ds] if isinstance(ds, str) else list(ds)
    return " + ".join(_DATASET_LABEL_BY_KEY.get(k, k) for k in keys)


def _dataset_chip_color(ds) -> str:
    if ds == "All" or not ds:
        return "grey"
    keys = [ds] if isinstance(ds, str) else list(ds)
    if len(keys) == 1:
        return _DATASET_CHIP_COLOR.get(keys[0], "grey")
    return "indigo"   # 여러 데이터셋을 함께 선택한 경우

# ── search mode options ───────────────────────────────────────────────────────────────────────────────────
_MODE_OPTIONS = [
    ("Hybrid", "hybrid"),
    ("Vector", "vector"),
    ("Text",   "text"),
]


def _add_subgraph_widget(graph_id: str, page_client) -> None:
    """현재 NiceGUI 컨텍스트 안에 서브그래프 토글 위젯을 추가한다."""
    state = {'shown': False, 'loaded': False}

    toggle_btn = (
        ui.button('서브그래프 보기', icon='account_tree')
        .props('flat dense no-caps')
        .style(
            'color:#6366f1; font-size:12px; font-weight:500; margin-top:8px;'
            'border:1px solid #e0e7ff; background:#f5f3ff; border-radius:6px; padding:2px 10px;'
        )
    )

    frame_container = ui.element('div').style(
        'width:100%; margin-top:6px; height:440px; '
        'border:1px solid #c7d2fe; border-radius:8px; overflow:hidden; background:white;'
    )
    with frame_container:
        # iframe 을 NiceGUI element 로 직접 만들어 src 를 Python prop 으로 제어한다.
        # (run_javascript / getElementById 방식은 슬롯 컨텍스트·타이밍 문제가 있었음)
        iframe = ui.element('iframe').style(
            'width:100%;height:100%;border:none;display:block;'
        ).props('src=about:blank')
    frame_container.set_visibility(False)

    async def _toggle():
        state['shown'] = not state['shown']
        frame_container.set_visibility(state['shown'])
        toggle_btn.props(f"icon={'expand_less' if state['shown'] else 'account_tree'}")
        if state['shown'] and not state['loaded']:
            state['loaded'] = True
            # 컨테이너가 보이게 된 뒤 iframe src 를 주입 → vis.js 가 올바른 크기로 렌더링
            await asyncio.sleep(0.05)
            iframe.props(f'src=/graph/{graph_id}')

    toggle_btn.on('click', _toggle)


class _PageState:
    def __init__(self):
        # ★ 여러 데이터셋을 동시에 선택할 수 있도록 set 으로 관리한다.
        #   {"All"} 이면 "전체"가 선택된 상태(= LLM 이 자동으로 데이터셋을 고름).
        #   특정 데이터셋을 하나 이상 고르면 "All"은 자동으로 빠지고, 고른
        #   데이터셋들만 검색 대상이 된다.
        self.dataset: set[str]  = {"All"}
        self.search_mode: str   = "hybrid"
        self.use_2hop: bool     = True   # TODO: wire to rag_engine when per-session hops are supported
        self.conversations: list[dict] = []   # {id, title, messages}
        self.active_conv_id: int | None = None
        self.messages: list[dict] = []


def build_chat_page(request: Request = None):
    state = _PageState()
    rag   = GraphRAG()

    # ★ 접속한 사용자의 IP — 검색 기록 로그(누가 무엇을 검색했는지)에 남기기 위함.
    #   NiceGUI 가 @ui.page 함수에 request 를 자동으로 주입해준다.
    session_ip = _client_ip(request) if request is not None else 'unknown'

    # capture client at page-build time — this is the only moment slot context is guaranteed
    from nicegui import context as _ctx
    _page_client = _ctx.client

    # ── global styles ─────────────────────────────────────────────────────────────────────────────────
    ui.add_head_html(r'''
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <script>
        // ★ MathJax v3 설정은 반드시 로더 스크립트보다 "먼저" 정의돼야 한다.
        //   - inlineMath 에 $...$ 도 추가 (기본은 \\(...\\) 만 인식)
        //   - code/pre/textarea 안의 $ 는 건드리지 않도록 skip
        //   - 사내망에서 CDN 이 막히면 수식은 그냥 원문 텍스트로 보이며 앱은 정상 동작
        window.MathJax = {
            tex: {
                // ★ '$' 자동 인식은 끈다. 진짜 수식은 _convert_math_delims 가 \(...\) /
                //   \[...\] 로 변환해 넘기므로, 여기서 '$' 를 켜두면 답변에 우연히 남은
                //   짝 안 맞는 '$' 두 개 사이의 일반 텍스트(한글 등)까지 수식으로 처리돼
                //   이탤릭·공백소실이 발생한다. \(...\) / \[...\] 만 처리하게 한다.
                inlineMath: [['\\(', '\\)']],
                displayMath: [['\\[', '\\]']]
            },
            // scale 0.9 로 주위 본문 글자 크기와 비슷하게 맞춘다 (기본값은 살짝 더 큼)
            chtml: { scale: 0.9, matchFontHeight: true },
            options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
        };
    </script>
    <!-- ★ 외부 CDN 대신 로컬로 서빙 (main.py 의 app.add_static_files('/mathjax', ...) 참고).
         사내망이 외부 CDN 을 막거나 지연시켜 수식이 렌더링 안 되는 문제를 근본적으로 없앤다. -->
    <script async src="/mathjax/tex-mml-chtml.js"></script>
    <style>
        *, *::before, *::after { font-family: "Inter", sans-serif; box-sizing: border-box; }
        body, html { margin: 0; padding: 0; overflow: hidden; background: #f8fafc; }
        .q-page { min-height: 0 !important; height: calc(100vh - 52px) !important; padding: 0 !important; }
        .nicegui-content { padding: 0 !important; margin: 0 !important; position: absolute; inset: 0; }
        .q-scrollarea__thumb--v { opacity: 0.4 !important; width: 4px !important; border-radius: 4px !important; }

        /* chat bubbles */
        .user-bubble { background: linear-gradient(135deg, #6366f1, #8b5cf6); word-break: break-word; white-space: pre-wrap; }
        .ai-bubble { background: #f1f5f9; overflow-x: auto; color: #334155; }
        /* ★ markdown2 는 단일 줄바꿈(\n)을 <br> 로 바꾸지 않고 그냥 이어붙인다.
           HTML 은 원래 줄바꿈 문자를 무시하고 한 줄로 붙여 보여주므로,
           pre-wrap 이 없으면 LLM 이 줄바꿈으로만 구분한 목록/문단이 전부
           한 줄로 뭉쳐 보인다. 표(<table>)/코드(<pre>)는 이미 자체 줄 구조가
           있는 블록 요소라 pre-wrap 의 영향을 받지 않는다. */
        .ai-bubble p, .ai-bubble li { color: #334155; margin: 0; white-space: pre-wrap; }
        .ai-bubble td { color: #334155; margin: 0; }
        .ai-bubble * { color: #334155; }
        .ai-bubble code { background: #e2e8f0; color: #1e293b; padding: 1px 5px; border-radius: 4px; }
        .ai-bubble pre { background: #e2e8f0; padding: 8px 12px; border-radius: 8px; overflow-x: auto; }
        /* links inside answers */
        .ai-bubble a, .ai-bubble a * { color: #4f46e5 !important; text-decoration: underline; cursor: pointer; word-break: break-all; }
        .ai-bubble a:hover { color: #6366f1 !important; }

        /* input field — force light mode regardless of Quasar dark setting */
        .q-field__native, .q-field__input { color: #1e293b !important; }
        .q-field__label { color: #64748b !important; }
        .q-field--outlined .q-field__control { background: white !important; }

        /* checkbox labels */
        .q-checkbox__label { color: #475569 !important; font-size: 13px; }

        /* scroll area inner content fills height so messages stay at bottom */
        .q-scrollarea__content { min-height: 100% !important; width: 100% !important; display: flex !important; flex-direction: column !important; }

        /* header-input (dark header 위 입력창) */
        .header-input .q-field__control:before { border-color: rgba(255,255,255,0.15) !important; }
        .header-input .q-field__control:hover:before { border-color: rgba(255,255,255,0.3) !important; }

        /* hover animations */
        .hover-btn { transition: all 0.2s ease; }
        .hover-btn:hover { transform: scale(1.05); filter: brightness(1.1); }

        /* sidebar resize handle */
        #kmap-sidebar-resizer:hover, #kmap-sidebar-resizer.resizing { background: #a5b4fc !important; }

        /* 인라인 수식이 본문 글자 크기와 어긋나지 않게 */
        .ai-bubble mjx-container { font-size: inherit !important; }
        .ai-bubble mjx-container[display="true"] { margin: 0.4em 0 !important; }

        /* 마크다운 강조(*...* / _..._)를 이탤릭 대신 굵게로 표시한다.
           한글은 전용 이탤릭 글꼴이 없어 기울이면 작고 어색하게 렌더링되기 때문. */
        .ai-bubble em, .ai-bubble i { font-style: normal !important; font-weight: 600; }
    </style>
    <script>
        // 답변이 렌더링/갱신된 뒤 호출하면 새로 들어온 수식을 다시 typeset 한다.
        window.kmapTypeset = function () {
            if (window.MathJax && window.MathJax.typesetPromise) {
                window.MathJax.typesetPromise().catch(function () {});
            }
        };

        // 답변(.ai-bubble) 안의 링크 클릭 시 새 탭에서 열기 (이벤트 위임 — 동적 콘텐츠에도 적용)
        document.addEventListener('click', function (e) {
            var a = e.target.closest && e.target.closest('.ai-bubble a');
            if (a && a.href) {
                e.preventDefault();
                window.open(a.href, '_blank', 'noopener,noreferrer');
            }
        });

        // ★ 사이드바 폭 마우스 드래그 리사이즈.
        //   폴링 없이 리사이저 엘리먼트가 DOM에 나타날 때까지 짧게 재시도만 하고,
        //   이후 로직은 전부 순수 클라이언트 이벤트라 타이밍 경쟁 문제가 없다.
        //   localStorage 에 폭을 저장해 새로고침 후에도 유지한다.
        (function () {
            function initSidebarResize() {
                var sidebar  = document.getElementById('kmap-sidebar');
                var resizer  = document.getElementById('kmap-sidebar-resizer');
                if (!sidebar || !resizer) {
                    setTimeout(initSidebarResize, 200);
                    return;
                }
                if (resizer.dataset.kmapBound) return;   // 중복 바인딩 방지
                resizer.dataset.kmapBound = '1';

                var saved = localStorage.getItem('kmap-sidebar-width');
                if (saved) sidebar.style.width = saved + 'px';

                var dragging = false;
                resizer.addEventListener('mousedown', function (e) {
                    dragging = true;
                    resizer.classList.add('resizing');
                    document.body.style.userSelect = 'none';
                    document.body.style.cursor = 'col-resize';
                    e.preventDefault();
                });
                document.addEventListener('mousemove', function (e) {
                    if (!dragging) return;
                    var rect = sidebar.getBoundingClientRect();
                    var w = e.clientX - rect.left;
                    w = Math.max(160, Math.min(480, w));
                    sidebar.style.width = w + 'px';
                });
                document.addEventListener('mouseup', function () {
                    if (!dragging) return;
                    dragging = false;
                    resizer.classList.remove('resizing');
                    document.body.style.userSelect = '';
                    document.body.style.cursor = '';
                    localStorage.setItem('kmap-sidebar-width', parseInt(sidebar.style.width, 10));
                });
            }
            initSidebarResize();
        })();
    </script>
    ''')

    # ── topbar ──────────────────────────────────────────────────────────────────────────────────────
    with ui.header().style(
        'height:52px; min-height:52px; padding:0;'
        'background:linear-gradient(135deg,#1e1b4b 0%,#4338ca 100%);'
        'border-bottom:1px solid rgba(255,255,255,0.08); box-shadow:0 1px 12px rgba(0,0,0,0.2);'
    ):
        with ui.element('div').style(
            'display:flex; align-items:center; width:100%; height:52px; padding:0 16px; gap:12px;'
        ):
            ui.html('<i class="material-icons" style="font-size:20px;color:rgba(255,255,255,0.9);line-height:1;">hub</i>')
            ui.label('KMap').style(
                'font-size:15px; font-weight:600; color:white; letter-spacing:-0.03em; white-space:nowrap;'
            )
            ui.label('SMEP-D').style(
                'font-size:11px; font-weight:500; color:white;'
                'padding:2px 8px; border-radius:999px; background:rgba(255,255,255,0.15); white-space:nowrap;'
            )
            ui.element('div').style('width:1px; height:20px; background:rgba(255,255,255,0.2);')

            dataset_btn_refs: dict[str, ui.button] = {}

            def _refresh_dataset_btn_styles():
                for k, btn in dataset_btn_refs.items():
                    if k in state.dataset:
                        btn.classes(remove='text-white/60 hover:text-white', add='bg-white/20 text-white')
                    else:
                        btn.classes(remove='bg-white/20 text-white', add='text-white/60 hover:text-white')

            def _on_dataset(key: str):
                # ★ 중복 선택 지원: "전체"를 누르면 단독 선택으로 리셋되고,
                #   개별 데이터셋은 토글(추가/해제)되며 서로 중복 선택 가능하다.
                #   개별 데이터셋을 하나라도 고르면 "전체"는 자동으로 빠지고,
                #   전부 해제되면 다시 "전체"로 되돌아간다(선택 없음 상태 방지).
                if key == "All":
                    state.dataset = {"All"}
                else:
                    if "All" in state.dataset:
                        state.dataset = set()
                    if key in state.dataset:
                        state.dataset.discard(key)
                    else:
                        state.dataset.add(key)
                    if not state.dataset:
                        state.dataset = {"All"}
                _refresh_dataset_btn_styles()

            for label, key in _DATASET_TABS:
                disabled = (key == "ExperimentsDB")
                btn = (
                    ui.button(label, on_click=lambda k=key: _on_dataset(k))
                    .props('flat dense')
                    .classes('px-3 py-1 rounded text-sm transition-colors hover-btn')
                )
                if key == "All":
                    btn.classes('bg-white/20 text-white')
                else:
                    btn.classes('text-white/60 hover:text-white')
                if disabled:
                    btn.disable()
                    btn.tooltip('준비중')
                dataset_btn_refs[key] = btn

            ui.element('div').style('flex:1;')

            mode_label = ui.label('hybrid').style('font-size:11px; color:rgba(255,255,255,0.5);')
            mode_label.bind_text_from(state, 'search_mode')

    # ── body ────────────────────────────────────────────────────────────────────────────────────────
    with ui.element('div').style(
        'display:flex; flex-direction:row; width:100%; height:calc(100vh - 52px); overflow:hidden;'
    ):
        # ── sidebar ─────────────────────────────────────────────────────────────────────────────────────
        # ★ 폭을 마우스 드래그로 조절할 수 있도록 id 를 부여하고, 오른쪽에 리사이즈
        #   핸들(얇은 세로 바)을 둔다. 실제 드래그 로직은 아래 ui.run_javascript 로
        #   한 번만 주입한다 (min 160px ~ max 480px, localStorage 에 폭 저장).
        with ui.element('div').props('id=kmap-sidebar').style(
            'width:225px; min-width:160px; max-width:480px; flex-shrink:0; background:#f1f5f9; '
            'border-right:1px solid #e2e8f0; padding:12px; display:flex; flex-direction:column; gap:12px; overflow-y:auto; overflow-x:hidden;'
        ):
            with ui.element('div').style('display:flex; align-items:center; justify-content:space-between;'):
                ui.label('대화 기록').style(
                    'font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.05em;'
                )
                ui.button(icon='add', on_click=lambda: _new_conversation()).props('flat round dense').classes(
                    'hover-btn'
                ).style('color:#64748b;').tooltip('새 대화')

            conv_list = ui.element('div').style('display:flex; flex-direction:column; gap:4px; width:100%;')

            ui.separator().style('border-color:#e2e8f0;')

            ui.label('검색 모드').style(
                'font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.05em;'
            )

            mode_btns: dict[str, ui.button] = {}

            def _on_mode(m: str):
                state.search_mode = m
                for k, b in mode_btns.items():
                    if k == m:
                        b.classes(remove='text-slate-500 hover:text-slate-800', add='bg-indigo-600 text-white')
                    else:
                        b.classes(remove='bg-indigo-600 text-white', add='text-slate-500 hover:text-slate-800')

            with ui.element('div').style('display:flex; flex-direction:row; gap:4px; width:100%;'):
                for label, val in _MODE_OPTIONS:
                    b = (
                        ui.button(label, on_click=lambda v=val: _on_mode(v))
                        .props('flat dense no-caps')
                        .classes('flex-1 rounded transition-colors hover-btn')
                        .style('font-size:12px; padding:4px 0; min-height:0;')
                    )
                    b.classes('bg-indigo-600 text-white' if val == state.search_mode else 'text-slate-500 hover:text-slate-800')
                    mode_btns[val] = b

            hop_chk = ui.checkbox('2-hop 탐색', value=state.use_2hop).props('dense').style(
                'color:#64748b; font-size:12px; margin-top:2px;'
            )
            hop_chk.bind_value(state, 'use_2hop')
            hop_chk.tooltip('2-hop: 검색된 노드의 이웃 노드까지 확장 탐색')

            ui.separator().style('border-color:#e2e8f0;')

            # ── 벡터 검색 컷오프 튜닝 (접힘 상태로 시작) ────────────────────────────
            # rag.entity_score_min / relative_gap / doc_score_min 는 GraphRAG 인스턴스
            # 속성이라, 여기서 바로 값을 바꾸면 다음 검색부터 즉시 반영된다.
            with ui.expansion('벡터 검색 튜닝', icon='tune').props('dense').classes('w-full').style(
                'font-size:12px; color:#64748b;'
            ):
                with ui.element('div').style('display:flex; flex-direction:column; gap:6px; padding-top:2px;'):
                    entity_min_input = ui.number(
                        value=rag.entity_score_min, min=0.0, max=1.0, step=0.01, format='%.2f'
                    ).props('dense outlined label="엔티티 최소 유사도"').style('width:100%;')
                    entity_min_input.on_value_change(
                        lambda e: setattr(rag, 'entity_score_min', e.value)
                    )

                    gap_input = ui.number(
                        value=rag.relative_gap, min=0.0, max=1.0, step=0.01, format='%.2f'
                    ).props('dense outlined label="상대 컷오프 (1등 대비 격차)"').style('width:100%;')
                    gap_input.on_value_change(
                        lambda e: setattr(rag, 'relative_gap', e.value)
                    )

                    doc_min_input = ui.number(
                        value=rag.doc_score_min, min=0.0, max=1.0, step=0.01, format='%.2f'
                    ).props('dense outlined label="문서 최소 유사도"').style('width:100%;')
                    doc_min_input.on_value_change(
                        lambda e: setattr(rag, 'doc_score_min', e.value)
                    )

                    ui.separator().style('border-color:#e2e8f0; margin:2px 0;')

                    # ★ 터미널 디버그 출력량 조절. rag.debug_level 도 인스턴스 속성이라
                    #   즉시 반영된다 (서버 재시작 불필요).
                    debug_level_select = ui.select(
                        {0: '0 - 에러만', 1: '1 - 요약 (기본)', 2: '2 - 보통', 3: '3 - 상세(전체)'},
                        value=rag.debug_level,
                    ).props('dense outlined label="디버그 출력량"').style('width:100%;')
                    debug_level_select.on_value_change(
                        lambda e: setattr(rag, 'debug_level', e.value)
                    )

                    # ★ 답변 생성용 LLM 모델 선택. rag.answer_llm 은 인스턴스 속성이라
                    #   다음 질문부터 즉시 반영된다 (서버 재시작 불필요).
                    llm_select = ui.select(
                        {
                            None:         'GPT-OSS 120B (기본)',
                            'GaussO4.1':  'GaussO4.1',
                            'Gemma4':     'Gemma4 (경량/빠름)',
                        },
                        value=rag.answer_llm,
                    ).props('dense outlined label="답변 생성 모델"').style('width:100%;')
                    llm_select.on_value_change(
                        lambda e: setattr(rag, 'answer_llm', e.value)
                    )

            ui.separator().style('border-color:#e2e8f0;')

        # ── sidebar resize handle ──────────────────────────────────────────────────────────────────────
        ui.element('div').props('id=kmap-sidebar-resizer').style(
            'width:5px; flex-shrink:0; cursor:col-resize; background:transparent; '
            'transition:background 0.15s;'
        )

        # ── chat area ────────────────────────────────────────────────────────────────────────────────────
        # ★ 챗봇 첫 화면 UX: 대화가 없을 때는 입력창이 화면 중앙에 크게 떠 있다가,
        #   첫 질문을 보내면 지금의 "상단 스크롤 + 하단 고정 입력창" 구조로 전환된다.
        #   input_box/send_btn 은 인스턴스를 하나만 만들고, 두 레이아웃 사이를
        #   ui 엘리먼트의 .move() 로 그대로 옮겨 재사용한다(값/이벤트 핸들러 유지).
        with ui.element('div').style(
            'flex:1; min-width:0; position:relative; overflow:hidden; background:#f8fafc;'
        ):
            # ── 중앙 시작 화면 (대화가 비어 있을 때) ──────────────────────────────────────
            center_wrap = ui.element('div').style(
                'position:absolute; inset:0; display:flex; flex-direction:column; '
                'align-items:center; justify-content:center; gap:20px; padding:24px; '
                'background:#f8fafc;'
            )
            with center_wrap:
                with ui.element('div').style('display:flex; flex-direction:column; align-items:center; gap:8px;'):
                    ui.avatar(icon='auto_awesome', color='indigo-1', text_color='indigo').style(
                        'width:48px; height:48px; font-size:24px;'
                    )
                    ui.label('KMap-Agent').style(
                        'font-size:20px; font-weight:600; color:#1e293b;'
                    )
                    ui.label('논문·내부문서·데이터를 GraphRAG/RAG로 검색합니다. 무엇이든 질문해보세요!').style(
                        'font-size:13px; color:#64748b;'
                    )
                center_input_slot = ui.element('div').style('width:100%; max-width:640px;')

            # ── 대화 화면 (첫 질문 이후) ─────────────────────────────────────────────────
            chat_layout = ui.element('div').style(
                'position:absolute; inset:0; display:none; flex-direction:column; overflow:hidden;'
            )
            with chat_layout:
                scroll_area = ui.scroll_area().style('flex:1; min-height:0; width:100%; background:#f8fafc;')
                with scroll_area:
                    chat_container = ui.element('div').style(
                        'display:flex; flex-direction:column; gap:8px; padding:20px; min-height:100%; width:100%;'
                    )
                    with chat_container:
                        ui.element('div').style('flex:1;')   # 메시지가 적을 때 아래로 붙게 하는 스페이서

                # ── bottom bar ─────────────────────────────────────────────────────────────────────────────
                bottom_bar = ui.element('div').style(
                    'flex-shrink:0; background:#f8fafc; border-top:1px solid #e2e8f0; padding:10px 16px 12px;'
                )

            # ── 입력창 (처음엔 중앙, 첫 질문 후 하단으로 이동) ────────────────────────────
            with center_input_slot:
                with ui.element('div').style('display:flex; align-items:flex-end; gap:8px; width:100%;') as input_row:
                    input_box = (
                        ui.textarea(placeholder='질문을 입력하세요… (Shift+Enter: 줄바꿈, Enter: 전송)')
                        .classes('flex-grow text-sm')
                        .style('font-size:14px;')
                        .props('outlined rounded dense autogrow')
                        # 줄바꿈 방지(preventDefault)는 아래 .on('keydown.enter.exact.prevent')
                        # 에서 처리한다 (NiceGUI 가 클라이언트 측 리스너를 올바른 엘리먼트에
                        # 직접 붙여주고 .prevent 도 브라우저에서 즉시 적용되므로, .props 로
                        # @keydown 을 넣는 방식보다 안정적으로 동작한다).
                    )
                    send_btn = (
                        ui.button(icon='arrow_upward')
                        .props('round unelevated')
                        .classes('hover-btn')
                        .style(
                            'background:linear-gradient(135deg,#6366f1,#8b5cf6);'
                            'color:white; min-width:36px; min-height:36px;'
                        )
                    )

            def _show_chat_layout():
                """첫 질문 전송 시: 중앙 화면 → 상단 스크롤+하단 입력창 구조로 전환."""
                if 'moved' not in _layout_state:
                    input_row.move(bottom_bar)
                    _layout_state['moved'] = True
                center_wrap.style('display:none')
                chat_layout.style('display:flex')

            def _show_center_layout():
                """새 대화 시작 시: 하단 입력창 → 중앙 화면으로 복귀."""
                input_row.move(center_input_slot)
                _layout_state.pop('moved', None)
                chat_layout.style('display:none')
                center_wrap.style('display:flex')

            _layout_state: dict = {}

        # ── helper: conversation list refresh ───────────────────────────────────────────────────────
        def _refresh_conv_list():
            conv_list.clear()
            with conv_list:
                for conv in reversed(state.conversations):
                    cid   = conv['id']
                    title = conv['title']
                    is_active = cid == state.active_conv_id
                    (
                        ui.button(title, on_click=lambda c=conv: _load_conversation(c))
                        .props('flat dense align=left')
                        .classes(
                            'w-full text-sm rounded px-2 py-1.5 truncate text-left hover-btn ' +
                            ('bg-indigo-100 text-indigo-800' if is_active else 'text-slate-500 hover:bg-slate-200')
                        )
                    )

        def _new_conversation():
            if state.messages:
                state.messages = []
                state.active_conv_id = None
                chat_container.clear()
                with chat_container:
                    ui.element('div').style('flex:1;')
                rag.clear_history()
                _refresh_conv_list()
                _show_center_layout()

        def _load_conversation(conv: dict):
            state.active_conv_id = conv['id']
            state.messages = conv['messages']
            chat_container.clear()
            with chat_container:
                for msg in state.messages:
                    _render_message(msg)
            if state.messages:
                _show_chat_layout()
            scroll_area.scroll_to(percent=1.0)
            _refresh_conv_list()
            # 불러온 대화의 수식도 다시 typeset
            try:
                _page_client.run_javascript('window.kmapTypeset && window.kmapTypeset()')
            except Exception:
                pass

        # ── helper: render a saved message ─────────────────────────────────────────────────────────────
        def _render_message(msg: dict):
            role    = msg.get('role', 'user')
            content = msg.get('content', '')
            ds      = msg.get('dataset', 'All')
            graph_id = msg.get('graph_id')

            if role == 'user':
                with ui.element('div').style('display:flex; flex-direction:column; align-items:flex-end; gap:4px; width:100%;'):
                    ui.badge(_dataset_display(ds), color=_dataset_chip_color(ds)).classes('text-xs')
                    with ui.element('div').classes(
                        'user-bubble rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm text-white'
                    ).style('max-width:80%;'):
                        ui.label(content)
            else:
                with ui.element('div').style('display:flex; align-items:flex-start; gap:8px;').classes('ai-msg'):
                    ui.avatar(icon='auto_awesome', color='indigo-1', text_color='indigo').style(
                        'width:24px; height:24px; min-width:24px; font-size:12px; flex-shrink:0;'
                    )
                    with ui.element('div').classes(
                        'ai-bubble rounded-2xl rounded-tl-sm px-4 py-2.5 text-sm'
                    ).style('max-width:calc(100% - 36px); color:#334155;'):
                        ui.markdown(_linkify(content), extras=['tables', 'fenced-code-blocks'])
                        if graph_id:
                            _add_subgraph_widget(graph_id, _page_client)

        # ── send handler ─────────────────────────────────────────────────────────────────────────────────────────
        async def on_send_message():
            query = input_box.value.strip()
            if not query:
                return

            input_box.value = ''
            send_btn.disable()

            # ★ "전체"만 선택돼 있으면 rag 쪽에는 "All" 문자열로(자동 데이터셋 선택),
            #   특정 데이터셋을 하나 이상 골랐으면 그 key 들의 리스트로 넘긴다.
            current_dataset = "All" if "All" in state.dataset else list(state.dataset)

            # ★ 검색 기록 로그: 누가(IP) 언제 무엇을 검색했는지 logs/search.log 에 남긴다.
            log_search(session_ip, _dataset_display(current_dataset), state.search_mode, query)
            current_mode    = state.search_mode

            # ── create conversation if first message ────────────────────────────────────────────────
            if not state.messages:
                conv = {
                    'id':       len(state.conversations),
                    'title':    query[:28] + ('…' if len(query) > 28 else ''),
                    'messages': state.messages,
                }
                state.conversations.append(conv)
                state.active_conv_id = conv['id']
                _refresh_conv_list()
                _show_chat_layout()   # ★ 첫 질문: 중앙 화면 → 대화 화면으로 전환
                # ★ input_row.move() 로 입력창 DOM 이 재배치되면서, 그 전에 비워둔 값이
                #   화면에 다시 첫 질문 텍스트로 남아 보이는 문제가 있었다. 이동 직후
                #   한 번 더 명시적으로 비워 확실히 지운다.
                input_box.value = ''

            # ── user bubble ─────────────────────────────────────────────────────────────────────────────────────
            user_msg = {'role': 'user', 'content': query, 'dataset': current_dataset}
            state.messages.append(user_msg)

            with chat_container:
                with ui.element('div').style('display:flex; flex-direction:column; align-items:flex-end; gap:4px; width:100%;'):
                    ui.badge(_dataset_display(current_dataset), color=_dataset_chip_color(current_dataset)).classes('text-xs')
                    with ui.element('div').classes(
                        'user-bubble rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm text-white'
                    ).style('max-width:80%;'):
                        ui.label(query)

            scroll_area.scroll_to(percent=1.0)

            # ── AI streaming bubble ──────────────────────────────────────────────────────────────────────────────
            ai_col_ref = None

            with chat_container:
                with ui.element('div').style(
                    'display:flex; align-items:flex-start; gap:8px;'
                ).classes('ai-msg'):
                    ui.avatar(icon='auto_awesome', color='indigo-1', text_color='indigo').style(
                        'width:24px; height:24px; min-width:24px; font-size:12px; flex-shrink:0;'
                    )
                    with ui.element('div').classes(
                        'ai-bubble rounded-2xl rounded-tl-sm px-4 py-2.5 text-sm'
                    ).style('max-width:calc(100% - 36px); color:#334155; min-width:60px;') as ai_col_ref:
                        # 진행 단계 표시: 스피너 + 상태 텍스트를 한 줄에 둔다.
                        # 첫 답변 청크가 오면 이 상태 박스를 지우고 markdown 으로 교체.
                        status_box = ui.element('div').style(
                            'display:flex; align-items:center; gap:8px;'
                        )
                        with status_box:
                            spinner = ui.spinner('dots', size='1.2em', color='indigo')
                            status_lbl = ui.label('').style('font-size:12px; color:#94a3b8;')

            await asyncio.sleep(0.05)
            scroll_area.scroll_to(percent=1.0)

            # ── run sync generator in executor ───────────────────────────────────────────────────────────────
            loop    = asyncio.get_event_loop()
            gen     = rag.answer_stream(query, current_dataset, current_mode)

            full_text    = ''
            chunk_buffer = ''
            chunk_count  = 0
            md_element   = None

            _SCROLL_JS = """
                var el = document.querySelector('.q-scrollarea__container');
                if (el) {
                    var nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
                    if (nearBottom) {
                        var msgs = document.querySelectorAll('.ai-msg');
                        if (msgs.length) msgs[msgs.length-1].scrollIntoView({behavior:'smooth',block:'end'});
                    }
                }
            """

            # ★ LLM 이 (특히 표를 그리다가) 같은 글자를 무한히 반복하며 답변이 끝나지
            #   않는 "반복 루프" 실패 모드에 빠지는 경우가 있다(약한/경량 모델에서
            #   컨텍스트가 크거나 표 형식을 요청할 때 관찰됨, 예: "|" 수천 개 반복).
            #   같은 글자가 임계값 넘게 연속되면 반복 루프로 간주하고 스트림 소비를
            #   중단해, 사용자가 끝없이 같은 글자만 보는 걸 막는다.
            #   ★★ 단, 표 구분선("|---|---|"), 구분선(------), 코드 들여쓰기(공백) 등은
            #   정상적인 마크다운에서도 같은 문자가 길게 연속될 수 있다 — 처음엔 이걸
            #   구분 안 하고 다 걸러서, 열이 많은 표를 그리는 도중에 답변이 통째로
            #   끊기는 문제가 있었다. 표/구분선에 흔히 쓰이는 문자는 이 검사에서
            #   제외하고, 그 외 문자(일반 글자/숫자)가 이상하게 반복될 때만 잡는다.
            #   표/구분선에 흔히 쓰이는 문자(-, |, = 등)는 그 자체로 반복되는 게
            #   비정상은 아니므로 임계값을 훨씬 높게 잡고(표가 아무리 넓어도 100개
            #   넘게 한 문자가 끊김 없이 이어지진 않음), 그 외 일반 문자/숫자는
            #   더 낮은 임계값으로 빠르게 잡는다.
            _TABLE_SAFE_CHARS = '-|=_*~. \t'
            _REPEAT_RE = re.compile(
                r'([^' + re.escape(_TABLE_SAFE_CHARS) + r'])\1{39,}$|'
                r'([' + re.escape(_TABLE_SAFE_CHARS) + r'])\2{99,}$',
                re.DOTALL
            )

            while True:
                event = await loop.run_in_executor(None, next, gen, None)
                if event is None:
                    if chunk_buffer:
                        full_text += chunk_buffer
                        if md_element:
                            md_element.set_content(full_text)
                    break

                # answer_stream 은 dict 이벤트를 내보낸다. (구버전 호환: 문자열이면 content 취급)
                if isinstance(event, dict):
                    etype = event.get('type', 'content')
                    etext = event.get('text', '')
                else:
                    etype, etext = 'content', str(event)

                # 진행 상태 이벤트: 상태줄만 갱신하고 다음 이벤트 대기
                if etype == 'status':
                    if md_element is None:      # 아직 답변 시작 전일 때만 표시
                        status_lbl.set_text(etext)
                        await asyncio.sleep(0)
                    continue

                chunk = etext

                # first content chunk: replace status box with markdown element
                if md_element is None:
                    status_box.delete()
                    with ai_col_ref:
                        md_element = ui.markdown('', extras=['tables', 'fenced-code-blocks'])

                chunk_buffer += chunk
                chunk_count  += 1

                # ★ 매 청크마다 반복 루프 여부를 검사한다 (누적되기 전에 빨리 잡아야
                #   불필요한 토큰 생성/네트워크 낭비와 화면 스팸을 최소화할 수 있음).
                if _REPEAT_RE.search(full_text + chunk_buffer):
                    # 반복된 꼬리 부분은 잘라내고, 반복 시작 직전까지만 남긴다.
                    combined = _REPEAT_RE.sub('', full_text + chunk_buffer)
                    full_text, chunk_buffer = combined, ''
                    full_text += "\n\n_(반복 오류가 감지되어 답변 생성을 중단했습니다. 다시 질문해 주세요.)_"
                    md_element.set_content(full_text)
                    print("[Debug] LLM 응답 반복 루프 감지 → 스트림 중단")
                    break

                if chunk_count % 5 == 0:
                    full_text    += chunk_buffer
                    chunk_buffer  = ''
                    md_element.set_content(full_text + '▌')
                    try:
                        await _page_client.run_javascript(_SCROLL_JS)
                    except Exception:
                        pass
                    await asyncio.sleep(0)

            if md_element is None:
                status_box.delete()
                with ai_col_ref:
                    md_element = ui.markdown('(응답을 받지 못했습니다)')
            else:
                md_element.set_content(_linkify(full_text))

            # 답변이 최종 확정된 뒤 수식(LaTeX)을 typeset (스트리밍 중엔 하지 않아 깜빡임 방지)
            try:
                await _page_client.run_javascript('window.kmapTypeset && window.kmapTypeset()')
            except Exception:
                pass

            # ── subgraph toggle ───────────────────────────────────────────────────────────────────────────────────
            # ★ rag.last_retrieved_nodes 는 "검색 단계에서 트리플이 나왔는지"만 알려준다 —
            #   검색은 됐지만 최종 답변이 문서 목록/표 등으로만 구성되어 그 관계를 실제로
            #   언급하지 않은 경우에도 non-empty 라서 위젯이 계속 뜨는 문제가 있었다.
            #   답변 텍스트에 검색된 엔티티 이름이 실제로 등장하는지 확인해, 하나도
            #   안 쓰였으면(=답변이 트리플을 활용하지 않았으면) 위젯을 띄우지 않는다.
            _norm_full_text = _normalize_for_mention_check(full_text)
            used_nodes = [
                n for n in rag.last_retrieved_nodes
                if n and _normalize_for_mention_check(n) in _norm_full_text
            ]
            graph_id = None
            if used_nodes and ai_col_ref is not None:
                try:
                    graph_id = _neo4j_viz.generate_rag_result_graph(
                        rag.last_retrieved_nodes, current_dataset
                    )
                    print(f"[subgraph] widget 생성: graph_id={graph_id}, "
                          f"nodes={len(rag.last_retrieved_nodes)} (답변에 언급된 노드 {len(used_nodes)}개)")
                    with ai_col_ref:
                        _add_subgraph_widget(graph_id, _page_client)
                except Exception as e:
                    print(f"서브그래프 생성 오류: {e}")
            elif rag.last_retrieved_nodes:
                print(f"[subgraph] 생략: 검색된 노드 {len(rag.last_retrieved_nodes)}개 중 "
                      f"답변에 언급된 노드가 없음")

            # ── save message ────────────────────────────────────────────────────────────────────────────────────────
            ai_msg = {'role': 'assistant', 'content': full_text, 'graph_id': graph_id}
            state.messages.append(ai_msg)

            scroll_area.scroll_to(percent=1.0)
            send_btn.enable()

        async def _on_enter(e):
            await on_send_message()

        send_btn.on('click', on_send_message)
        # ★ 단일 바인딩으로 "줄바꿈 방지 + 전송" 을 모두 처리한다.
        #   - .exact  : Shift/Ctrl/Alt/Meta 등 보조키 없이 순수 Enter 일 때만 발동
        #               (→ Shift+Enter 는 자연스럽게 줄바꿈으로 통과)
        #   - .prevent: NiceGUI 가 클라이언트 리스너에 preventDefault 를 걸어주므로,
        #               서버 왕복 전에 브라우저에서 즉시 줄바꿈 기본동작을 막는다
        #               (→ 첫 질문에서 줄바꿈이 새던 타이밍 문제 해소)
        input_box.on('keydown.enter.exact.prevent', _on_enter)
