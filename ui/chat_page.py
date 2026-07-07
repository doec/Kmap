from __future__ import annotations

import asyncio
import os
import re

from nicegui import ui

from rag_engine import GraphRAG, REPORTS_DATASET, PAPERS_DATASET, CONFLUENCE_DATASET, DATASETS
from retriever.neo4j_retriever import Neo4jRetriever

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


def _linkify(text: str) -> str:
    """맨 URL과 DOI 문자열을 클릭 가능한 마크다운 링크로 변환한다. (URL에 붙은 ** 제거)"""
    if not text:
        return text
    text = _BARE_URL_RE.sub(lambda m: f'[{m.group(1)}]({m.group(1)})', text)
    text = _DOI_RE.sub(
        lambda m: f'[{m.group(1)}](https://doi.org/{m.group(2)})', text
    )
    return text

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
_DATASET_TABS = [
    ("전체",    "All"),
    ("ReportsDB", REPORTS_DATASET),
    ("논문",    PAPERS_DATASET),
    ("Confluence", CONFLUENCE_DATASET),
    ("실험",    "ExperimentsDB"),   # future — disabled
]

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
        self.dataset: str       = "All"
        self.search_mode: str   = "hybrid"
        self.use_2hop: bool     = True   # TODO: wire to rag_engine when per-session hops are supported
        self.conversations: list[dict] = []   # {id, title, messages}
        self.active_conv_id: int | None = None
        self.messages: list[dict] = []


def build_chat_page():
    state = _PageState()
    rag   = GraphRAG()

    # capture client at page-build time — this is the only moment slot context is guaranteed
    from nicegui import context as _ctx
    _page_client = _ctx.client

    # ── global styles ─────────────────────────────────────────────────────────────────────────────────
    ui.add_head_html('''
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <style>
        *, *::before, *::after { font-family: "Inter", sans-serif; box-sizing: border-box; }
        body, html { margin: 0; padding: 0; overflow: hidden; background: #f8fafc; }
        .q-page { min-height: 0 !important; height: calc(100vh - 52px) !important; padding: 0 !important; }
        .nicegui-content { padding: 0 !important; margin: 0 !important; position: absolute; inset: 0; }
        .q-scrollarea__thumb--v { opacity: 0.4 !important; width: 4px !important; border-radius: 4px !important; }

        /* chat bubbles */
        .user-bubble { background: linear-gradient(135deg, #6366f1, #8b5cf6); word-break: break-word; white-space: pre-wrap; }
        .ai-bubble { background: #f1f5f9; overflow-x: auto; color: #334155; }
        .ai-bubble p, .ai-bubble li, .ai-bubble td { color: #334155; margin: 0; }
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
    </style>
    <script>
        // 답변(.ai-bubble) 안의 링크 클릭 시 새 탭에서 열기 (이벤트 위임 — 동적 콘텐츠에도 적용)
        document.addEventListener('click', function (e) {
            var a = e.target.closest && e.target.closest('.ai-bubble a');
            if (a && a.href) {
                e.preventDefault();
                window.open(a.href, '_blank', 'noopener,noreferrer');
            }
        });
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

            def _on_dataset(key: str):
                state.dataset = key
                for k, btn in dataset_btn_refs.items():
                    if k == key:
                        btn.classes(remove='text-white/60 hover:text-white', add='bg-white/20 text-white')
                    else:
                        btn.classes(remove='bg-white/20 text-white', add='text-white/60 hover:text-white')

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
                    btn.tooltip('준비 중 (Qdrant RAG 2단계)')
                dataset_btn_refs[key] = btn

            ui.element('div').style('flex:1;')

            mode_label = ui.label('hybrid').style('font-size:11px; color:rgba(255,255,255,0.5);')
            mode_label.bind_text_from(state, 'search_mode')

    # ── body ────────────────────────────────────────────────────────────────────────────────────────
    with ui.element('div').style(
        'display:flex; flex-direction:row; width:100%; height:calc(100vh - 52px); overflow:hidden;'
    ):
        # ── sidebar ─────────────────────────────────────────────────────────────────────────────────────
        with ui.element('div').style(
            'width:220px; min-width:220px; flex-shrink:0; background:#f1f5f9; '
            'border-right:1px solid #e2e8f0; padding:12px; display:flex; flex-direction:column; gap:12px; overflow-y:auto;'
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
                ui.label('엔티티 최소 유사도').style('font-size:11px; color:#94a3b8; margin-top:4px;')
                entity_min_input = ui.number(
                    value=rag.entity_score_min, min=0.0, max=1.0, step=0.01, format='%.2f'
                ).props('dense outlined').style('width:100%;')
                entity_min_input.on_value_change(
                    lambda e: setattr(rag, 'entity_score_min', e.value)
                )

                ui.label('상대 컷오프 (1등 대비 격차)').style('font-size:11px; color:#94a3b8; margin-top:4px;')
                gap_input = ui.number(
                    value=rag.relative_gap, min=0.0, max=1.0, step=0.01, format='%.2f'
                ).props('dense outlined').style('width:100%;')
                gap_input.on_value_change(
                    lambda e: setattr(rag, 'relative_gap', e.value)
                )

                ui.label('문서 최소 유사도').style('font-size:11px; color:#94a3b8; margin-top:4px;')
                doc_min_input = ui.number(
                    value=rag.doc_score_min, min=0.0, max=1.0, step=0.01, format='%.2f'
                ).props('dense outlined').style('width:100%;')
                doc_min_input.on_value_change(
                    lambda e: setattr(rag, 'doc_score_min', e.value)
                )

            ui.separator().style('border-color:#e2e8f0;')

        # ── chat area ────────────────────────────────────────────────────────────────────────────────────
        with ui.element('div').style(
            'flex:1; min-width:0; display:flex; flex-direction:column; overflow:hidden; background:#f8fafc;'
        ):
            scroll_area = ui.scroll_area().style('flex:1; min-height:0; width:100%; background:#f8fafc;')
            with scroll_area:
                chat_container = ui.element('div').style(
                    'display:flex; flex-direction:column; gap:8px; padding:20px; min-height:100%; width:100%;'
                )
                with chat_container:
                    # spacer — pushes messages to bottom when few messages exist
                    ui.element('div').style('flex:1;')
                    # welcome message
                    with ui.element('div').style('display:flex; align-items:flex-start; gap:8px;').classes('ai-msg'):
                        ui.avatar(icon='auto_awesome', color='indigo-1', text_color='indigo').style(
                            'width:24px; height:24px; min-width:24px; font-size:12px; flex-shrink:0;'
                        )
                        with ui.element('div').classes(
                            'ai-bubble rounded-2xl rounded-tl-sm px-4 py-2.5 text-sm'
                        ).style('color:#334155;'):
                            ui.markdown(
                                "안녕하세요! **KMap 연구 어시스턴트**입니다.  \n"
                                "논문·내부문서·데이터를 GraphRAG/RAG로 검색합니다. 무엇이든 질문해보세요!"
                            )

            # ── bottom bar ─────────────────────────────────────────────────────────────────────────────
            with ui.element('div').style(
                'flex-shrink:0; background:#f8fafc; border-top:1px solid #e2e8f0; padding:10px 16px 12px;'
            ):
                # input
                with ui.element('div').style('display:flex; align-items:flex-end; gap:8px; width:100%;'):
                    input_box = (
                        ui.textarea(placeholder='질문을 입력하세요… (Shift+Enter: 줄바꿈, Enter: 전송)')
                        .classes('flex-grow text-sm')
                        .style('font-size:14px;')
                        .props('outlined rounded dense autogrow')
                        # ★ Shift 없는 Enter 는 기본 동작(줄바꿈)을 막는다.
                        #   Vue 템플릿 속성으로 컴포넌트 자체에 바로 붙기 때문에, 별도
                        #   스크립트를 나중에 주입해 붙이는 방식(ui.run_javascript 폴링)과
                        #   달리 렌더링 시점에 곧바로 적용되어 타이밍 경쟁(race condition)이
                        #   없다. (기존 방식은 페이지 로드 직후 아주 빠르게 첫 질문을
                        #   입력하면 패치가 붙기 전에 Enter 가 눌려 줄바꿈이 새는 문제가 있었음)
                        .props('''@keydown="$event.key === 'Enter' && !$event.shiftKey && $event.preventDefault()"''')
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
                    with ui.element('div').style('display:flex; align-items:flex-start; gap:8px;').classes('ai-msg'):
                        ui.avatar(icon='auto_awesome', color='indigo-1', text_color='indigo').style(
                            'width:24px; height:24px; min-width:24px; font-size:12px; flex-shrink:0;'
                        )
                        with ui.element('div').classes(
                            'ai-bubble rounded-2xl rounded-tl-sm px-4 py-2.5 text-sm'
                        ).style('color:#334155;'):
                            ui.markdown("새 대화를 시작합니다. 무엇이든 질문해보세요!")
                rag.clear_history()
                _refresh_conv_list()

        def _load_conversation(conv: dict):
            state.active_conv_id = conv['id']
            state.messages = conv['messages']
            chat_container.clear()
            with chat_container:
                for msg in state.messages:
                    _render_message(msg)
            scroll_area.scroll_to(percent=1.0)
            _refresh_conv_list()

        # ── helper: render a saved message ─────────────────────────────────────────────────────────────
        def _render_message(msg: dict):
            role    = msg.get('role', 'user')
            content = msg.get('content', '')
            ds      = msg.get('dataset', 'All')
            graph_id = msg.get('graph_id')

            if role == 'user':
                chip_color = _DATASET_CHIP_COLOR.get(ds, 'grey')
                with ui.element('div').style('display:flex; flex-direction:column; align-items:flex-end; gap:4px; width:100%;'):
                    ui.badge(ds, color=chip_color).classes('text-xs')
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
                        ui.markdown(_linkify(content))
                        if graph_id:
                            _add_subgraph_widget(graph_id, _page_client)

        # ── send handler ─────────────────────────────────────────────────────────────────────────────────────────
        async def on_send_message():
            query = input_box.value.strip()
            if not query:
                return

            input_box.value = ''
            send_btn.disable()

            current_dataset = state.dataset
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

            # ── user bubble ─────────────────────────────────────────────────────────────────────────────────────
            chip_color = _DATASET_CHIP_COLOR.get(current_dataset, 'grey')
            user_msg = {'role': 'user', 'content': query, 'dataset': current_dataset}
            state.messages.append(user_msg)

            with chat_container:
                with ui.element('div').style('display:flex; flex-direction:column; align-items:flex-end; gap:4px; width:100%;'):
                    ui.badge(current_dataset, color=chip_color).classes('text-xs')
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
                        md_element = ui.markdown('')

                chunk_buffer += chunk
                chunk_count  += 1
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

            # ── subgraph toggle ───────────────────────────────────────────────────────────────────────────────────
            graph_id = None
            if rag.last_retrieved_nodes and ai_col_ref is not None:
                try:
                    graph_id = _neo4j_viz.generate_rag_result_graph(
                        rag.last_retrieved_nodes, current_dataset
                    )
                    print(f"[subgraph] widget 생성: graph_id={graph_id}, nodes={len(rag.last_retrieved_nodes)}")
                    with ai_col_ref:
                        _add_subgraph_widget(graph_id, _page_client)
                except Exception as e:
                    print(f"서브그래프 생성 오류: {e}")

            # ── save message ────────────────────────────────────────────────────────────────────────────────────────
            ai_msg = {'role': 'assistant', 'content': full_text, 'graph_id': graph_id}
            state.messages.append(ai_msg)

            scroll_area.scroll_to(percent=1.0)
            send_btn.enable()

        async def _on_enter(e):
            if not e.args.get('shiftKey'):
                await on_send_message()

        send_btn.on('click', on_send_message)
        # keydown.enter 만 등록 (prevent 없이) — 줄바꿈 방지는 위 input_box 의 @keydown
        # Vue 속성이 렌더링 시점에 즉시 처리하므로, 여기서는 전송(send)만 담당한다.
        input_box.on('keydown.enter', _on_enter)
