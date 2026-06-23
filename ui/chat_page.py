from __future__ import annotations

import asyncio
import os

from nicegui import ui

from rag_engine import GraphRAG, REPORTS_DATASET, PAPERS_DATASET, DATASETS
from retriever.neo4j_retriever import Neo4jRetriever

# ── module-level singleton for graph visualization ────────────────────────────────────────────
_neo4j_viz = Neo4jRetriever()

# ── source chip colours ────────────────────────────────────────────────────────────────────────────────
_DATASET_CHIP_COLOR = {
    REPORTS_DATASET: "blue",
    PAPERS_DATASET:  "teal",
    "ExperimentsDB": "amber",
}

# ── dataset tab labels ──────────────────────────────────────────────────────────────────────────────────
_DATASET_TABS = [
    ("전체",    "All"),
    ("주간보고", REPORTS_DATASET),
    ("논문",    PAPERS_DATASET),
    ("실험",    "ExperimentsDB"),   # future — disabled
]

# ── search mode options ───────────────────────────────────────────────────────────────────────────────────
_MODE_OPTIONS = [
    ("Hybrid",  "hybrid"),
    ("Dense",   "vector"),
    ("Graph",   "text"),
]


class _PageState:
    def __init__(self):
        self.dataset: str       = "All"
        self.search_mode: str   = "hybrid"
        self.use_2hop: bool     = True   # TODO: wire to rag_engine when per-session hops are supported
        self.use_reports: bool  = True
        self.use_papers: bool   = True
        self.conversations: list[dict] = []   # {id, title, messages}
        self.active_conv_id: int | None = None
        self.messages: list[dict] = []


def build_chat_page():
    state = _PageState()
    rag   = GraphRAG()

    # ── global styles ─────────────────────────────────────────────────────────────────────────────────
    ui.add_head_html('''
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <style>
        *, *::before, *::after { font-family: "Inter", sans-serif; box-sizing: border-box; }
        body, html { margin: 0; padding: 0; overflow: hidden; background: #111827; }
        .q-page { min-height: 0 !important; height: calc(100vh - 52px) !important; padding: 0 !important; }
        .nicegui-content { padding: 0 !important; margin: 0 !important; position: absolute; inset: 0; }
        .q-scrollarea__thumb--v { opacity: 0.4 !important; width: 4px !important; border-radius: 4px !important; }
        .user-bubble { background: linear-gradient(135deg, #2563eb, #4f46e5); word-break: break-word; }
        .ai-bubble { background: #1f2937; overflow-x: auto; }
        .ai-bubble p { margin: 0; }
        .ai-bubble .q-markdown { color: #e5e7eb; }
    </style>
    ''')

    # ── topbar ──────────────────────────────────────────────────────────────────────────────────────
    with ui.header().style(
        'height:52px; min-height:52px; padding:0;'
        'background:#111827; border-bottom:1px solid #374151;'
    ):
        with ui.element('div').style(
            'display:flex; align-items:center; width:100%; height:52px; padding:0 16px; gap:12px;'
        ):
            ui.label('KMap').style(
                'font-size:16px; font-weight:700; color:#f9fafb; letter-spacing:-0.02em; white-space:nowrap;'
            )
            ui.element('div').style('width:1px; height:20px; background:#374151;')

            dataset_btn_refs: dict[str, ui.button] = {}

            def _on_dataset(key: str):
                state.dataset = key
                for k, btn in dataset_btn_refs.items():
                    if k == key:
                        btn.classes(remove='text-gray-400 hover:text-white', add='bg-blue-600 text-white')
                    else:
                        btn.classes(remove='bg-blue-600 text-white', add='text-gray-400 hover:text-white')

            for label, key in _DATASET_TABS:
                disabled = (key == "ExperimentsDB")
                btn = (
                    ui.button(label, on_click=lambda k=key: _on_dataset(k))
                    .props('flat dense')
                    .classes('px-3 py-1 rounded text-sm transition-colors')
                )
                if key == "All":
                    btn.classes('bg-blue-600 text-white')
                else:
                    btn.classes('text-gray-400 hover:text-white')
                if disabled:
                    btn.disable()
                    btn.tooltip('준비 중 (Qdrant RAG 2단계)')
                dataset_btn_refs[key] = btn

            ui.element('div').style('flex:1;')

            mode_label = ui.label('hybrid').style('font-size:11px; color:#6b7280;')
            mode_label.bind_text_from(state, 'search_mode')

    # ── body ────────────────────────────────────────────────────────────────────────────────────────
    with ui.row().classes('w-full').style('height: calc(100vh - 52px); overflow: hidden;'):

        # ── sidebar ─────────────────────────────────────────────────────────────────────────────────────
        with ui.column().style(
            'width:250px; min-width:250px; background:#0f172a; '
            'border-right:1px solid #1e293b; padding:12px; gap:12px; overflow-y:auto;'
        ):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('대화 기록').style(
                    'font-size:11px; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:0.05em;'
                )
                ui.button(icon='add', on_click=lambda: _new_conversation()).props('flat round dense').style(
                    'color:#6b7280; font-size:14px;'
                ).tooltip('새 대화')

            conv_list = ui.column().classes('w-full gap-1')

            ui.separator().style('border-color:#1e293b;')

            ui.label('검색 모드').style(
                'font-size:11px; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:0.05em;'
            )

            mode_btns: dict[str, ui.button] = {}

            def _on_mode(m: str):
                state.search_mode = m
                for k, b in mode_btns.items():
                    if k == m:
                        b.classes(remove='text-gray-400 hover:text-white', add='bg-teal-700 text-white')
                    else:
                        b.classes(remove='bg-teal-700 text-white', add='text-gray-400 hover:text-white')

            for label, val in _MODE_OPTIONS:
                b = (
                    ui.button(label, on_click=lambda v=val: _on_mode(v))
                    .props('flat dense align=left')
                    .classes('w-full text-sm rounded px-2 py-1.5 text-left transition-colors')
                )
                b.classes('bg-teal-700 text-white' if val == state.search_mode else 'text-gray-400 hover:text-white')
                mode_btns[val] = b

        # ── chat area ────────────────────────────────────────────────────────────────────────────────────
        with ui.column().style('flex:1; overflow:hidden; display:flex; flex-direction:column;'):

            scroll_area = ui.scroll_area().style('flex:1; width:100%;')
            with scroll_area:
                chat_container = ui.column().classes('w-full p-4 gap-3')

                # welcome message
                with chat_container:
                    with ui.row().classes('w-full justify-start items-start gap-2 ai-msg'):
                        ui.avatar(icon='auto_awesome', color='indigo-1', text_color='indigo').style(
                            'width:28px; height:28px; min-width:28px; font-size:14px;'
                        )
                        with ui.column().classes('gap-0.5').style('max-width: 100%; min-width: 0;'):
                            with ui.element('div').classes(
                                'ai-bubble rounded-2xl rounded-tl-sm px-4 py-3 text-sm'
                            ):
                                ui.markdown(
                                    "안녕하세요! **KMap 연구 어시스턴트**입니다.  \n"
                                    "GraphRAG (Neo4j KG) 기반으로 논문 및 주간보고 데이터를 검색합니다."
                                )

            # ── bottom bar ─────────────────────────────────────────────────────────────────────────────
            with ui.column().style(
                'width:100%; background:#0f172a; border-top:1px solid #1e293b; padding:8px 16px; gap:8px; flex-shrink:0;'
            ):
                # toggles
                with ui.row().classes('items-center gap-4'):
                    hop_chk = ui.checkbox('2-hop', value=state.use_2hop).style('color:#9ca3af; font-size:13px;')
                    hop_chk.bind_value(state, 'use_2hop')
                    hop_chk.tooltip('2-hop 탐색 (per-session 제어는 2단계 예정)')

                    rpt_chk = ui.checkbox('ReportsDB', value=state.use_reports).style('color:#9ca3af; font-size:13px;')
                    rpt_chk.bind_value(state, 'use_reports')

                    ppr_chk = ui.checkbox('PapersDB', value=state.use_papers).style('color:#9ca3af; font-size:13px;')
                    ppr_chk.bind_value(state, 'use_papers')

                # input
                with ui.row().classes('w-full items-center gap-2'):
                    input_box = (
                        ui.textarea(placeholder='질문을 입력하세요… (Shift+Enter: 줄바꾸음, Enter: 전송)')
                        .classes('flex-1 rounded-xl text-sm')
                        .style('font-size: 14px;')
                        .props('rows=2 outlined dense dark')
                    )
                    send_btn = (
                        ui.button(icon='arrow_upward')
                        .props('round unelevated')
                        .style(
                            'background: linear-gradient(135deg, #2563eb, #4f46e5);'
                            'color: white; min-width: 36px; min-height: 36px;'
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
                            'w-full text-sm rounded px-2 py-1.5 truncate text-left ' +
                            ('bg-gray-700 text-white' if is_active else 'text-gray-400 hover:bg-gray-800')
                        )
                    )

        def _new_conversation():
            if state.messages:
                state.messages = []
                state.active_conv_id = None
                chat_container.clear()
                with chat_container:
                    with ui.row().classes('w-full justify-start items-start gap-2 ai-msg'):
                        ui.avatar(icon='auto_awesome', color='indigo-1', text_color='indigo').style(
                            'width:28px; height:28px; min-width:28px; font-size:14px;'
                        )
                        with ui.column().classes('gap-0.5').style('max-width: 100%; min-width: 0;'):
                            with ui.element('div').classes('ai-bubble rounded-2xl rounded-tl-sm px-4 py-3 text-sm'):
                                ui.markdown("새 대화를 시작합니다.")
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
                with ui.column().classes('w-full items-end gap-1'):
                    ui.badge(ds, color=chip_color).classes('text-xs')
                    with ui.element('div').classes(
                        'user-bubble rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm text-white'
                    ).style('max-width:80%;'):
                        ui.label(content)
            else:
                with ui.row().classes('w-full justify-start items-start gap-2 ai-msg'):
                    ui.avatar(icon='auto_awesome', color='indigo-1', text_color='indigo').style(
                        'width:28px; height:28px; min-width:28px; font-size:14px;'
                    )
                    with ui.column().classes('gap-0.5').style('max-width: 100%; min-width: 0;'):
                        with ui.element('div').classes('ai-bubble rounded-2xl rounded-tl-sm px-4 py-3 text-sm'):
                            ui.markdown(content)
                        if graph_id:
                            with ui.expansion('서브그래프 보기', icon='account_tree').classes('w-full mt-1'):
                                ui.html(
                                    f'<iframe src="/graph/{graph_id}" '
                                    f'style="width:100%;height:400px;border:none;border-radius:8px;">'
                                    f'</iframe>'
                                )

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
                with ui.column().classes('w-full items-end gap-1'):
                    ui.badge(current_dataset, color=chip_color).classes('text-xs')
                    with ui.element('div').classes(
                        'user-bubble rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm text-white'
                    ).style('max-width:80%;'):
                        ui.label(query)

            scroll_area.scroll_to(percent=1.0)

            # ── AI streaming bubble ──────────────────────────────────────────────────────────────────────────────
            ai_col_ref = None

            with chat_container:
                with ui.row().classes('w-full justify-start items-start gap-2 ai-msg'):
                    ui.avatar(icon='auto_awesome', color='indigo-1', text_color='indigo').style(
                        'width:28px; height:28px; min-width:28px; font-size:14px;'
                    )
                    with ui.column().classes('gap-0.5').style('max-width: 100%; min-width: 0;') as ai_col_ref:
                        with ui.element('div').classes(
                            'ai-bubble rounded-2xl rounded-tl-sm px-4 py-3 text-sm'
                        ) as ai_bubble:
                            spinner    = ui.spinner('dots', size='1.2em', color='indigo')
                            md_element = ui.markdown('')

            await asyncio.sleep(0.05)
            scroll_area.scroll_to(percent=1.0)

            # ── run sync generator in executor ───────────────────────────────────────────────────────────────
            loop    = asyncio.get_event_loop()
            gen     = rag.answer_stream(query, current_dataset, current_mode)

            spinner.delete()
            full_text    = ''
            chunk_buffer = ''
            chunk_count  = 0

            while True:
                chunk = await loop.run_in_executor(None, next, gen, None)
                if chunk is None:
                    if chunk_buffer:
                        full_text    += chunk_buffer
                        md_element.set_content(full_text)
                    break
                chunk_buffer += chunk
                chunk_count  += 1
                if chunk_count % 5 == 0:
                    full_text    += chunk_buffer
                    chunk_buffer  = ''
                    md_element.set_content(full_text + '▌')
                    await ui.run_javascript("""
                        var el = document.querySelector('.q-scrollarea__container');
                        if (el) {
                            var nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
                            if (nearBottom) {
                                var msgs = document.querySelectorAll('.ai-msg');
                                if (msgs.length) msgs[msgs.length-1].scrollIntoView({behavior:'smooth',block:'end'});
                            }
                        }
                    """)
                    await asyncio.sleep(0)

            md_element.set_content(full_text)

            # ── subgraph toggle ───────────────────────────────────────────────────────────────────────────────────
            graph_id = None
            if rag.last_retrieved_nodes and ai_col_ref is not None:
                try:
                    graph_id = _neo4j_viz.generate_rag_result_graph(
                        rag.last_retrieved_nodes, current_dataset
                    )
                    with ai_col_ref:
                        with ui.expansion('서브그래프 보기', icon='account_tree').classes('w-full mt-1'):
                            ui.html(
                                f'<iframe src="/graph/{graph_id}" '
                                f'style="width:100%;height:400px;border:none;border-radius:8px;"></iframe>'
                            )
                except Exception as e:
                    print(f"서브그래프 생성 오류: {e}")

            # ── save message ────────────────────────────────────────────────────────────────────────────────────────
            ai_msg = {'role': 'assistant', 'content': full_text, 'graph_id': graph_id}
            state.messages.append(ai_msg)

            scroll_area.scroll_to(percent=1.0)
            send_btn.enable()

        send_btn.on('click', on_send_message)
        input_box.on(
            'keydown',
            lambda e: asyncio.ensure_future(on_send_message())
            if (e.args.get('key') == 'Enter' and not e.args.get('shiftKey'))
            else None,
        )
