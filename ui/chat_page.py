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
    ("Hybrid", "hybrid"),
    ("Vector", "vector"),
    ("Text",   "text"),
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
        .user-bubble { background: linear-gradient(135deg, #6366f1, #8b5cf6); word-break: break-word; }
        .ai-bubble { background: #f1f5f9; overflow-x: auto; color: #334155; }
        .ai-bubble p, .ai-bubble li, .ai-bubble td { color: #334155; margin: 0; }
        .ai-bubble * { color: #334155; }
        .ai-bubble code { background: #e2e8f0; color: #1e293b; padding: 1px 5px; border-radius: 4px; }
        .ai-bubble pre { background: #e2e8f0; padding: 8px 12px; border-radius: 8px; overflow-x: auto; }

        /* input field — force light mode regardless of Quasar dark setting */
        .q-field__native, .q-field__input { color: #1e293b !important; }
        .q-field__label { color: #64748b !important; }
        .q-field--outlined .q-field__control { background: white !important; }

        /* checkbox labels */
        .q-checkbox__label { color: #475569 !important; font-size: 13px; }

        /* scroll area inner content fills height so messages stay at bottom */
        .q-scrollarea__content { min-height: 100% !important; display: flex !important; flex-direction: column !important; }

        /* header-input (dark header 위 입력창) */
        .header-input .q-field__control:before { border-color: rgba(255,255,255,0.15) !important; }
        .header-input .q-field__control:hover:before { border-color: rgba(255,255,255,0.3) !important; }

        /* hover animations */
        .hover-btn { transition: all 0.2s ease; }
        .hover-btn:hover { transform: scale(1.05); filter: brightness(1.1); }
    </style>
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

            for label, val in _MODE_OPTIONS:
                b = (
                    ui.button(label, on_click=lambda v=val: _on_mode(v))
                    .props('flat dense align=left')
                    .classes('w-full text-sm rounded px-2 py-1.5 text-left transition-colors hover-btn')
                )
                b.classes('bg-indigo-600 text-white' if val == state.search_mode else 'text-slate-500 hover:text-slate-800')
                mode_btns[val] = b

        # ── chat area ────────────────────────────────────────────────────────────────────────────────────
        with ui.element('div').style(
            'flex:1; min-width:0; display:flex; flex-direction:column; overflow:hidden; background:#f8fafc;'
        ):
            scroll_area = ui.scroll_area().style('flex:1; min-height:0; width:100%; background:#f8fafc;')
            with scroll_area:
                chat_container = ui.element('div').style(
                    'display:flex; flex-direction:column; gap:8px; padding:20px; min-height:100%;'
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
                                "논문·주간보고 데이터를 GraphRAG로 검색합니다. 무엇이든 질문해보세요!"
                            )

            # ── bottom bar ─────────────────────────────────────────────────────────────────────────────
            with ui.element('div').style(
                'flex-shrink:0; background:#f8fafc; border-top:1px solid #e2e8f0; padding:8px 16px 12px;'
            ):
                # toggles
                with ui.element('div').style('display:flex; align-items:center; gap:16px; margin-bottom:6px;'):
                    hop_chk = ui.checkbox('2-hop', value=state.use_2hop).style('color:#64748b; font-size:13px;')
                    hop_chk.bind_value(state, 'use_2hop')
                    hop_chk.tooltip('2-hop 탐색 (per-session 제어는 2단계 예정)')

                    rpt_chk = ui.checkbox('ReportsDB', value=state.use_reports).style('color:#64748b; font-size:13px;')
                    rpt_chk.bind_value(state, 'use_reports')

                    ppr_chk = ui.checkbox('PapersDB', value=state.use_papers).style('color:#64748b; font-size:13px;')
                    ppr_chk.bind_value(state, 'use_papers')

                # input
                with ui.element('div').style('display:flex; align-items:flex-end; gap:8px; width:100%;'):
                    input_box = (
                        ui.textarea(placeholder='질문을 입력하세요… (Shift+Enter: 줄바꿈, Enter: 전송)')
                        .classes('flex-grow text-sm')
                        .style('font-size:14px;')
                        .props('outlined rounded dense autogrow')
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
                        spinner = ui.spinner('dots', size='1.2em', color='indigo')

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
                chunk = await loop.run_in_executor(None, next, gen, None)
                if chunk is None:
                    if chunk_buffer:
                        full_text += chunk_buffer
                        if md_element:
                            md_element.set_content(full_text)
                    break

                # first chunk: replace spinner with markdown element
                if md_element is None:
                    spinner.delete()
                    with ai_col_ref:
                        md_element = ui.markdown('')

                chunk_buffer += chunk
                chunk_count  += 1
                if chunk_count % 5 == 0:
                    full_text    += chunk_buffer
                    chunk_buffer  = ''
                    md_element.set_content(full_text + '▌')
                    await _page_client.run_javascript(_SCROLL_JS)
                    await asyncio.sleep(0)

            if md_element is None:
                spinner.delete()
                with ai_col_ref:
                    md_element = ui.markdown('(응답을 받지 못했습니다)')
            else:
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

        async def _on_enter(e):
            if not e.args.get('shiftKey'):
                await on_send_message()

        send_btn.on('click', on_send_message)
        input_box.on('keydown.enter.prevent', _on_enter)
