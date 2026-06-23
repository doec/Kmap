from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncIterator

from nicegui import ui

import config
from retriever.qdrant_retriever import QdrantRetriever
from retriever.neo4j_retriever import Neo4jRetriever

qdrant = QdrantRetriever()
neo4j = Neo4jRetriever()

# ── chip colour map ──────────────────────────────────────────────────────────
SOURCE_COLORS: dict[str, str] = {
    "paper":      "teal",
    "report":     "blue",
    "experiment": "amber",
}

# ── mock LLM stream ──────────────────────────────────────────────────────────
async def _mock_stream(query: str) -> AsyncIterator[str]:
    tokens = f"[KMap 답변] '{query}'에 대한 검색 결과입니다. 현재 LLM 연동 전 목업 응답입니다. 실제 구현 시 ask_llm_stream_iter_messages()를 사용하세요.".split()
    for token in tokens:
        yield token + " "
        await asyncio.sleep(0.05)


async def ask_llm_stream_iter_messages(messages: list[dict]) -> AsyncIterator[str]:
    """Placeholder — replace with Samsung internal API call."""
    query = messages[-1].get("content", "") if messages else ""
    async for token in _mock_stream(query):
        yield token


# ── state ────────────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.search_mode: str = config.DEFAULT_SEARCH_MODE
        self.use_2hop: bool = config.DEFAULT_SEARCH_HOPS == 2
        self.use_reports: bool = True
        self.use_papers: bool = True
        self.dataset_filter: str = "전체"  # 전체 | 주간보고 | 논문 | 실험
        self.conversations: list[dict] = []   # {id, title, messages}
        self.active_conv_id: int | None = None
        self.messages: list[dict] = []        # current conversation messages


# ── source chip ──────────────────────────────────────────────────────────────
def render_source_chip(source: dict):
    src_type = source.get("type", "paper")
    color = SOURCE_COLORS.get(src_type, "grey")
    label = source.get("label", src_type)
    ui.badge(label, color=color).classes("text-xs cursor-pointer").tooltip(source.get("tooltip", ""))


# ── message card ─────────────────────────────────────────────────────────────
def render_message(msg: dict):
    role = msg.get("role", "user")
    content = msg.get("content", "")
    sources: list[dict] = msg.get("sources", [])
    subgraph_html: str = msg.get("subgraph_html", "")

    if role == "user":
        with ui.row().classes("w-full justify-end"):
            ui.label(content).classes(
                "bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-2 max-w-xl text-sm"
            )
    else:
        with ui.row().classes("w-full justify-start items-start gap-2"):
            ui.icon("smart_toy").classes("text-teal-500 mt-1 text-xl")
            with ui.column().classes("gap-0.5").style("max-width: 100%; min-width: 0;"):
                ui.markdown(content).classes(
                    "bg-gray-800 text-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 text-sm"
                )
                if sources:
                    with ui.row().classes("flex-wrap gap-1 mt-1"):
                        for src in sources:
                            render_source_chip(src)
                if subgraph_html:
                    with ui.expansion("서브그래프 보기", icon="account_tree").classes("w-full mt-1"):
                        ui.html(subgraph_html).classes("w-full")


# ── main page builder ────────────────────────────────────────────────────────
def build_chat_page():
    state = AppState()

    ui.query("body").style("background-color: #111827; color: #f9fafb;")
    ui.add_head_html('<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">')

    # ── topbar ────────────────────────────────────────────────────────────────
    with ui.header().classes("bg-gray-900 border-b border-gray-700 px-4 py-2 flex items-center gap-4"):
        ui.label("KMap").classes("text-white font-bold text-lg mr-4")

        dataset_tabs = ["전체", "주간보고", "논문", "실험"]
        dataset_btn_refs: dict[str, ui.button] = {}

        def on_dataset(tab: str):
            state.dataset_filter = tab
            for t, btn in dataset_btn_refs.items():
                btn.classes(
                    remove="bg-blue-600 text-white",
                    add="text-gray-300 hover:text-white",
                ) if t != tab else btn.classes(
                    remove="text-gray-300 hover:text-white",
                    add="bg-blue-600 text-white",
                )

        for tab in dataset_tabs:
            btn = (
                ui.button(tab, on_click=lambda t=tab: on_dataset(t))
                .classes("px-3 py-1 rounded text-sm transition-colors")
                .props("flat dense")
            )
            if tab == "전체":
                btn.classes("bg-blue-600 text-white")
            else:
                btn.classes("text-gray-300 hover:text-white")
            dataset_btn_refs[tab] = btn

        ui.space()
        ui.label("").bind_text_from(state, "search_mode").classes("text-gray-400 text-sm")

    # ── body: sidebar + chat ──────────────────────────────────────────────────
    with ui.row().classes("w-full flex-1 overflow-hidden").style("height: calc(100vh - 56px);"):

        # ── sidebar ───────────────────────────────────────────────────────────
        with ui.column().classes("bg-gray-900 border-r border-gray-700 p-3 gap-3 overflow-y-auto").style("width: 250px; min-width: 250px;"):
            ui.label("대화 기록").classes("text-gray-400 text-xs font-semibold uppercase tracking-wide")

            conv_list = ui.column().classes("gap-1 w-full")

            def refresh_conv_list():
                conv_list.clear()
                with conv_list:
                    for conv in reversed(state.conversations):
                        cid = conv["id"]
                        title = conv["title"]
                        is_active = cid == state.active_conv_id
                        (
                            ui.button(title, on_click=lambda c=conv: load_conversation(c))
                            .classes(
                                "w-full text-left text-sm rounded px-2 py-1.5 truncate " +
                                ("bg-gray-700 text-white" if is_active else "text-gray-300 hover:bg-gray-800")
                            )
                            .props("flat dense align=left")
                        )

            ui.separator().classes("border-gray-700")
            ui.label("검색 모드").classes("text-gray-400 text-xs font-semibold uppercase tracking-wide")

            mode_options = [("Hybrid", "hybrid"), ("Dense", "dense"), ("Graph", "graph")]
            mode_btns: dict[str, ui.button] = {}

            def on_mode(m: str):
                state.search_mode = m
                for k, b in mode_btns.items():
                    b.classes(
                        remove="bg-teal-700 text-white",
                        add="text-gray-300 hover:text-white",
                    ) if k != m else b.classes(
                        remove="text-gray-300 hover:text-white",
                        add="bg-teal-700 text-white",
                    )

            for label, val in mode_options:
                b = (
                    ui.button(label, on_click=lambda v=val: on_mode(v))
                    .classes("w-full text-sm rounded px-2 py-1.5 text-left transition-colors")
                    .props("flat dense align=left")
                )
                if val == state.search_mode:
                    b.classes("bg-teal-700 text-white")
                else:
                    b.classes("text-gray-300 hover:text-white")
                mode_btns[val] = b

        # ── chat area ─────────────────────────────────────────────────────────
        with ui.column().classes("flex-1 flex flex-col overflow-hidden"):

            # scrollable message list
            scroll_area = ui.scroll_area().classes("flex-1 w-full")
            with scroll_area:
                chat_container = ui.column().classes("w-full p-4 gap-2")

            # ── bottom toolbar + input ─────────────────────────────────────────
            with ui.column().classes("w-full bg-gray-900 border-t border-gray-700 px-4 py-2 gap-2"):

                # toggles row
                with ui.row().classes("items-center gap-3"):
                    hop_toggle = (
                        ui.toggle(["1-hop", "2-hop"], value="2-hop" if state.use_2hop else "1-hop")
                        .classes("text-xs")
                    )

                    def on_hop(e):
                        state.use_2hop = e.value == "2-hop"

                    hop_toggle.on("update:model-value", on_hop)

                    reports_chk = ui.checkbox("ReportsDB", value=state.use_reports).classes("text-sm text-gray-300")
                    reports_chk.bind_value(state, "use_reports")

                    papers_chk = ui.checkbox("PapersDB", value=state.use_papers).classes("text-sm text-gray-300")
                    papers_chk.bind_value(state, "use_papers")

                # input row
                with ui.row().classes("w-full items-center gap-2"):
                    input_box = (
                        ui.textarea(placeholder="질문을 입력하세요…")
                        .classes("flex-1 rounded-xl bg-gray-800 text-gray-100 border-gray-600")
                        .style("font-size: 15px;")
                        .props("rows=2 outlined dense")
                    )
                    send_btn = (
                        ui.button(icon="send")
                        .classes("bg-blue-600 hover:bg-blue-500 text-white rounded-xl")
                        .props("round")
                    )

            # ── send logic ────────────────────────────────────────────────────
            async def on_send_message():
                query = input_box.value.strip()
                if not query:
                    return
                input_box.value = ""
                send_btn.disable()

                # save / create conversation
                if not state.messages:
                    conv = {
                        "id": len(state.conversations),
                        "title": query[:30] + ("…" if len(query) > 30 else ""),
                        "messages": state.messages,
                    }
                    state.conversations.append(conv)
                    state.active_conv_id = conv["id"]
                    refresh_conv_list()

                # user bubble
                user_msg = {"role": "user", "content": query}
                state.messages.append(user_msg)
                with chat_container:
                    render_message(user_msg)

                scroll_area.scroll_to(percent=1.0)

                # AI streaming bubble
                ai_content = ""
                with chat_container:
                    with ui.row().classes("w-full justify-start items-start gap-2") as ai_row:
                        ui.icon("smart_toy").classes("text-teal-500 mt-1 text-xl")
                        with ui.column().classes("gap-0.5").style("max-width: 100%; min-width: 0;"):
                            ai_label = ui.markdown("▌").classes(
                                "bg-gray-800 text-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 text-sm"
                            )

                llm_messages = [{"role": m["role"], "content": m["content"]} for m in state.messages]
                async for token in ask_llm_stream_iter_messages(llm_messages):
                    ai_content += token
                    ai_label.set_content(ai_content + "▌")
                    scroll_area.scroll_to(percent=1.0)

                ai_label.set_content(ai_content)

                ai_msg = {"role": "assistant", "content": ai_content, "sources": []}
                state.messages.append(ai_msg)
                send_btn.enable()

            send_btn.on("click", on_send_message)
            input_box.on(
                "keydown",
                lambda e: asyncio.ensure_future(on_send_message())
                if (e.args.get("key") == "Enter" and not e.args.get("shiftKey"))
                else None,
            )

            # ── load conversation ──────────────────────────────────────────────
            def load_conversation(conv: dict):
                state.active_conv_id = conv["id"]
                state.messages = conv["messages"]
                chat_container.clear()
                with chat_container:
                    for msg in state.messages:
                        render_message(msg)
                scroll_area.scroll_to(percent=1.0)
                refresh_conv_list()
