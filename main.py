import os
import tempfile

from nicegui import ui, app
from fastapi.responses import HTMLResponse

from ui.chat_page import build_chat_page, _neo4j_viz


@app.get('/graph/{graph_id}')
def serve_graph(graph_id: str):
    # sanitize: allow only hex characters (uuid4().hex)
    safe_id = ''.join(c for c in graph_id if c in '0123456789abcdef')
    path = os.path.join(tempfile.gettempdir(), f'kmap_graph_{safe_id}.html')
    if not os.path.exists(path):
        return HTMLResponse('<p style="font-family:sans-serif;color:#666">Graph not found</p>', status_code=404)
    html = _neo4j_viz.inject_custom_html(path)
    return HTMLResponse(content=html)


app.on_shutdown(_neo4j_viz.close)

# ★ 수식(LaTeX) 렌더링용 MathJax 를 외부 CDN 대신 로컬 파일로 직접 서빙한다.
#   사내망이 외부 CDN(jsdelivr 등)을 차단/지연시켜 수식이 렌더링되지 않는 문제를
#   막기 위함 — static/mathjax/tex-mml-chtml.js (npm mathjax@3.2.2 의
#   es5 combined-component 단일 파일, 외부 의존성 없이 그 자체로 완결됨).
app.add_static_files('/mathjax', 'static/mathjax')

ui.page('/')(build_chat_page)

ui.run(title='KMap', port=8080, reload=False, dark=False,
       show=False, favicon='./favicon_kg_t.svg')
