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

ui.page('/')(build_chat_page)

ui.run(title='KMap', port=8080, reload=False, dark=False, favicon='./favicon_kg_t.svg')
