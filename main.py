import os
import tempfile
from pathlib import Path

from nicegui import ui, app
from fastapi.responses import HTMLResponse
from starlette.requests import Request

from ui.chat_page import build_chat_page, _neo4j_viz
from access_log import log_access, client_ip

_ROOT = Path(__file__).parent


# ★ 접속 로그(IP/시간): 모든 HTTP 요청에 대해 logs/access.log 에 기록한다.
#   정적 파일(js/css/이미지 등) 요청까지 다 찍으면 로그가 지나치게 많아지므로,
#   페이지 이동/API 성격의 요청만 남기고 정적 자산 확장자는 걸러낸다.
_STATIC_EXT = ('.js', '.css', '.png', '.jpg', '.svg', '.ico', '.woff', '.woff2', '.map')


@app.middleware('http')
async def _log_requests(request: Request, call_next):
    if not request.url.path.endswith(_STATIC_EXT):
        log_access(client_ip(request), request.method, request.url.path)
    return await call_next(request)


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
_mathjax_dir = _ROOT / 'static' / 'mathjax'
if not (_mathjax_dir / 'tex-mml-chtml.js').exists():
    print(f"[WARN] MathJax 파일을 찾을 수 없음: {_mathjax_dir / 'tex-mml-chtml.js'} "
          f"— 수식이 렌더링되지 않습니다.")
app.add_static_files('/mathjax', str(_mathjax_dir))

ui.page('/')(build_chat_page)

ui.run(title='KMap', port=8080, reload=False, dark=False,
       show=False, favicon='./favicon_kg_t.svg')
