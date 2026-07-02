from __future__ import annotations

import os
import time
import uuid
import tempfile

from pyvis.network import Network
from neo4j import GraphDatabase

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')

NEO4J_URI      = os.getenv('NEO4J_URI')
NEO4J_USER     = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

_TYPE_PALETTE = [
    '#1D9E75', '#378ADD', '#EF9F27', '#D85A30',
    '#D4537E', '#7F77DD', '#06B6D4', '#A855F7',
    '#84CC16', '#EAB308', '#FF6B6B', '#4ECDC4',
]
_UNKNOWN_COLOR = '#888780'

_HIGHLIGHT_JS = """
<script>
(function () {
    var origNode = {}, origEdge = {}, origSize = {}, active = false;
    var nodeFontSz = 44, edgeFontSz = 28, edgeW = 3, nodeScale = 1.0;

    function init() {
        if (typeof network === 'undefined' || !network) { setTimeout(init, 100); return; }
        nodes.forEach(function (n) {
            origNode[n.id] = { color: n.color, fontColor: (n.font && n.font.color) || '#000000' };
            origSize[n.id] = n.size || 35;
        });
        edges.forEach(function (e) {
            origEdge[e.id] = { color: e.color, fontColor: (e.font && e.font.color) || '#333333', label: e.label || '' };
        });
        network.on('selectNode', function (p) { if (p.nodes.length) { highlight(p.nodes[0]); showInfo(p.nodes[0]); } });
        network.on('deselectNode', function() { reset(); hideInfo(); });

        var infoPanel = document.createElement('div');
        infoPanel.id = 'nodeInfoPanel';
        infoPanel.style.cssText = 'position:fixed;top:16px;right:16px;min-width:180px;max-width:260px;'
            + 'background:rgba(255,255,255,0.95);border:1px solid #e2e8f0;border-radius:8px;'
            + 'padding:10px 14px;box-shadow:0 2px 8px rgba(0,0,0,0.08);z-index:1000;'
            + 'font-family:Inter,sans-serif;display:none;';
        document.body.appendChild(infoPanel);
    }

    function showInfo(id) {
        var n = nodes.get(id);
        if (!n) return;
        var panel = document.getElementById('nodeInfoPanel');
        var title = n.title || '';
        var rows = title.split('|').map(function(s) { return s.trim(); }).filter(Boolean);
        var html = '<div style="font-size:11px;font-weight:600;color:#64748b;margin-bottom:7px;letter-spacing:0.05em;">NODE INFO</div>';
        html += '<div style="font-size:13px;font-weight:600;color:#1e293b;margin-bottom:6px;">' + (n.label || id) + '</div>';
        rows.forEach(function(row) {
            var parts = row.split(':');
            var key = parts[0].trim();
            var val = parts.slice(1).join(':').trim();
            html += '<div style="display:flex;gap:6px;margin-bottom:3px;font-size:12px;">'
                + '<span style="color:#64748b;white-space:nowrap;">' + key + ':</span>'
                + '<span style="color:#334155;">' + val + '</span></div>';
        });
        panel.innerHTML = html;
        panel.style.display = 'block';
    }

    function hideInfo() {
        var panel = document.getElementById('nodeInfoPanel');
        if (panel) panel.style.display = 'none';
    }

    window.scaleAll = function(d) {
        nodeScale = Math.max(0.3, Math.min(4.0, nodeScale + d * 0.08));
        nodes.update(nodes.get().map(function(n) {
            return {id: n.id, size: Math.round(origSize[n.id] * nodeScale)};
        }));
        nodeFontSz = Math.max(10, nodeFontSz + d * 4);
        edgeFontSz = Math.max(8,  edgeFontSz + d * 3);
        edgeW = Math.max(1, edgeW + d);
        network.setOptions({
            nodes: {font: {size: nodeFontSz}},
            edges: {font: {size: edgeFontSz}, width: edgeW}
        });
    };
    window.resetScale = function() {
        nodeScale = 1.0; nodeFontSz = 44; edgeFontSz = 28; edgeW = 3;
        nodes.update(nodes.get().map(function(n) {
            return {id: n.id, size: origSize[n.id]};
        }));
        network.setOptions({
            nodes: {font: {size: 44}},
            edges: {font: {size: 28}, width: 3}
        });
    };

    function highlight(id) {
        active = true;
        var nbrs = new Set(network.getConnectedNodes(id));
        nbrs.add(id);
        var nbEdges = new Set(network.getConnectedEdges(id));
        nodes.update(nodes.get().map(function (n) {
            if (nbrs.has(n.id))
                return { id: n.id, color: origNode[n.id].color, font: { color: '#000000' }, borderWidth: 3 };
            return { id: n.id,
                color: { background: '#d8d8d8', border: '#c4c4c4',
                         highlight: { background: '#d8d8d8', border: '#c4c4c4' } },
                font: { color: '#aaaaaa' }, borderWidth: 1 };
        }));
        edges.update(edges.get().map(function (e) {
            if (nbEdges.has(e.id))
                return { id: e.id, color: origEdge[e.id].color,
                    label: '<b>' + origEdge[e.id].label + '</b>',
                    font: { color: '#333333', multi: 'html' }, width: edgeW + 2 };
            return { id: e.id,
                color: { color: '#cccccc', highlight: '#cccccc' },
                font: { color: '#bbbbbb', multi: false }, width: edgeW };
        }));
        var frontEdgeIds = Array.from(nbEdges);
        var frontEdgeData = edges.get(frontEdgeIds);
        edges.remove(frontEdgeIds);
        edges.add(frontEdgeData);
        network.storePositions();
        var frontIds = Array.from(nbrs);
        var frontData = nodes.get(frontIds);
        nodes.remove(frontIds);
        nodes.add(frontData);
    }

    function reset() {
        if (!active) return;
        active = false;
        nodes.update(nodes.get().map(function (n) {
            return { id: n.id, color: origNode[n.id].color, font: { color: origNode[n.id].fontColor }, borderWidth: 1 };
        }));
        edges.update(edges.get().map(function (e) {
            return { id: e.id,
                color: origEdge[e.id].color !== undefined ? origEdge[e.id].color : null,
                label: origEdge[e.id].label,
                font: { color: origEdge[e.id].fontColor, multi: false }, width: edgeW };
        }));
    }

    init();
})();
</script>
"""


class Neo4jRetriever:
    """Graph visualization manager (pyvis). Search logic lives in GraphRAG."""

    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            print("Neo4j 연결 성공 (Neo4jRetriever)")
        except Exception as e:
            print(f"Neo4j 연결 오류 (Neo4jRetriever): {e}")
            self.driver = None
        self._type_color_map = self._build_type_colors()

    def close(self):
        if self.driver:
            self.driver.close()

    def _build_type_colors(self) -> dict:
        if not self.driver:
            return {}
        try:
            with self.driver.session() as session:
                # 엔티티 타입만 색상 팔레트에 매핑한다.
                # ★ Paper/Document 는 메타 노드 타입이라 서브그래프에 그려지지 않으므로
                #   범례(legend)에서도 제외해 목록을 엔티티 타입만으로 유지한다.
                types = [
                    r['t'] for r in session.run(
                        "MATCH (n) WHERE n.type IS NOT NULL "
                        "RETURN DISTINCT n.type AS t ORDER BY t"
                    ) if r['t'] and r['t'] not in ('Unknown', 'Paper', 'Document')
                ]
            return {t: _TYPE_PALETTE[i % len(_TYPE_PALETTE)] for i, t in enumerate(types)}
        except Exception as e:
            print(f"node_type 조회 오류: {e}")
            return {}

    def get_node_count(self, dataset: str = None) -> int:
        if not self.driver:
            return 0
        try:
            with self.driver.session() as session:
                query = "MATCH (n)"
                if dataset and dataset != 'All':
                    query += " WHERE $dataset IN labels(n)"
                query += " RETURN count(n) AS cnt"
                return session.run(query, dataset=dataset).single()["cnt"]
        except Exception:
            return 0

    def get_edge_count(self, dataset: str = None) -> int:
        if not self.driver:
            return 0
        try:
            with self.driver.session() as session:
                query = "MATCH (n)-[r]->(m)"
                if dataset and dataset != 'All':
                    query += " WHERE $dataset IN labels(n) AND $dataset IN labels(m)"
                query += " RETURN count(r) AS cnt"
                return session.run(query, dataset=dataset).single()["cnt"]
        except Exception:
            return 0

    def get_dataset_labels(self) -> list[str]:
        if not self.driver:
            return []
        try:
            with self.driver.session() as session:
                return sorted([
                    r['label'] for r in session.run("CALL db.labels() YIELD label RETURN label")
                ])
        except Exception:
            return []

    def _type_color(self, props: dict) -> str:
        return self._type_color_map.get(props.get('type', 'Unknown'), _UNKNOWN_COLOR)

    def _get_node_label(self, props: dict, node_id: str) -> str:
        for key in ('name', 'title', 'label', 'id'):
            if props.get(key):
                return str(props[key])
        return node_id

    def _node_size(self, degree: int) -> int:
        return max(35, min(100, 35 + degree * 12))

    def _make_net(self) -> Network:
        net = Network(height='100vh', width='100%', bgcolor='#ffffff',
                      font_color='black', cdn_resources='in_line')
        net.set_options("""var options = {
          "nodes": { "font": { "size": 44, "strokeWidth": 4, "strokeColor": "#ffffff" } },
          "edges": { "font": { "size": 28 }, "width": 3, "smooth": { "type": "dynamic" } },
          "physics": {
            "solver": "barnesHut",
            "barnesHut": { "gravitationalConstant": -8000, "springConstant": 0.001, "springLength": 200 }
          }
        }""")
        return net

    def _make_legend_html(self) -> str:
        items = ''.join(
            f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:5px;">'
            f'<div style="width:13px;height:13px;border-radius:50%;background:{color};flex-shrink:0;border:1px solid rgba(0,0,0,0.1);"></div>'
            f'<span style="font-size:12px;color:#334155;">{ntype}</span></div>'
            for ntype, color in sorted(self._type_color_map.items())
        )
        items += (
            '<div style="display:flex;align-items:center;gap:7px;">'
            f'<div style="width:13px;height:13px;border-radius:50%;background:{_UNKNOWN_COLOR};flex-shrink:0;border:1px solid rgba(0,0,0,0.1);"></div>'
            '<span style="font-size:12px;color:#334155;">Unknown</span></div>'
        )
        return (
            '<div id="legend" style="position:fixed;bottom:16px;left:16px;'
            'max-height:calc(100vh - 32px);display:flex;flex-direction:column;'
            'background:rgba(255,255,255,0.93);border:1px solid #e2e8f0;border-radius:8px;'
            'padding:10px 14px;box-shadow:0 2px 8px rgba(0,0,0,0.08);z-index:1000;'
            'font-family:Inter,sans-serif;">'
            '<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;cursor:pointer;user-select:none;" '
            'onclick="(function(){'
                'var b=document.getElementById(\'legend-body\');'
                'var ic=document.getElementById(\'legend-toggle-ic\');'
                'var collapsed=b.style.display===\'none\';'
                'b.style.display=collapsed?\'block\':\'none\';'
                'ic.textContent=collapsed?\'▴\':\'▾\';'
            '})()">' 
            '<span style="font-size:11px;font-weight:600;color:#64748b;letter-spacing:0.05em;">NODE TYPE</span>'
            '<span id="legend-toggle-ic" style="font-size:12px;color:#94a3b8;line-height:1;">▴</span>'
            '</div>'
            '<div id="legend-body" style="margin-top:7px;overflow-y:auto;flex:1;min-height:0;display:none;">' + items + '</div>'
            '</div>'
        )

    def inject_custom_html(self, html_path: str, dataset: str = None) -> str:
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()

        html = html.replace(
            '<style type="text/css">',
            '<style type="text/css">\n'
            '        body, html { margin: 0; padding: 0; height: 100vh; width: 100%; overflow: hidden; }\n'
            '        #mynetwork { height: 100vh !important; width: 100% !important; border: none !important; }\n'
            '        .card { border: none !important; border-radius: 0 !important; margin: 0 !important; box-shadow: none !important; }\n'
            '        .card-body { padding: 0 !important; border: none !important; }'
        )

        total_nodes = self.get_node_count(dataset)
        total_edges = self.get_edge_count(dataset)

        info_js = f"""
<script>
(function() {{
    function tryAddCounts() {{
        if (typeof nodes !== 'undefined' && nodes && typeof edges !== 'undefined' && edges) {{
            var div = document.createElement('div');
            div.style.cssText = 'position:fixed;top:16px;left:16px;font-family:Inter,sans-serif;font-size:11px;color:#94a3b8;pointer-events:none;';
            div.innerHTML = '<span style="margin-right:10px;">nodes ' + nodes.length + ' / {total_nodes}</span>'
                          + '<span>edges ' + edges.length + ' / {total_edges}</span>';
            document.body.appendChild(div);
        }} else {{ setTimeout(tryAddCounts, 100); }}
    }}
    tryAddCounts();
}})();
</script>
"""
        physics_js = """
<script>
(function() {
    function addPhysicsToggle() {
        if (typeof network === 'undefined' || !network) { setTimeout(addPhysicsToggle, 100); return; }
        var saved = localStorage.getItem('physics_enabled');
        var enabled = saved === null ? true : saved === 'true';
        network.setOptions({physics: {enabled: enabled}});
        var div = document.createElement('div');
        div.style.cssText = 'position:fixed;top:36px;left:16px;font-family:Inter,sans-serif;font-size:11px;color:#94a3b8;display:flex;align-items:center;gap:5px;';
        var cb = document.createElement('input');
        cb.type = 'checkbox'; cb.id = 'physics-toggle'; cb.checked = enabled;
        cb.style.cssText = 'width:12px;height:12px;cursor:pointer;accent-color:#4338ca;';
        var lbl = document.createElement('label');
        lbl.htmlFor = 'physics-toggle'; lbl.textContent = 'Physics';
        lbl.style.cssText = 'cursor:pointer;color:#94a3b8;';
        cb.addEventListener('change', function() {
            network.setOptions({physics: {enabled: this.checked}});
            localStorage.setItem('physics_enabled', this.checked);
        });
        div.appendChild(cb); div.appendChild(lbl);
        document.body.appendChild(div);
    }
    addPhysicsToggle();
})();
</script>
"""
        html = html.replace(
            '</body>',
            self._make_legend_html() + _HIGHLIGHT_JS + info_js + physics_js + '</body>'
        )
        return html

    def generate_rag_result_graph(self, node_names: list[str],
                                   dataset: str = None,
                                   edge_limit: int = 200) -> str:
        """
        RAG 검색 결과 노드들의 서브그래프를 생성하고 graph_id를 반환.
        HTML은 tempdir에 kmap_graph_{graph_id}.html 로 저장됨.
        """
        graph_id  = uuid.uuid4().hex
        temp_path = os.path.join(tempfile.gettempdir(), f'kmap_graph_{graph_id}.html')

        net = self._make_net()

        if not self.driver or not node_names:
            net.add_node("empty", label="검색 결과 없음", color='#94a3b8', size=30)
            net.save_graph(temp_path)
            return graph_id

        try:
            with self.driver.session() as session:
                # ★ 스키마 변경 대응:
                #   - FROM_PAPER / FROM_DOC 는 엔티티→메타 노드 출처 관계이므로
                #     시각화에서 제외한다 (의미 트리플만 그린다).
                #   - 메타 노드(Paper/Document)는 이름이 doc_id 라서 보통 $names 에
                #     안 잡히지만, 방어적으로 레이블로도 제외한다.
                query = """
                    MATCH (s)-[r]->(o)
                    WHERE s.name IN $names AND o.name IN $names
                      AND NOT type(r) IN ['FROM_PAPER', 'FROM_DOC']
                      AND NOT s:Paper AND NOT s:Document
                      AND NOT o:Paper AND NOT o:Document
                """
                if dataset and dataset != 'All':
                    query += " AND $dataset IN labels(s) AND $dataset IN labels(o)"
                query += """
                    RETURN elementId(s) AS sid, properties(s) AS sp,
                           type(r) AS rel,
                           elementId(o) AS tid, properties(o) AS tp
                    LIMIT $edge_limit
                """
                records = list(session.run(
                    query, names=node_names, dataset=dataset, edge_limit=edge_limit
                ))

                if not records:
                    node_q = "MATCH (n) WHERE n.name IN $names"
                    if dataset and dataset != 'All':
                        node_q += " AND $dataset IN labels(n)"
                    node_q += " RETURN elementId(n) AS nid, properties(n) AS np"
                    for rec in session.run(node_q, names=node_names, dataset=dataset):
                        np = rec['np']
                        net.add_node(rec['nid'],
                            label=np.get('name', ''),
                            color=self._type_color(np),
                            size=35,
                            title=f"Type: {np.get('type', 'Unknown')}")
                else:
                    degree: dict[str, int] = {}
                    for rec in records:
                        degree[rec['sid']] = degree.get(rec['sid'], 0) + 1
                        degree[rec['tid']] = degree.get(rec['tid'], 0) + 1

                    added: set[str] = set()
                    for rec in records:
                        sid, tid = rec['sid'], rec['tid']
                        if sid not in added:
                            sp = rec['sp']
                            net.add_node(sid,
                                label=self._get_node_label(sp, sid),
                                color=self._type_color(sp),
                                size=self._node_size(degree.get(sid, 1)),
                                title=f"Type: {sp.get('type', 'Unknown')} | Degree: {degree.get(sid, 1)}")
                            added.add(sid)
                        if tid not in added:
                            tp = rec['tp']
                            net.add_node(tid,
                                label=self._get_node_label(tp, tid),
                                color=self._type_color(tp),
                                size=self._node_size(degree.get(tid, 1)),
                                title=f"Type: {tp.get('type', 'Unknown')} | Degree: {degree.get(tid, 1)}")
                            added.add(tid)
                        net.add_edge(sid, tid, label=rec['rel'], title=rec['rel'])

        except Exception as e:
            print(f"RAG 그래프 생성 오류: {e}")
            net.add_node("error", label=f"오류: {e}", color='#ff0000', size=30)

        net.save_graph(temp_path)
        return graph_id
