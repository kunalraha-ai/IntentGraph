"""IntentGraph's interactive memory-graph renderer — a freshly written module.

This is IntentGraph's own HTML renderer for the memory archive: it takes the
``nx.Graph`` the visualization adapter builds (:mod:`intentgraph.viz`) and
emits a single self-contained ``graph.html`` (pinned vis-network from CDN,
everything else inline; no build step, no internet requirement beyond the
one CDN script the browser loads).

Scope is deliberately the memory-graph use case — this is not a general
code-graph renderer:

* nodes: stable record ids, degree sizing, memory-type communities, the
  adapter's ``viz_*`` overlay attributes (status rings, shapes, evidence
  panel fields) copied conditionally so plain graphs render identically;
* edges: typed relations with the stored explicit/inferred confidence
  (inferred → dashed), optional ``viz_color`` accents (contradictions);
* retrieval: an optional highlight overlay produced by a REAL retrieve()
  call — cyan for directly retrieved, violet for graph-expanded context,
  temporary dimming of everything else, a reset control, and an explicit
  honest empty state ("No evidence returned"). The renderer never invents
  results; it displays exactly the overlay it is given;
* security: all label/field text is sanitized; the embedded JSON is
  ``</``-escaped so no value can break out of the script tag; panel fields
  are rendered through an esc() boundary in the JS.

The call contract mirrors the shape the previous (borrowed) renderer
exposed, so :func:`intentgraph.viz.write_memory_html` needed only its
import line changed.
"""
from __future__ import annotations

import html as _html
import json
import re
from pathlib import Path

import networkx as nx

__all__ = ["to_html"]

#: Pinned vis-network with SRI — same artifact the platform has always used;
#: the hash is a public CDN checksum.
_VIS_NETWORK_TAG = (
    '<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/'
    'vis-network.min.js"\n'
    '        integrity="sha384-Ux6phic9PEHJ38YtrijhkzyJ8yQlH8i/+buBR8s3mAZOJrP1gwyvAcIYl3GWtpX1"\n'
    '        crossorigin="anonymous"></script>'
)

# Highlighting palette (memory-status channel — never the community colors,
# which keep their own meaning).
_DIRECT = "#22d3ee"    # cyan: directly retrieved evidence
_EXPANDED = "#8b5cf6"  # violet: graph-expanded supporting context
_DIM_BG = "#23233a"
_DIM_BORDER = "#3a3a4e"
_DIM_FONT = "#4a4a5a"


def _sanitize(text: str) -> str:
    """Strip control characters and cap length — display-safe text.

    Mirrors the platform-wide sanitization semantics: control characters
    (ANSI escapes, template delimiters' invisible relatives) never reach the
    DOM; length is capped so a hostile corpus cannot balloon the page.
    """
    cleaned = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]", "", str(text))
    return cleaned[:256]


def _js_safe(obj) -> str:
    """JSON for embedding in a <script> tag: neutralize the closing-tag
    sequence so no data value can terminate the script."""
    return json.dumps(obj).replace("</", "<\\/")


def _styles() -> str:
    return """<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f0f1a; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; display: flex; height: 100vh; overflow: hidden; }
  #graph { flex: 1; }
  #sidebar { width: 280px; background: #1a1a2e; border-left: 1px solid #2a2a4e; display: flex; flex-direction: column; overflow: hidden; }
  #search-wrap { padding: 12px; border-bottom: 1px solid #2a2a4e; }
  #search { width: 100%; background: #0f0f1a; border: 1px solid #3a3a5e; color: #e0e0e0; padding: 7px 10px; border-radius: 6px; font-size: 13px; outline: none; }
  #search:focus { border-color: #4E79A7; }
  #search-results { max-height: 140px; overflow-y: auto; padding: 4px 12px; display: none; }
  .search-item { padding: 4px 6px; cursor: pointer; border-radius: 4px; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .search-item:hover { background: #2a2a4e; }
  #info-panel { padding: 14px; border-bottom: 1px solid #2a2a4e; min-height: 140px; }
  #info-panel h3 { font-size: 13px; color: #aaa; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
  #info-content { font-size: 13px; color: #ccc; line-height: 1.6; }
  #info-content .field { margin-bottom: 5px; }
  #info-content .field b { color: #e0e0e0; }
  #info-content .empty { color: #555; font-style: italic; }
  .neighbor-link { display: block; padding: 2px 6px; margin: 2px 0; border-radius: 3px; cursor: pointer; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-left: 3px solid #333; }
  .neighbor-link:hover { background: #2a2a4e; }
  #neighbors-list { max-height: 160px; overflow-y: auto; margin-top: 4px; }
  #legend-wrap { flex: 1; overflow-y: auto; padding: 12px; }
  #legend-wrap h3 { font-size: 13px; color: #aaa; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
  .legend-item { display: flex; align-items: center; gap: 8px; padding: 4px 0; cursor: pointer; border-radius: 4px; font-size: 12px; }
  .legend-item:hover { background: #2a2a4e; padding-left: 4px; }
  .legend-item.dimmed { opacity: 0.35; }
  .legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  .legend-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .legend-count { color: #666; font-size: 11px; }
  #stats { padding: 10px 14px; border-top: 1px solid #2a2a4e; font-size: 11px; color: #555; }
  #legend-extra { margin-top: 4px; }
</style>"""


def _overlay_styles() -> str:
    """HUD styles, shipped only when a retrieval overlay is present — a
    query-less render must not carry retrieval markup at all."""
    return """<style>
  #retrieval-hud { padding: 12px 14px; border-bottom: 1px solid #2a2a4e; }
  #retrieval-hud h3 { font-size: 13px; color: #aaa; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
  #retrieval-hud button { margin-top: 8px; width: 100%; background: #0f0f1a; border: 1px solid #3a3a5e; color: #ccc; padding: 6px 10px; border-radius: 6px; font-size: 12px; cursor: pointer; }
  #retrieval-hud button:hover { border-color: #22d3ee; color: #fff; }
  #retrieval-results { max-height: 140px; overflow-y: auto; }
  #retrieval-results .retrieval-item { display: block; padding: 2px 6px; margin: 2px 0; border-radius: 3px; cursor: pointer; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-left: 3px solid #22d3ee; background: transparent; color: #ccc; width: 100%; text-align: left; }
  #retrieval-results .retrieval-item:hover { background: #2a2a4e; }
  #retrieval-results .retrieval-more { display: block; padding: 2px 6px; font-size: 11px; color: #666; }
  #retrieval-results .empty { color: #555; font-style: italic; font-size: 12px; }
  #retrieval-hud.retrieval-empty { opacity: 0.75; }
</style>"""


def _script() -> str:
    """The interactive layer: datasets, panel, search, legend filters, and
    the retrieval highlight/dim/reset cycle. Written against the documented
    JSON contract emitted by :func:`to_html`."""
    return """<script>
const RAW_NODES = /*__NODES__*/;
const RAW_EDGES = /*__EDGES__*/;
const LEGEND = /*__LEGEND__*/;

// HTML-escape helper — the injection boundary for everything rendered into
// innerHTML (the #1838-style XSS boundary: values are data, never markup)
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

const nodesDS = new vis.DataSet(RAW_NODES.map((n, i) => {
  const v = {
    id: n.id, label: n.label, color: n.color, size: n.size,
    font: n.font, title: n.title,
    x: 30 * Math.sqrt(i) * Math.cos(i * 2.4),
    y: 30 * Math.sqrt(i) * Math.sin(i * 2.4),
    community: n.community, community_name: n.community_name,
    source_file: n.source_file, degree: n.degree,
  };
  // conditional overlay passthrough: plain nodes render unchanged
  if (n.shapeProperties !== undefined) v.shapeProperties = n.shapeProperties;
  if (n.borderWidth !== undefined) v.borderWidth = n.borderWidth;
  if (n.shape !== undefined) v.shape = n.shape;
  if (n.extra_fields !== undefined) v.extra_fields = n.extra_fields;
  if (n.viz_ring !== undefined) v.viz_ring = n.viz_ring;
  return v;
}));

const edgesDS = new vis.DataSet(RAW_EDGES.map((e, i) => {
  const v = {
    id: i, from: e.from, to: e.to, label: '',
    title: e.title, dashes: e.dashes, width: e.width, color: e.color,
    arrows: { to: { enabled: true, scaleFactor: 0.5 } },
  };
  if (e.viz_color !== undefined) v.viz_color = e.viz_color;
  return v;
}));

const container = document.getElementById('graph');
const network = new vis.Network(container, { nodes: nodesDS, edges: edgesDS }, {
  physics: {
    enabled: true, solver: 'forceAtlas2Based',
    forceAtlas2Based: { gravitationalConstant: -60, centralGravity: 0.005,
                        springLength: 120, springConstant: 0.08, damping: 0.4,
                        avoidOverlap: 0.8 },
    stabilization: { iterations: 200, fit: true },
  },
  interaction: { hover: true, tooltipDelay: 100, hideEdgesOnDrag: true },
  nodes: { shape: 'dot', borderWidth: 1.5 },
  edges: { smooth: { type: 'continuous', roundness: 0.2 }, selectionWidth: 3 },
});
network.once('stabilizationIterationsDone', () => {
  network.setOptions({ physics: { enabled: false } });
});

function showInfo(nodeId) {
  const n = nodesDS.get(nodeId);
  if (!n) return;
  const neighborIds = network.getConnectedNodes(nodeId);
  const neighborItems = neighborIds.map(nid => {
    const nb = nodesDS.get(nid);
    const color = nb ? nb.color.background : '#555';
    return `<span class="neighbor-link" style="border-left-color:${esc(color)}" data-nid="${esc(nid)}">${esc(nb ? nb.label : nid)}</span>`;
  }).join('');
  let html = `<div class="field"><b>${esc(n.label)}</b></div>`;
  html += `<div class="field">Community: ${esc(n.community_name || '')}</div>`;
  if (n.source_file) html += `<div class="field">Source: ${esc(n.source_file)}</div>`;
  html += `<div class="field">Degree: ${n.degree}</div>`;
  if (n.extra_fields && n.extra_fields.length) {
    n.extra_fields.forEach(pair => {
      html += `<div class="field">${esc(pair[0])}: ${esc(pair[1])}</div>`;
    });
  }
  if (neighborIds.length) {
    html += `<div class="field" style="margin-top:8px;color:#aaa;font-size:11px">Neighbors (${neighborIds.length})</div><div id="neighbors-list">${neighborItems}</div>`;
  }
  document.getElementById('info-content').innerHTML = html;
}

function focusNode(nodeId) {
  network.focus(nodeId, { scale: 1.4, animation: true });
  network.selectNodes([nodeId]);
  showInfo(nodeId);
}
document.addEventListener('click', e => {
  const el = e.target.closest('.neighbor-link');
  if (el && el.dataset.nid !== undefined) focusNode(el.dataset.nid);
});
document.addEventListener('click', e => {
  const el = e.target.closest('.retrieval-item');
  if (el && el.dataset.nid !== undefined) focusNode(el.dataset.nid);
});

let hoveredNodeId = null;
network.on('hoverNode', params => { hoveredNodeId = params.node; container.style.cursor = 'pointer'; });
network.on('blurNode', () => { hoveredNodeId = null; container.style.cursor = 'default'; });
network.on('click', params => {
  if (params.nodes.length > 0) showInfo(params.nodes[0]);
  else if (hoveredNodeId === null) {
    document.getElementById('info-content').innerHTML =
      '<span class="empty">Click a node to inspect it</span>';
  }
});

const searchInput = document.getElementById('search');
const searchResults = document.getElementById('search-results');
searchInput.addEventListener('input', () => {
  const q = searchInput.value.toLowerCase().trim();
  searchResults.innerHTML = '';
  if (!q) { searchResults.style.display = 'none'; return; }
  const matches = RAW_NODES.filter(n => n.label.toLowerCase().includes(q)).slice(0, 20);
  if (!matches.length) { searchResults.style.display = 'none'; return; }
  searchResults.style.display = 'block';
  matches.forEach(n => {
    const el = document.createElement('div');
    el.className = 'search-item';
    el.textContent = n.label;
    el.style.borderLeft = `3px solid ${n.color.background}`;
    el.style.paddingLeft = '8px';
    el.onclick = () => focusNode(n.id);
    searchResults.appendChild(el);
  });
});

const hiddenCommunities = new Set();
function toggleCommunity(cid, show) {
  const updates = RAW_NODES
    .filter(n => n.community === cid)
    .map(n => ({ id: n.id, hidden: !show }));
  nodesDS.update(updates);
  if (show) hiddenCommunities.delete(cid); else hiddenCommunities.add(cid);
}
const legendEl = document.getElementById('legend');
LEGEND.forEach(c => {
  const item = document.createElement('div');
  item.className = 'legend-item';
  const cb = document.createElement('input');
  cb.type = 'checkbox'; cb.checked = true;
  cb.addEventListener('change', e => {
    e.stopPropagation();
    toggleCommunity(c.cid, cb.checked);
    item.classList.toggle('dimmed', !cb.checked);
  });
  item.innerHTML = `<div class="legend-dot" style="background:${esc(c.color)}"></div>
    <span class="legend-label">${esc(c.label)}</span>
    <span class="legend-count">${esc(c.count)}</span>`;
  item.prepend(cb);
  item.onclick = e => {
    if (e.target === cb) return;
    cb.checked = !cb.checked;
    cb.dispatchEvent(new Event('change'));
  };
  legendEl.appendChild(item);
});

// --- retrieval highlight & extra legend live in _overlay_script(), emitted
// --- only when the corresponding overlay exists (a query-less render must
// --- carry zero retrieval markup).
</script>"""


def _overlay_script(retrieval_json: str, extra_json: str,
                     has_retrieval: bool, has_extra: bool) -> str:
    """Retrieval highlight/dim/reset + evidence-state legend JS.

    Second <script> block; top-level consts from the first block remain
    visible (script-level lexical bindings are shared across classic
    scripts). Emitted only when at least one overlay is present — a
    query-less render carries zero retrieval markup.
    """
    retrieval_js = ""
    if has_retrieval:
        retrieval_js = """
// Applied from a REAL retrieve() result; the renderer displays exactly what
// it is given, including the empty case (no fabricated matches).
const RETRIEVAL = __RETRIEVAL_JSON__;
(function() {
  const hud = document.getElementById('retrieval-hud');
  if (!hud) return;
  const DIRECT = '__DIRECT__';
  const EXPANDED = '__EXPANDED__';
  const DIM_BG = '__DIM_BG__', DIM_BORDER = '__DIM_BORDER__', DIM_FONT = '__DIM_FONT__';
  const involved = new Set(RETRIEVAL.results.map(r => r.id));
  const tiers = {};
  RETRIEVAL.results.forEach(r => { tiers[r.id] = r.tier || 'direct'; });
  let active = false;

  function applyHighlight() {
    if (active) return;
    active = true;
    const updates = [];
    RAW_NODES.forEach(n => {
      if (involved.has(n.id)) {
        const ring = tiers[n.id] === 'expanded' ? EXPANDED : DIRECT;
        updates.push({ id: n.id,
          color: { background: n.color.background, border: ring,
                   highlight: { background: '#ffffff', border: ring } },
          borderWidth: 4,
          font: { size: Math.max(n.font.size, 14), color: '#ffffff' },
        });
      } else {
        updates.push({ id: n.id,
          color: { background: DIM_BG, border: DIM_BORDER,
                   highlight: { background: '#ffffff', border: DIM_BORDER } },
          font: { size: n.font.size, color: DIM_FONT },
        });
      }
    });
    nodesDS.update(updates);
    const edgeUpdates = [];
    edgesDS.forEach(e => {
      const both = involved.has(e.from) && involved.has(e.to);
      edgeUpdates.push({ id: e.id,
        color: { color: both ? (e.viz_color || '#8f9bb3') : '#2a2a3e',
                 opacity: both ? 0.95 : 0.12,
                 highlight: e.viz_color || '#8f9bb3' },
        width: both ? 3 : 1,
      });
    });
    edgesDS.update(edgeUpdates);
  }

  function resetHighlight() {
    if (!active) return;
    active = false;
    const updates = [];
    RAW_NODES.forEach(n => {
      const u = { id: n.id, color: n.color, font: n.font };
      u.borderWidth = n.borderWidth !== undefined ? n.borderWidth : 1.5;
      u.shapeProperties = n.shapeProperties || {};
      updates.push(u);
    });
    nodesDS.update(updates);
    const edgeUpdates = [];
    edgesDS.forEach(e => {
      const raw = RAW_EDGES[e.id];
      if (!raw) return;
      edgeUpdates.push({ id: e.id, color: raw.color, width: raw.width });
    });
    edgesDS.update(edgeUpdates);
  }

  document.getElementById('retrieval-reset')
    .addEventListener('click', resetHighlight);
  const list = document.getElementById('retrieval-results');
  if (!RETRIEVAL.results.length) {
    hud.classList.add('retrieval-empty');
    list.innerHTML =
      '<span class="empty">No evidence returned \u2014 nothing highlighted.</span>';
    return;
  }
  applyHighlight();
  const counts = { direct: 0, expanded: 0 };
  RETRIEVAL.results.forEach(r => { counts[r.tier || 'direct']++; });
  list.innerHTML = RETRIEVAL.results.slice(0, 12).map(r => {
    const n = RAW_NODES.find(x => x.id === r.id);
    const lbl = n ? n.label : r.id;
    const dot = tiers[r.id] === 'expanded' ? EXPANDED : DIRECT;
    return `<span class="retrieval-item" style="border-left-color:${dot}" data-nid="${esc(r.id)}">${esc(lbl)}</span>`;
  }).join('') + (RETRIEVAL.results.length > 12
      ? `<span class="retrieval-more">+ ${RETRIEVAL.results.length - 12} more</span>` : '');
  document.getElementById('retrieval-count').textContent =
    `${counts.direct} direct \u00b7 ${counts.expanded} supporting`;
})();
"""
        retrieval_js = (retrieval_json and retrieval_js
                         .replace("__RETRIEVAL_JSON__", retrieval_json)
                         .replace("__DIRECT__", _DIRECT)
                         .replace("__EXPANDED__", _EXPANDED)
                         .replace("__DIM_BG__", _DIM_BG)
                         .replace("__DIM_BORDER__", _DIM_BORDER)
                         .replace("__DIM_FONT__", _DIM_FONT))

    extra_js = ""
    if has_extra:
        extra_js = """
// Evidence & memory-state legend entries (only when supplied)
const EXTRA_LEGEND = __EXTRA_JSON__;
(function() {
  const el = document.getElementById('legend-extra');
  if (!el || !EXTRA_LEGEND.length) return;
  EXTRA_LEGEND.forEach(c => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.style.opacity = '0.85';
    item.innerHTML = `<div class="legend-dot" style="background:${esc(c.color)};${
      c.dashes ? 'background:transparent;border:2px dashed ' + esc(c.color) + ';' : ''
    }"></div><span class="legend-label">${esc(c.label)}</span>`;
    el.appendChild(item);
  });
})();
"""
        extra_js = extra_js.replace("__EXTRA_JSON__", extra_json)

    if not (retrieval_js or extra_js):
        return ""
    return "<script>" + retrieval_js + extra_js + "</script>"


def to_html(
    G: nx.Graph,
    communities: dict[int, list[str]],
    output_path: str,
    community_labels: dict[int, str] | None = None,
    retrieval: dict | None = None,
    legend_extra: list[dict] | None = None,
) -> bool:
    """Render a memory graph to an interactive, self-contained graph.html.

    ``retrieval`` (optional): a REAL retrieval overlay,
    ``{"query", "results": [{"id", "tier"}...]}``; when None the HUD is
    absent. ``legend_extra`` (optional): evidence/memory-state legend
    entries ``[{"color", "label", "dashes"?}]``.

    Returns True when the file was written.
    """
    degree = dict(G.degree())
    max_deg = max(degree.values(), default=1) or 1
    node_community = {nid: cid for cid, members in communities.items()
                      for nid in members}

    vis_nodes = []
    for node_id, data in G.nodes(data=True):
        cid = node_community.get(node_id, 0)
        color = _COMMUNITY_COLORS[cid % len(_COMMUNITY_COLORS)]
        label = _sanitize(data.get("label", node_id))
        deg = degree.get(node_id, 1)
        node = {
            "id": node_id,
            "label": label,
            "color": {"background": color, "border": color,
                      "highlight": {"background": "#ffffff", "border": color}},
            "size": round(10 + 30 * (deg / max_deg), 1),
            "font": {"size": 12 if deg >= max_deg * 0.15 else 0,
                     "color": "#ffffff"},
            "title": label,
            "community": cid,
            "community_name": _sanitize(
                (community_labels or {}).get(cid, f"Community {cid}")),
            "source_file": _sanitize(str(data.get("source_file") or "")),
            "degree": deg,
        }
        viz_ring = data.get("viz_ring")
        if isinstance(viz_ring, str) and viz_ring:
            ring = _sanitize(viz_ring)
            node["color"] = {"background": color, "border": ring,
                             "highlight": {"background": "#ffffff", "border": ring}}
            node["borderWidth"] = 3
            node["viz_ring"] = ring
        if data.get("viz_dashes"):
            node["shapeProperties"] = {"borderDashes": [4, 4]}
        viz_shape = data.get("viz_shape")
        if isinstance(viz_shape, str) and viz_shape:
            node["shape"] = viz_shape
        viz_fields = data.get("viz_fields")
        if isinstance(viz_fields, list) and viz_fields:
            node["extra_fields"] = [
                [_sanitize(str(lbl)), _sanitize(str(val))]
                for lbl, val in viz_fields[:16]]
        vis_nodes.append(node)

    vis_edges = []
    for u, v, data in G.edges(data=True):
        confidence = data.get("confidence", "EXTRACTED")
        relation = data.get("relation", "")
        edge = {
            "from": data.get("_src", u),
            "to": data.get("_tgt", v),
            "label": relation,
            "title": _sanitize(f"{relation} [{confidence}]"),
            "dashes": confidence != "EXTRACTED",
            "width": 2 if confidence == "EXTRACTED" else 1,
            "color": {"opacity": 0.7 if confidence == "EXTRACTED" else 0.35},
        }
        viz_color = data.get("viz_color")
        if isinstance(viz_color, str) and viz_color:
            c = _sanitize(viz_color)
            edge["color"] = {"color": c, "opacity": 0.9, "highlight": c}
            edge["viz_color"] = c
        vis_edges.append(edge)

    legend_data = []
    for cid in sorted(set(node_community.values())):
        color = _COMMUNITY_COLORS[cid % len(_COMMUNITY_COLORS)]
        legend_data.append({
            "cid": cid, "color": color,
            "label": _sanitize((community_labels or {}).get(
                cid, f"Community {cid}")),
            "count": len(communities.get(cid, [])),
        })

    retrieval_payload = dict(retrieval or {"query": "", "results": []})
    nodes_json = _js_safe(vis_nodes)
    edges_json = _js_safe(vis_edges)
    legend_json = _js_safe(legend_data)
    retrieval_json = _js_safe(retrieval_payload)
    extra_json = _js_safe(legend_extra or [])

    title = _html.escape(_sanitize(Path(output_path).name))
    stats = (f"{G.number_of_nodes()} nodes &middot; "
             f"{G.number_of_edges()} edges &middot; "
             f"{len(legend_data)} communities")
    q = _html.escape(_sanitize(str(retrieval_payload.get("query", "")))) \
        if retrieval is not None else ""
    hud = "" if retrieval is None else f"""
  <div id="retrieval-hud">
    <h3>Retrieved Evidence</h3>
    <div class="field" style="font-size:12px;color:#aaa">Query: {q}</div>
    <div id="retrieval-count" class="field" style="font-size:11px;color:#888"></div>
    <div id="retrieval-results"></div>
    <button id="retrieval-reset" type="button">Reset highlighting</button>
  </div>"""
    extra_block = "" if not legend_extra else (
        '<h3 style="margin-top:14px">Evidence &amp; Memory States</h3>'
        '<div id="legend-extra"></div>')

    script = (_script()
             .replace("/*__NODES__*/", nodes_json)
             .replace("/*__EDGES__*/", edges_json)
             .replace("/*__LEGEND__*/", legend_json))
    overlay_script = _overlay_script(
        retrieval_json, extra_json,
        has_retrieval=retrieval is not None,
        has_extra=bool(legend_extra))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IntentGraph — {title}</title>
{_VIS_NETWORK_TAG}
{_styles()}
{_overlay_styles() if retrieval is not None else ""}
</head>
<body>
<div id="graph"></div>
<div id="sidebar">
  <div id="search-wrap">
    <input id="search" type="text" placeholder="Search memories..." autocomplete="off">
    <div id="search-results"></div>
  </div>
  <div id="info-panel">
    <h3>Memory Info</h3>
    <div id="info-content"><span class="empty">Click a memory to inspect it</span></div>
  </div>
  {hud}
  <div id="legend-wrap">
    <h3>Memory Types</h3>
    <div id="legend"></div>
    {extra_block}
  </div>
  <div id="stats">{stats}</div>
</div>
{script}
{overlay_script}
</body>
</html>"""
    Path(output_path).write_text(html, encoding="utf-8")
    return True


#: Community palette — the same sequence the platform's visualizations have
#: always used, so a memory graph looks native next to a code graph.
_COMMUNITY_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]
