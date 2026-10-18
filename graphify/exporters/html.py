"""html — moved verbatim from graphify/export.py."""
from __future__ import annotations

from graphify.exporters.base import COMMUNITY_COLORS  # noqa: E402,F401
from pathlib import Path
import html as _html
from graphify.analyze import _node_community_map
from graphify.paths import write_text_atomic
import json
import networkx as nx
from graphify.security import sanitize_label


MAX_NODES_FOR_VIZ = 5_000
_HTML_STALE_MARKER = ".graph.html.stale"

def _viz_node_limit() -> int:
    """Return the effective viz node limit, honoring GRAPHIFY_VIZ_NODE_LIMIT env var.

    Falls back to MAX_NODES_FOR_VIZ when the env var is unset, empty, or non-integer.
    Set to 0 to disable HTML viz unconditionally (useful for CI runners).
    """
    import os
    raw = os.environ.get("GRAPHIFY_VIZ_NODE_LIMIT")
    if raw is None or not raw.strip():
        return MAX_NODES_FOR_VIZ
    try:
        return int(raw)
    except ValueError:
        return MAX_NODES_FOR_VIZ

def _html_styles() -> str:
    return """<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f0f1a; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; display: flex; height: 100vh; overflow: hidden; }
  #graph { flex: 1; }
  #sidebar { width: 280px; background: #1a1a2e; border-left: 1px solid #2a2a4e; display: flex; flex-direction: column; overflow: hidden; }
  #search-wrap { padding: 12px; border-bottom: 1px solid #2a2a4e; }
  #search { width: 100%; background: #0f0f1a; border: 1px solid #3a3a5e; color: #e0e0e0; padding: 7px 10px; border-radius: 6px; font-size: 13px; outline: none; }
  #search:focus { border-color: #4E79A7; }
  #search-results { max-height: 140px; overflow-y: auto; padding: 4px 12px; border-bottom: 1px solid #2a2a4e; display: none; }
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
  #legend-controls { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; padding: 4px 0; }
  #legend-controls label { display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 12px; color: #aaa; user-select: none; }
  #legend-controls label:hover { color: #e0e0e0; }
  .legend-cb, #select-all-cb { appearance: none; -webkit-appearance: none; width: 14px; height: 14px; border: 1.5px solid #3a3a5e; border-radius: 3px; background: #0f0f1a; cursor: pointer; position: relative; flex-shrink: 0; }
  .legend-cb:checked, #select-all-cb:checked { background: #4E79A7; border-color: #4E79A7; }
  .legend-cb:checked::after, #select-all-cb:checked::after { content: ''; position: absolute; left: 3.5px; top: 1px; width: 4px; height: 7px; border: solid #fff; border-width: 0 2px 2px 0; transform: rotate(45deg); }
  #select-all-cb:indeterminate { background: #4E79A7; border-color: #4E79A7; }
  #select-all-cb:indeterminate::after { content: ''; position: absolute; left: 2px; top: 5px; width: 8px; height: 2px; background: #fff; border: none; transform: none; }
</style>"""


def _html_overlay_styles() -> str:
    """Styles for the retrieval HUD / evidence-state legend. Emitted ONLY
    when an overlay (retrieval or legend_extra) is present, so the default
    render stays byte-identical to the pre-feature HTML."""
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
  #legend-extra { margin-top: 4px; }
</style>"""

def _hyperedge_script(hyperedges_json: str) -> str:
    return f"""<script>
// Render hyperedges as shaded regions
const hyperedges = {hyperedges_json};
// afterDrawing passes ctx already transformed to network coordinate space.
// Draw node positions raw — no manual pan/zoom/DPR math needed.

// Andrew's monotone chain. Returns the hull in counter-clockwise order, which
// is what the perimeter must be traced in. Collinear and duplicate points
// collapse to the extremes, so degenerate member sets render as a segment
// rather than a zero-area crossed path.
function convexHull(pts) {{
    const p = pts.slice().sort((a, b) => (a.x - b.x) || (a.y - b.y));
    if (p.length < 3) return p;
    const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    const build = seq => {{
        const out = [];
        for (const q of seq) {{
            while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
            out.push(q);
        }}
        out.pop();
        return out;
    }};
    const hull = build(p).concat(build(p.slice().reverse()));
    return hull.length >= 3 ? hull : p;
}}
network.on('afterDrawing', function(ctx) {{
    hyperedges.forEach(h => {{
        const positions = h.nodes
            .map(nid => network.getPositions([nid])[nid])
            .filter(p => p !== undefined);
        if (positions.length < 2) return;
        ctx.save();
        ctx.globalAlpha = 0.12;
        ctx.fillStyle = '#6366f1';
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        ctx.beginPath();
        // Centroid and expanded hull in network coordinates.
        // The perimeter must follow hull order, not h.nodes order: tracing the
        // raw member order self-intersects whenever the layout does not happen
        // to place members in angular order, filling as crossed wedges.
        const cx = positions.reduce((s, p) => s + p.x, 0) / positions.length;
        const cy = positions.reduce((s, p) => s + p.y, 0) / positions.length;
        const hull = convexHull(positions);
        const expanded = hull.map(p => ({{
            x: cx + (p.x - cx) * 1.15,
            y: cy + (p.y - cy) * 1.15
        }}));
        ctx.moveTo(expanded[0].x, expanded[0].y);
        expanded.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 0.4;
        ctx.stroke();
        // Label
        ctx.globalAlpha = 0.8;
        ctx.fillStyle = '#4f46e5';
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(h.label, cx, cy - 5);
        ctx.restore();
    }});
}});
</script>"""

def _retrieval_script(retrieval_json: str) -> str:
    """Highlight/dim/reset block for a REAL retrieval result (additive).

    The renderer displays exactly what it is given — including the empty
    case, which shows an explicit "no evidence returned" notice and changes
    nothing. Direct results get a cyan status ring; graph-expanded
    supporting context a violet one (community background colors are never
    touched by highlighting); unrelated nodes are temporarily dimmed via
    font/background updates that the reset control restores from the
    RAW_NODES snapshot. A retrieval rank is a retrieval fact, never a
    truth score — the HUD says 'evidence', not 'correct'.
    """
    return f"""<script>
const RETRIEVAL = {retrieval_json};
// Only render the HUD when a real query ran.
(function() {{
  const hud = document.getElementById('retrieval-hud');
  if (!hud) return;
  const DIRECT = '#22d3ee';   // cyan: directly retrieved evidence
  const EXPANDED = '#8b5cf6';  // violet: graph-expanded supporting context
  const DIM_BG = '#23233a', DIM_BORDER = '#3a3a4e', DIM_FONT = '#4a4a5a';
  const involved = new Set(RETRIEVAL.results.map(r => r.id));
  const tiers = {{}};
  RETRIEVAL.results.forEach(r => {{ tiers[r.id] = r.tier || 'direct'; }});
  let active = false;

  function nodeBase(nid) {{
    const raw = RAW_NODES.find(n => n.id === nid);
    if (!raw) return null;
    return {{
      color: raw.color, font: raw.font,
      borderWidth: raw.borderWidth !== undefined ? raw.borderWidth : undefined,
      shapeProperties: raw.shapeProperties,
    }};
  }}

  function applyHighlight() {{
    if (active) return;
    active = true;
    const updates = [];
    RAW_NODES.forEach(n => {{
      if (involved.has(n.id)) {{
        const ring = tiers[n.id] === 'expanded' ? EXPANDED : DIRECT;
        updates.push({{ id: n.id,
          color: {{ background: n.color.background, border: ring,
                   highlight: {{ background: '#ffffff', border: ring }} }},
          borderWidth: 4,
          font: {{ size: Math.max(n.font.size, 14), color: '#ffffff' }},
        }});
      }} else {{
        updates.push({{ id: n.id,
          color: {{ background: DIM_BG, border: DIM_BORDER,
                   highlight: {{ background: '#ffffff', border: DIM_BORDER }} }},
          font: {{ size: n.font.size, color: DIM_FONT }},
        }});
      }}
    }});
    nodesDS.update(updates);
    // Raise visibility of edges among involved nodes (they exist in the
    // graph by construction); dim the rest temporarily.
    const edgeUpdates = [];
    edgesDS.forEach(e => {{
      const both = involved.has(e.from) && involved.has(e.to);
      edgeUpdates.push({{ id: e.id,
        color: {{ color: both ? (e.viz_color || '#8f9bb3') : '#2a2a3e',
                 opacity: both ? 0.95 : 0.12,
                 highlight: e.viz_color || '#8f9bb3' }},
        width: both ? 3 : 1,
      }});
    }});
    edgesDS.update(edgeUpdates);
  }}

  function resetHighlight() {{
    if (!active) return;
    active = false;
    const updates = [];
    RAW_NODES.forEach(n => {{
      const base = nodeBase(n.id);
      if (!base) return;
      const u = {{ id: n.id, color: base.color, font: base.font }};
      u.borderWidth = base.borderWidth !== undefined ? base.borderWidth : 1.5;
      u.shapeProperties = base.shapeProperties || {{}};
      updates.push(u);
    }});
    nodesDS.update(updates);
    const edgeUpdates = [];
    edgesDS.forEach(e => {{
      const raw = RAW_EDGES[e.id];
      if (!raw) return;
      edgeUpdates.push({{ id: e.id, color: raw.color, width: raw.width }});
    }});
    edgesDS.update(edgeUpdates);
  }}

  document.getElementById('retrieval-reset').addEventListener('click', resetHighlight);
  const list = document.getElementById('retrieval-results');
  if (!RETRIEVAL.results.length) {{
    hud.classList.add('retrieval-empty');
    list.innerHTML = '<span class="empty">No evidence returned — nothing highlighted.</span>';
    return;  // honest empty state: no fake matches, no dimming
  }}
  applyHighlight();
  const counts = {{ direct: 0, expanded: 0 }};
  RETRIEVAL.results.forEach(r => {{ counts[r.tier || 'direct']++; }});
  list.innerHTML =
    RETRIEVAL.results.slice(0, 12).map(r => {{
      const n = RAW_NODES.find(x => x.id === r.id);
      const lbl = n ? n.label : r.id;
      const dot = tiers[r.id] === 'expanded' ? EXPANDED : DIRECT;
      return `<span class="retrieval-item" style="border-left-color:${{dot}}" data-nid="${{esc(r.id)}}">${{esc(lbl)}}</span>`;
    }}).join('')
    + (RETRIEVAL.results.length > 12
       ? `<span class="retrieval-more">+ ${{RETRIEVAL.results.length - 12}} more</span>` : '');
  document.getElementById('retrieval-count').textContent =
    `${{counts.direct}} direct · ${{counts.expanded}} supporting`;
}})();
</script>"""


def _legend_extra_script(legend_extra_json: str) -> str:
    """Second legend block (memory/evidence states), emitted only when the
    caller supplies entries — the community legend above is untouched."""
    return f"""<script>
const EXTRA_LEGEND = {legend_extra_json};
(function() {{
  const el = document.getElementById('legend-extra');
  if (!el || !EXTRA_LEGEND.length) return;
  EXTRA_LEGEND.forEach(c => {{
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.style.opacity = '0.85';
    item.innerHTML = `<div class="legend-dot" style="background:${{esc(c.color)}};${{
      c.dashes ? 'background:transparent;border:2px dashed ' + esc(c.color) + ';' : ''
    }}"></div><span class="legend-label">${{esc(c.label)}}</span>`;
    el.appendChild(item);
  }});
}})();
</script>"""


def _html_script(nodes_json: str, edges_json: str, legend_json: str) -> str:
    return f"""<script>
const RAW_NODES = {nodes_json};
const RAW_EDGES = {edges_json};
const LEGEND = {legend_json};

// HTML-escape helper — prevents XSS when injecting graph data into innerHTML
function esc(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}}

// Build vis datasets
const nodesDS = new vis.DataSet(RAW_NODES.map((n, i) => {{
  const v = {{
  id: n.id, label: n.label, color: n.color, size: n.size,
  font: n.font, title: n.title,
  // Fermat/golden-angle spiral seed positions (#3699): spreading nodes out
  // before physics runs keeps avoidOverlap from computing near-zero-distance
  // repulsion that blows the BarnesHut recursion into a stack overflow on large
  // graphs. `i` is the map index — physics still settles small graphs identically.
  x: 30 * Math.sqrt(i) * Math.cos(i * 2.4),
  y: 30 * Math.sqrt(i) * Math.sin(i * 2.4),
  community: n.community, community_name: n.community_name,
  source_file: n.source_file, file_type: n.file_type, degree: n.degree,
  member_count: n.member_count,
  }};
  // Optional overlay styling fields (additive): nodes without them render
  // exactly as before — these conditionals existed for none of the original
  // nodes. shapeProperties/borderWidth/shape carry status rings & shapes;
  // extra_fields carries evidence-panel rows; viz_ring/viz_color mark the
  // pre-highlight state the retrieval reset restores to.
  if (n.shapeProperties !== undefined) v.shapeProperties = n.shapeProperties;
  if (n.borderWidth !== undefined) v.borderWidth = n.borderWidth;
  if (n.shape !== undefined) v.shape = n.shape;
  if (n.extra_fields !== undefined) v.extra_fields = n.extra_fields;
  if (n.viz_ring !== undefined) v.viz_ring = n.viz_ring;
  return v;
}})));

const edgesDS = new vis.DataSet(RAW_EDGES.map((e, i) => {{
  const v = {{
  id: i, from: e.from, to: e.to,
  label: '',
  title: e.title,
  dashes: e.dashes,
  width: e.width,
  color: e.color,
  arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
  }};
  if (e.viz_color !== undefined) v.viz_color = e.viz_color;
  return v;
}})));

const container = document.getElementById('graph');
const network = new vis.Network(container, {{ nodes: nodesDS, edges: edgesDS }}, {{
  physics: {{
    enabled: true,
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{
      gravitationalConstant: -60,
      centralGravity: 0.005,
      springLength: 120,
      springConstant: 0.08,
      damping: 0.4,
      avoidOverlap: 0.8,
    }},
    stabilization: {{ iterations: 200, fit: true }},
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 100,
    hideEdgesOnDrag: true,
    navigationButtons: false,
    keyboard: false,
  }},
  nodes: {{ shape: 'dot', borderWidth: 1.5 }},
  edges: {{ smooth: {{ type: 'continuous', roundness: 0.2 }}, selectionWidth: 3 }},
}});

network.once('stabilizationIterationsDone', () => {{
  network.setOptions({{ physics: {{ enabled: false }} }});
}});

function showInfo(nodeId) {{
  const n = nodesDS.get(nodeId);
  if (!n) return;
  const neighborIds = network.getConnectedNodes(nodeId);
  const neighborItems = neighborIds.map(nid => {{
    const nb = nodesDS.get(nid);
    const color = nb ? nb.color.background : '#555';
    return `<span class="neighbor-link" style="border-left-color:${{esc(color)}}" data-nid="${{esc(nid)}}">${{esc(nb ? nb.label : nid)}}</span>`;
  }}).join('');
  const isMetaNode = !n.file_type && !n.source_file;
  let html = `<div class="field"><b>${{esc(n.label)}}</b></div>`;
  if (!isMetaNode) {{
    html += `<div class="field">Type: ${{esc(n.file_type || 'unknown')}}</div>`;
  }}
  html += `<div class="field">Community: ${{esc(n.community_name || '')}}</div>`;
  if (!isMetaNode) {{
    html += `<div class="field">Source: ${{esc(n.source_file || '-')}}</div>`;
  }} else if (n.member_count !== undefined && n.member_count !== null) {{
    html += `<div class="field">Members: ${{esc(n.member_count)}}</div>`;
  }}
  html += `<div class="field">Degree: ${{n.degree}}</div>`;
  // Evidence rows from an overlay (additive): ordered label/value pairs the
  // adapter supplied — memory type, record id, event/ingest times, source
  // session, provenance, status, retrieval rank/score. esc() is the same
  // #1838 XSS boundary the rest of the panel uses.
  if (n.extra_fields && n.extra_fields.length) {{
    n.extra_fields.forEach(pair => {{
      html += `<div class="field">${{esc(pair[0])}}: ${{esc(pair[1])}}</div>`;
    }});
  }}
  if (neighborIds.length) {{
    html += `<div class="field" style="margin-top:8px;color:#aaa;font-size:11px">Neighbors (${{neighborIds.length}})</div><div id="neighbors-list">${{neighborItems}}</div>`;
  }}
  document.getElementById('info-content').innerHTML = html;
}}

function focusNode(nodeId) {{
  network.focus(nodeId, {{ scale: 1.4, animation: true }});
  network.selectNodes([nodeId]);
  showInfo(nodeId);
}}

// Neighbor links use a data attribute + one delegated listener rather than an
// inline onclick. A node id/label sourced from a document or a scraped URL
// (graphify add) can contain a double-quote; dropping the stringified id
// unescaped into a quoted onclick both broke every link and allowed a hostile
// source to inject an event handler into the local report (stored XSS, #1838).
// esc() on data-nid keeps the value inside the attribute; the listener reads it
// back verbatim. Bound to document so it survives the innerHTML rebuild that
// recreates #neighbors-list on each showInfo().
document.addEventListener('click', e => {{
  const el = e.target.closest('.neighbor-link');
  if (el && el.dataset.nid !== undefined) focusNode(el.dataset.nid);
}});

// Track hovered node — hover detection is more reliable than click params
let hoveredNodeId = null;
network.on('hoverNode', params => {{
  hoveredNodeId = params.node;
  container.style.cursor = 'pointer';
}});
network.on('blurNode', () => {{
  hoveredNodeId = null;
  container.style.cursor = 'default';
}});
container.addEventListener('click', () => {{
  if (hoveredNodeId !== null) {{
    showInfo(hoveredNodeId);
    network.selectNodes([hoveredNodeId]);
  }}
}});
network.on('click', params => {{
  if (params.nodes.length > 0) {{
    showInfo(params.nodes[0]);
  }} else if (hoveredNodeId === null) {{
    document.getElementById('info-content').innerHTML = '<span class="empty">Click a node to inspect it</span>';
  }}
}});

const searchInput = document.getElementById('search');
const searchResults = document.getElementById('search-results');
searchInput.addEventListener('input', () => {{
  const q = searchInput.value.toLowerCase().trim();
  searchResults.innerHTML = '';
  if (!q) {{ searchResults.style.display = 'none'; return; }}
  const matches = RAW_NODES.filter(n => n.label.toLowerCase().includes(q)).slice(0, 20);
  if (!matches.length) {{ searchResults.style.display = 'none'; return; }}
  searchResults.style.display = 'block';
  matches.forEach(n => {{
    const el = document.createElement('div');
    el.className = 'search-item';
    el.textContent = n.label;
    el.style.borderLeft = `3px solid ${{n.color.background}}`;
    el.style.paddingLeft = '8px';
    el.onclick = () => {{
      network.focus(n.id, {{ scale: 1.5, animation: true }});
      network.selectNodes([n.id]);
      showInfo(n.id);
      searchResults.style.display = 'none';
      searchInput.value = '';
    }};
    searchResults.appendChild(el);
  }});
}});
document.addEventListener('click', e => {{
  if (!searchResults.contains(e.target) && e.target !== searchInput)
    searchResults.style.display = 'none';
}});

const hiddenCommunities = new Set();

const selectAllCb = document.getElementById('select-all-cb');

function updateSelectAllState() {{
  const total = LEGEND.length;
  const hidden = hiddenCommunities.size;
  selectAllCb.checked = hidden === 0;
  selectAllCb.indeterminate = hidden > 0 && hidden < total;
}}

function toggleAllCommunities(hide) {{
  document.querySelectorAll('.legend-item').forEach(item => {{
    hide ? item.classList.add('dimmed') : item.classList.remove('dimmed');
  }});
  document.querySelectorAll('.legend-cb').forEach(cb => {{
    cb.checked = !hide;
  }});
  LEGEND.forEach(c => {{
    if (hide) hiddenCommunities.add(c.cid); else hiddenCommunities.delete(c.cid);
  }});
  const updates = RAW_NODES.map(n => ({{ id: n.id, hidden: hide }}));
  nodesDS.update(updates);
  updateSelectAllState();
}}

const legendEl = document.getElementById('legend');
LEGEND.forEach(c => {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.className = 'legend-cb';
  cb.checked = true;
  cb.addEventListener('change', (e) => {{
    e.stopPropagation();
    if (cb.checked) {{
      hiddenCommunities.delete(c.cid);
      item.classList.remove('dimmed');
    }} else {{
      hiddenCommunities.add(c.cid);
      item.classList.add('dimmed');
    }}
    const updates = RAW_NODES
      .filter(n => n.community === c.cid)
      .map(n => ({{ id: n.id, hidden: !cb.checked }}));
    nodesDS.update(updates);
    updateSelectAllState();
  }});
  item.innerHTML = `<div class="legend-dot" style="background:${{c.color}}"></div>
    <span class="legend-label">${{c.label}}</span>
    <span class="legend-count">${{c.count}}</span>`;
  item.prepend(cb);
  item.onclick = (e) => {{
    if (e.target === cb) return;
    cb.checked = !cb.checked;
    cb.dispatchEvent(new Event('change'));
  }};
  legendEl.appendChild(item);
}});
</script>"""


def _html_document_title(output_path: str) -> str:
    """Return a portable label for the graph.html <title>.

    Tracked artifacts must not embed the generator host absolute path
    (regression of #433; reported again as #2598 on Windows). Keep from the
    configured output-dir bare name (``graphify-out`` / ``GRAPHIFY_OUT``
    basename) onward — portable in every case; otherwise fall back to a
    cwd-relative label, and finally the filename only.
    """
    from graphify.paths import GRAPHIFY_OUT_NAME

    raw = str(output_path).replace("\\", "/")
    # Drop Windows drive prefix so Path parts are comparable on any OS.
    if len(raw) >= 3 and raw[1] == ":" and raw[0].isalpha() and raw[2] == "/":
        raw = raw[2:]  # "/Users/..." style after drive strip
    p = Path(raw)

    parts = list(Path(raw).parts)
    # Path("C:/Users/..") on POSIX may keep "C:" as first part — strip it.
    if parts and len(parts[0]) == 2 and parts[0][1] == ":" and parts[0][0].isalpha():
        parts = parts[1:]
    # Prefer keeping from the output-dir marker onward: portable in every
    # case, whereas a cwd-relative path still leaks host/user segments when
    # the graph is built from a directory ABOVE the project (#2598 follow-up).
    marker = GRAPHIFY_OUT_NAME
    for i, part in enumerate(parts):
        if part == marker or part.startswith("graphify-out"):
            return "/".join(parts[i:])

    # No standard out-dir marker (fully custom output path): fall back to a
    # cwd-relative label when the target is under cwd, else the bare filename.
    try:
        resolved = p if p.is_absolute() else (Path.cwd() / p)
        rel = resolved.resolve().relative_to(Path.cwd().resolve())
        label = rel.as_posix()
        if label and label != ".":
            return label
    except (ValueError, OSError, RuntimeError):
        pass

    name = p.name
    return name if name else "graph.html"

def to_html(
    G: nx.Graph,
    communities: dict[int, list[str]],
    output_path: str,
    community_labels: dict[int, str] | None = None,
    member_counts: dict[int, int] | None = None,
    node_limit: int | None = None,
    learning_overlay: dict | None = None,
    retrieval: dict | None = None,
    legend_extra: list[dict] | None = None,
) -> bool:
    """Generate an interactive vis.js HTML visualization of the graph.

    Features: node size by degree, click-to-inspect panel, search box,
    community filter, physics clustering by community, confidence-styled edges.
    Raises ValueError if graph exceeds MAX_NODES_FOR_VIZ.

    If member_counts is provided (aggregated community view), node sizes are
    based on community member counts rather than graph degree.

    If node_limit is set and the graph exceeds it, automatically builds an
    aggregated community-level meta-graph instead of raising ValueError.

    ``retrieval`` (optional, additive — the IntentGraph/Phase-4 convention):
    a dict describing a REAL retrieval result to highlight on the graph,
    ``{"query": str, "results": [{"id", "tier"}...], "note": str?}`` where
    tier is "direct" or "expanded". When None (the default, and every
    existing caller), the emitted HTML is byte-identical to the pre-feature
    render: the highlight script, HUD, and reset control are only emitted
    when a retrieval overlay is supplied. The renderer never fabricates
    results — it displays exactly what it is given, including the empty
    case ("no evidence returned — nothing highlighted").

    ``legend_extra`` (optional): extra legend entries for a second legend
    block (e.g. memory states), ``[{"color", "label", "dashes"?}]``. Only
    rendered when non-empty, after the communities legend.

    Returns True when the output was written. Returns False when an aggregated
    view would contain fewer than two communities and is intentionally skipped.
    """
    limit = node_limit if node_limit is not None else _viz_node_limit()
    if G.number_of_nodes() > limit:
        if node_limit is not None:
            # Build aggregated community meta-graph
            from collections import Counter as _Counter
            import networkx as _nx
            print(f"Graph has {G.number_of_nodes()} nodes (above {limit} limit). Building aggregated community view...")
            node_to_community = {nid: cid for cid, members in communities.items() for nid in members}
            meta = _nx.Graph()
            for cid, members in communities.items():
                meta.add_node(str(cid), label=(community_labels or {}).get(cid, f"Community {cid}"))
            edge_counts = _Counter()
            for u, v in G.edges():
                cu, cv = node_to_community.get(u), node_to_community.get(v)
                if cu is not None and cv is not None and cu != cv:
                    edge_counts[(min(cu, cv), max(cu, cv))] += 1
            for (cu, cv), w in edge_counts.items():
                meta.add_edge(str(cu), str(cv), weight=w,
                              relation=f"{w} cross-community edges", confidence="AGGREGATED")
            if meta.number_of_nodes() <= 1:
                print("Single community - aggregated view not useful. Skipping graph.html.")
                return False
            meta_communities = {cid: [str(cid)] for cid in communities}
            mc = {cid: len(members) for cid, members in communities.items()}
            # Remap hyperedges from semantic node IDs to community IDs
            raw_hyperedges = G.graph.get("hyperedges", [])
            if raw_hyperedges:
                remapped = []
                for he in raw_hyperedges:
                    he_members = he.get("nodes", [])
                    comm_ids, seen = [], set()
                    for nid in he_members:
                        c = node_to_community.get(nid)
                        if c is None:
                            continue
                        s = str(c)
                        if s in seen:
                            continue
                        seen.add(s)
                        comm_ids.append(s)
                    if len(comm_ids) < 2:
                        continue
                    remapped.append({
                        "id": he.get("id", ""),
                        "label": he.get("label") or he.get("relation", "").replace("_", " "),
                        "nodes": comm_ids,
                    })
                meta.graph["hyperedges"] = remapped
            written = to_html(meta, meta_communities, output_path,
                              community_labels=community_labels, member_counts=mc)
            if not written:
                return False
            print(f"graph.html written (aggregated: {meta.number_of_nodes()} community nodes, {meta.number_of_edges()} cross-community edges)")
            print("Tip: run with --obsidian for full node-level detail.")
            return True
        raise ValueError(
            f"Graph has {G.number_of_nodes()} nodes - too large for HTML viz "
            f"(limit: {limit}). Use --no-viz, raise GRAPHIFY_VIZ_NODE_LIMIT, "
            f"or reduce input size."
        )

    node_community = _node_community_map(communities)
    degree = dict(G.degree())
    max_deg = max(degree.values(), default=1) or 1
    max_mc = (max(member_counts.values(), default=1) or 1) if member_counts else 1

    # Work-memory overlay (derived sidecar). When not passed explicitly, load it
    # best-effort from the sibling .graphify_learning.json next to the output
    # graph.html (which lives beside graph.json). Empty/missing => no learning
    # fields, so the un-annotated render is byte-identical to pre-feature.
    if learning_overlay is None:
        learning_overlay = {}
        try:
            from graphify.reflect import load_learning_overlay as _llo
            learning_overlay = _llo(Path(output_path))
        except Exception:
            learning_overlay = {}
    # Status -> ring color. preferred=green, contested=amber. Tentative gets no
    # ring (it's not yet trustworthy enough to highlight in the map).
    _RING = {"preferred": "#22c55e", "contested": "#f59e0b"}

    # Build nodes list for vis.js
    vis_nodes = []
    for node_id, data in G.nodes(data=True):
        cid = node_community.get(node_id, 0)
        color = COMMUNITY_COLORS[cid % len(COMMUNITY_COLORS)]
        label = sanitize_label(data.get("label", node_id))
        deg = degree.get(node_id, 1)
        if member_counts:
            mc = member_counts.get(cid, 1)
            size = 10 + 30 * (mc / max_mc)
            font_size = 12
        else:
            size = 10 + 30 * (deg / max_deg)
            # Only show label for high-degree nodes by default; others show on hover
            font_size = 12 if deg >= max_deg * 0.15 else 0
        node = {
            "id": node_id,
            "label": label,
            "color": {"background": color, "border": color, "highlight": {"background": "#ffffff", "border": color}},
            "size": round(size, 1),
            "font": {"size": font_size, "color": "#ffffff"},
            # Tooltip `title` is a STRING, which vis-network renders via
            # Popup.setText -> `frame.innerText = t` (verified in the pinned
            # 9.1.6 bundle: the only non-Element branch is innerText, and the
            # bundle has zero `innerHTML = <var>` sinks). So raw special chars
            # are shown literally and must NOT be html-escaped here or the user
            # sees `&amp;`/`&lt;` in the tooltip (#3664/#3686). This is the
            # #1838 stored-XSS boundary: never pass an HTMLElement as `title`,
            # and do not change the vis-network pin below without re-checking
            # Popup rendering (the test_export version-pin guard enforces this).
            "title": label,
            "community": cid,
            "community_name": sanitize_label((community_labels or {}).get(cid, f"Community {cid}")),
            "source_file": sanitize_label(str(data.get("source_file") or "")),
            "file_type": data.get("file_type", ""),
            "degree": deg,
        }
        # --- generic visualization-overlay attributes (additive) --------------
        # An overlay (e.g. IntentGraph memory viz) may set viz_* attributes on
        # nx node data; they are copied conditionally so nodes without them
        # keep the exact pre-feature dict shape. The renderer stays generic:
        # it knows the viz_* convention, not any particular overlay's semantics.
        #   viz_shape       -> vis node shape (e.g. 'diamond' for conversation
        #                      events vs the default 'dot')
        #   viz_ring        -> status ring color (border); does NOT touch the
        #                      community background color, which keeps its
        #                      meaning (the status must not overload it)
        #   viz_dashes      -> dashed status ring
        #   viz_fields      -> [(label, value), ...] panel rows, sanitized here
        #                      so no overlay value can inject markup
        viz_ring = data.get("viz_ring")
        if isinstance(viz_ring, str) and viz_ring:
            node["color"] = {"background": color, "border": viz_ring,
                             "highlight": {"background": "#ffffff", "border": viz_ring}}
            node["borderWidth"] = 3
            node["viz_ring"] = sanitize_label(viz_ring)
        if data.get("viz_dashes"):
            node["shapeProperties"] = {"borderDashes": [4, 4]}
        viz_shape = data.get("viz_shape")
        if isinstance(viz_shape, str) and viz_shape:
            node["shape"] = viz_shape
        viz_fields = data.get("viz_fields")
        if isinstance(viz_fields, list) and viz_fields:
            node["extra_fields"] = [
                [sanitize_label(str(lbl)), sanitize_label(str(val))]
                for lbl, val in viz_fields[:16]
            ]
        if member_counts:
            node["member_count"] = member_counts.get(cid, len(communities.get(cid, [])))
        # Conditional learning fields — only present for annotated nodes, so
        # un-annotated output keeps the exact pre-feature node dict shape.
        entry = learning_overlay.get(str(node_id)) if learning_overlay else None
        if entry:
            status = sanitize_label(str(entry.get("status", "")))
            stale = bool(entry.get("stale"))
            node["learning_status"] = status
            node["learning_stale"] = stale
            ring = _RING.get(status)
            if ring:
                # Status-colored ring via the border; stale => desaturated +
                # dashed (vis.js supports per-node `shapeProperties.borderDashes`).
                if stale:
                    ring = "#9ca3af"
                    node["shapeProperties"] = {"borderDashes": [4, 4]}
                node["borderWidth"] = 3
                node["color"] = {
                    "background": color, "border": ring,
                    "highlight": {"background": "#ffffff", "border": ring},
                }
            # Lesson line appended to the hover title.
            if status == "contested":
                lesson = f"Lesson: contested (useful {entry.get('uses', 0)} / dead-end {entry.get('neg', 0)})"
            elif status == "preferred":
                lesson = f"Lesson: preferred source ({entry.get('uses', 0)} useful, score={entry.get('score', 0)})"
            else:
                lesson = f"Lesson: {status} ({entry.get('uses', 0)} useful)"
            if stale:
                lesson += " [code changed — re-verify]"
            node["title"] = f"{label}\n{sanitize_label(lesson)}"
        vis_nodes.append(node)

    # Build edges list. Restore original edge direction from _src/_tgt
    # (stashed by build.py for exactly this reason): undirected NetworkX
    # canonicalizes endpoint order, which would otherwise flip the arrow
    # for `calls` and `rationale_for` in the rendered graph (#563).
    vis_edges = []
    for u, v, data in G.edges(data=True):
        confidence = data.get("confidence", "EXTRACTED")
        relation = data.get("relation", "")
        true_src = data.get("_src", u)
        true_tgt = data.get("_tgt", v)
        edge = {
            "from": true_src,
            "to": true_tgt,
            "label": relation,
            "title": sanitize_label(f"{relation} [{confidence}]"),
            "dashes": confidence != "EXTRACTED",
            "width": 2 if confidence == "EXTRACTED" else 1,
            "color": {"opacity": 0.7 if confidence == "EXTRACTED" else 0.35},
            "confidence": confidence,
        }
        # generic overlay edge accent (additive): an overlay may mark an edge
        # with viz_color (e.g. amber for a declared contradiction) — applied
        # as the edge's border color at existing opacity, never replacing the
        # dash/width confidence semantics
        viz_color = data.get("viz_color")
        if isinstance(viz_color, str) and viz_color:
            edge["color"] = {"color": sanitize_label(viz_color),
                             "opacity": 0.9, "highlight": sanitize_label(viz_color)}
            edge["viz_color"] = sanitize_label(viz_color)
        vis_edges.append(edge)

    # Build community legend data
    legend_data = []
    for cid in sorted((community_labels or {}).keys()):
        color = COMMUNITY_COLORS[cid % len(COMMUNITY_COLORS)]
        lbl = _html.escape(sanitize_label((community_labels or {}).get(cid, f"Community {cid}")))
        n = member_counts.get(cid, len(communities.get(cid, []))) if member_counts else len(communities.get(cid, []))
        legend_data.append({"cid": cid, "color": color, "label": lbl, "count": n})

    # Escape </script> sequences so embedded JSON cannot break out of the script tag
    def _js_safe(obj) -> str:
        return json.dumps(obj).replace("</", "<\\/")

    nodes_json = _js_safe(vis_nodes)
    edges_json = _js_safe(vis_edges)
    legend_json = _js_safe(legend_data)
    hyperedges_json = _js_safe(getattr(G, "graph", {}).get("hyperedges", []))
    title = _html.escape(sanitize_label(_html_document_title(output_path)))
    stats = f"{G.number_of_nodes()} nodes &middot; {G.number_of_edges()} edges &middot; {len(communities)} communities"

    # --- additive retrieval HUD / extra legend (only when an overlay exists) --
    # retrieval and legend_extra default to None/[] at every existing caller,
    # so these blocks are absent and the emitted HTML is byte-identical to
    # the pre-feature render unless an overlay explicitly opts in.
    retrieval_json = _js_safe(retrieval or {"query": "", "results": []})
    legend_extra_json = _js_safe(legend_extra or [])
    retrieval_block = ""
    if retrieval is not None:
        q = _html.escape(sanitize_label(str(retrieval.get("query", ""))))
        note = _html.escape(sanitize_label(str(retrieval.get("note", ""))))
        retrieval_block = f"""<script>
const RETRIEVAL_QUERY = "{q}";
const RETRIEVAL_NOTE = "{note}";
</script>""" + _retrieval_script(retrieval_json)
    legend_extra_block = _legend_extra_script(legend_extra_json) if legend_extra else ""
    # overlay styles ship only with an overlay — default render unchanged
    overlay_styles = _html_overlay_styles() if (retrieval is not None or legend_extra) else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>graphify - {title}</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"
        integrity="sha384-Ux6phic9PEHJ38YtrijhkzyJ8yQlH8i/+buBR8s3mAZOJrP1gwyvAcIYl3GWtpX1"
        crossorigin="anonymous"></script>
{_html_styles()}
{overlay_styles}
</head>
<body>
<div id="graph"></div>
<div id="sidebar">
  <div id="search-wrap">
    <input id="search" type="text" placeholder="Search nodes..." autocomplete="off">
    <div id="search-results"></div>
  </div>
  <div id="info-panel">
    <h3>Node Info</h3>
    <div id="info-content"><span class="empty">Click a node to inspect it</span></div>
  </div>
  {f'''<div id="retrieval-hud">
    <h3>Retrieved Evidence</h3>
    <div class="field" style="font-size:12px;color:#aaa">{f"Query: {q}" if q else ""}</div>
    <div id="retrieval-count" class="field" style="font-size:11px;color:#888"></div>
    <div id="retrieval-results"></div>
    <button id="retrieval-reset" type="button">Reset highlighting</button>
  </div>''' if retrieval is not None else ""}
  <div id="legend-wrap">
    <h3>Communities</h3>
    <div id="legend-controls">
      <label><input type="checkbox" id="select-all-cb" checked onchange="toggleAllCommunities(!this.checked)">Select All</label>
    </div>
    <div id="legend"></div>
    {f'<h3 style="margin-top:14px">Evidence &amp; Memory States</h3><div id="legend-extra"></div>' if legend_extra else ""}
  </div>
  <div id="stats">{stats}</div>
</div>
{_html_script(nodes_json, edges_json, legend_json)}
{_hyperedge_script(hyperedges_json)}
{retrieval_block}
{legend_extra_block}
</body>
</html>"""

    write_text_atomic(output_path, html)
    return True
