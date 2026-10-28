/* Evidence Replay — tight, focused, fast.
   Focuses the graph on just the replay nodes by dimming everything else,
   so the trail is immediately visible without camera gymnastics across
   a 1,300-node graph. Voice narration is one short sentence per step. */

(function() {
  "use strict";

  var replay = null;
  var stepIdx = -1;
  var playing = false;
  var speed = 1.0;
  var activeNode = null;
  var speech = window.speechSynthesis || null;
  var timer = null;

  var FOCAL = { border: "#827AFF", bg: "#2d2455" };
  var GLOW = { border: "#59E8C7", bg: "#1a3a35" };
  var TRAIL = { border: "#2a4a44", bg: "#152522" };
  var DIM = { bg: "#131829", border: "#1e2438", font: "#2a3048" };
  var EDGE_LIT = "#59E8C7";
  var EDGE_DIM = "#1a1e30";

  function el(id) { return document.getElementById(id); }
  function ds() { return typeof nodesDS !== "undefined" ? nodesDS : null; }
  function eds() { return typeof edgesDS !== "undefined" ? edgesDS : null; }
  function net() { return typeof network !== "undefined" ? network : null; }

  // ---- build the overlay ----
  function build() {
    if (el("replay-bar")) return;
    var css = document.createElement("style");
    css.textContent = `
#replay-bar { position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
  z-index: 2000; display: flex; gap: 8px; align-items: center;
  background: rgba(14,18,31,0.94); border: 1px solid #303B59; border-radius: 12px;
  padding: 8px 12px; backdrop-filter: blur(8px); max-width: 70vw; }
#replay-bar input { background: #0f0f1a; border: 1px solid #303B59; color: #e0e0e0;
  border-radius: 8px; padding: 7px 12px; font-size: 13px; outline: none; width: 360px; }
#replay-bar input:focus { border-color: #827AFF; }
#replay-bar button { background: #827AFF; border: none; color: #fff; border-radius: 8px;
  padding: 7px 14px; font-size: 13px; font-weight: 600; cursor: pointer; white-space: nowrap; }
#replay-bar button:hover { background: #948aff; }
#replay-bar button:disabled { background: #303B59; color: #666; cursor: default; }
#replay-ctrl { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
  z-index: 2000; display: none; flex-direction: column; gap: 6px;
  background: rgba(14,18,31,0.94); border: 1px solid #303B59; border-radius: 14px;
  padding: 10px 14px; backdrop-filter: blur(8px); min-width: 420px; max-width: 80vw; }
#replay-ctrl.visible { display: flex; }
#replay-ctrl-row { display: flex; gap: 8px; align-items: center; }
#replay-ctrl button { background: #171C30; border: 1px solid #303B59; color: #e0e0e0;
  border-radius: 8px; padding: 5px 10px; font-size: 13px; cursor: pointer; }
#replay-ctrl button:hover { border-color: #59E8C7; }
#replay-ctrl button.active { border-color: #59E8C7; color: #59E8C7; }
#replay-timeline { flex: 1; height: 5px; -webkit-appearance: none; appearance: none;
  background: #303B59; border-radius: 3px; outline: none; cursor: pointer; }
#replay-timeline::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 12px;
  border-radius: 50%; background: #59E8C7; cursor: pointer; }
#replay-narration { font-size: 13.5px; color: #e0e0e0; line-height: 1.4;
  max-height: 48px; overflow-y: auto; font-family: system-ui, sans-serif; }
#replay-narration .kind { font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; color: #59E8C7; margin-right: 6px; }`;
    document.head.appendChild(css);

    var bar = document.createElement("div");
    bar.id = "replay-bar";
    bar.innerHTML = `
      <span style="color:#59E8C7;font-size:13px;font-weight:700;white-space:nowrap">▶</span>
      <input id="replay-query" type="text" placeholder="ask a question…" autocomplete="off">
      <button id="replay-go" onclick="Replay.start()">Replay</button>`;

    var ctrl = document.createElement("div");
    ctrl.id = "replay-ctrl";
    ctrl.innerHTML = `
      <div id="replay-ctrl-row">
        <button id="replay-prev" onclick="Replay.prev()" title="prev">⏮</button>
        <button id="replay-play" onclick="Replay.toggle()" title="play/pause">⏸</button>
        <button id="replay-next" onclick="Replay.next()" title="next">⏭</button>
        <input id="replay-timeline" type="range" min="0" max="0" value="0" step="1"
               oninput="Replay.seek(this.value)">
        <button id="replay-speed" onclick="Replay.cycleSpeed()">1×</button>
        <button id="replay-stop" onclick="Replay.stop()">⏹</button>
      </div>
      <div id="replay-narration"></div>`;

    document.body.appendChild(bar);
    document.body.appendChild(ctrl);
    el("replay-query").addEventListener("keydown", function(e) {
      if (e.key === "Enter") Replay.start();
    });
  }

  // ---- fetch ----
  function fetchReplay(q) {
    var key = localStorage.getItem("ig_api_key") ||
              new URLSearchParams(location.search).get("key") || "";
    var h = { "Content-Type": "application/json" };
    if (key) h["X-API-Key"] = key;
    return fetch("/v1/replay", { method: "POST", headers: h,
               body: JSON.stringify({ query: q }) })
      .then(function(r) { if (!r.ok) throw new Error(r.status); return r.json(); });
  }

  // ---- focus mode: dim everything not in the replay path ----
  function collectReplayIds() {
    return replay.steps
      .filter(function(s) { return s.id; })
      .map(function(s) { return s.id; });
  }

  function enterFocusMode() {
    var n = ds(), e = eds(), net_ = net();
    if (!n) return;
    var ids = new Set(collectReplayIds());
    var updates = [];
    n.get().forEach(function(node) {
      if (!ids.has(node.id)) {
        updates.push({ id: node.id, hidden: false,
          color: { background: DIM.bg, border: DIM.border,
                   highlight: { background: "#ffffff", border: DIM.border } },
          font: { size: 0, color: DIM.font } });
      }
    });
    n.update(updates);
    if (e) {
      var edgeUpdates = [];
      e.get().forEach(function(edge) {
        var lit = ids.has(edge.from) && ids.has(edge.to);
        edgeUpdates.push({ id: edge.id,
          color: lit ? { color: "#3a4a68", opacity: 0.6, highlight: EDGE_LIT }
                     : { color: EDGE_DIM, opacity: 0.08, highlight: EDGE_DIM },
          width: lit ? 2.5 : 0.5 });
      });
      e.update(edgeUpdates);
    }
    if (net_) {
      var nodeList = collectReplayIds();
      if (nodeList.length > 0) {
        net_.fit({ nodes: nodeList, animation: { duration: 800, easingFunction: "easeInOutQuad" } });
      }
    }
  }

  function exitFocusMode() {
    var n = ds(), e = eds();
    if (typeof RAW_NODES === "undefined" || !n) return;
    RAW_NODES.forEach(function(rn) {
      n.update({ id: rn.id,
        borderWidth: rn.borderWidth !== undefined ? rn.borderWidth : 1.5,
        color: rn.color, font: rn.font });
    });
    if (typeof RAW_EDGES === "undefined" || !e) return;
    RAW_EDGES.forEach(function(re, i) {
      var c = re.color && re.color.color ? re.color : { color: "#8f9bb3", opacity: 0.7 };
      e.update({ id: i, color: c, width: re.width || 2 });
    });
  }

  // ---- glow ----
  function glow(id, colors) {
    var n = ds();
    if (!n || !n.get(id)) return false;
    n.update({ id: id, borderWidth: 5,
      color: { background: colors.bg, border: colors.border,
               highlight: { background: "#ffffff", border: colors.border } },
      font: { size: 14, color: "#ffffff" } });
    return true;
  }

  function dimTrail(id) {
    var n = ds();
    if (!n || !n.get(id)) return;
    n.update({ id: id, borderWidth: 3,
      color: { background: TRAIL.bg, border: TRAIL.border } });
  }

  function litEdge(from, to) {
    var e = eds();
    if (!e) return;
    e.get({ filter: function(edge) {
      return (edge.from === from && edge.to === to) ||
             (edge.from === to && edge.to === from);
    }}).forEach(function(edge) {
      e.update({ id: edge.id,
        color: { color: EDGE_LIT, opacity: 0.95, highlight: EDGE_LIT },
        width: 4 });
    });
  }

  // ---- voice: wait for speech to actually finish, never cut off ----
  function speak(text, onDone) {
    var fired = false;
    function done() { if (!fired) { fired = true; if (onDone) onDone(); } }
    if (!speech) { setTimeout(done, Math.max(2000, text.length * 60)); return; }
    speech.cancel();
    var u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0 * speed;
    u.pitch = 1.0;
    u.volume = 0.9;
    u.onend = done;
    speech.speak(u);
    // generous fallback: if onend doesn't fire within a generous window
    // (some browsers have quirks), advance anyway — but only after the
    // speech has had plenty of time to complete
    var words = text.split(/\s+/).length;
    var estimate = (words / 145) * 60 * 1000 / speed;
    setTimeout(done, estimate + 2000);
  }

  // ---- step engine ----
  function runStep(idx) {
    if (!replay || idx < 0 || idx >= replay.steps.length) return;
    clearTimeout(timer);
    var s = replay.steps[idx];
    stepIdx = idx;
    el("replay-timeline").value = idx;
    el("replay-timeline").max = replay.steps.length - 1;

    var kindLabel = s.kind === "focal" ? "●" : s.kind === "code" ? "◈"
                   : s.kind === "verdict" ? "✓" : "○";
    el("replay-narration").innerHTML =
      '<span class="kind">' + kindLabel + '</span>' + s.narration;

    if (s.kind === "verdict") {
      speak(s.narration, function() { if (playing) stop(); });
      return;
    }

    if (activeNode && activeNode !== s.id) dimTrail(activeNode);
    if (s.edge_from) litEdge(s.edge_from, s.id);
    if (glow(s.id, s.kind === "focal" ? FOCAL : GLOW)) activeNode = s.id;

    // speak the full narration, wait for it to actually finish, then advance
    speak(s.narration, function() {
      if (playing && stepIdx < replay.steps.length - 1) {
        runStep(stepIdx + 1);
      } else if (playing) {
        stop();
      }
    });
  }

  // ---- controls ----
  function start() {
    var q = el("replay-query").value.trim();
    if (!q) return;
    el("replay-go").textContent = "…";
    el("replay-go").disabled = true;
    fetchReplay(q).then(function(script) {
      replay = script;
      activeNode = null;
      enterFocusMode();
      el("replay-ctrl").classList.add("visible");
      el("replay-timeline").max = replay.steps.length - 1;
      playing = true;
      el("replay-go").textContent = "Replay";
      el("replay-go").disabled = false;
      setTimeout(function() { runStep(0); }, 900);
    }).catch(function(err) {
      el("replay-go").textContent = "Replay";
      el("replay-go").disabled = false;
      el("replay-narration").textContent = "Error: " + err.message;
      el("replay-ctrl").classList.add("visible");
    });
  }

  function toggle() {
    playing = !playing;
    el("replay-play").textContent = playing ? "⏸" : "▶";
    if (playing) { if (stepIdx < 0) runStep(0); else runStep(stepIdx); }
    else { clearTimeout(timer); if (speech) speech.cancel(); }
  }

  function next() { clearTimeout(timer); if (speech) speech.cancel();
    if (replay && stepIdx < replay.steps.length - 1) { playing = true; el("replay-play").textContent = "⏸"; runStep(stepIdx + 1); } }
  function prev() { clearTimeout(timer); if (speech) speech.cancel();
    if (replay && stepIdx > 0) { playing = true; el("replay-play").textContent = "⏸"; runStep(stepIdx - 1); } }
  function seek(i) { clearTimeout(timer); if (speech) speech.cancel();
    var v = parseInt(i, 10); if (replay && v >= 0 && v < replay.steps.length) { playing = false; el("replay-play").textContent = "▶"; runStep(v); } }
  function cycleSpeed() { speed = speed === 0.5 ? 1 : speed === 1 ? 2 : 0.5; el("replay-speed").textContent = speed + "×"; }
  function stop() { playing = false; clearTimeout(timer); if (speech) speech.cancel(); el("replay-play").textContent = "▶"; }

  window.Replay = { start: start, toggle: toggle, next: next, prev: prev,
                    seek: seek, cycleSpeed: cycleSpeed, stop: stop };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();
