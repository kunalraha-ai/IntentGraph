/* Evidence Replay — tight, focused, with Gnani voice + language selection.
   Uses the Gnani Timbre TTS API when the server has GNANI_API_KEY; falls
   back to the browser's built-in speechSynthesis when not configured. */

(function() {
  "use strict";

  var replay = null;
  var stepIdx = -1;
  var playing = false;
  var speed = 1.0;
  var activeNode = null;
  var speech = window.speechSynthesis || null;
  var timer = null;
  var gnaniAvailable = false;
  var currentAudio = null;

  var FOCAL = { border: "#827AFF", bg: "#2d2455" };
  var GLOW = { border: "#59E8C7", bg: "#1a3a35" };
  var TRAIL = { border: "#2a4a44", bg: "#152522" };
  var DIM = { bg: "#131829", border: "#1e2438", font: "#2a3048" };
  var EDGE_LIT = "#59E8C7";
  var EDGE_DIM = "#1a1e30";

  var LANGUAGES = [
    ["auto", "Auto-detect"], ["en-IN", "English"], ["hi-IN", "Hindi"],
    ["hi-en", "Hinglish"], ["ta-IN", "Tamil"], ["te-IN", "Telugu"],
    ["kn-IN", "Kannada"], ["ml-IN", "Malayalam"], ["mr-IN", "Marathi"],
    ["pa-IN", "Punjabi"], ["bn-IN", "Bengali"], ["gu-IN", "Gujarati"],
  ];
  var VOICES = [
    ["Shivani", "Shivani (warm)"], ["Avinash", "Avinash (firm)"],
    ["Lakshmi", "Lakshmi (expressive)"], ["Kalpana", "Kalpana (calm)"],
    ["Shubham", "Shubham (energetic)"], ["Tarika", "Tarika (reassuring)"],
    ["Devansh", "Devansh (young)"], ["Prakash", "Prakash (steady)"],
  ];

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
  z-index: 2000; display: flex; gap: 6px; align-items: center;
  background: rgba(14,18,31,0.94); border: 1px solid #303B59; border-radius: 12px;
  padding: 7px 10px; backdrop-filter: blur(8px); max-width: 75vw; flex-wrap: wrap; }
#replay-bar input { background: #0f0f1a; border: 1px solid #303B59; color: #e0e0e0;
  border-radius: 8px; padding: 6px 10px; font-size: 13px; outline: none; width: 280px; }
#replay-bar input:focus { border-color: #827AFF; }
#replay-bar select { background: #0f0f1a; border: 1px solid #303B59; color: #e0e0e0;
  border-radius: 8px; padding: 6px 8px; font-size: 12px; outline: none; cursor: pointer; }
#replay-bar button { background: #827AFF; border: none; color: #fff; border-radius: 8px;
  padding: 6px 12px; font-size: 13px; font-weight: 600; cursor: pointer; white-space: nowrap; }
#replay-bar button:hover { background: #948aff; }
#replay-bar button:disabled { background: #303B59; color: #666; cursor: default; }
#replay-ctrl { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
  z-index: 2000; display: none; flex-direction: column; gap: 6px;
  background: rgba(14,18,31,0.94); border: 1px solid #303B59; border-radius: 14px;
  padding: 10px 14px; backdrop-filter: blur(8px); min-width: 420px; max-width: 80vw; }
#replay-ctrl.visible { display: flex; }
#replay-ctrl-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
#replay-ctrl button { background: #171C30; border: 1px solid #303B59; color: #e0e0e0;
  border-radius: 8px; padding: 5px 10px; font-size: 13px; cursor: pointer; }
#replay-ctrl button:hover { border-color: #59E8C7; }
#replay-ctrl button.active { border-color: #59E8C7; color: #59E8C7; }
#replay-ctrl select { background: #0f0f1a; border: 1px solid #303B59; color: #e0e0e0;
  border-radius: 8px; padding: 4px 8px; font-size: 12px; outline: none; cursor: pointer; }
#replay-timeline { flex: 1; height: 5px; -webkit-appearance: none; appearance: none;
  background: #303B59; border-radius: 3px; outline: none; cursor: pointer; min-width: 100px; }
#replay-timeline::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 12px;
  border-radius: 50%; background: #59E8C7; cursor: pointer; }
#replay-narration { font-size: 13.5px; color: #e0e0e0; line-height: 1.4;
  max-height: 52px; overflow-y: auto; font-family: system-ui, sans-serif; }
#replay-narration .kind { font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; color: #59E8C7; margin-right: 6px; }`;
    document.head.appendChild(css);

    var langOpts = LANGUAGES.map(function(l) {
      return '<option value="' + l[0] + '">' + l[1] + "</option>";
    }).join("");
    var voiceOpts = VOICES.map(function(v) {
      return '<option value="' + v[0] + '">' + v[1] + "</option>";
    }).join("");

    var bar = document.createElement("div");
    bar.id = "replay-bar";
    bar.innerHTML = `
      <span style="color:#59E8C7;font-size:13px;font-weight:700">▶</span>
      <input id="replay-query" type="text" placeholder="ask a question…" autocomplete="off">
      <select id="replay-lang" title="narration language">${langOpts}</select>
      <select id="replay-voice" title="voice">${voiceOpts}</select>
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
        <select id="replay-ctrl-lang" title="language during replay">${langOpts}</select>
      </div>
      <div id="replay-narration"></div>`;

    document.body.appendChild(bar);
    document.body.appendChild(ctrl);

    el("replay-query").addEventListener("keydown", function(e) {
      if (e.key === "Enter") Replay.start();
    });
    el("replay-ctrl-lang").addEventListener("change", function() {
      el("replay-lang").value = this.value;
    });
    el("replay-lang").addEventListener("change", function() {
      el("replay-ctrl-lang").value = this.value;
    });

    // check if Gnani voice is available on the server
    checkVoiceProvider();
  }

  function checkVoiceProvider() {
    var key = localStorage.getItem("ig_api_key") ||
              new URLSearchParams(location.search).get("key") || "";
    var h = {};
    if (key) h["X-API-Key"] = key;
    fetch("/v1/status", { headers: h })
      .then(function(r) { return r.json(); })
      .then(function(st) {
        gnaniAvailable = st.voice && st.voice.available;
        if (!gnaniAvailable) {
          el("replay-lang").style.display = "none";
          el("replay-voice").style.display = "none";
        }
      })
      .catch(function() {});
  }

  // ---- fetch the replay script ----
  function fetchReplay(q) {
    var key = localStorage.getItem("ig_api_key") ||
              new URLSearchParams(location.search).get("key") || "";
    var h = { "Content-Type": "application/json" };
    if (key) h["X-API-Key"] = key;
    return fetch("/v1/replay", { method: "POST", headers: h,
               body: JSON.stringify({ query: q }) })
      .then(function(r) { if (!r.ok) throw new Error(r.status); return r.json(); });
  }

  // ---- Gnani voice: text → audio → play ----
  function speakGnani(text, onDone) {
    var key = localStorage.getItem("ig_api_key") ||
              new URLSearchParams(location.search).get("key") || "";
    var lang = el("replay-lang").value || "auto";
    var voice = el("replay-voice").value || "Shivani";
    var h = { "Content-Type": "application/json" };
    if (key) h["X-API-Key"] = key;
    fetch("/v1/voice", { method: "POST", headers: h,
      body: JSON.stringify({ text: text, language: lang, voice: voice, speed: speed })
    }).then(function(r) {
      if (!r.ok) throw new Error("voice " + r.status);
      return r.json();
    }).then(function(body) {
      // hex → bytes → blob URL → Audio element
      var bytes = new Uint8Array(body.audio.match(/../g).map(function(h) {
        return parseInt(h, 16);
      }));
      var blob = new Blob([bytes], { type: "audio/mpeg" });
      var url = URL.createObjectURL(blob);
      currentAudio = new Audio(url);
      currentAudio.onended = function() { URL.revokeObjectURL(url); onDone(); };
      currentAudio.onerror = function() { URL.revokeObjectURL(url); onDone(); };
      currentAudio.play();
    }).catch(function(err) {
      // Gnani failed → fall back to browser TTS for this step
      speakBrowser(text, onDone);
    });
  }

  // ---- browser TTS fallback ----
  function speakBrowser(text, onDone) {
    var fired = false;
    function done() { if (!fired) { fired = true; if (onDone) onDone(); } }
    if (!speech) { setTimeout(done, Math.max(2000, text.length * 60)); return; }
    speech.cancel();
    var u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0 * speed;
    u.onend = done;
    speech.speak(u);
    var words = text.split(/\s+/).length;
    var estimate = (words / 145) * 60 * 1000 / speed;
    setTimeout(done, estimate + 2000);
  }

  // ---- unified speak: Gnani if available, browser otherwise ----
  function speak(text, onDone) {
    if (currentAudio) { currentAudio.pause(); currentAudio = null; }
    if (speech) speech.cancel();
    if (gnaniAvailable) speakGnani(text, onDone);
    else speakBrowser(text, onDone);
  }

  // ---- focus mode ----
  function collectReplayIds() {
    return replay.steps.filter(function(s) { return s.id; })
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

  // ---- step engine ----
  function runStep(idx) {
    if (!replay || idx < 0 || idx >= replay.steps.length) return;
    clearTimeout(timer);
    if (currentAudio) { currentAudio.pause(); currentAudio = null; }
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

    speak(s.narration, function() {
      if (playing && stepIdx < replay.steps.length - 1) runStep(stepIdx + 1);
      else if (playing) stop();
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
    if (playing) { if (stepIdx < 0) runStep(0); }
    else { clearTimeout(timer); if (speech) speech.cancel();
           if (currentAudio) currentAudio.pause(); }
  }

  function next() { clearTimeout(timer); if (speech) speech.cancel();
    if (currentAudio) currentAudio.pause();
    if (replay && stepIdx < replay.steps.length - 1) { playing = true; el("replay-play").textContent = "⏸"; runStep(stepIdx + 1); } }
  function prev() { clearTimeout(timer); if (speech) speech.cancel();
    if (currentAudio) currentAudio.pause();
    if (replay && stepIdx > 0) { playing = true; el("replay-play").textContent = "⏸"; runStep(stepIdx - 1); } }
  function seek(i) { clearTimeout(timer); if (speech) speech.cancel();
    if (currentAudio) currentAudio.pause();
    var v = parseInt(i, 10); if (replay && v >= 0 && v < replay.steps.length) { playing = false; el("replay-play").textContent = "▶"; runStep(v); } }
  function cycleSpeed() { speed = speed === 0.5 ? 1 : speed === 1 ? 2 : 0.5; el("replay-speed").textContent = speed + "×"; }
  function stop() { playing = false; clearTimeout(timer);
    if (speech) speech.cancel();
    if (currentAudio) { currentAudio.pause(); currentAudio = null; }
    el("replay-play").textContent = "▶"; }

  window.Replay = { start: start, toggle: toggle, next: next, prev: prev,
                    seek: seek, cycleSpeed: cycleSpeed, stop: stop };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();
