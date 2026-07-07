/* Grievance Model Explainer — the whole BiLSTM+Attention network as one live canvas.
   Every activation comes from a real forward pass of the production model. */
(function () {
  const $ = s => document.querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  const URG_COLORS = { Low: "#2e9e5b", Medium: "#d9a416", High: "#e07b39", Critical: "#d64545" };

  let MODEL = null, TR = null;

  /* ---------------- svg helpers ---------------- */
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(x, y, cls, content, parent, anchor) {
    const t = el("text", { x, y, class: cls, ...(anchor ? { "text-anchor": anchor } : {}) }, parent);
    t.textContent = content;
    return t;
  }
  const rib = (x0, y0, x1, y1) => `M ${x0} ${y0} C ${(x0 + x1) / 2} ${y0}, ${(x0 + x1) / 2} ${y1}, ${x1} ${y1}`;
  function heat(v, scale = 1.6) { // blue = negative, orange = positive (light theme)
    const x = Math.max(-1, Math.min(1, v / scale));
    return x >= 0 ? `rgba(226, 122, 60, ${(0.12 + 0.8 * x).toFixed(3)})`
                  : `rgba(74, 134, 232, ${(0.12 + 0.8 * -x).toFixed(3)})`;
  }

  /* ---------------- geometry ---------------- */
  const X = { tok: 118, emb: 172, lstmF: 352, lstmB: 410, hid: 486, attn: 682, ctx: 828,
              h1: 968, h2: 1042, h3: 1116, probName: 1268, probBar: 1278, probEnd: 1462 };
  const ROW = 42, TOP = 118;

  /* ---------------- API ---------------- */
  async function api(path, body) {
    const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    return res.json();
  }

  /* ---------------- main draw ---------------- */
  function draw() {
    const svg = $("#flow");
    svg.innerHTML = "";
    if (!TR || !MODEL) return;
    const t = TR, cfg = MODEL.config, n = t.tokens.length;
    const yTok = i => TOP + i * ROW + ROW / 2;
    const yMid = TOP + (n * ROW) / 2;
    const H = Math.max(720, TOP + n * ROW + 140, yMid + 320);
    svg.setAttribute("viewBox", `0 0 1560 ${H}`);
    svg.setAttribute("width", "1560");

    // gradients
    const defs = el("defs", {}, svg);
    const g1 = el("linearGradient", { id: "gradAttn", x1: 0, y1: 0, x2: 1, y2: 0 }, defs);
    el("stop", { offset: "0%", "stop-color": "#f4b09e" }, g1);
    el("stop", { offset: "100%", "stop-color": "#e2543a" }, g1);

    const glowLayer = el("g", { id: "glow-layer" }, svg);   // guide highlights (behind)
    const ribbons = el("g", {}, svg);                        // ribbons behind nodes
    const nodes = el("g", {}, svg);

    /* ----- column headers ----- */
    const heads = [
      [X.tok - 20, "Input", `${n} tokens`, "tok"],
      [X.emb + 44, "Embedding", `emb_dim = ${cfg.emb_dim}`, "emb"],
      [(X.lstmF + X.lstmB) / 2, "BiLSTM", `hidden = ${cfg.hidden_dim} × ${cfg.bidirectional ? "2 dirs" : "1 dir"}`, "lstm"],
      [X.hid + 50, "Hidden states", `${t.lstm_dim} / token`, "hid"],
      [X.attn, "Attention", `attn_dim = ${cfg.attn_dim} → α`, "attn"],
      [X.ctx, "Context", `${t.context_dim} dims`, "ctx"],
      [X.h2, "Two MLP heads", `${cfg.head_hidden}·${cfg.head_hidden} + BatchNorm`, "heads"],
      [(X.probName + X.probEnd) / 2, "Probabilities", "softmax", "out"],
    ];
    heads.forEach(([x, title, sub, id]) => {
      txt(x, 44, "col-title", title, nodes, "middle").dataset.col = id;
      txt(x, 62, "col-sub", sub, nodes, "middle");
    });

    /* ----- tokens ----- */
    t.tokens.forEach((tok, i) => {
      const g = el("g", { "data-i": i }, nodes);
      txt(X.tok, yTok(i) + 4, "tok-label", tok, g, "end");
      txt(X.tok, yTok(i) + 16, "tok-id", `id ${t.token_ids[i]}${t.in_vocab[i] ? "" : " · UNK"}`, g, "end");
      // token -> embedding ribbon
      el("path", { d: rib(X.tok + 8, yTok(i), X.emb - 6, yTok(i)), class: "ribbon rib-emb draw",
                   "stroke-width": 2, "data-i": i, style: `animation-delay:${i * 40}ms` }, ribbons);
    });

    /* ----- embedding heat strips ----- */
    t.embedding_preview.forEach((vec, i) => {
      const g = el("g", { "data-i": i, class: "pop", style: `animation-delay:${i * 40}ms` }, nodes);
      vec.forEach((v, j) => {
        el("rect", { x: X.emb + j * 11, y: yTok(i) - 8, width: 9, height: 16, rx: 2.5, fill: heat(v),
                     "data-tip": `${t.tokens[i]} · dim ${j}: ${v}` }, g);
      });
      txt(X.emb + 92, yTok(i) + 4, "small-note", `‖${t.embedding_norms[i]}‖`, g);
      // embedding -> fwd & bwd lstm
      el("path", { d: rib(X.emb + 90, yTok(i), X.lstmF - 11, yTok(i)), class: "ribbon rib-fwd draw",
                   "stroke-width": 2, "data-i": i, style: `animation-delay:${120 + i * 40}ms` }, ribbons);
      el("path", { d: rib(X.emb + 90, yTok(i), X.lstmB - 11, yTok(i)), class: "ribbon rib-bwd draw",
                   "stroke-width": 1.6, "data-i": i, style: `animation-delay:${160 + i * 40}ms` }, ribbons);
    });

    /* ----- BiLSTM columns: recurrent chains ----- */
    const maxN = Math.max(...t.lstm_fwd_norms, ...t.lstm_bwd_norms, 1e-9);
    // recurrence arrows (forward chain goes down, backward chain up)
    for (let i = 0; i < n - 1; i++) {
      el("line", { x1: X.lstmF, y1: yTok(i) + 11, x2: X.lstmF, y2: yTok(i + 1) - 11,
                   stroke: "#b7cdf6", "stroke-width": 2, "marker-end": "url(#arrF)" }, ribbons);
      el("line", { x1: X.lstmB, y1: yTok(i + 1) - 11, x2: X.lstmB, y2: yTok(i) + 11,
                   stroke: "#d3c2f7", "stroke-width": 2, "marker-end": "url(#arrB)" }, ribbons);
    }
    ["arrF#4a86e8", "arrB#9b6df2"].forEach(s => {
      const [id, color] = s.split("#");
      const m = el("marker", { id, viewBox: "0 0 8 8", refX: 6, refY: 4, markerWidth: 5, markerHeight: 5, orient: "auto" }, defs);
      el("path", { d: "M0,0 L8,4 L0,8 z", fill: "#" + color }, m);
    });
    t.tokens.forEach((tok, i) => {
      const g = el("g", { "data-i": i }, nodes);
      const rF = 5 + (t.lstm_fwd_norms[i] / maxN) * 6, rB = 5 + (t.lstm_bwd_norms[i] / maxN) * 6;
      el("circle", { cx: X.lstmF, cy: yTok(i), r: rF.toFixed(1), fill: "#4a86e8", opacity: .85, class: "pop",
                     style: `animation-delay:${200 + i * 40}ms`, "data-tip": `forward LSTM ‖h⃗‖ = ${t.lstm_fwd_norms[i]}` }, g);
      el("circle", { cx: X.lstmB, cy: yTok(i), r: rB.toFixed(1), fill: "#9b6df2", opacity: .85, class: "pop",
                     style: `animation-delay:${240 + i * 40}ms`, "data-tip": `backward LSTM ‖h⃖‖ = ${t.lstm_bwd_norms[i]}` }, g);
      // to concat bar
      el("path", { d: rib(X.lstmB + 10, yTok(i), X.hid - 4, yTok(i)), class: "ribbon rib-hid draw",
                   "stroke-width": 3, "data-i": i, style: `animation-delay:${260 + i * 40}ms` }, ribbons);
    });
    txt(X.lstmF, TOP - 26, "dim-tag", "forward →", nodes, "middle");
    txt(X.lstmB, TOP - 14, "dim-tag", "← backward", nodes, "middle");

    /* ----- concatenated hidden states (stacked two-color bars) ----- */
    t.tokens.forEach((tok, i) => {
      const g = el("g", { "data-i": i, class: "fade", style: `animation-delay:${300 + i * 30}ms` }, nodes);
      const total = t.lstm_fwd_norms[i] + t.lstm_bwd_norms[i];
      const wF = total ? (t.lstm_fwd_norms[i] / total) * 96 : 48;
      el("rect", { x: X.hid, y: yTok(i) - 6, width: wF.toFixed(1), height: 12, rx: 3, fill: "#7fa8ec",
                   "data-tip": `forward part ${t.lstm_fwd_norms[i]}` }, g);
      el("rect", { x: X.hid + wF, y: yTok(i) - 6, width: (96 - wF).toFixed(1), height: 12, rx: 3, fill: "#b79bf0",
                   "data-tip": `backward part ${t.lstm_bwd_norms[i]}` }, g);
      // hidden -> attention scorer
      el("path", { d: rib(X.hid + 100, yTok(i), X.attn - 12, yTok(i)), class: "ribbon rib-hid draw",
                   "stroke-width": 2, "data-i": i, style: `animation-delay:${340 + i * 30}ms` }, ribbons);
    });

    /* ----- attention: the Sankey moment ----- */
    const maxA = Math.max(...t.attention, 1e-9);
    t.tokens.forEach((tok, i) => {
      const a = t.attention[i];
      const g = el("g", { "data-i": i }, nodes);
      const r = 4 + (a / maxA) * 9;
      el("circle", { cx: X.attn, cy: yTok(i), r: r.toFixed(1), fill: "#e2543a",
                     opacity: (0.35 + 0.65 * a / maxA).toFixed(2), class: "pop",
                     style: `animation-delay:${420 + i * 30}ms`,
                     "data-tip": `α(${tok}) = ${(a * 100).toFixed(2)}% of the model's focus` }, g);
      txt(X.attn + 16, yTok(i) + 3.5, "alpha-label", `${(a * 100).toFixed(1)}%`, g);
      // ribbon to context — WIDTH ∝ attention weight (the weighted sum, visible)
      el("path", { d: rib(X.attn + 12, yTok(i), X.ctx - 20, yMid), class: "ribbon rib-attn draw",
                   "stroke-width": Math.max(1.2, a * 34).toFixed(1), "data-i": i,
                   style: `animation-delay:${480 + i * 30}ms`,
                   "data-tip": `${tok} contributes ${(a * 100).toFixed(1)}% of the context vector` }, ribbons);
    });
    txt(X.attn, TOP + n * ROW + 26, "small-note", "ribbon width = attention weight α (sums to 100%)", nodes, "middle");

    /* ----- context vector ----- */
    const ctxH = 12 * 17;
    const gCtx = el("g", { class: "fade", style: "animation-delay:700ms" }, nodes);
    t.context_preview.forEach((v, j) => {
      el("rect", { x: X.ctx - 14, y: yMid - ctxH / 2 + j * 17, width: 28, height: 15, rx: 3, fill: heat(v, 2.5),
                   "data-tip": `context dim ${j}: ${v}` }, gCtx);
    });
    txt(X.ctx, yMid + ctxH / 2 + 18, "small-note", `… ${t.context_dim} dims · ‖c‖ = ${t.context_norm}`, gCtx, "middle");

    /* ----- two heads ----- */
    const yA = Math.max(yMid - 170, 150), yB = Math.min(yMid + 170, H - 150);
    const catsSorted = Object.entries(t.category_probs).sort((a, b) => b[1] - a[1]);
    const urgEntries = MODEL.urgency_labels.map(u => [u, t.urgency_probs[u] ?? 0]);

    function drawHead(yC, label, color, ribCls, delay) {
      const g = el("g", { class: "fade", style: `animation-delay:${delay}ms` }, nodes);
      el("path", { d: rib(X.ctx + 16, yMid, X.h1 - 26, yC), class: `ribbon ${ribCls} draw`, "stroke-width": 7,
                   style: `animation-delay:${delay}ms` }, ribbons);
      [X.h1, X.h2, X.h3].forEach((x, k) => {
        el("rect", { x: x - 22, y: yC - 30, width: 44, height: 60, rx: 9, fill: "#fff", stroke: color,
                     "stroke-width": 1.6, "data-tip": k < 2 ? `Linear ${k ? MODEL.config.head_hidden : t.context_dim}→${MODEL.config.head_hidden} + BatchNorm + ReLU + dropout` : "Output linear layer → logits" }, g);
        // little neuron dots
        for (let d = 0; d < 4; d++) el("circle", { cx: x, cy: yC - 18 + d * 12, r: 2.6, fill: color, opacity: .55 }, g);
        if (k < 2) for (let r2 = 0; r2 < 3; r2++)
          el("line", { x1: x + 22, y1: yC - 14 + r2 * 14, x2: [X.h1, X.h2, X.h3][k + 1] - 22, y2: yC - 14 + r2 * 14,
                       stroke: color, "stroke-width": 1, opacity: .35 }, g);
      });
      txt(X.h2, yC - 44, "head-label", label, g, "middle");
      return g;
    }
    drawHead(yA, "HEAD A — CATEGORY", "#199f86", "rib-headB", 760);
    drawHead(yB, "HEAD B — URGENCY", "#7a5af8", "rib-head", 820);

    /* ----- probability lists ----- */
    function probList(entries, yStart, gap, colorFn, winIdx, fromY, delay, tipFn) {
      entries.forEach(([name, p], k) => {
        const y = yStart + k * gap;
        const g = el("g", { class: "fade", style: `animation-delay:${delay + k * 45}ms` }, nodes);
        txt(X.probName, y + 4, "prob-name" + (k === winIdx ? " win" : ""), name, g, "end");
        const w = Math.max(2, p * (X.probEnd - X.probBar));
        el("rect", { x: X.probBar, y: y - 4, width: (X.probEnd - X.probBar), height: 8, rx: 4, fill: "#f0f3f8" }, g);
        el("rect", { x: X.probBar, y: y - 4, width: w.toFixed(1), height: 8, rx: 4, fill: colorFn(name, k),
                     opacity: k === winIdx ? 1 : .45, "data-tip": tipFn(name, p) }, g);
        txt(X.probEnd + 8, y + 4, "prob-pct", (p * 100).toFixed(p >= 0.1 ? 1 : 2) + "%", g);
        // ribbon from head output to top entries
        if (k < 4) el("path", { d: rib(X.h3 + 24, fromY, X.probBar - 58, y), class: "ribbon rib-hid draw",
                                "stroke-width": k === winIdx ? 3 : 1.4, style: `animation-delay:${delay + k * 45}ms` }, ribbons);
      });
    }
    const deptShow = catsSorted.slice(0, 9);
    txt(X.probName, yA - 96, "head-label", "DEPARTMENT (top 9 of " + catsSorted.length + ")", nodes, "end");
    probList(deptShow, yA - 76, 26, () => "#199f86", 0, yA, 900, (nm, p) => `P(${nm}) = ${(p * 100).toFixed(2)}%`);
    txt(X.probName, yB - 62, "head-label", "URGENCY", nodes, "end");
    probList(urgEntries, yB - 40, 30, nm => URG_COLORS[nm], urgEntries.map(e => e[1]).indexOf(Math.max(...urgEntries.map(e => e[1]))), yB, 1050,
             (nm, p) => `P(${nm}) = ${(p * 100).toFixed(2)}%`);

    // decision stamp
    const win = urgEntries.slice().sort((a, b) => b[1] - a[1])[0];
    const stamp = el("g", { class: "pop", style: "animation-delay:1250ms" }, nodes);
    el("rect", { x: X.probName - 60, y: yB + 96, width: 320, height: 44, rx: 12, fill: "#fff",
                 stroke: URG_COLORS[win[0]], "stroke-width": 1.8 }, stamp);
    const st = txt(X.probName + 100, yB + 123, "col-title", `→ ${win[0]} urgency · route to ${deptShow[0][0]}`, stamp, "middle");
    st.style.fill = URG_COLORS[win[0]]; st.style.fontWeight = "700";

    wireHover(svg);
    drawGuideGlow();
  }

  /* ---------------- hover tracing + tooltip ---------------- */
  function wireHover(svg) {
    svg.addEventListener("mouseover", e => {
      const g = e.target.closest("[data-i]");
      if (!g) return;
      svg.classList.add("tracing");
      svg.querySelectorAll(`[data-i="${g.dataset.i}"]`).forEach(x => x.classList.add("hl"));
    });
    svg.addEventListener("mouseout", e => {
      const g = e.target.closest("[data-i]");
      if (!g) return;
      svg.classList.remove("tracing");
      svg.querySelectorAll(".hl").forEach(x => x.classList.remove("hl"));
    });
  }
  const tip = $("#tip");
  document.addEventListener("mousemove", e => {
    const t = e.target.closest && e.target.closest("[data-tip]");
    if (t) {
      tip.style.display = "block";
      tip.textContent = t.getAttribute("data-tip");
      tip.style.left = Math.min(e.clientX + 14, innerWidth - 280) + "px";
      tip.style.top = (e.clientY + 16) + "px";
    } else tip.style.display = "none";
  });

  /* ---------------- guided tour ---------------- */
  const GUIDE = [
    { t: "The whole model, on one canvas", col: null,
      p: "This is the exact neural network that triages grievances on the citizen portal — a BiLSTM with additive attention and two parallel classification heads. Type any complaint above and press Classify: everything you see re-computes from a real forward pass. Use ‹ › to walk through the layers.", hp: "~435k trained parameters · PyTorch" },
    { t: "Input — words become ids", col: "tok",
      p: "The complaint is lower-cased and split into tokens. Each token is looked up in a vocabulary learned from 4,840 real grievances; words the model never saw become <UNK>. Hover any token to trace its entire journey across the canvas.", hp: "max_len · max_vocab · min_freq · lowercase" },
    { t: "Embedding — ids become meaning", col: "emb",
      p: "Each id indexes a trainable matrix and becomes a dense vector (first 8 of its dimensions shown as colored cells — blue negative, orange positive). Words used in similar complaints drift toward similar vectors during training.", hp: "emb_dim = embedding size per token" },
    { t: "BiLSTM — reading in both directions", col: "lstm",
      p: "Two recurrent chains: the forward pass (blue, arrows downward) carries context left→right; the backward pass (violet, arrows upward) right→left. Node size shows activation strength. After this, every token knows the whole sentence.", hp: "hidden_dim · num_layers · bidirectional · dropout" },
    { t: "Hidden states — the concatenation", col: "hid",
      p: "For each token the forward and backward states are glued together into one vector (bar shows the blue/violet share). This is 'Layer 3' of the architecture — no parameters, just concatenation.", hp: "2 × hidden_dim per token" },
    { t: "Attention — where the magic focuses", col: "attn",
      p: "A tiny network scores every token; softmax turns scores into weights α that sum to 100%. The ribbon width flowing right IS the weight — watch words like 'stopped', 'urgent' or 'pension' carry the thickest ribbons. These exact weights are shown to citizens as the explanation.", hp: "attn_dim = scoring projection size" },
    { t: "Context vector — one point for one complaint", col: "ctx",
      p: "The α-weighted sum squeezes the whole complaint into a single vector. Every complaint ever filed lands somewhere in this space; urgent ones cluster together — that's what the heads exploit next.", hp: "dimension = 2 × hidden_dim" },
    { t: "Two heads — two answers at once", col: "heads",
      p: "Two independent MLPs read the same context vector: Head A picks the responsible department, Head B the urgency level. Each has two hidden layers with BatchNorm + ReLU + dropout. Multi-task learning: both losses train the shared layers below.", hp: "head_hidden neurons ×2 layers · w_category / w_urgency" },
    { t: "Softmax probabilities — the decision", col: "out",
      p: "Logits become probabilities. The department routes the complaint (with CPGRAMS codes); the urgency sets its queue position and SLA. The rare Critical class is caught reliably because training multiplied its loss weight — see the Observatory's Experiments tab for the proof.", hp: "critical_boost · use_class_weights" },
    { t: "Now it's your turn", col: null,
      p: "Try the examples menu, or write your own complaint — mention durations ('for 8 months') and vulnerable people ('elderly mother') and watch the attention ribbons shift. Then visit the Project Story for the full journey, or the Observatory to retrain the model yourself.", hp: "/story · /admin · /" },
  ];
  let gStep = 0;
  const COL_RANGES = { tok: [24, 132], emb: [148, 282], lstm: [322, 440], hid: [470, 596], attn: [628, 742],
                       ctx: [796, 862], heads: [930, 1152], out: [1180, 1520] };

  function drawGuideGlow() {
    const layer = $("#glow-layer");
    if (!layer) return;
    layer.innerHTML = "";
    const col = GUIDE[gStep].col;
    if (!col || !TR) return;
    const [x0, x1] = COL_RANGES[col];
    const vb = $("#flow").viewBox.baseVal;
    el("rect", { x: x0, y: 26, width: x1 - x0, height: vb.height - 60, rx: 14, class: "col-glow" }, layer);
  }
  function renderGuide() {
    const g = GUIDE[gStep];
    $("#g-kicker").textContent = `GUIDED TOUR — LAYER WALK`;
    $("#g-title").textContent = g.t;
    $("#g-text").textContent = g.p;
    $("#g-hp").textContent = g.hp ? "knobs: " + g.hp : "";
    $("#g-count").textContent = `${gStep + 1} / ${GUIDE.length}`;
    $("#g-bar").style.width = ((gStep + 1) / GUIDE.length * 100) + "%";
    drawGuideGlow();
  }
  $("#g-prev").addEventListener("click", () => { gStep = (gStep - 1 + GUIDE.length) % GUIDE.length; renderGuide(); });
  $("#g-next").addEventListener("click", () => { gStep = (gStep + 1) % GUIDE.length; renderGuide(); });
  $("#guide-close").addEventListener("click", () => { $("#guide-card").style.display = "none"; $("#guide-reopen").style.display = "block"; });
  $("#guide-reopen").addEventListener("click", () => { $("#guide-card").style.display = "block"; $("#guide-reopen").style.display = "none"; });

  /* ---------------- run ---------------- */
  async function classify() {
    const btn = $("#classify-btn");
    btn.disabled = true; btn.textContent = "Running…";
    try {
      TR = await api("/api/admin/explain", { text: $("#text-in").value });
      draw();
    } catch (e) { alert(e.message); }
    finally { btn.disabled = false; btn.textContent = "Classify ▶"; }
  }
  $("#classify-btn").addEventListener("click", classify);
  $("#text-in").addEventListener("keydown", e => { if (e.key === "Enter") classify(); });
  $("#examples").addEventListener("change", () => {
    if ($("#examples").value) { $("#text-in").value = $("#examples").value; classify(); $("#examples").selectedIndex = 0; }
  });

  (async () => {
    try {
      MODEL = await api("/api/admin/model");
      $("#top-meta").innerHTML = `model <b>${MODEL.run_name}</b> · ${(MODEL.n_params || 0).toLocaleString()} params · Critical-recall ${(MODEL.test_metrics.critical_recall ?? 0).toFixed(2)}<br>
        <a href="/admin">Observatory</a><a href="/story">Project story</a><a href="/">Portal</a>`;
      renderGuide();
      await classify();
    } catch (e) {
      $("#top-meta").innerHTML = "⚠ no trained model — run <code>python -m src.main train</code>";
    }
  })();
})();
