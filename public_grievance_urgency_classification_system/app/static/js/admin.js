/* Model Observatory — admin dashboard logic */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const URG_COLORS = { Low: "#3fbf7f", Medium: "#e9b949", High: "#f08a4b", Critical: "#e0596b" };
  const SERIES_COLORS = ["#6d8dff", "#39c6d6", "#e9b949", "#e0596b", "#9d6dff"];
  const fmtPct = v => (v * 100).toFixed(1) + "%";

  async function api(path, body) {
    const res = await fetch(path, body ? {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    } : undefined);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    return res.json();
  }

  /* ---------------- navigation ---------------- */
  $$(".side-nav button").forEach(b => b.addEventListener("click", () => {
    $$(".side-nav button").forEach(x => x.classList.toggle("active", x === b));
    $$(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + b.dataset.view));
    if (b.dataset.view === "experiments") loadRuns();
    if (b.dataset.view === "magic" && window.Magic) Magic.refreshReplay();
  }));

  /* ================= OVERVIEW ================= */
  let MODEL = null;

  function metricCard(label, value, note, hero) {
    return `<div class="metric ${hero ? "hero-metric" : ""}">
      <div class="m-label">${label}</div><div class="m-value">${value}</div><div class="m-note">${note || ""}</div></div>`;
  }

  async function loadOverview() {
    MODEL = await api("/api/admin/model");
    const t = MODEL.test_metrics || {};
    $("#side-model").innerHTML = `Production model<b>${MODEL.run_name || MODEL.run_id}</b>
      ${(MODEL.n_params || 0).toLocaleString()} params · ${MODEL.vocab_size.toLocaleString()} vocab`;
    $("#ov-metrics").innerHTML =
      metricCard("Critical recall (headline)", (t.critical_recall ?? 0).toFixed(3), "share of true emergencies caught", true) +
      metricCard("Urgency macro-F1", (t.urgency_macro_f1 ?? 0).toFixed(3), "balanced quality across 4 classes") +
      metricCard("Urgency accuracy", fmtPct(t.urgency_accuracy ?? 0), `on ${t.n_test || "—"} held-out complaints`) +
      metricCard("Category accuracy", fmtPct(t.category_accuracy ?? 0), "department routing head");

    const cfg = MODEL.config || {};
    $("#ov-card").innerHTML = `
      <tr><td>Run</td><td class="mono">${MODEL.run_name} (${MODEL.run_id})</td></tr>
      <tr><td>Trained at</td><td class="mono">${MODEL.trained_at || "—"}</td></tr>
      <tr><td>Architecture</td><td>Embedding ${cfg.emb_dim} → BiLSTM ${cfg.hidden_dim}×${cfg.num_layers} (bi) → Attention ${cfg.attn_dim} → 2× MLP heads (${cfg.head_hidden}·${cfg.head_hidden})</td></tr>
      <tr><td>Sequence length</td><td class="mono">max_len = ${cfg.max_len}</td></tr>
      <tr><td>Vocabulary</td><td class="mono">${MODEL.vocab_size.toLocaleString()} tokens (max ${cfg.max_vocab}, min_freq ${cfg.min_freq})</td></tr>
      <tr><td>Optimiser</td><td class="mono">${cfg.optimizer} · lr ${cfg.lr} · wd ${cfg.weight_decay} · ${cfg.scheduler} scheduler</td></tr>
      <tr><td>Loss</td><td class="mono">${cfg.w_category}·CE(category) + ${cfg.w_urgency}·CE(urgency, class-weighted, Critical ×${cfg.critical_boost})</td></tr>
      <tr><td>Outputs</td><td>${MODEL.categories.length} categories · ${MODEL.urgency_labels.join(" / ")}</td></tr>`;

    Charts.barChart($("#ov-params"), (MODEL.layer_params || []).map((l, i) => ({
      label: `${l.module} (${l.class})`, value: l.params, color: SERIES_COLORS[i % SERIES_COLORS.length],
    })), { horizontal: true, format: v => v.toLocaleString() });

    const h = MODEL.history || [];
    if (h.length) {
      const pts = k => h.map(r => ({ x: r.epoch, y: r[k] }));
      Charts.lineChart($("#ov-loss"), [
        { name: "train", color: "#6d8dff", points: pts("train_loss") },
        { name: "val", color: "#39c6d6", points: pts("val_loss") },
      ], { xLabel: "epoch" });
      Charts.lineChart($("#ov-quality"), [
        { name: "val macro-F1", color: "#e9b949", points: pts("val_macro_f1") },
        { name: "val Critical-recall", color: "#e0596b", points: pts("val_critical_recall") },
      ], { xLabel: "epoch", y0: 0 });
    }
    loadEvaluation(t);
  }

  /* ================= ARCHITECTURE EXPLORER ================= */
  const STAGE_META = [
    { key: "tokens", chip: "LAYER 0", color: "#5c6a8c", title: "Tokenisation — text becomes integers",
      hp: "max_len, max_vocab, min_freq",
      desc: "The complaint is lower-cased and split on whitespace; each word is looked up in the learned vocabulary (dashed = out-of-vocabulary → <UNK>)." },
    { key: "emb", chip: "LAYER 1", color: "#6d8dff", title: "Embedding — each token becomes a dense vector",
      hp: "emb_dim",
      desc: "Every token id indexes a trainable matrix. Columns below show the first 8 of the vector's dimensions for each token (blue = negative, orange = positive)." },
    { key: "lstm", chip: "LAYER 2–3", color: "#9d6dff", title: "Bidirectional LSTM — context in both directions",
      hp: "hidden_dim, num_layers, dropout",
      desc: "A forward pass reads left→right, a backward pass right→left; their hidden states are concatenated. Bars show the activation magnitude (norm) of each direction per token." },
    { key: "attn", chip: "LAYER 4–5", color: "#e0596b", title: "Additive attention — which words matter",
      hp: "attn_dim",
      desc: "The model scores every position, softmaxes to weights α, and blends the LSTM states into one context vector. This is the explainability layer — these exact weights are shown to citizens." },
    { key: "ctx", chip: "CONTEXT", color: "#39c6d6", title: "Context vector — the complaint as one vector",
      hp: "2 × hidden_dim",
      desc: "The attention-weighted summary that both classification heads read." },
    { key: "heads", chip: "HEADS A+B", color: "#3fbf7f", title: "Parallel MLP heads — two predictions at once",
      hp: "head_hidden ×2 layers, BatchNorm, ReLU",
      desc: "Head A predicts the department/category; Head B the urgency. Softmax turns logits into the probabilities below." },
  ];

  function heatColor(v, scale = 1.6) {
    const x = Math.max(-1, Math.min(1, v / scale));
    return x >= 0 ? `rgba(240,138,75,${(0.12 + 0.85 * x).toFixed(3)})`
                  : `rgba(109,141,255,${(0.12 + 0.85 * -x).toFixed(3)})`;
  }

  function renderArch(tr) {
    const flow = $("#arch-flow");
    const hl = i => `data-i="${i}" onmouseenter="window.__archHL(${i},true)" onmouseleave="window.__archHL(${i},false)"`;
    const stages = {};

    stages.tokens = `<div class="tok-row">${tr.tokens.map((t, i) =>
      `<span class="tok arch-tok ${tr.in_vocab[i] ? "" : "oov"}" ${hl(i)}>${t}<span class="tid">id ${tr.token_ids[i]}${tr.in_vocab[i] ? "" : " (UNK)"}</span></span>`).join("")}</div>`;

    stages.emb = `<div class="emb-grid">${tr.embedding_preview.map((vec, i) =>
      `<div class="emb-col arch-tok" ${hl(i)}><div class="emb-cells">${vec.map(v =>
        `<span class="emb-cell" style="background:${heatColor(v)}" title="${v}"></span>`).join("")}</div>
        <span class="cap">${tr.tokens[i]}</span></div>`).join("")}
      </div><p class="desc" style="margin:8px 0 0">…each column continues to ${tr.embedding_dim} dimensions (‖v‖ shown on hover: ${tr.embedding_norms.slice(0, 6).join(", ")}…)</p>`;

    const maxN = Math.max(...tr.lstm_fwd_norms, ...tr.lstm_bwd_norms, 1e-9);
    stages.lstm = `<div class="lstm-row">${tr.tokens.map((t, i) =>
      `<div class="lstm-col arch-tok" ${hl(i)}><div class="lstm-bars">
        <span class="lstm-bar fwd" style="height:${(tr.lstm_fwd_norms[i] / maxN * 100).toFixed(1)}%" title="forward ‖h‖ = ${tr.lstm_fwd_norms[i]}"></span>
        <span class="lstm-bar bwd" style="height:${(tr.lstm_bwd_norms[i] / maxN * 100).toFixed(1)}%" title="backward ‖h‖ = ${tr.lstm_bwd_norms[i]}"></span>
      </div><span class="cap" style="font-size:9.5px;color:var(--dim);max-width:56px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${t}</span></div>`).join("")}</div>
      <p class="desc" style="margin:8px 0 0"><span class="legend-dot" style="background:#6d8dff"></span>forward &nbsp;<span class="legend-dot" style="background:#9d6dff"></span>backward · output dimension per token: ${tr.lstm_dim}</p>`;

    const maxA = Math.max(...tr.attention, 1e-9);
    stages.attn = `<div class="attn-row">${tr.tokens.map((t, i) =>
      `<div class="attn-col arch-tok" ${hl(i)}>
        <span class="attn-pct">${(tr.attention[i] * 100).toFixed(1)}%</span>
        <div class="attn-bar-track"><span class="attn-bar" style="height:${(tr.attention[i] / maxA * 100).toFixed(1)}%"></span></div>
        <span class="cap" style="font-size:9.5px;color:var(--dim);max-width:56px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${t}</span></div>`).join("")}</div>`;

    stages.ctx = `<div class="ctx-strip">${tr.context_preview.map(v =>
      `<span class="ctx-cell" style="background:${heatColor(v, 2.5)}" title="${v}"></span>`).join("")}<span class="cap" style="align-self:center;font-size:10.5px;color:var(--dim)">&nbsp;… ${tr.context_dim} dims, ‖c‖ = ${tr.context_norm}</span></div>`;

    const catSorted = Object.fromEntries(Object.entries(tr.category_probs).sort((a, b) => b[1] - a[1]).slice(0, 6));
    stages.heads = `<div class="heads-grid">
      <div class="head-panel"><h5>HEAD A — Category / Department (top 6 of ${Object.keys(tr.category_probs).length})</h5><div id="arch-cat"></div></div>
      <div class="head-panel urg"><h5>HEAD B — Urgency</h5><div id="arch-urg"></div></div></div>`;

    flow.innerHTML = STAGE_META.map((s, i) => `
      ${i ? '<div class="arch-connector"></div>' : ""}
      <div class="arch-stage">
        <h4><span class="stage-chip" style="background:${s.color}">${s.chip}</span> ${s.title}
          <span class="stage-hp">${s.hp}</span></h4>
        <p class="desc">${s.desc}</p>
        ${stages[s.key]}
      </div>`).join("");

    Charts.probBars($("#arch-cat"), catSorted, null);
    $$("#arch-cat .prob-fill").forEach(f => f.style.background = "#39c6d6");
    Charts.probBars($("#arch-urg"), tr.urgency_probs, URG_COLORS);

    window.__archHL = (i, on) => $$(`.arch-tok[data-i="${i}"]`).forEach(e => e.classList.toggle("hl", on));
  }

  async function runArch() {
    const btn = $("#arch-run");
    btn.disabled = true; btn.textContent = "Running…";
    try { renderArch(await api("/api/admin/explain", { text: $("#arch-input").value })); }
    catch (e) { alert(e.message); }
    finally { btn.disabled = false; btn.textContent = "Run forward pass ▶"; }
  }
  $("#arch-run").addEventListener("click", runArch);
  $("#arch-input").addEventListener("keydown", e => { if (e.key === "Enter") runArch(); });
  $$(".ex-btn").forEach(b => b.addEventListener("click", () => { $("#arch-input").value = b.dataset.ex; runArch(); }));

  /* ================= TRAINING STUDIO ================= */
  let HP_SPEC = {};
  const hpState = {};

  function hpControl(name, spec) {
    if (spec.type === "choice")
      return `<div class="hp-item"><label>${name}</label>
        <select data-hp="${name}">${spec.choices.map(c => `<option ${c === spec.default ? "selected" : ""}>${c}</option>`).join("")}</select>
        <div class="help">${spec.help}</div></div>`;
    if (spec.type === "bool")
      return `<div class="hp-item"><label>${name}</label>
        <select data-hp="${name}"><option value="true" ${spec.default ? "selected" : ""}>true</option><option value="false" ${!spec.default ? "selected" : ""}>false</option></select>
        <div class="help">${spec.help}</div></div>`;
    const isLog = name === "lr" || name === "weight_decay";
    return `<div class="hp-item"><label>${name}<span class="val" id="hpv-${name}">${spec.default}</span></label>
      <input type="range" data-hp="${name}" data-log="${isLog}" min="${isLog ? Math.log10(Math.max(spec.min, 1e-6)) : spec.min}"
        max="${isLog ? Math.log10(spec.max) : spec.max}" step="${isLog ? 0.1 : spec.step}"
        value="${isLog ? Math.log10(Math.max(spec.default, 1e-6)) : spec.default}">
      <div class="help">${spec.help}</div></div>`;
  }

  async function buildHpForm() {
    HP_SPEC = await api("/api/admin/hyperparameters");
    const groups = {};
    Object.entries(HP_SPEC).forEach(([k, v]) => (groups[v.group] = groups[v.group] || []).push([k, v]));
    $("#hp-form").innerHTML = Object.entries(groups).map(([g, items]) =>
      `<div class="hp-group"><b>${g}</b><div class="hp-grid">${items.map(([k, v]) => hpControl(k, v)).join("")}</div></div>`).join("");

    $$("#hp-form [data-hp]").forEach(el => {
      const name = el.dataset.hp, spec = HP_SPEC[name];
      const read = () => {
        if (spec.type === "choice") return el.value;
        if (spec.type === "bool") return el.value === "true";
        let v = parseFloat(el.value);
        if (el.dataset.log === "true") v = Math.pow(10, v);
        return spec.type === "int" ? Math.round(v) : +v.toPrecision(3);
      };
      hpState[name] = spec.default;
      el.addEventListener("input", () => {
        hpState[name] = read();
        const lbl = $("#hpv-" + name);
        if (lbl) lbl.textContent = typeof hpState[name] === "number" && hpState[name] < 0.01
          ? hpState[name].toExponential(1) : hpState[name];
      });
    });
  }

  let pollTimer = null, liveEpochs = [];

  function setPill(cls, text) {
    const p = $("#studio-pill");
    p.className = "pill " + cls;
    $("#studio-status").textContent = text;
  }

  async function startTraining() {
    const overrides = {};
    Object.entries(hpState).forEach(([k, v]) => { if (v !== HP_SPEC[k].default) overrides[k] = v; });
    // always send epochs/seed so the run is fully specified in the log
    overrides.epochs = hpState.epochs; overrides.seed = hpState.seed;
    try {
      const r = await api("/api/admin/train", {
        overrides, run_name: $("#run-name").value.trim(), promote: $("#promote-check").checked,
      });
      liveEpochs = [];
      $("#train-start").disabled = true; $("#train-stop").disabled = false;
      $("#studio-summary").textContent = "";
      setPill("running", `training "${r.run_name}" — epoch 0/${hpState.epochs}`);
      poll();
    } catch (e) { alert(e.message); }
  }

  async function poll() {
    clearTimeout(pollTimer);
    try {
      const s = await api(`/api/admin/train/status?since_epoch=${liveEpochs.length}`);
      liveEpochs.push(...s.epochs);
      if (liveEpochs.length) renderLive(liveEpochs);
      const total = s.total_epochs_planned || hpState.epochs;
      $("#studio-progress").style.width = Math.min(100, liveEpochs.length / total * 100) + "%";
      $("#studio-epoch-count").textContent = liveEpochs.length ? `epoch ${liveEpochs.length}/${total}` : "";
      if (s.running) {
        setPill("running", `training "${s.run_name}" — epoch ${liveEpochs.length}/${total}`);
        pollTimer = setTimeout(poll, 1200);
      } else {
        $("#train-start").disabled = false; $("#train-stop").disabled = true;
        if (s.error) { setPill("error", "failed: " + s.error); return; }
        if (s.summary) {
          const t = s.summary.test || {};
          setPill("done", `finished "${s.run_name}" — ${s.summary.epochs_ran} epochs${s.summary.stopped_early ? " (early stop)" : ""}`);
          $("#studio-summary").innerHTML =
            `<b style="color:var(--text)">Held-out test:</b> Critical-recall <b style="color:var(--critical)">${(t.critical_recall ?? 0).toFixed(3)}</b> ·
             macro-F1 <b>${(t.urgency_macro_f1 ?? 0).toFixed(3)}</b> · urgency acc ${fmtPct(t.urgency_accuracy ?? 0)} ·
             category acc ${fmtPct(t.category_accuracy ?? 0)} ${s.promote ? "· <span class='tag diff'>promoted to production</span>" : ""}`;
          if (s.promote) loadOverview();
        } else if (liveEpochs.length === 0) setPill("", "idle — no run in progress");
      }
    } catch (e) { pollTimer = setTimeout(poll, 2500); }
  }

  function renderLive(eps) {
    const pts = k => eps.map(r => ({ x: r.epoch, y: r[k] }));
    Charts.lineChart($("#st-loss"), [
      { name: "train", color: "#6d8dff", points: pts("train_loss") },
      { name: "val", color: "#39c6d6", points: pts("val_loss") }], { xLabel: "epoch" });
    Charts.lineChart($("#st-acc"), [
      { name: "train", color: "#6d8dff", points: pts("train_acc") },
      { name: "val", color: "#39c6d6", points: pts("val_acc") }], { xLabel: "epoch", y0: 0 });
    Charts.lineChart($("#st-f1"), [
      { name: "train F1", color: "#6d8dff", points: pts("train_macro_f1") },
      { name: "val F1", color: "#e9b949", points: pts("val_macro_f1") },
      { name: "Critical-recall", color: "#e0596b", points: pts("val_critical_recall") }], { xLabel: "epoch", y0: 0 });
    Charts.lineChart($("#st-lr"), [
      { name: "lr", color: "#9d6dff", points: pts("lr") }], { xLabel: "epoch", logY: true });

    const last = eps[eps.length - 1];
    const labels = ["Low", "Medium", "High", "Critical"];
    const cm = labels.map((_, i) => labels.map((_, j) => last[`cm_${i}_${j}`] ?? 0));
    Charts.heatmap($("#st-cm"), cm, labels, { color: [109, 141, 255] });

    $("#st-table").innerHTML = `<tr><th>ep</th><th>train loss</th><th>val loss</th><th>val acc</th><th>val F1</th><th>crit recall</th><th>cat acc</th><th>lr</th><th>sec</th></tr>` +
      [...eps].reverse().map(r => `<tr><td>${r.epoch}</td><td>${r.train_loss.toFixed(4)}</td><td>${r.val_loss.toFixed(4)}</td>
        <td>${r.val_acc.toFixed(3)}</td><td>${r.val_macro_f1.toFixed(3)}</td>
        <td style="color:var(--critical);font-weight:700">${r.val_critical_recall.toFixed(3)}</td>
        <td>${(r.val_cat_acc ?? 0).toFixed(3)}</td>
        <td class="mono">${(+r.lr).toExponential(1)}</td><td>${r.epoch_seconds}</td></tr>`).join("");
  }

  $("#train-start").addEventListener("click", startTraining);
  $("#train-stop").addEventListener("click", () => api("/api/admin/train/stop", {}).catch(() => {}));

  /* ================= EXPERIMENTS ================= */
  let RUNS = [], selected = new Set();
  const DIFF_KEYS = ["emb_dim", "hidden_dim", "num_layers", "bidirectional", "attn_dim",
    "dropout", "head_hidden", "max_len", "lowercase", "batch_size", "optimizer", "scheduler",
    "weight_decay", "grad_clip", "patience", "critical_boost", "w_category", "w_urgency",
    "use_class_weights", "seed"];
  const same = (a, b) => String(a).toLowerCase() === String(b).toLowerCase();

  async function loadRuns() {
    RUNS = await api("/api/admin/runs");
    const defaults = {}; Object.entries(HP_SPEC).forEach(([k, v]) => defaults[k] = v.default);
    $("#runs-table").innerHTML = `<tr><th></th><th>run</th><th>epochs</th><th>changed hyperparameters</th>
      <th>val loss</th><th>val F1</th><th>best crit recall</th><th>time</th></tr>` +
      RUNS.map(r => {
        const changed = DIFF_KEYS.filter(k => defaults[k] !== undefined &&
            r.hyperparameters[k] !== undefined && r.hyperparameters[k] !== "" &&
            !same(r.hyperparameters[k], defaults[k]))
          .map(k => `<span class="tag diff">${k}=${r.hyperparameters[k]}</span>`).join(" ") || `<span class="tag">notebook defaults</span>`;
        return `<tr><td><input type="checkbox" class="run-check" data-run="${r.run_id}" ${selected.has(r.run_id) ? "checked" : ""}></td>
          <td><b>${r.run_name}</b><br><span class="mono" style="color:var(--dim)">${r.run_id}</span></td>
          <td>${r.n_epochs}</td><td>${changed}</td>
          <td>${r.final.val_loss.toFixed(4)}</td><td>${r.final.val_macro_f1.toFixed(3)}</td>
          <td style="color:var(--critical);font-weight:700">${r.best_val_critical_recall.toFixed(3)} <span class="tag">ep ${r.best_epoch}</span></td>
          <td>${r.total_seconds}s</td></tr>`;
      }).join("");
    $$("#runs-table .run-check").forEach(c => c.addEventListener("change", () => {
      c.checked ? selected.add(c.dataset.run) : selected.delete(c.dataset.run);
      if (selected.size > 3) { selected.delete(c.dataset.run); c.checked = false; }
      renderCompare();
    }));
    renderCompare();
  }

  let compareSeq = 0;
  async function renderCompare() {
    const area = $("#compare-area");
    if (selected.size < 1) { area.style.display = "none"; return; }
    area.style.display = "block";
    const seq = ++compareSeq;
    const details = await Promise.all([...selected].map(id => api("/api/admin/runs/" + id)));
    if (seq !== compareSeq) return; // a newer selection superseded this render
    const mk = key => details.map((d, i) => ({
      name: d.run_name, color: SERIES_COLORS[i],
      points: d.epochs.map(e => ({ x: e.epoch, y: e[key] })),
    }));
    Charts.lineChart($("#cmp-loss"), mk("val_loss"), { xLabel: "epoch" });
    Charts.lineChart($("#cmp-f1"), mk("val_macro_f1"), { xLabel: "epoch", y0: 0 });
    Charts.lineChart($("#cmp-crit"), mk("val_critical_recall"), { xLabel: "epoch", y0: 0 });

    const rows = details.map(d => {
      const meta = RUNS.find(r => r.run_id === d.run_id);
      return { name: d.run_name, hp: meta.hyperparameters, final: meta.final };
    });
    // ignore runs logged before a column existed (empty after schema migration)
    const diffKeys = DIFF_KEYS.filter(k => new Set(rows.map(r => String(r.hp[k]).toLowerCase())
      .filter(v => v !== "" && v !== "undefined" && v !== "null")).size > 1);
    $("#cmp-diff").innerHTML = `<tr><th>run</th>${DIFF_KEYS.map(k =>
        `<th ${diffKeys.includes(k) ? 'style="color:var(--amber)"' : ""}>${k}</th>`).join("")}<th>final crit recall</th></tr>` +
      rows.map((r, i) => `<tr><td><span class="legend-dot" style="background:${SERIES_COLORS[i]}"></span><b>${r.name}</b></td>
        ${DIFF_KEYS.map(k => `<td class="mono" ${diffKeys.includes(k) ? 'style="color:var(--amber);font-weight:700"' : ""}>${r.hp[k]}</td>`).join("")}
        <td style="color:var(--critical);font-weight:700">${r.final.val_critical_recall.toFixed(3)}</td></tr>`).join("");
  }

  /* ================= EVALUATION ================= */
  function loadEvaluation(t) {
    if (!t || !t.per_class) return;
    $("#ev-metrics").innerHTML =
      metricCard("Critical recall", t.critical_recall.toFixed(3), "missed emergencies = 1 − this", true) +
      metricCard("Macro-F1", t.urgency_macro_f1.toFixed(3), "urgency head") +
      metricCard("Urgency accuracy", fmtPct(t.urgency_accuracy), `${t.n_test} test complaints`) +
      metricCard("Category accuracy", fmtPct(t.category_accuracy), "routing head");
    const labels = t.per_class.map(c => c.label);
    Charts.heatmap($("#ev-cm"), t.confusion_matrix, labels, { color: [109, 141, 255] });
    const norm = t.confusion_matrix.map(row => {
      const s = Math.max(row.reduce((a, b) => a + b, 0), 1); return row.map(v => v / s);
    });
    Charts.heatmap($("#ev-cmn"), norm, labels, { normalized: true, color: [224, 89, 107] });
    Charts.groupedBarChart($("#ev-perclass"),
      t.per_class.map(c => ({ label: c.label, values: [c.precision, c.recall, c.f1] })),
      ["precision", "recall", "F1"], ["#6d8dff", "#3fbf7f", "#e9b949"], { max: 1.05 });
    const cal = t.calibration;
    Charts.groupedBarChart($("#ev-calib"),
      cal.bins.map((b, i) => ({ label: i % 4 === 0 ? b.toFixed(1) : "", values: [cal.correct[i], cal.wrong[i]] })),
      ["correct", "wrong"], ["#3fbf7f", "#e0596b"], {});
    $("#ev-report").innerHTML = `<tr><th>class</th><th>precision</th><th>recall</th><th>F1</th><th>support</th></tr>` +
      t.per_class.map(c => `<tr><td><span class="legend-dot" style="background:${URG_COLORS[c.label]}"></span><b>${c.label}</b></td>
        <td>${c.precision.toFixed(3)}</td><td ${c.label === "Critical" ? 'style="color:var(--critical);font-weight:700"' : ""}>${c.recall.toFixed(3)}</td>
        <td>${c.f1.toFixed(3)}</td><td>${c.support}</td></tr>`).join("");
  }

  /* ================= DATASET ================= */
  async function loadDataset() {
    const d = await api("/api/admin/dataset");
    Charts.barChart($("#ds-urg"), Object.entries(d.by_urgency).map(([k, v]) =>
      ({ label: k, value: v, color: URG_COLORS[k] })), {});
    Charts.barChart($("#ds-len"), d.length_hist.counts.map((c, i) =>
      ({ label: `${d.length_hist.edges[i]}–${d.length_hist.edges[i + 1]}`, value: c, color: "#39c6d6" })), {});
    Charts.barChart($("#ds-cat"), Object.entries(d.by_category).map(([k, v]) =>
      ({ label: k, value: v, color: "#6d8dff" })), { horizontal: true, height: 420 });
  }

  /* ================= MAGIC OF THE ARCHITECTURE ================= */
  const Magic = (() => {
    let TR = null, step = 0, autoTimer = null;

    /* ---------- Act 1: guided forward-pass show ---------- */
    const STEPS = [
      { key: "input", lbl: "Complaint", title: "A citizen writes a complaint",
        hp: [],
        text: () => `Everything starts with plain text. The model cannot read words — the next stages turn this sentence into numbers, step by step, exactly as it happens in production every time someone uses the portal.` },
      { key: "tok", lbl: "Tokenise", title: "Layer 0 — Tokenisation",
        hp: ["lowercase", "max_len", "max_vocab", "min_freq"],
        text: t => `The text is ${MODEL.config.lowercase ? "lower-cased and " : ""}split into ${t.tokens.length} tokens, each looked up in the learned vocabulary (${MODEL.vocab_size.toLocaleString()} entries). Unknown words become &lt;UNK&gt; (dashed). Sequences are padded/truncated to max_len = ${MODEL.config.max_len}.` },
      { key: "emb", lbl: "Embedding", title: "Layer 1 — Embedding lookup",
        hp: ["emb_dim", "dropout"],
        text: t => `Each token id indexes a trainable matrix and becomes a dense vector of ${t.embedding_dim} numbers. Words that behave similarly ("stopped", "denied") end up with similar vectors — this is where meaning begins. Blue = negative, orange = positive values (first 8 of ${t.embedding_dim} dims shown).` },
      { key: "lstm", lbl: "BiLSTM", title: "Layers 2–3 — Bidirectional LSTM",
        hp: ["hidden_dim", "num_layers", "bidirectional", "dropout"],
        text: t => `Watch the two sweeps: a forward pass reads left→right (blue), a backward pass right→left (violet). Each token now carries context from the WHOLE sentence — "stopped" knows it is about a pension. Their states are concatenated into ${t.lstm_dim} dims per token.` },
      { key: "attn", lbl: "Attention", title: "Layers 4–5 — Additive attention",
        hp: ["attn_dim"],
        text: t => { const m = t.tokens[t.attention.indexOf(Math.max(...t.attention))]; return `The model scores every position and softmaxes the scores into weights α that sum to 100%. Here it focuses hardest on “${m}” (${(Math.max(...t.attention) * 100).toFixed(1)}%). These exact weights are the highlights citizens see — the network explaining itself.`; } },
      { key: "ctx", lbl: "Context", title: "The context vector",
        hp: ["hidden_dim", "bidirectional"],
        text: t => `The attention weights blend all token states into ONE vector of ${t.context_dim} numbers — the entire complaint compressed into a single point in space. Every complaint ever filed lands somewhere in this space; urgent ones cluster together.` },
      { key: "heads", lbl: "Two heads", title: "Heads A + B — two predictions in parallel",
        hp: ["head_hidden", "dropout", "w_category", "w_urgency"],
        text: () => `Two separate MLPs (${MODEL.config.head_hidden}→${MODEL.config.head_hidden} neurons, BatchNorm + ReLU) read the same context vector. Head A picks the department out of ${MODEL.categories.length}; Head B the urgency out of 4. Softmax turns raw logits into the probabilities below.` },
      { key: "verdict", lbl: "Decision", title: "The decision — and what happens with it",
        hp: ["critical_boost", "use_class_weights"],
        text: t => { const u = Object.entries(t.urgency_probs).sort((a, b) => b[1] - a[1])[0]; return `Final call: ${u[0]} urgency at ${(u[1] * 100).toFixed(1)}% confidence. In the live portal this sets the queue position, response SLA and escalation path. The rare Critical class only gets caught this reliably because training boosted its loss weight ×${MODEL.config.critical_boost}.`; } },
    ];

    function stepper() {
      $("#magic-stepper").innerHTML = STEPS.map((s, i) =>
        `<div class="mstep ${i < step ? "done" : ""} ${i === step ? "current" : ""}" data-step="${i}">
           <div class="dot"></div><div class="lbl">${s.lbl}</div></div>`).join("");
      $$("#magic-stepper .mstep").forEach(el => el.addEventListener("click", () => { stopAuto(); go(+el.dataset.step); }));
    }

    function narrate() {
      const s = STEPS[step];
      $("#mn-kicker").textContent = `STAGE ${step} / ${STEPS.length - 1}`;
      $("#mn-title").textContent = s.title;
      $("#mn-text").innerHTML = TR ? s.text(TR) : "Load a complaint first.";
      $("#mn-hp").innerHTML = s.hp.map(h => `<span class="tag diff">${h}</span>`).join("") ||
        `<span class="tag">no tunable knobs at this stage</span>`;
    }

    const stag = (i, ms = 90) => `style="animation-delay:${i * ms}ms"`;

    function renderStage() {
      const c = $("#magic-canvas");
      if (!TR) { c.innerHTML = "<div class='chart-empty'>Load a complaint to begin the show</div>"; return; }
      const t = TR, key = STEPS[step].key;

      if (key === "input") {
        c.innerHTML = `<div class="pop" style="font-size:17px;line-height:1.8;padding:14px 6px">“${$("#magic-input").value}”</div>
          <p class="sub" style="margin-top:8px">…heading into a network of ${(MODEL.n_params || 0).toLocaleString()} trained weights.</p>`;

      } else if (key === "tok") {
        c.innerHTML = `<div class="tok-row">${t.tokens.map((tok, i) =>
          `<span class="tok pop ${t.in_vocab[i] ? "" : "oov"}" ${stag(i)}>${tok}<span class="tid">id ${t.token_ids[i]}${t.in_vocab[i] ? "" : " (UNK)"}</span></span>`).join("")}</div>`;

      } else if (key === "emb") {
        c.innerHTML = `<div class="emb-grid">${t.embedding_preview.map((vec, i) =>
          `<div class="emb-col pop" ${stag(i)}><div class="emb-cells">${vec.map(v =>
            `<span class="emb-cell" style="background:${heatColor(v)}" title="${v}"></span>`).join("")}</div>
           <span class="cap">${t.tokens[i]}</span></div>`).join("")}</div>
          <p class="sub" style="margin-top:10px">vector norms ‖v‖: ${t.embedding_norms.join(" · ")}</p>`;

      } else if (key === "lstm") {
        const n = t.tokens.length, maxN = Math.max(...t.lstm_fwd_norms, ...t.lstm_bwd_norms, 1e-9);
        c.innerHTML = `<div class="tok-row" id="lstm-sweep">${t.tokens.map((tok, i) =>
            `<span class="tok" data-i="${i}">${tok}</span>`).join("")}</div>
          <p class="sub" style="margin:12px 0 6px">forward sweep <span class="legend-dot" style="background:#6d8dff"></span> then backward sweep <span class="legend-dot" style="background:#9d6dff"></span> — resulting activation strength per token:</p>
          <div class="lstm-row">${t.tokens.map((tok, i) =>
            `<div class="lstm-col pop" style="animation-delay:${n * 240 + i * 60}ms"><div class="lstm-bars">
               <span class="lstm-bar fwd" style="height:${(t.lstm_fwd_norms[i] / maxN * 100).toFixed(1)}%" title="fwd ‖h‖=${t.lstm_fwd_norms[i]}"></span>
               <span class="lstm-bar bwd" style="height:${(t.lstm_bwd_norms[i] / maxN * 100).toFixed(1)}%" title="bwd ‖h‖=${t.lstm_bwd_norms[i]}"></span>
             </div><span class="cap" style="font-size:9.5px;color:var(--dim)">${tok}</span></div>`).join("")}</div>`;
        $$("#lstm-sweep .tok").forEach((el, i) => {
          el.classList.add("sweep-f"); el.style.animationDelay = `${i * 110}ms`;
          setTimeout(() => { el.classList.remove("sweep-f"); void el.offsetWidth;
            el.classList.add("sweep-b"); el.style.animationDelay = `${(n - 1 - i) * 110}ms`; }, n * 110 + 500);
        });

      } else if (key === "attn") {
        const maxA = Math.max(...t.attention, 1e-9);
        c.innerHTML = `<div class="attn-row">${t.tokens.map((tok, i) =>
          `<div class="attn-col"><span class="attn-pct">${(t.attention[i] * 100).toFixed(1)}%</span>
             <div class="attn-bar-track"><span class="attn-bar" data-h="${(t.attention[i] / maxA * 100).toFixed(1)}" style="height:0%"></span></div>
             <span class="cap" style="font-size:9.5px;color:var(--dim)">${tok}</span></div>`).join("")}</div>`;
        requestAnimationFrame(() => setTimeout(() =>
          $$(".attn-bar", c).forEach(b => b.style.height = b.dataset.h + "%"), 60));

      } else if (key === "ctx") {
        c.innerHTML = `<p class="sub" style="margin:0 0 10px">α-weighted sum of all token states →</p>
          <div class="ctx-strip">${t.context_preview.map((v, i) =>
            `<span class="ctx-cell pop" ${stag(i, 50)} style="background:${heatColor(v, 2.5)};animation-delay:${i * 50}ms" title="${v}"></span>`).join("")}
          <span class="cap" style="align-self:center;font-size:11px;color:var(--dim)">&nbsp;… ${t.context_dim} dims · ‖c‖ = ${t.context_norm}</span></div>`;

      } else if (key === "heads") {
        const catTop = Object.fromEntries(Object.entries(t.category_probs).sort((a, b) => b[1] - a[1]).slice(0, 5));
        c.innerHTML = `<div class="heads-grid">
          <div class="head-panel pop"><h5>HEAD A — Category / Department</h5><div id="mg-cat"></div></div>
          <div class="head-panel urg pop" style="animation-delay:120ms"><h5>HEAD B — Urgency</h5><div id="mg-urg"></div></div></div>`;
        Charts.probBars($("#mg-cat"), catTop, null);
        $$("#mg-cat .prob-fill").forEach(f => f.style.background = "#39c6d6");
        Charts.probBars($("#mg-urg"), t.urgency_probs, URG_COLORS);

      } else if (key === "verdict") {
        const u = Object.entries(t.urgency_probs).sort((a, b) => b[1] - a[1])[0];
        const cat = Object.entries(t.category_probs).sort((a, b) => b[1] - a[1])[0];
        c.innerHTML = `<div class="magic-final pop">
            <div class="tag">FORWARD PASS COMPLETE — ~${(MODEL.n_params || 0).toLocaleString()} multiplications later</div>
            <div class="verdict" style="color:${URG_COLORS[u[0]]}">${u[0]} urgency · ${(u[1] * 100).toFixed(1)}%</div>
            <div style="color:var(--muted)">routed to <b style="color:var(--cyan)">${cat[0]}</b> (${(cat[1] * 100).toFixed(1)}% confident)</div>
          </div>
          <div class="magic-verdict-grid">
            <div class="head-panel pop" style="animation-delay:150ms"><h5>WHAT THE CITIZEN SEES</h5>
              <p style="font-size:12px;color:var(--muted);margin:0">Urgency gauge + score breakdown, the attention words highlighted, department & CPGRAMS codes, SLAs and next steps.</p></div>
            <div class="head-panel urg pop" style="animation-delay:280ms"><h5>WHAT THE SYSTEM DOES</h5>
              <p style="font-size:12px;color:var(--muted);margin:0">${u[0] === "Critical" ? "Auto-flags the Nodal Grievance Officer — first response due in 24 h." : u[0] === "High" ? "Places it in the priority queue — first response in 3 working days." : "Queues it with a " + (u[0] === "Medium" ? "7" : "15") + "-day first-response target."} Reference ID issued; escalation clock starts.</p></div>
          </div>`;
      }
    }

    function go(i) { step = Math.max(0, Math.min(STEPS.length - 1, i)); stepper(); narrate(); renderStage(); }

    function stopAuto() { clearInterval(autoTimer); autoTimer = null; $("#magic-play").textContent = "⏵ Auto-play the magic"; }

    async function load() {
      const btn = $("#magic-load");
      btn.disabled = true; btn.textContent = "Loading…";
      try { TR = await api("/api/admin/explain", { text: $("#magic-input").value }); go(0); }
      catch (e) { alert(e.message); }
      finally { btn.disabled = false; btn.textContent = "Load complaint"; }
    }

    /* ---------- Act 2: hyperparameter anatomy ---------- */
    const ANATOMY = {
      max_len:        { at: ["input"], up: "Longer complaints kept whole; slower, more padding on short ones.", down: "Faster, but long complaints get truncated and may lose the urgent part." },
      max_vocab:      { at: ["input"], up: "Rarer words get their own vectors; bigger embedding table.", down: "More words collapse to <UNK>; model becomes blinder to rare terms." },
      min_freq:       { at: ["input"], up: "Cleaner vocabulary, fewer one-off typos learned.", down: "Vocabulary absorbs noise and typos; risk of overfitting rare tokens." },
      lowercase:      { at: ["input"], up: "ON: 'Pension' = 'pension' — smaller vocab, more data per word.", down: "OFF: case kept; 'URGENT' differs from 'urgent' — richer but sparser." },
      emb_dim:        { at: ["emb"], up: "Richer word meaning; more params, slower, can overfit small data.", down: "Compact & fast; words may become indistinguishable." },
      dropout:        { at: ["emb", "lstm", "headA", "headB"], up: "Stronger regularisation — use when train F1 ≫ val F1.", down: "Model memorises more; watch val loss turning upward." },
      hidden_dim:     { at: ["lstm"], up: "More capacity to track context (raise if underfitting: 128→256).", down: "Faster and lighter; may underfit complex phrasing." },
      num_layers:     { at: ["lstm"], up: "Deeper stack captures higher-order patterns; slower, needs dropout.", down: "Single layer is fast and usually enough for short complaints." },
      bidirectional:  { at: ["lstm"], up: "ON: context from both directions; context vector = 2×hidden.", down: "OFF: left-to-right only; halves the context size and quality." },
      attn_dim:       { at: ["attn"], up: "More expressive scoring of which tokens matter.", down: "Simpler attention; may blur focus across tokens." },
      head_hidden:    { at: ["headA", "headB"], up: "Stronger classifier heads on top of the context vector.", down: "Leaner heads; fine when the BiLSTM features are already rich." },
      batch_size:     { at: ["train"], up: "Smoother gradients, better GPU use; sometimes worse generalisation.", down: "Noisier updates that can escape bad minima; slower per epoch." },
      epochs:         { at: ["train"], up: "More passes to converge — early stopping guards the excess.", down: "May stop before the model has fully learned." },
      lr:             { at: ["train"], up: "Learns faster but can spike/NaN — the classic instability knob.", down: "Stable but slow; may stall at a plateau (lower if loss spikes)." },
      weight_decay:   { at: ["train"], up: "Shrinks weights — another overfitting brake.", down: "Weights roam free; pair with dropout to control variance." },
      optimizer:      { at: ["train"], up: "adamw: adaptive + proper decay (default).", down: "sgd+momentum: simpler, sometimes generalises better, needs tuning." },
      scheduler:      { at: ["train"], up: "plateau: halves LR when Critical-recall stalls.", down: "cosine: smooth decay; none: constant LR." },
      grad_clip:      { at: ["train"], up: "Looser clipping; only wild spikes are cut.", down: "Tighter safety net against exploding LSTM gradients." },
      patience:       { at: ["train"], up: "Waits longer for a comeback before stopping.", down: "Stops quickly — snappy experiments, may quit too early." },
      seed:           { at: ["train"], up: "Different seed = different init & split shuffle.", down: "Same seed = exactly reproducible run." },
      w_category:     { at: ["loss", "headA"], up: "Routing accuracy matters more in the shared loss.", down: "Category head learns more slowly." },
      w_urgency:      { at: ["loss", "headB"], up: "Urgency dominates the gradient budget.", down: "Urgency head under-trained — dangerous for triage." },
      use_class_weights: { at: ["loss", "headB"], up: "ON: rare classes weigh more — protects Critical.", down: "OFF: majority classes dominate; Critical-recall collapses." },
      critical_boost: { at: ["loss", "headB"], up: "Missing a Critical costs the model dearly → recall ↑ (some false alarms).", down: "At 1.0 the ablation shows ~6% of emergencies get missed." },
    };

    function buildAnatomy() {
      const cfg = MODEL.config, H = cfg.hidden_dim * (cfg.bidirectional ? 2 : 1);
      const blocks = [
        { id: "input", name: "Input · tokens", dim: `≤ ${cfg.max_len} ids` },
        { id: "emb", name: "Embedding", dim: `${MODEL.vocab_size} × ${cfg.emb_dim}` },
        { id: "lstm", name: `BiLSTM ×${cfg.num_layers}`, dim: `${cfg.hidden_dim} × ${cfg.bidirectional ? 2 : 1} dirs → ${H}/token` },
        { id: "attn", name: "Additive attention", dim: `score dim ${cfg.attn_dim} → α weights` },
        { id: "ctx", name: "Context vector", dim: `${H} dims` },
      ];
      const arrow = '<div class="ma-arrow"></div>';
      $("#mini-arch").innerHTML =
        blocks.map(b => `<div class="ma-block" data-b="${b.id}"><div class="ma-name">${b.name}</div><div class="ma-dim">${b.dim}</div></div>`).join(arrow) +
        arrow +
        `<div class="ma-split">
           <div class="ma-block" data-b="headA"><div class="ma-name">Head A · category</div><div class="ma-dim">${H}→${cfg.head_hidden}→${cfg.head_hidden}→${MODEL.categories.length}</div></div>
           <div class="ma-block" data-b="headB"><div class="ma-name">Head B · urgency</div><div class="ma-dim">${H}→${cfg.head_hidden}→${cfg.head_hidden}→4</div></div>
         </div>` + arrow +
        `<div class="ma-block" data-b="loss"><div class="ma-name">Multi-task loss</div><div class="ma-dim">${cfg.w_category}·CE(cat) + ${cfg.w_urgency}·CE(urg, weighted ×${cfg.critical_boost})</div></div>` + arrow +
        `<div class="ma-block" data-b="train"><div class="ma-name">Training loop</div><div class="ma-dim">${cfg.optimizer} · lr ${cfg.lr} · batch ${cfg.batch_size} · ${cfg.scheduler}</div></div>`;

      $("#anatomy-chips").innerHTML = Object.keys(ANATOMY)
        .filter(k => HP_SPEC[k]) // only show knobs that exist in the Studio
        .map(k => `<span class="hp-chip" data-anat="${k}">${k}</span>`).join("");
      $$("#anatomy-chips .hp-chip").forEach(ch => ch.addEventListener("click", () => selectAnatomy(ch.dataset.anat)));
    }

    function selectAnatomy(k) {
      $$("#anatomy-chips .hp-chip").forEach(c => c.classList.toggle("sel", c.dataset.anat === k));
      const a = ANATOMY[k], spec = HP_SPEC[k] || {};
      $$("#mini-arch .ma-block").forEach(b => b.classList.toggle("lit", a.at.includes(b.dataset.b)));
      const cur = MODEL.config[k];
      $("#anatomy-card").innerHTML = `<b>${k}</b> <span class="tag">production value: ${cur}</span>
        <span class="tag">default: ${spec.default}</span>
        <p>${spec.help || ""} Acts on: ${a.at.map(x => `<span class="tag diff">${x}</span>`).join(" ")}</p>
        <div class="fx-row">
          <div class="fx up"><b>▲ IF YOU RAISE / TURN ON</b>${a.up}</div>
          <div class="fx down"><b>▼ IF YOU LOWER / TURN OFF</b>${a.down}</div>
        </div>`;
    }

    /* ---------- Act 3: training loop + loss ---------- */
    function buildTrainLoop() {
      const cfg = MODEL.config;
      $("#train-loop").innerHTML = [
        { n: "STEP 1", t: "Forward pass", p: `A batch of ${cfg.batch_size} complaints flows through the network (exactly the journey in Act 1) producing category + urgency probabilities for each.` },
        { n: "STEP 2", t: "Loss — how wrong were we?", p: "Cross-entropy compares predictions to the true labels. Urgency errors on rare classes are amplified by the class weights below — a missed Critical hurts the most." },
        { n: "STEP 3", t: "Backpropagation", p: `The error flows backwards through both heads, the attention, the BiLSTM and the embeddings, computing a gradient for every one of the ${(MODEL.n_params || 0).toLocaleString()} weights (clipped at norm ${cfg.grad_clip}).` },
        { n: "STEP 4", t: `Update (${cfg.optimizer})`, p: `Each weight takes a small step against its gradient (lr ${cfg.lr}, weight decay ${cfg.weight_decay}). Then validate; the '${cfg.scheduler}' scheduler and early stopping (patience ${cfg.patience}) watch Critical-recall.` },
      ].map(c => `<div class="tl-card"><div class="n">${c.n}</div><h5>${c.t}</h5><p>${c.p}</p></div>`).join("");
      $("#loss-formula").innerHTML =
        `L &nbsp;=&nbsp; <span class="hl-c">${cfg.w_category} · CE( category )</span> &nbsp;+&nbsp; ` +
        `<span class="hl-u">${cfg.w_urgency} · CE( urgency, class-weights )</span>` +
        `&nbsp;&nbsp;with&nbsp; <span class="hl-r">weight(Critical) ×${cfg.critical_boost}</span>` +
        `<small>Class weights are inverse-frequency: rare classes (Critical ≈ 2.4% of data) get large weights so the optimiser cannot ignore them. This single term is why test Critical-recall reaches 1.00.</small>`;
    }

    /* ---------- Act 4: training replay ---------- */
    let replayEpochs = [], replayTimer = null;

    async function buildReplay() {
      const runs = await api("/api/admin/runs");
      const sel = $("#replay-run");
      sel.innerHTML = runs.map(r =>
        `<option value="${r.run_id}" ${MODEL.run_id === r.run_id ? "selected" : ""}>${r.run_name} — ${r.n_epochs} epochs (crit ${r.best_val_critical_recall.toFixed(2)})</option>`).join("");
      sel.onchange = () => loadReplayRun(sel.value);
      if (runs.length) loadReplayRun(sel.value || runs[0].run_id);
    }

    async function loadReplayRun(runId) {
      stopReplay();
      const d = await api("/api/admin/runs/" + runId);
      replayEpochs = d.epochs;
      const sl = $("#replay-slider");
      sl.max = replayEpochs.length; sl.value = replayEpochs.length;
      sl.oninput = () => { stopReplay(); showEpoch(+sl.value); };
      showEpoch(replayEpochs.length);
    }

    function replayComment(k) {
      const e = replayEpochs[k - 1], prev = replayEpochs[k - 2];
      const bits = [];
      if (k === 1) bits.push(`Epoch 1 — weights start ~random: loss ${e.train_loss.toFixed(2)}. Anything better than guessing is progress.`);
      else {
        const d = e.val_loss - prev.val_loss;
        if (d < -1e-4) bits.push(`Val loss fell ${prev.val_loss.toFixed(4)} → ${e.val_loss.toFixed(4)} — the model is genuinely generalising.`);
        else if (d > 1e-4) bits.push(`Val loss rose ${prev.val_loss.toFixed(4)} → ${e.val_loss.toFixed(4)} while train kept falling — an early overfitting sign the regularisers must absorb.`);
        else bits.push(`Val loss held steady at ${e.val_loss.toFixed(4)} — the model has converged on this data.`);
        if (e.lr < prev.lr) bits.push(`⚙ The plateau scheduler halved the learning rate to ${(+e.lr).toExponential(1)} — Critical-recall had stalled.`);
      }
      const best = Math.max(...replayEpochs.slice(0, k).map(x => x.val_critical_recall));
      if (e.val_critical_recall >= best && (!prev || e.val_critical_recall > Math.max(...replayEpochs.slice(0, k - 1).map(x => x.val_critical_recall))))
        bits.push(`★ New best Critical-recall ${e.val_critical_recall.toFixed(3)} — checkpoint saved.`);
      if (k === replayEpochs.length) {
        const planned = e.epochs_planned;
        bits.push(planned && k < planned
          ? `⏹ Early stopping: no Critical-recall improvement for ${e.patience} epochs — best checkpoint restored, ${planned - k} planned epochs saved.`
          : `✔ Run complete after ${k} epochs.`);
      }
      return bits.join(" ");
    }

    function showEpoch(k) {
      if (!replayEpochs.length) return;
      k = Math.max(1, Math.min(replayEpochs.length, k));
      $("#replay-slider").value = k;
      $("#replay-epoch-label").textContent = `epoch ${k}/${replayEpochs.length}`;
      const upto = replayEpochs.slice(0, k);
      const pts = key => upto.map(r => ({ x: r.epoch, y: r[key] }));
      Charts.lineChart($("#rp-loss"), [
        { name: "train", color: "#6d8dff", points: pts("train_loss") },
        { name: "val", color: "#39c6d6", points: pts("val_loss") }], { xLabel: "epoch" });
      Charts.lineChart($("#rp-qual"), [
        { name: "val F1", color: "#e9b949", points: pts("val_macro_f1") },
        { name: "Critical-recall", color: "#e0596b", points: pts("val_critical_recall") }], { xLabel: "epoch", y0: 0 });
      const last = upto[upto.length - 1], labels = ["Low", "Medium", "High", "Critical"];
      Charts.heatmap($("#rp-cm"), labels.map((_, i) => labels.map((_, j) => last[`cm_${i}_${j}`] ?? 0)),
        labels, { color: [157, 109, 255] });
      $("#replay-narration").innerHTML = `<b>Epoch ${k}:</b> ${replayComment(k)}`;
    }

    function stopReplay() { clearInterval(replayTimer); replayTimer = null; $("#replay-play").textContent = "⏵ Replay training"; }

    function playReplay() {
      if (replayTimer) { stopReplay(); return; }
      if (!replayEpochs.length) return;
      let k = 0;
      $("#replay-play").textContent = "⏸ Pause";
      replayTimer = setInterval(() => {
        k += 1; showEpoch(k);
        if (k >= replayEpochs.length) stopReplay();
      }, 1000);
    }

    /* ---------- Act 5: project flow ---------- */
    function buildProjectFlow() {
      const link = `<div class="pf-link"><svg viewBox="0 0 34 16"><line class="dashline" x1="0" y1="8" x2="34" y2="8"/></svg></div>`;
      $("#project-flow").innerHTML = [
        { ico: "🧑", b: "Citizen", p: "Describes a problem in plain language on the Jan Samadhan portal — with a live AI preview while typing." },
        { ico: "🧹", b: "Preprocessing", p: "Lower-case → tokenise → vocabulary ids → pad to max_len. Same code path in training and serving." },
        { ico: "🧠", b: "BiLSTM + Attention", p: "The trained network from Act 1 — one forward pass, ~7 ms on CPU.", cls: "model" },
        { ico: "🔀", b: "Two predictions", p: "Head A → department + CPGRAMS codes. Head B → urgency, fused with the explainable 0–100 oracle score.", cls: "out-a" },
        { ico: "🧭", b: "Escalation engine", p: "Urgency maps to routing level, response SLAs, officer escalation path and citizen next steps.", cls: "out-b" },
        { ico: "🏛️", b: "Officer queue + citizen plan", p: "Critical cases surface within 24 h; the citizen gets a reference ID, tracking and a printable acknowledgement." },
      ].map(n => `<div class="pf-node ${n.cls || ""}"><span class="ico">${n.ico}</span><b>${n.b}</b><p>${n.p}</p></div>`).join(link);
    }

    /* ---------- wiring ---------- */
    function init() {
      if (!MODEL) return;
      stepper(); narrate();
      buildAnatomy(); selectAnatomy("critical_boost");
      buildTrainLoop(); buildReplay().catch(() => {}); buildProjectFlow();
      $("#magic-load").addEventListener("click", load);
      $("#magic-input").addEventListener("keydown", e => { if (e.key === "Enter") load(); });
      $$("button[data-mex]").forEach(b => b.addEventListener("click", () => { $("#magic-input").value = b.dataset.mex; load(); }));
      $("#replay-play").addEventListener("click", playReplay);
      $("#magic-prev").addEventListener("click", () => { stopAuto(); go(step - 1); });
      $("#magic-next").addEventListener("click", () => { stopAuto(); go(step + 1); });
      $("#magic-play").addEventListener("click", async () => {
        if (autoTimer) { stopAuto(); return; }
        if (!TR) await load();
        $("#magic-play").textContent = "⏸ Pause";
        go(0);
        autoTimer = setInterval(() => { if (step >= STEPS.length - 1) stopAuto(); else go(step + 1); }, 3400);
      });
      load().catch(() => {});
    }
    return { init, refreshReplay: () => buildReplay().catch(() => {}) };
  })();

  /* ---------------- boot ---------------- */
  (async function boot() {
    try { await buildHpForm(); } catch (e) { console.error(e); }
    try { await loadOverview(); } catch (e) {
      $("#side-model").innerHTML = "⚠ No trained model yet.<br>Run: <b class='mono'>python -m src.main train</b>";
    }
    try { await loadDataset(); } catch (e) { console.error(e); }
    try { await loadRuns(); } catch (e) { console.error(e); }
    try { window.Magic = Magic; Magic.init(); } catch (e) { console.error(e); }
    runArch().catch(() => {});
    poll(); // pick up any in-flight run after a page reload
  })();
})();
