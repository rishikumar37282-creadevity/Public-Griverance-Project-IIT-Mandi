/* Jan Samadhan citizen portal logic */
(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const URGENCY_COLORS = { Low: "#2e9e5b", Medium: "#e9b949", High: "#e07b39", Critical: "#d64545" };

  const SAMPLES = [
    "My elderly mother's pension has been stopped for 8 months. I have complained repeatedly but there is no response. Medical bills are unpaid and the situation is urgent.",
    "No clean water supply in our colony for 12 days, children are falling sick and one neighbour was taken to hospital. The situation is unsafe and dangerous.",
    "Suggestion to add more counters at the post office and improve the website for booking appointments.",
  ];

  async function api(path, body) {
    const res = await fetch(path, body ? {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    } : undefined);
    if (!res.ok) {
      const detail = (await res.json().catch(() => ({}))).detail || res.statusText;
      throw new Error(detail);
    }
    return res.json();
  }

  /* ---------------- tabs ---------------- */
  $$("nav.tabs button").forEach(btn => btn.addEventListener("click", () => {
    $$("nav.tabs button").forEach(b => b.classList.toggle("active", b === btn));
    $$(".tab-panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab));
  }));

  /* ---------------- header stats + departments ---------------- */
  api("/api/stats").then(s => { $("#chip-total").textContent = s.total.toLocaleString(); }).catch(() => {});
  api("/api/departments").then(depts => {
    $("#chip-depts").textContent = depts.length;
    $("#dept-grid").innerHTML = depts.map(d => `
      <div class="dept-card"><b>${d.department}</b>
        <div>Category code <span class="code-tag">${d.category_code}</span></div>
        <div style="margin-top:4px">Org code <span class="code-tag">${d.org_code}</span></div>
      </div>`).join("");
  }).catch(() => {});

  /* ---------------- composer: live preview ---------------- */
  const textEl = $("#complaint-text");
  const submitBtn = $("#submit-btn");
  let debounceTimer = null, lastPreviewText = "";

  textEl.addEventListener("input", () => {
    $("#char-count").textContent = textEl.value.length;
    submitBtn.disabled = textEl.value.trim().split(/\s+/).filter(Boolean).length < 3;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(updatePreview, 550);
  });

  $$("button[data-sample]").forEach(b => b.addEventListener("click", () => {
    textEl.value = SAMPLES[+b.dataset.sample];
    textEl.dispatchEvent(new Event("input"));
  }));

  /* voice input (graceful fallback) */
  const micBtn = $("#mic-btn");
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) micBtn.style.display = "none";
  else {
    let rec = null;
    micBtn.addEventListener("click", () => {
      if (rec) { rec.stop(); return; }
      rec = new SR();
      rec.lang = "en-IN"; rec.interimResults = false; rec.continuous = true;
      micBtn.classList.add("active"); micBtn.textContent = "⏹ Stop recording";
      rec.onresult = e => {
        textEl.value = (textEl.value + " " + [...e.results].map(r => r[0].transcript).join(" ")).trim();
        textEl.dispatchEvent(new Event("input"));
      };
      rec.onend = () => { micBtn.classList.remove("active"); micBtn.textContent = "🎙️ Speak instead of typing"; rec = null; };
      rec.start();
    });
  }

  async function updatePreview() {
    const text = textEl.value.trim();
    const body = $("#preview-body");
    if (text.split(/\s+/).filter(Boolean).length < 4 || text === lastPreviewText) return;
    lastPreviewText = text;
    try {
      const r = await api("/api/analyze", { text });
      body.innerHTML = `
        <div style="text-align:center;margin-bottom:4px">
          <span class="urgency-pill u-${r.urgency}"><i></i>${r.urgency} urgency</span>
        </div>
        <div id="pv-gauge"></div>
        <div class="preview-cat">
          <span>Routed to <b>${r.department}</b></span>
          <span class="conf-tag">${(r.category_confidence * 100).toFixed(0)}% confident</span>
        </div>
        <div id="pv-probs"></div>
        <ul class="signal-list">
          <li><span>Severity keywords found</span><b>${r.matched_signals.lexicon.slice(0, 4).join(", ") || "none"}</b></li>
          <li><span>Vulnerable group mentioned</span><b>${r.matched_signals.vulnerable.join(", ") || "no"}</b></li>
          <li><span>Time-pending signal</span><b>+${r.score_breakdown.temporal} pts</b></li>
        </ul>`;
      Charts.gauge($("#pv-gauge"), r.urgency_score, "urgency score (0–100)");
      Charts.probBars($("#pv-probs"), r.urgency_probs, URGENCY_COLORS);
    } catch (e) { /* keep last preview on transient errors */ }
  }

  /* ---------------- submit + result ---------------- */
  submitBtn.addEventListener("click", async () => {
    submitBtn.disabled = true;
    submitBtn.innerHTML = "Analysing with BiLSTM + Attention<span class='loading-dots'></span>";
    try {
      const r = await api("/api/classify", {
        text: textEl.value.trim(),
        name: $("#f-name").value.trim(),
        location: $("#f-location").value.trim(),
        contact: $("#f-contact").value.trim(),
      });
      renderResult(r);
    } catch (e) {
      alert("Could not submit: " + e.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Submit grievance for AI triage →";
    }
  });

  function attentionHTML(tokens, weights) {
    return tokens.map((t, i) => {
      const w = weights[i] || 0;
      const bg = `rgba(226,84,58,${(0.06 + 0.6 * w).toFixed(3)})`;
      const bold = w > 0.65 ? "font-weight:700;" : "";
      return `<span class="attn-tok" style="background:${bg};${bold}" title="attention ${(w * 100).toFixed(0)}%">${t}</span>`;
    }).join(" ");
  }

  function renderResult(r) {
    const c = r.classification, plan = r.action_plan;
    $("#compose-area").style.display = "none";
    const area = $("#result-area");
    area.style.display = "block";
    area.innerHTML = `
      <div class="result-head">
        <svg class="ok-ico" viewBox="0 0 24 24" fill="none" stroke="#7ee2a8" stroke-width="2"><circle cx="12" cy="12" r="10" opacity=".4"/><path d="m8 12.5 2.7 2.7L16.5 9"/></svg>
        <div>
          <h2>Grievance registered successfully</h2>
          <p>Filed ${r.filed_at.replace("T", " at ")} · AI triage complete · queued for officer review</p>
        </div>
        <div class="ref-box"><div class="lbl">Reference ID</div><div class="ref" id="ref-id">${r.reference_id}</div></div>
      </div>

      <div class="result-grid">
        <div class="card">
          <h3>🏛️ Routing decision</h3>
          <p class="sub">Where your complaint is going, and why.</p>
          <dl class="kv">
            <dt>Department</dt><dd>${c.department}</dd>
            <dt>Confidence</dt><dd>${(c.category_confidence * 100).toFixed(1)}% ${c.top_categories.length > 1 ? `<span class="conf-tag">(next: ${c.top_categories[1].category} ${(c.top_categories[1].prob * 100).toFixed(0)}%)</span>` : ""}</dd>
            <dt>CPGRAMS codes</dt><dd><span class="code-tag">${c.category_code}</span> <span class="code-tag">${c.org_code}</span></dd>
            <dt>Routing level</dt><dd>${plan.routing_level}</dd>
          </dl>
          ${r.subcategories.length ? `<p class="sub" style="margin:16px 0 2px"><b>Suggested specific sub-category</b> (from the official CPGRAMS tree):</p>` +
            r.subcategories.map(s => `<div class="subcat">${s.description}
              <div class="st">code ${s.code} · level ${s.stage} · matched: ${s.matched.join(", ")}</div></div>`).join("") : ""}
        </div>

        <div class="card">
          <h3>🚦 Urgency assessment</h3>
          <p class="sub">Neural prediction + explainable 0–100 score.</p>
          <div style="text-align:center"><span class="urgency-pill u-${c.urgency}"><i></i>${c.urgency} — ${(c.urgency_confidence * 100).toFixed(1)}% confident</span></div>
          <div id="res-gauge"></div>
          <div id="res-probs"></div>
          <ul class="signal-list" style="margin-top:14px">
            ${Object.entries(c.score_breakdown).map(([k, v]) =>
              `<li><span>${{ category: "Category severity", lexicon: "Severity keywords", temporal: "Time pending", vulnerable: "Vulnerable group", structural: "Prior escalations" }[k] || k}</span><b>+${v} pts</b></li>`).join("")}
          </ul>
        </div>

        <div class="card full">
          <h3>🔍 Why the AI decided this</h3>
          <p class="sub">Darker words carried more weight in the model's attention layer.</p>
          <div class="attn-text">${attentionHTML(c.tokens, c.attention_norm)}</div>
          <div class="attn-legend">low influence <span class="attn-scale"></span> high influence
            &nbsp;·&nbsp; keywords detected: ${[...c.matched_signals.lexicon, ...c.matched_signals.vulnerable].map(k => `<span class="code-tag">${k}</span>`).join(" ") || "—"}</div>
        </div>

        <div class="card">
          <h3>📋 Your next steps</h3>
          <ol class="timeline">${plan.citizen_steps.map(s => `<li>${s}</li>`).join("")}</ol>
          <div class="sla-chips">
            <div class="sla-chip">First response<b>${plan.first_response}</b></div>
            <div class="sla-chip">Resolution target<b>${plan.resolution_target}</b></div>
            <div class="sla-chip">Priority rank<b>#${plan.priority_rank} of 4</b></div>
          </div>
        </div>

        <div class="card">
          <h3>📈 Escalation path (officer side)</h3>
          <p class="sub">${plan.note}</p>
          <ol class="esc-path">${plan.escalation_path.map(s => `<li>${s}</li>`).join("")}</ol>
          <div class="note-banner">ℹ️ <span>If deadlines pass without action, quote <b>${r.reference_id}</b> on the Track page to trigger escalation.</span></div>
        </div>
      </div>

      <div class="result-actions">
        <button class="btn-ghost" onclick="navigator.clipboard.writeText('${r.reference_id}').then(()=>this.textContent='✓ Copied')">📋 Copy reference ID</button>
        <button class="btn-ghost" onclick="window.print()">🖨️ Print acknowledgement</button>
        <button class="btn-ghost" id="new-complaint">＋ File another grievance</button>
      </div>`;

    Charts.gauge($("#res-gauge"), c.urgency_score, "urgency score (0–100)");
    Charts.probBars($("#res-probs"), c.urgency_probs, URGENCY_COLORS);
    $("#new-complaint").addEventListener("click", () => {
      area.style.display = "none"; area.innerHTML = "";
      $("#compose-area").style.display = "";
      textEl.value = ""; lastPreviewText = "";
      textEl.dispatchEvent(new Event("input"));
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    api("/api/stats").then(s => { $("#chip-total").textContent = s.total.toLocaleString(); }).catch(() => {});
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  /* ---------------- track ---------------- */
  $("#track-btn").addEventListener("click", track);
  $("#track-input").addEventListener("keydown", e => { if (e.key === "Enter") track(); });

  async function track() {
    const ref = $("#track-input").value.trim().toUpperCase();
    const out = $("#track-result");
    if (!ref) return;
    out.innerHTML = "<p class='sub loading-dots'>Looking up</p>";
    try {
      const r = await api("/api/complaints/" + encodeURIComponent(ref));
      out.innerHTML = `
        <div style="margin-top:18px">
          <span class="status-badge">● ${r.status}</span>
          <dl class="kv" style="margin-top:14px">
            <dt>Reference</dt><dd>${r.reference_id}</dd>
            <dt>Filed</dt><dd>${r.filed_at.replace("T", " at ")}</dd>
            <dt>Department</dt><dd>${r.department} <span class="code-tag">${r.org_code}</span></dd>
            <dt>Urgency</dt><dd><span class="urgency-pill u-${r.urgency}" style="padding:2px 10px"><i></i>${r.urgency}</span> (score ${r.urgency_score}/100)</dd>
            <dt>First response due</dt><dd>${r.first_response_due}</dd>
            <dt>Resolution target</dt><dd>${r.resolution_target}</dd>
          </dl>
          <p class="sub" style="margin-top:14px"><b>Complaint:</b> ${r.text}</p>
        </div>`;
    } catch (e) {
      out.innerHTML = `<p class="error-text">✖ ${e.message}</p>`;
    }
  }
})();
