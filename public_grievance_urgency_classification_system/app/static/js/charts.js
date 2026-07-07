/* Tiny dependency-free SVG chart library used by both platforms.
   All charts render into a container element and are responsive via viewBox. */
(function (global) {
  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function niceTicks(min, max, n) {
    if (min === max) { max = min + 1; }
    const span = max - min;
    const step = Math.pow(10, Math.floor(Math.log10(span / n)));
    const err = (span / n) / step;
    const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    const s = mult * step;
    const ticks = [];
    for (let v = Math.ceil(min / s) * s; v <= max + 1e-12; v += s) ticks.push(+v.toFixed(12));
    return ticks;
  }

  function fmt(v) {
    if (Math.abs(v) >= 1000) return v.toLocaleString();
    if (Number.isInteger(v)) return String(v);
    if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(0);
    return String(+v.toFixed(3));
  }

  /* Multi-series line chart.
     series: [{name, color, points:[{x,y}], dash?}]  opts: {title, xLabel, yLabel, y0, logY} */
  function lineChart(container, series, opts = {}) {
    container.innerHTML = "";
    const W = 460, H = 260, m = { t: opts.title ? 34 : 16, r: 14, b: 40, l: 52 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" }, container);
    const pts = series.flatMap(s => s.points);
    if (!pts.length) { container.innerHTML = "<div class='chart-empty'>No data yet</div>"; return; }
    let xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    let xmin = Math.min(...xs), xmax = Math.max(...xs);
    let ymin = opts.y0 !== undefined ? opts.y0 : Math.min(...ys), ymax = Math.max(...ys);
    if (opts.logY) { ymin = Math.log10(Math.max(1e-9, Math.min(...ys))); ymax = Math.log10(Math.max(...ys)); }
    if (ymin === ymax) { ymin -= 0.5; ymax += 0.5; }
    if (xmin === xmax) { xmin -= 0.5; xmax += 0.5; }
    const pad = (ymax - ymin) * 0.06; ymin -= (opts.y0 !== undefined ? 0 : pad); ymax += pad;
    const X = x => m.l + (x - xmin) / (xmax - xmin) * (W - m.l - m.r);
    const Y = y => { const v = opts.logY ? Math.log10(Math.max(1e-9, y)) : y; return H - m.b - (v - ymin) / (ymax - ymin) * (H - m.t - m.b); };

    niceTicks(ymin, ymax, 5).forEach(t => {
      const y = H - m.b - (t - ymin) / (ymax - ymin) * (H - m.t - m.b);
      el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "grid" }, svg);
      el("text", { x: m.l - 6, y: y + 3, class: "tick", "text-anchor": "end" }, svg)
        .textContent = opts.logY ? fmt(Math.pow(10, t)) : fmt(t);
    });
    niceTicks(xmin, xmax, 6).forEach(t => {
      el("text", { x: X(t), y: H - m.b + 16, class: "tick", "text-anchor": "middle" }, svg).textContent = fmt(t);
    });
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
    el("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "axis" }, svg);
    if (opts.title) el("text", { x: m.l, y: 18, class: "chart-title" }, svg).textContent = opts.title;
    if (opts.xLabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 6, class: "axis-label", "text-anchor": "middle" }, svg).textContent = opts.xLabel;

    series.forEach(s => {
      if (!s.points.length) return;
      const d = s.points.map((p, i) => (i ? "L" : "M") + X(p.x).toFixed(1) + "," + Y(p.y).toFixed(1)).join(" ");
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2.2, "stroke-dasharray": s.dash || "none", "stroke-linejoin": "round" }, svg);
      s.points.forEach(p => {
        const c = el("circle", { cx: X(p.x), cy: Y(p.y), r: 3, fill: s.color, class: "dot" }, svg);
        el("title", {}, c).textContent = `${s.name} — ${opts.xLabel || "x"} ${p.x}: ${fmt(p.y)}`;
      });
    });
    // legend
    let lx = m.l + 4;
    series.forEach(s => {
      el("rect", { x: lx, y: m.t - 10, width: 10, height: 3, fill: s.color, rx: 1.5 }, svg);
      const t = el("text", { x: lx + 14, y: m.t - 5, class: "legend" }, svg);
      t.textContent = s.name;
      lx += 22 + s.name.length * 6.2;
    });
  }

  /* Horizontal or vertical bar chart. data: [{label, value, color?}] */
  function barChart(container, data, opts = {}) {
    container.innerHTML = "";
    const horizontal = opts.horizontal;
    const W = 460, H = opts.height || (horizontal ? Math.max(120, data.length * 30 + 50) : 260);
    const m = { t: opts.title ? 32 : 14, r: 16, b: horizontal ? 26 : 62, l: horizontal ? 130 : 52 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" }, container);
    const vmax = opts.max !== undefined ? opts.max : Math.max(...data.map(d => d.value), 1e-9) * 1.08;
    if (opts.title) el("text", { x: m.l, y: 18, class: "chart-title" }, svg).textContent = opts.title;

    if (horizontal) {
      const bh = (H - m.t - m.b) / data.length;
      data.forEach((d, i) => {
        const y = m.t + i * bh;
        const w = (d.value / vmax) * (W - m.l - m.r);
        el("rect", { x: m.l, y: y + bh * 0.15, width: Math.max(w, 0.5), height: bh * 0.7, rx: 3, fill: d.color || "#5B8DEF" }, svg);
        el("text", { x: m.l - 6, y: y + bh / 2 + 4, class: "tick", "text-anchor": "end" }, svg).textContent = d.label;
        el("text", { x: m.l + w + 5, y: y + bh / 2 + 4, class: "bar-val" }, svg).textContent = opts.format ? opts.format(d.value) : fmt(d.value);
      });
    } else {
      const bw = (W - m.l - m.r) / data.length;
      niceTicks(0, vmax, 4).forEach(t => {
        const y = H - m.b - (t / vmax) * (H - m.t - m.b);
        el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "grid" }, svg);
        el("text", { x: m.l - 6, y: y + 3, class: "tick", "text-anchor": "end" }, svg).textContent = fmt(t);
      });
      data.forEach((d, i) => {
        const h = (d.value / vmax) * (H - m.t - m.b);
        const x = m.l + i * bw;
        const r = el("rect", { x: x + bw * 0.12, y: H - m.b - h, width: bw * 0.76, height: Math.max(h, 0.5), rx: 3, fill: d.color || "#5B8DEF" }, svg);
        el("title", {}, r).textContent = `${d.label}: ${fmt(d.value)}`;
        el("text", { x: x + bw / 2, y: H - m.b - h - 5, class: "bar-val", "text-anchor": "middle" }, svg).textContent = opts.format ? opts.format(d.value) : fmt(d.value);
        const lbl = el("text", { x: x + bw / 2, y: H - m.b + 14, class: "tick", "text-anchor": "middle" }, svg);
        lbl.textContent = d.label.length > 11 ? d.label.slice(0, 10) + "…" : d.label;
        if (data.length > 6) lbl.setAttribute("transform", `rotate(-28 ${x + bw / 2} ${H - m.b + 14})`);
      });
      el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
    }
  }

  /* Grouped bar chart: data: [{label, values:[..]}], seriesNames, colors */
  function groupedBarChart(container, data, seriesNames, colors, opts = {}) {
    container.innerHTML = "";
    const W = 460, H = 260, m = { t: opts.title ? 34 : 20, r: 14, b: 44, l: 46 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" }, container);
    const vmax = opts.max || Math.max(...data.flatMap(d => d.values), 1e-9) * 1.1;
    if (opts.title) el("text", { x: m.l, y: 18, class: "chart-title" }, svg).textContent = opts.title;
    niceTicks(0, vmax, 4).forEach(t => {
      const y = H - m.b - (t / vmax) * (H - m.t - m.b);
      el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "grid" }, svg);
      el("text", { x: m.l - 6, y: y + 3, class: "tick", "text-anchor": "end" }, svg).textContent = fmt(t);
    });
    const gw = (W - m.l - m.r) / data.length;
    data.forEach((d, i) => {
      const bw = gw * 0.8 / d.values.length;
      d.values.forEach((v, j) => {
        const h = (v / vmax) * (H - m.t - m.b);
        const x = m.l + i * gw + gw * 0.1 + j * bw;
        const r = el("rect", { x, y: H - m.b - h, width: bw * 0.9, height: Math.max(h, 0.5), rx: 2, fill: colors[j] }, svg);
        el("title", {}, r).textContent = `${d.label} · ${seriesNames[j]}: ${fmt(v)}`;
      });
      el("text", { x: m.l + i * gw + gw / 2, y: H - m.b + 15, class: "tick", "text-anchor": "middle" }, svg).textContent = d.label;
    });
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
    let lx = m.l + 4;
    seriesNames.forEach((n, j) => {
      el("rect", { x: lx, y: m.t - 12, width: 9, height: 9, rx: 2, fill: colors[j] }, svg);
      el("text", { x: lx + 13, y: m.t - 4, class: "legend" }, svg).textContent = n;
      lx += 20 + n.length * 6.2;
    });
  }

  /* Confusion-matrix heatmap. matrix: 2-D array; labels: axis labels. */
  function heatmap(container, matrix, labels, opts = {}) {
    container.innerHTML = "";
    const n = matrix.length;
    const cell = 52, m = { t: opts.title ? 56 : 34, r: 12, b: 22, l: 66 };
    const W = m.l + n * cell + m.r, H = m.t + n * cell + m.b;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" }, container);
    if (opts.title) el("text", { x: m.l, y: 18, class: "chart-title" }, svg).textContent = opts.title;
    const rowMax = matrix.map(r => Math.max(...r, 1e-9));
    const base = opts.color || [37, 99, 235]; // blue
    matrix.forEach((row, i) => row.forEach((v, j) => {
      const frac = opts.normalized ? v : v / rowMax[i];
      const a = 0.06 + 0.94 * Math.min(1, frac);
      const rect = el("rect", {
        x: m.l + j * cell, y: m.t + i * cell, width: cell - 3, height: cell - 3, rx: 5,
        fill: `rgba(${base[0]},${base[1]},${base[2]},${a.toFixed(3)})`,
        class: i === j ? "hm-diag" : (i === n - 1 ? "" : "")
      }, svg);
      el("title", {}, rect).textContent = `true ${labels[i]} → predicted ${labels[j]}: ${opts.normalized ? (v * 100).toFixed(1) + "%" : v}`;
      el("text", {
        x: m.l + j * cell + (cell - 3) / 2, y: m.t + i * cell + cell / 2 + 3,
        "text-anchor": "middle", class: "hm-val",
        fill: a > 0.55 ? "#fff" : "currentColor"
      }, svg).textContent = opts.normalized ? (v * 100).toFixed(0) + "%" : fmt(v);
    }));
    labels.forEach((l, i) => {
      el("text", { x: m.l - 8, y: m.t + i * cell + cell / 2 + 3, "text-anchor": "end", class: "tick" }, svg).textContent = l;
      el("text", { x: m.l + i * cell + cell / 2, y: m.t - 8, "text-anchor": "middle", class: "tick" }, svg).textContent = l;
    });
    el("text", { x: m.l - 8, y: m.t - 22, "text-anchor": "end", class: "axis-label" }, svg).textContent = "true ↓";
    el("text", { x: m.l, y: H - 4, class: "axis-label" }, svg).textContent = "predicted →";
  }

  /* Semi-circular gauge for the 0-100 urgency score. */
  function gauge(container, value, label, opts = {}) {
    container.innerHTML = "";
    const W = 220, H = 132, cx = W / 2, cy = 112, r = 84;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart gauge" }, container);
    const segs = [
      { from: 0, to: 25, color: "#2E9E5B" }, { from: 25, to: 45, color: "#E9B949" },
      { from: 45, to: 70, color: "#E07B39" }, { from: 70, to: 100, color: "#D64545" },
    ];
    const ang = v => Math.PI * (1 + v / 100);
    const arc = (a0, a1, rad) => {
      const x0 = cx + rad * Math.cos(a0), y0 = cy + rad * Math.sin(a0);
      const x1 = cx + rad * Math.cos(a1), y1 = cy + rad * Math.sin(a1);
      return `M ${x0} ${y0} A ${rad} ${rad} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${x1} ${y1}`;
    };
    segs.forEach(s => el("path", { d: arc(ang(s.from) + 0.015, ang(s.to) - 0.015, r), stroke: s.color, "stroke-width": 15, fill: "none", "stroke-linecap": "round", opacity: 0.32 }, svg));
    const v = Math.max(0, Math.min(100, value));
    const seg = segs.find(s => v <= s.to) || segs[3];
    el("path", { d: arc(ang(0), ang(Math.max(v, 1.5)), r), stroke: seg.color, "stroke-width": 15, fill: "none", "stroke-linecap": "round" }, svg);
    const na = ang(v);
    el("line", { x1: cx + (r - 26) * Math.cos(na), y1: cy + (r - 26) * Math.sin(na), x2: cx + (r - 8) * Math.cos(na), y2: cy + (r - 8) * Math.sin(na), stroke: "currentColor", "stroke-width": 2.5, "stroke-linecap": "round" }, svg);
    el("text", { x: cx, y: cy - 22, "text-anchor": "middle", class: "gauge-value", fill: seg.color }, svg).textContent = Math.round(v);
    el("text", { x: cx, y: cy - 4, "text-anchor": "middle", class: "gauge-label" }, svg).textContent = label || "urgency score";
    el("text", { x: cx - r - 2, y: cy + 14, "text-anchor": "middle", class: "tick" }, svg).textContent = "0";
    el("text", { x: cx + r + 2, y: cy + 14, "text-anchor": "middle", class: "tick" }, svg).textContent = "100";
  }

  /* Probability bars: data {label: prob} with per-label colors. */
  function probBars(container, probs, colors, opts = {}) {
    container.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "prob-bars";
    Object.entries(probs).forEach(([label, p]) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `<span class="prob-label">${label}</span>
        <span class="prob-track"><span class="prob-fill" style="width:${(p * 100).toFixed(1)}%;background:${(colors && colors[label]) || "#5B8DEF"}"></span></span>
        <span class="prob-pct">${(p * 100).toFixed(1)}%</span>`;
      wrap.appendChild(row);
    });
    container.appendChild(wrap);
  }

  global.Charts = { lineChart, barChart, groupedBarChart, heatmap, gauge, probBars };
})(window);
