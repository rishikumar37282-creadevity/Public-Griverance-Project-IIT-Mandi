/* Project Story deck — navigation, reveal animations, real-data charts */
(function () {
  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];
  const slides = $$(".slide");
  let cur = 0;
  const drawn = new Set();

  /* ---------------- navigation ---------------- */
  const dots = $("#dk-dots");
  slides.forEach((s, i) => {
    const d = document.createElement("i");
    d.title = s.dataset.title || `Slide ${i + 1}`;
    d.addEventListener("click", () => go(i));
    dots.appendChild(d);
  });

  function go(i) {
    cur = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((s, k) => {
      s.classList.toggle("active", k === cur);
      s.classList.toggle("prev", k < cur);
    });
    $$("#dk-dots i").forEach((d, k) => d.classList.toggle("on", k === cur));
    $("#dk-count").textContent = `${cur + 1} / ${slides.length}`;
    $("#dk-bar").style.width = ((cur + 1) / slides.length * 100) + "%";
    onSlide(cur);
  }
  $("#dk-prev").addEventListener("click", () => go(cur - 1));
  $("#dk-next").addEventListener("click", () => go(cur + 1));
  document.addEventListener("keydown", e => {
    if (e.key === "ArrowRight" || e.key === " " || e.key === "PageDown") go(cur + 1);
    if (e.key === "ArrowLeft" || e.key === "PageUp") go(cur - 1);
    if (e.key === "Home") go(0);
    if (e.key === "End") go(slides.length - 1);
  });
  // touch swipe
  let tx = null;
  document.addEventListener("touchstart", e => tx = e.touches[0].clientX, { passive: true });
  document.addEventListener("touchend", e => {
    if (tx === null) return;
    const dx = e.changedTouches[0].clientX - tx;
    if (Math.abs(dx) > 60) go(cur + (dx < 0 ? 1 : -1));
    tx = null;
  }, { passive: true });

  /* ---------------- count-up animation ---------------- */
  function countUp(elm) {
    const to = parseFloat(elm.dataset.to), dec = +elm.dataset.dec;
    const t0 = performance.now(), dur = 1400;
    function tick(t) {
      const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      elm.textContent = (to * e).toFixed(dec);
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ---------------- real-data charts (lazy per slide) ---------------- */
  let MODEL = null;
  async function fetchModel() {
    if (MODEL) return MODEL;
    const res = await fetch("/api/admin/model");
    if (res.ok) MODEL = await res.json();
    return MODEL;
  }

  async function onSlide(i) {
    const s = slides[i];
    // count-ups restart each visit
    s.querySelectorAll(".countup").forEach(countUp);

    const title = s.dataset.title;
    if (drawn.has(title)) return;

    if (title === "Watching it learn") {
      drawn.add(title);
      try {
        const m = await fetchModel();
        const h = m.history || [];
        if (h.length) {
          const pts = k => h.map(r => ({ x: r.epoch, y: r[k] }));
          Charts.lineChart($("#s-loss"), [
            { name: "train loss", color: "#6d8dff", points: pts("train_loss") },
            { name: "val loss", color: "#39c6d6", points: pts("val_loss") }], { xLabel: "epoch" });
          Charts.lineChart($("#s-qual"), [
            { name: "val macro-F1", color: "#e9b949", points: pts("val_macro_f1") },
            { name: "val Critical-recall", color: "#e0596b", points: pts("val_critical_recall") }], { xLabel: "epoch", y0: 0 });
          $("#s-learn-note").innerHTML = `<b>How to read it:</b> run <b>${m.run_name}</b> trained ${h.length} epochs
            (early stopping); val loss bottomed at ${Math.min(...h.map(r => r.val_loss)).toFixed(3)} and the kept checkpoint
            has val Critical-recall ${Math.max(...h.map(r => r.val_critical_recall)).toFixed(3)} with macro-F1
            ${Math.max(...h.map(r => r.val_macro_f1)).toFixed(3)}.`;
        }
      } catch (e) { /* charts keep their placeholder */ }
    }

    if (title === "Results") {
      drawn.add(title);
      try {
        const m = await fetchModel();
        const t = m.test_metrics;
        if (t && t.confusion_matrix) {
          Charts.heatmap($("#s-cm"), t.confusion_matrix, t.per_class.map(c => c.label), { color: [109, 141, 255] });
        }
      } catch (e) { /* keep placeholder */ }
    }
  }

  go(0);
})();
