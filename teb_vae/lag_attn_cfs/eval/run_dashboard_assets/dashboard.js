/* The run dashboard. The payload holds finished Plotly figures, so this script only lays the page
   out, picks the variant each selector asks for, and applies the colour theme. */
(function () {
  "use strict";
  const RUN = window.__RUN__;
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, html) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (html != null) node.innerHTML = html;
    return node;
  };
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const state = { section: null, population: RUN.default_population, local: {}, search: "" };
  const rendered = new Map();

  // ---------------------------------------------------------------------------------- decoding
  function decode(packed) {
    // {d: shape, o: offset, s: scale, b: base64 int16}; the sentinel -32768 is a missing value.
    const raw = atob(packed.b);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const ints = new Int16Array(bytes.buffer);
    const out = new Array(ints.length);
    for (let i = 0; i < ints.length; i++) out[i] = ints[i] === -32768 ? null : packed.o + packed.s * ints[i];
    return out;
  }
  function rows(flat, width) {
    const out = [];
    for (let i = 0; i < flat.length; i += width) out.push(flat.slice(i, i + width));
    return out;
  }

  // ------------------------------------------------------------------------------------- theme
  function themed(layout) {
    const text = css("--text-primary"), muted = css("--text-secondary");
    const grid = css("--plot-grid"), line = css("--border");
    const out = Object.assign({}, layout);
    out.paper_bgcolor = "rgba(0,0,0,0)";
    out.plot_bgcolor = "rgba(0,0,0,0)";
    out.font = Object.assign({ family: 'Inter, "Segoe UI", Arial, sans-serif', size: 12, color: text }, layout.font || {});
    out.margin = Object.assign({ l: 62, r: 22, t: 26, b: 48 }, layout.margin || {});
    out.hoverlabel = { bgcolor: css("--surface-1"), bordercolor: line, font: { color: text, size: 12 } };
    out.legend = Object.assign({ orientation: "h", y: 1.02, yanchor: "bottom", x: 0, font: { size: 11.5, color: muted }, bgcolor: "rgba(0,0,0,0)" }, layout.legend || {});
    Object.keys(out).filter((key) => /^[xy]axis\d*$/.test(key)).forEach((key) => {
      out[key] = Object.assign({ gridcolor: grid, zerolinecolor: line, linecolor: line, tickcolor: line, automargin: true, title: {} }, out[key]);
      if (typeof out[key].title === "string") out[key].title = { text: out[key].title };
      out[key].title = Object.assign({ font: { size: 12, color: muted }, standoff: 6 }, out[key].title);
    });
    (out.annotations || []).forEach((note) => { note.font = Object.assign({ color: muted, size: 11.5 }, note.font || {}); });
    return out;
  }
  const CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"], toImageButtonOptions: { format: "png", scale: 2 } };

  // ---------------------------------------------------------------------------------- variants
  function pickVariant(panel) {
    const variants = panel.variants || [];
    const wantsPop = variants.some((v) => v.pop != null);
    const wantsOpt = variants.some((v) => v.opt != null);
    const opt = state.local[panel.id] != null ? state.local[panel.id] : (wantsOpt ? variants.find((v) => v.opt != null).opt : null);
    // A plot that follows the Anchors filter never falls back to another population: a run with
    // no anchor in the selected population has no variant for it, and the card then says so.
    return variants.find((v) => (!wantsPop || v.pop === state.population) && (!wantsOpt || v.opt === opt)) || null;
  }
  function drawPlot(panel, holder) {
    const variant = pickVariant(panel);
    if (!variant) {
      Plotly.purge(holder);
      const population = RUN.populations.find((pop) => pop.key === state.population);
      holder.innerHTML = "<div class='missing'>This run has no data for this plot on the population '"
        + (population ? population.label : state.population) + "'. Choose another entry of the Anchors filter.</div>";
      holder.dataset.empty = "1";
      return;
    }
    if (holder.dataset.empty) { holder.innerHTML = ""; delete holder.dataset.empty; }
    const layout = themed(Object.assign({ height: panel.height || 380 }, variant.layout || {}));
    Plotly.react(holder, variant.data, layout, CONFIG);
  }

  // ------------------------------------------------------------------------------------ panels
  function explainBlock(panel) {
    const box = el("div", "explain");
    const text = panel.explain || {};
    if (text.run) box.appendChild(el("div", "run-note", "<h4>In this run</h4><p>" + text.run + "</p>"));
    [["how", "How it is made"], ["shows", "What it shows"], ["read", "How to read it"]].forEach(([key, heading]) => {
      if (text[key]) box.appendChild(el("div", null, "<h4>" + heading + "</h4><p>" + text[key] + "</p>"));
    });
    return box;
  }
  function footer(panel) {
    const links = (items, label) => items && items.length
      ? "<span><span class='lab'>" + label + "</span> " + items.map((item) => "<a href='" + encodeURI(item) + "' target='_blank' rel='noopener'>" + item.split("/").pop() + "</a>").join(", ") + "</span>"
      : "";
    const html = links(panel.pdfs, "PDF") + links(panel.sources, "Data");
    return html ? el("div", "card-foot", html) : null;
  }
  function header(panel) {
    const head = el("div", "card-head");
    head.appendChild(el("h2", null, panel.title));
    const variants = panel.variants || [];
    if (panel.kind === "plot" || panel.kind === "explorer") {
      const followsPop = variants.some((v) => v.pop != null);
      const scope = panel.kind === "explorer" ? "single segments, own controls" : (panel.scope || "all anchors only");
      head.appendChild(el("span", "badge" + (followsPop ? " pop" : ""), followsPop ? "follows the Anchors filter" : scope));
    }
    const options = [...new Set(variants.filter((v) => v.opt != null).map((v) => v.opt))];
    if (options.length > 1) {
      const wrap = el("label", "local", (panel.opt_label || "View") + " ");
      const select = el("select");
      options.forEach((name) => { const o = el("option", null, name); o.value = name; select.appendChild(o); });
      select.value = state.local[panel.id] != null ? state.local[panel.id] : options[0];
      select.addEventListener("change", () => { state.local[panel.id] = select.value; redraw(panel.id); });
      wrap.appendChild(select);
      head.appendChild(wrap);
    }
    return head;
  }
  function tableNode(panel) {
    const wrap = el("div", "table-wrap");
    const head = "<thead><tr>" + panel.columns.map((c) => "<th>" + c + "</th>").join("") + "</tr></thead>";
    const body = panel.rows.map((row) => "<tr>" + row.map((cell) => {
      const numeric = typeof cell === "number";
      const shown = cell == null ? "n/a" : numeric ? (Number.isInteger(cell) ? cell.toLocaleString("en-US") : Math.abs(cell) >= 1000 ? cell.toFixed(1) : cell.toPrecision(4)) : cell;
      return "<td" + (numeric ? " class='num'" : "") + ">" + shown + "</td>";
    }).join("") + "</tr>").join("");
    wrap.innerHTML = "<table class='data'>" + head + "<tbody>" + body + "</tbody></table>";
    return wrap;
  }
  function tilesNode(panel) {
    const grid = el("section", "tiles");
    panel.tiles.forEach((tile) => {
      grid.appendChild(el("div", "tile " + (tile.tone || ""), "<div class='name'>" + tile.name + "</div><div class='value'>" + tile.value + "</div>" + (tile.note ? "<div class='note'>" + tile.note + "</div>" : "")));
    });
    return grid;
  }

  // ---------------------------------------------------------------------------------- explorer
  const explorerCache = new Map();
  function explorerSegment(panel, index) {
    if (!explorerCache.has(index)) {
      const seg = panel.segments[index];
      const cached = { meta: seg };
      ["fhr", "ua", "gate", "kl", "gain"].forEach((name) => { cached[name] = decode(seg[name]); });
      ["kl_map", "attention_map"].forEach((name) => { cached[name] = seg[name] ? rows(decode(seg[name]), panel.lag_s.length) : null; });
      explorerCache.set(index, cached);
    }
    return explorerCache.get(index);
  }
  function drawExplorer(panel, holder, tools) {
    const index = +tools.querySelector(".ex-seg").value;
    const onlyOn = tools.querySelector(".ex-only").checked;
    const mapName = tools.querySelector(".ex-map").value;
    const seg = explorerSegment(panel, index);
    const on = seg.gate.map((g) => g != null && g >= panel.gate_shut_below);
    const rawT = seg.fhr.map((_, i) => i * panel.raw_dt);
    const anchorT = seg.meta.anchors.map((a) => a * panel.dt);
    // The gate of the raw sample at time t is the gate of the anchor whose step holds t.
    const first = seg.meta.anchors[0];
    const onAt = (t) => { const k = Math.floor(t / panel.dt) - first; return k >= 0 && k < on.length ? on[k] : false; };
    const mask = (values, times) => onlyOn ? values.map((v, i) => (onAt(times[i]) ? v : null)) : values;
    const maskA = (values) => onlyOn ? values.map((v, i) => (on[i] ? v : null)) : values;
    const shapes = [];
    let start = null;
    on.forEach((flag, i) => {
      if (flag && start == null) start = anchorT[i];
      if ((!flag || i === on.length - 1) && start != null) {
        const stop = flag ? anchorT[i] + panel.dt : anchorT[i];
        shapes.push({ type: "rect", xref: "x", yref: "paper", x0: start, x1: stop, y0: 0, y1: 1, fillcolor: "rgba(42,120,214,.11)", line: { width: 0 }, layer: "below" });
        start = null;
      }
    });
    (seg.meta.contractions || []).forEach((span) => {
      shapes.push({ type: "rect", xref: "x", yref: "y2 domain", x0: span[0], x1: span[1], y0: 0, y1: 1, fillcolor: "rgba(27,175,122,.20)", line: { width: 0 }, layer: "below" });
    });
    shapes.push({ type: "line", xref: "x", yref: "paper", x0: anchorT[0], x1: anchorT[0], y0: 0, y1: 1, line: { color: css("--text-muted"), width: 1, dash: "dot" } });
    const z = seg[mapName] ? seg[mapName].map((row, i) => (onlyOn && !on[i] ? row.map(() => null) : row)) : null;
    const zT = z ? panel.lag_s.map((_, l) => z.map((row) => row[l])) : null;
    const data = [
      { x: rawT, y: mask(seg.fhr, rawT), name: "FHR", yaxis: "y", line: { color: "#2a78d6", width: 1.8 }, hovertemplate: "%{y:.1f}<extra>FHR</extra>" },
      { x: rawT, y: mask(seg.ua, rawT), name: "UA", yaxis: "y2", line: { color: "#1baf7a", width: 1.8 }, hovertemplate: "%{y:.1f}<extra>UA</extra>" },
      { x: anchorT, y: seg.gate, name: "gate g", yaxis: "y3", line: { color: "#6250d6", width: 2.4 }, fill: "tozeroy", fillcolor: "rgba(98,80,214,.12)", hovertemplate: "%{y:.2f}<extra>gate</extra>" },
      { x: anchorT, y: maskA(seg.kl), name: "KL (nats)", yaxis: "y4", line: { color: "#e4572e", width: 2.2 }, hovertemplate: "%{y:.2f}<extra>KL</extra>" },
      { x: anchorT, y: maskA(seg.gain), name: "forecast gain (nats)", yaxis: "y5", line: { color: "#eda100", width: 2.2 }, hovertemplate: "%{y:.1f}<extra>gain</extra>" },
    ];
    if (zT) data.push({ type: "heatmap", x: anchorT, y: panel.lag_s, z: zT, yaxis: "y6", colorscale: "Viridis", zmin: 0, colorbar: { len: 0.2, y: 0.09, thickness: 10, title: { text: mapName === "kl_map" ? "nats" : "weight", side: "right" } }, hovertemplate: "t %{x:.0f} s, lag %{y:.0f} s: %{z:.3f}<extra></extra>" });
    const layout = themed({
      height: 860, shapes, showlegend: true, hovermode: "x unified",
      margin: { l: 66, r: 70, t: 30, b: 46 },
      xaxis: { title: "seconds from the start of the segment", range: [0, rawT[rawT.length - 1]], anchor: "y6" },
      yaxis: { domain: [0.80, 1.0], title: "FHR (" + panel.fhr_unit + ")" },
      yaxis2: { domain: [0.62, 0.78], title: "UA (" + panel.ua_unit + ")" },
      yaxis3: { domain: [0.46, 0.60], title: "gate g", range: [0, 1.02] },
      yaxis4: { domain: [0.24, 0.44], title: "KL (nats)", rangemode: "tozero" },
      yaxis5: { domain: [0.24, 0.44], title: "gain (nats)", overlaying: "y4", side: "right", showgrid: false },
      yaxis6: { domain: [0.0, 0.20], title: "lag (s)" },
    });
    layout.shapes.push({ type: "line", xref: "paper", yref: "y3", x0: 0, x1: 1, y0: panel.gate_shut_below, y1: panel.gate_shut_below, line: { color: css("--text-muted"), width: 1, dash: "dash" } });
    layout.shapes.push({ type: "line", xref: "paper", yref: "y3", x0: 0, x1: 1, y0: panel.gate_open_from, y1: panel.gate_open_from, line: { color: css("--text-muted"), width: 1, dash: "dash" } });
    Plotly.react(holder, data, layout, CONFIG);
    const meta = seg.meta;
    tools.querySelector(".ex-meta").innerHTML = meta.class + " · " + meta.hours.toFixed(2) + " h before delivery · " + meta.n_scored + " scored anchors · gate on at " + (100 * on.filter(Boolean).length / Math.max(meta.n_scored, 1)).toFixed(0) + "% of them" + (meta.page ? " · <a href='" + encodeURI(meta.page) + "' target='_blank' rel='noopener'>open the sample dashboard</a>" : "");
  }
  function explorerNode(panel) {
    const wrap = el("div");
    const tools = el("div", "explorer-tools");
    const options = panel.segments.map((seg, i) => "<option value='" + i + "'>" + seg.label + "</option>").join("");
    tools.innerHTML = "<label>Segment <select class='ex-seg'>" + options + "</select></label>"
      + "<label><input type='checkbox' class='ex-only'> Hide the gate-off parts</label>"
      + "<label>Lag map <select class='ex-map'><option value='kl_map'>KL attribution per lag</option><option value='attention_map'>attention per lag</option></select></label>"
      + "<span class='ex-meta muted'></span>";
    const holder = el("div", "plot");
    wrap.appendChild(tools);
    wrap.appendChild(el("div", "plot-wrap")).appendChild(holder);
    const draw = () => drawExplorer(panel, holder, tools);
    tools.querySelectorAll("select, input").forEach((node) => node.addEventListener("change", draw));
    return { node: wrap, draw };
  }

  // -------------------------------------------------------------------------------------- page
  function cardFor(panel) {
    const card = el("article", "card");
    card.id = "panel-" + panel.id;
    card.dataset.search = (panel.title + " " + Object.values(panel.explain || {}).join(" ")).toLowerCase();
    if (panel.kind === "tiles") { const node = tilesNode(panel); node.dataset.search = card.dataset.search; return node; }
    card.appendChild(header(panel));
    if (panel.kind === "missing") {
      card.appendChild(el("div", "missing", panel.reason));
    } else if (panel.kind === "html") {
      card.appendChild(el("div", "prose", panel.html));
    } else if (panel.kind === "table") {
      const body = el("div", "card-body full");
      body.appendChild(tableNode(panel));
      if (panel.explain) body.appendChild(explainBlock(panel));
      card.appendChild(body);
    } else if (panel.kind === "explorer") {
      const body = el("div", "card-body full");
      const explorer = explorerNode(panel);
      body.appendChild(explorer.node);
      body.appendChild(explainBlock(panel));
      card.appendChild(body);
      rendered.set(panel.id, { panel, draw: explorer.draw, done: false, card });
    } else {
      const body = el("div", "card-body" + (panel.wide ? " full" : ""));
      const holder = el("div", "plot");
      holder.style.height = (panel.height || 380) + "px";
      body.appendChild(el("div", "plot-wrap")).appendChild(holder);
      body.appendChild(explainBlock(panel));
      card.appendChild(body);
      rendered.set(panel.id, { panel, draw: () => drawPlot(panel, holder), done: false, card });
    }
    const foot = footer(panel);
    if (foot) card.appendChild(foot);
    return card;
  }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const id = entry.target.id.replace(/^panel-/, "");
      const item = rendered.get(id);
      if (item && !item.done) { item.done = true; item.draw(); }
    });
  }, { rootMargin: "500px 0px" });
  function redraw(id) {
    const item = rendered.get(id);
    if (item && item.done) item.draw();
  }
  function redrawAll() { rendered.forEach((item) => { if (item.done) item.draw(); }); }

  function showSection(id) {
    const searching = state.search.length > 0;
    const sections = searching ? RUN.sections : RUN.sections.filter((s) => s.id === id);
    state.section = id;
    observer.disconnect();
    rendered.clear();
    const content = $("content");
    // Release the graphs of the section that is leaving, or each switch keeps them in memory.
    content.querySelectorAll(".js-plotly-plot").forEach((node) => Plotly.purge(node));
    content.innerHTML = "";
    sections.forEach((section) => {
      const panels = section.panels.filter((panel) => !searching || (panel.title + " " + Object.values(panel.explain || {}).join(" ")).toLowerCase().includes(state.search));
      if (!panels.length) return;
      const head = el("div", "section-head");
      head.appendChild(el("h1", null, section.title));
      if (section.intro && !searching) head.appendChild(el("p", null, section.intro));
      content.appendChild(head);
      panels.forEach((panel) => {
        const card = cardFor(panel);
        content.appendChild(card);
        if (rendered.has(panel.id)) observer.observe(card);
      });
    });
    if (!content.children.length) content.appendChild(el("p", "muted", "No plot matches the search."));
    document.querySelectorAll(".sidebar button.sec").forEach((button) => button.classList.toggle("active", !searching && button.dataset.id === id));
    if (!searching) history.replaceState(null, "", "#" + id);
    window.scrollTo(0, 0);
  }

  function build() {
    document.title = RUN.meta.title;
    $("run-title").textContent = RUN.meta.title;
    $("run-sub").textContent = " " + RUN.meta.subtitle;
    const side = $("sidebar");
    side.appendChild(el("div", "nav-title", "Sections"));
    RUN.sections.forEach((section) => {
      const button = el("button", "sec", "<span>" + section.title + "</span><span class='count'>" + section.panels.length + "</span>");
      button.dataset.id = section.id;
      button.addEventListener("click", () => { $("search").value = ""; state.search = ""; showSection(section.id); });
      side.appendChild(button);
    });
    const links = el("div", "side-links");
    links.appendChild(el("div", "nav-title", "Other pages"));
    (RUN.meta.links || []).forEach((link) => links.appendChild(el("a", null, link.label)).setAttribute("href", encodeURI(link.href)));
    side.appendChild(links);

    const pops = $("pop-select");
    RUN.populations.forEach((pop) => {
      const button = el("button", pop.key === state.population ? "active" : "", "<span class='dot' style='background:" + pop.color + "'></span>" + pop.label);
      button.title = pop.rule;
      button.addEventListener("click", () => {
        state.population = pop.key;
        pops.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
        button.classList.add("active");
        redrawAll();
      });
      pops.appendChild(button);
    });
    $("search").addEventListener("input", (event) => { state.search = event.target.value.trim().toLowerCase(); showSection(state.section); });
    const toggle = $("theme-toggle");
    const apply = (theme) => {
      document.documentElement.dataset.theme = theme;
      toggle.textContent = theme === "dark" ? "Light" : "Dark";
      try { localStorage.setItem("run-dashboard-theme", theme); } catch (error) { /* storage is optional */ }
      redrawAll();
    };
    toggle.addEventListener("click", () => apply(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
    let saved = null;
    try { saved = localStorage.getItem("run-dashboard-theme"); } catch (error) { saved = null; }
    document.documentElement.dataset.theme = saved || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    toggle.textContent = document.documentElement.dataset.theme === "dark" ? "Light" : "Dark";
    const wanted = location.hash.replace(/^#/, "");
    showSection(RUN.sections.some((s) => s.id === wanted) ? wanted : RUN.sections[0].id);
  }
  build();
})();
