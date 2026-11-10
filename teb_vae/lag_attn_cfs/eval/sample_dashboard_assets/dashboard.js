(function () {
  "use strict";
  const D = window.__DASH__;
  const G = D.geometry;
  const AX = D.axes;
  // The right margin holds the right axis and the colour bar. It is wider than both need, because
  // plotly widens a figure's margin for a colour bar that does not fit, and one figure with a wider
  // margin has a narrower plot area, which would break the column-for-column alignment of the page.
  const PLOT_MARGIN = { l: 64, r: 156, t: 38, b: 46 };
  const COLORBAR_OFFSET_PX = 74;
  const PALETTE_SIZE = 8;
  const TRACE_PLOT_PX = 360, MAP_PLOT_PX = 412;
  // Major and minor grid steps in seconds. Every segment starts on a whole minute, so a grid anchored
  // at the segment start also falls on round values of the minutes-to-delivery axis.
  const GRID_STEPS = [[1, 0.25], [2, 0.5], [4, 1], [10, 2], [20, 4], [30, 5], [60, 10], [120, 20], [300, 60], [600, 120]];
  const MAX_MAJOR_LINES = 12;
  const COLORSCALES = { Viridis: "Viridis", Cividis: "Cividis", Magma: "Magma", Hot: "Hot", "Red-Blue": "RdBu", Blues: "Blues" };

  // ---- Data access ------------------------------------------------------------------------------
  const decoded = {};
  function arr(name) {
    if (decoded[name]) return decoded[name];
    const e = D.arrays[name];
    const bytes = Uint8Array.from(atob(e.b), c => c.charCodeAt(0));
    const q = new Int16Array(bytes.buffer);
    const out = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) out[i] = q[i] === -32768 ? NaN : e.o + e.s * q[i];
    decoded[name] = { v: out, d: e.d };
    return decoded[name];
  }
  const nAnchors = AX.anchors.length;
  const anchorIndexOfStep = {};
  AX.anchors.forEach((a, i) => { if (AX.anchor_valid[i]) anchorIndexOfStep[a] = i; });

  function column(a, c) {
    const [n, m] = a.d;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = a.v[i * m + c];
    return out;
  }
  const TRUTH_FIELDS = new Set(["truth", "err_full", "err_base", "skill", "z_full"]);
  function cellScored(tch, tau) { return arr("cell_mask").v[tau * arr("mu_full").d[2] + tch] > 0.5; }
  function stepScored(i, tau) { return arr("forecast_mask").v[i * G.H + tau] > 0.5; }
  function fcField(field, tch, tau) {
    // One forecast cell over the anchors, (A,). A cell outside the model's scored set is blank, and
    // an entry that reads the truth is also blank where the objective masks the step.
    const out = new Float32Array(nAnchors);
    const H = G.H;
    const C = arr("mu_full").d[2];
    const scored = cellScored(tch, tau);
    const usesTruth = TRUTH_FIELDS.has(field);
    const at = (name, i) => arr(name).v[(i * H + tau) * C + tch];
    const truth = i => {
      const a = AX.anchors[i] + 1 + tau;
      return a < G.T ? arr("truth").v[a * C + tch] : NaN;
    };
    for (let i = 0; i < nAnchors; i++) {
      if (!AX.anchor_valid[i] || !scored || (usesTruth && !stepScored(i, tau))) { out[i] = NaN; continue; }
      const y = truth(i);
      switch (field) {
        case "truth": out[i] = y; break;
        case "mu_full": out[i] = at("mu_full", i); break;
        case "mu_base": out[i] = at("mu_base", i); break;
        case "sigma_full": out[i] = Math.exp(0.5 * at("logvar_full", i)); break;
        case "sigma_base": out[i] = Math.exp(0.5 * at("logvar_base", i)); break;
        case "err_full": out[i] = Math.abs(y - at("mu_full", i)); break;
        case "err_base": out[i] = Math.abs(y - at("mu_base", i)); break;
        case "skill": out[i] = Math.abs(y - at("mu_base", i)) - Math.abs(y - at("mu_full", i)); break;
        case "z_full": out[i] = at("z_full", i); break;
        default: out[i] = NaN;
      }
    }
    return out;
  }

  // ---- Clocks -----------------------------------------------------------------------------------
  function toX(seconds) {
    switch (S.xMode) {
      case "seg_s": return seconds;
      case "seg_min": return seconds / 60;
      case "step": return seconds / G.dt;
      default: return (G.t0 + seconds) / 60;
    }
  }
  function fromX(x) {
    switch (S.xMode) {
      case "seg_s": return x;
      case "seg_min": return x * 60;
      case "step": return x * G.dt;
      default: return x * 60 - G.t0;
    }
  }
  const X_TITLES = { to_delivery: "Minutes relative to delivery", seg_min: "Minutes from segment start", seg_s: "Seconds from segment start", step: "Stored step" };

  // ---- State -------------------------------------------------------------------------------------
  const traceById = {}; D.traces.forEach(t => traceById[t.id] = t);
  const mapById = {}; D.heatmaps.forEach(h => mapById[h.id] = h);
  const has = id => !!traceById[id];
  const PRESETS = [
    { name: "FHR and KL", left: ["fhr"], right: ["kld"] },
    { name: "UA and KL", left: ["up"], right: ["kld"] },
    { name: "UA and max lag", left: ["up"], right: ["attn_argmax", "kl_argmax"] },
    { name: "FHR and UA", left: ["fhr"], right: ["up"] },
    { name: "UA and event gate", left: ["up"], right: ["event_gate"] },
    { name: "UA and forecast gap", left: ["up"], right: ["pred_gap"] },
    { name: "FHR and forecast gap", left: ["fhr"], right: ["pred_gap"] },
    { name: "Forecast error", left: ["mse_base", "mse_full", "mse_persist"], right: ["fhr"] },
    { name: "Forecast and truth", left: ["truth_tau", "fc_mu_full", "fc_mu_base"], right: ["fc_sigma_full"] },
    { name: "Latent shift", left: ["delta_mu_norm", "kld"], right: ["fhr"] },
  ];
  function defaultFig(kind, n) {
    if (kind === "trace") {
      const p = PRESETS[n === 0 ? 0 : 2];
      return { left: p.left.filter(has), right: p.right.filter(has), params: { L: {}, R: {} }, scale: { L: "auto", R: "auto" }, smooth: 0, colors: {} };
    }
    const first = n === 0 ? "attn" : "kl_lag";
    return { map: mapById[first] ? first : D.heatmaps[0].id, overlay: ["up"].filter(has), params: {}, overlayParams: {}, cmode: "auto", scale: null, log: null, colors: {} };
  }
  function freshState() {
    return {
      xMode: "to_delivery", theme: "light",
      overlays: { contractions: true, decelerations: false, warmup: true, gaps: false, unscored: true },
      figs: { t1: defaultFig("trace", 0), t2: defaultFig("trace", 1), h1: defaultFig("map", 0), h2: defaultFig("map", 1) },
    };
  }
  function loadState() {
    let saved = null;
    try {
      const hash = decodeURIComponent(location.hash.replace(/^#view=/, ""));
      if (hash) saved = JSON.parse(atob(hash));
    } catch (e) { saved = null; }
    if (!saved) {
      try { saved = JSON.parse(localStorage.getItem("sample-dashboard-view") || "null"); } catch (e) { saved = null; }
    }
    const fresh = freshState();
    if (!saved || !saved.figs) return fresh;
    const out = Object.assign(fresh, { xMode: saved.xMode || fresh.xMode, theme: saved.theme || fresh.theme, overlays: Object.assign(fresh.overlays, saved.overlays || {}) });
    for (const key of Object.keys(fresh.figs)) {
      const f = saved.figs[key];
      if (!f) continue;
      const merged = Object.assign(fresh.figs[key], f);
      if (merged.left) merged.left = merged.left.filter(has);
      if (merged.right) merged.right = merged.right.filter(has);
      if (merged.map && !mapById[merged.map]) merged.map = D.heatmaps[0].id;
      if (merged.overlay) merged.overlay = merged.overlay.filter(has);
      out.figs[key] = merged;
    }
    return out;
  }
  const S = loadState();
  function saveState() {
    try { localStorage.setItem("sample-dashboard-view", JSON.stringify(S)); } catch (e) { /* no storage */ }
  }
  document.documentElement.dataset.theme = S.theme;

  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function seriesColour(i) { return cssVar("--series-" + (i % PALETTE_SIZE + 1)); }

  function assignColour(fig, id) {
    if (fig.colors[id] !== undefined) return fig.colors[id];
    const used = new Set(Object.values(fig.colors));
    let slot = 0;
    while (used.has(slot) && slot < PALETTE_SIZE) slot++;
    fig.colors[id] = slot % PALETTE_SIZE;
    return fig.colors[id];
  }
  function releaseColours(fig, live) {
    for (const id of Object.keys(fig.colors)) if (!live.includes(id)) delete fig.colors[id];
  }

  // ---- Trace evaluation ---------------------------------------------------------------------------
  function paramOf(store, p) { return store[p] !== undefined ? store[p] : defaultParam(p); }
  function defaultParam(p) {
    if (p === "tch") return Math.min(10, D.params.tch.options.length - 1);
    if (p === "sch") return Math.min(10, D.params.sch.options.length - 1);
    if (p === "tau") return 0;
    return 0;
  }
  function evalTrace(spec, store) {
    // Returns {y, x0, dx} on a regular grid in seconds, or {y, xs} on an explicit grid.
    const param = p => paramOf(store, p);
    let y, grid;
    if (spec.k === "v") y = arr(spec.a).v;
    else if (spec.k === "col") y = column(arr(spec.a), param(spec.p[0]));
    else if (spec.k === "fc") {
      y = fcField(spec.f, param("tch"), param("tau"));
      const tau = param("tau");
      const xs = new Float64Array(nAnchors);
      for (let i = 0; i < nAnchors; i++) xs[i] = (AX.anchors[i] + 1 + tau) * G.dt;
      return { y, xs };
    }
    if (spec.c === "raw") grid = { x0: 0, dx: G.raw_dt };
    else if (spec.c === "anchor") {
      const xs = new Float64Array(nAnchors);
      for (let i = 0; i < nAnchors; i++) xs[i] = AX.anchors[i] * G.dt;
      return { y, xs };
    } else grid = { x0: 0, dx: G.dt };
    return Object.assign({ y }, grid);
  }
  function smoothed(y, spec, seconds, grid) {
    // A centred moving average of the finite samples in the window, by prefix sums.
    if (!seconds) return y;
    const half = Math.max(1, Math.round(seconds / (grid.dx || G.dt) / 2));
    const n = y.length;
    const sum = new Float64Array(n + 1), count = new Uint32Array(n + 1);
    for (let i = 0; i < n; i++) {
      const finite = Number.isFinite(y[i]);
      sum[i + 1] = sum[i] + (finite ? y[i] : 0);
      count[i + 1] = count[i] + (finite ? 1 : 0);
    }
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const lo = Math.max(0, i - half), hi = Math.min(n, i + half + 1);
      const c = count[hi] - count[lo];
      out[i] = c > 0 && Number.isFinite(y[i]) ? (sum[hi] - sum[lo]) / c : NaN;
    }
    return out;
  }
  function zscored(y) {
    let s = 0, s2 = 0, n = 0;
    for (const v of y) if (Number.isFinite(v)) { s += v; s2 += v * v; n++; }
    const mean = s / Math.max(n, 1), sd = Math.sqrt(Math.max(s2 / Math.max(n, 1) - mean * mean, 1e-12));
    return Float32Array.from(y, v => (v - mean) / sd);
  }
  function quantiles(values, lo, hi) {
    const f = [];
    for (const v of values) if (Number.isFinite(v)) f.push(v);
    if (!f.length) return [0, 1];
    f.sort((a, b) => a - b);
    return [f[Math.floor(lo * (f.length - 1))], f[Math.floor(hi * (f.length - 1))]];
  }

  // ---- Shared figure furniture ----------------------------------------------------------------------
  function font() { return { family: "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif", size: 11, color: cssVar("--text-secondary") }; }
  function axisBase() {
    return { gridcolor: cssVar("--plot-grid"), gridwidth: 0.6, minor: { showgrid: true, gridcolor: cssVar("--plot-grid-minor"), gridwidth: 0.5, nticks: 3 }, zeroline: false, linecolor: cssVar("--border"), ticks: "outside", ticklen: 3, tickcolor: cssVar("--border"), showline: true, automargin: false };
  }
  function secondsShown() { return viewRange || [0, G.T * G.dt]; }
  function timeGrid() {
    const [a, b] = secondsShown();
    return GRID_STEPS.find(step => (b - a) / step[0] <= MAX_MAJOR_LINES) || GRID_STEPS[GRID_STEPS.length - 1];
  }
  function decimals(step) {
    for (let p = 0; p <= 2; p++) { const v = step * 10 ** p; if (Math.abs(v - Math.round(v)) < 1e-6) return p; }
    return 2;
  }
  function timeTicks() {
    // The two clocks share one set of ticks. The top axis reads seconds from the segment start,
    // unless the bottom axis already does, and then it reads minutes relative to delivery.
    const [major, minor] = timeGrid();
    const [a, b] = secondsShown();
    const top = S.xMode === "seg_s" ? (t => (G.t0 + t) / 60) : (t => t);
    const axis = of => {
      const origin = of(0), d = of(major) - origin;
      return { tickmode: "linear", tick0: origin, dtick: d, tickformat: "." + decimals(d) + "f", minor: { tick0: origin, dtick: of(minor) - origin, ticks: "outside", ticklen: 2, tickcolor: cssVar("--border") } };
    };
    return { bottom: axis(toX), top: Object.assign(axis(top), { range: [top(a), top(b)] }), title: S.xMode === "seg_s" ? X_TITLES.to_delivery : X_TITLES.seg_s };
  }
  function timeUpdate(F) {
    // What a figure needs when the visible window changes: the two ranges, their ticks, and a map's grid.
    const t = timeTicks();
    const out = { "xaxis.range": currentRange(), "xaxis.autorange": false, "xaxis2.range": t.top.range, "xaxis2.autorange": false };
    [["xaxis", t.bottom], ["xaxis2", t.top]].forEach(([name, c]) => {
      out[name + ".tick0"] = c.tick0;
      out[name + ".dtick"] = c.dtick;
      out[name + ".tickformat"] = c.tickformat;
      out[name + ".minor.tick0"] = c.minor.tick0;
      out[name + ".minor.dtick"] = c.minor.dtick;
    });
    if (F.type === "map") out.shapes = shapes(true, F.map.yCoord, F.yFactor);
    return out;
  }
  function anchorTraces() {
    // Plotly draws an axis only when a trace uses it, and it leaves stale layers behind when the set
    // of used axis pairs changes between redraws. These traces have no visible points and keep every
    // pair in use, whichever lists the reader has emptied.
    const blank = (xaxis, yaxis) => ({ type: "scatter", mode: "lines", x: timeTicks().top.range, y: [null, null], xaxis, yaxis, hoverinfo: "skip", showlegend: false, line: { width: 0 } });
    return [blank("x", "y"), blank("x", "y2"), blank("x2", "y")];
  }
  function valueGrid(values) {
    // Round major and minor steps for a map's y axis. An integer axis never gets a fractional line.
    const lo = Math.min(...values), hi = Math.max(...values);
    if (!(hi > lo)) return null;
    const rough = (hi - lo) / 8;
    const power = 10 ** Math.floor(Math.log10(rough));
    const lead = [1, 2, 5, 10].find(n => n * power >= rough);
    const integer = values.every(v => Number.isInteger(v));
    const major = integer ? Math.max(1, lead * power) : lead * power;
    const minor = major / (lead === 2 ? 4 : 5);
    return { lo, hi, major, minor: integer && !Number.isInteger(minor) ? major : minor };
  }
  // ---- Paired y axes: every right tick sits on the same grid line as a left tick --------------------
  function niceAbove(x) {
    const p = 10 ** Math.floor(Math.log10(x));
    return [1, 2, 2.5, 5, 10].map(m => m * p).find(v => v >= x * (1 - 1e-9));
  }
  function minorStep(step) {
    const lead = step / 10 ** Math.floor(Math.log10(step) + 1e-9);
    return step / (Math.abs(lead - 2) < 1e-6 ? 4 : 5);
  }
  function extentOf(series) {
    let lo = Infinity, hi = -Infinity;
    series.forEach(values => { for (const v of values) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } });
    return lo <= hi ? [lo, hi] : [0, 1];
  }
  function targetRange(mode, series, ctg) {
    // The window one axis has to show. A signal that never goes negative keeps a zero baseline.
    if (mode === "ctg" && ctg) return [ctg[0], ctg[1]];
    const [dataLo, dataHi] = extentOf(series);
    let lo = dataLo, hi = dataHi, pad = 0.04;
    if (mode === "robust") { [lo, hi] = quantiles(series.flatMap(v => Array.from(v)), 0.01, 0.99); pad = 0.05; }
    if (!(hi > lo)) { const half = 0.5 * (Math.abs(lo) || 1); lo -= half; hi += half; }
    const margin = pad * (hi - lo);
    return [dataLo >= 0 ? Math.max(0, lo - margin) : lo - margin, hi + margin];
  }
  function fitAxis(lo, hi, intervals) {
    let step = niceAbove((hi - lo) / intervals);
    for (let tries = 0; tries < 8; tries++) {
      const start = Math.floor(lo / step + 1e-9) * step;
      if (start + intervals * step >= hi - 1e-9 * step) return { range: [start, start + intervals * step], step, tick0: start };
      step = niceAbove(step * 1.0001);
    }
    const start = Math.floor(lo / step) * step;
    return { range: [start, Math.max(hi, start + intervals * step)], step, tick0: start };
  }
  function alignAxes(targets) {
    // One interval count for every axis, chosen to waste the least height.
    let best = null, bestCost = Infinity;
    for (let intervals = 5; intervals <= 10; intervals++) {
      const fits = targets.map(([lo, hi]) => fitAxis(lo, hi, intervals));
      const cost = fits.reduce((sum, f, i) => sum + (f.range[1] - f.range[0]) / (targets[i][1] - targets[i][0]), 0) + 0.02 * Math.abs(intervals - 8);
      if (cost < bestCost) { bestCost = cost; best = fits; }
    }
    return best || [];
  }
  function fitToTicks(range, major, target) {
    // A right axis whose ticks land on the major lines of a left axis that cannot move, as on a heatmap.
    const first = Math.ceil(range[0] / major - 1e-9) * major;
    const intervals = Math.floor((range[1] - first) / major + 1e-9);
    if (intervals < 1) return null;
    const [lo, hi] = target;
    let step = niceAbove((hi - lo) / intervals);
    for (let tries = 0; tries < 12; tries++) {
      const start = Math.floor(lo / step + 1e-9) * step;
      const at = y => start + (y - first) * step / major;
      if (at(range[0]) <= lo + 1e-9 * step && at(range[1]) >= hi - 1e-9 * step) return { range: [at(range[0]), at(range[1])], step, tick0: start };
      step = niceAbove(step * 1.0001);
    }
    return null;
  }
  // Plotly writes zoomed values into the range array it receives, so it always gets a copy.
  function tickProps(fit) { return { range: fit.range.slice(), autorange: false, tickmode: "linear", tick0: fit.tick0, dtick: fit.step }; }
  function restoreY(F) {
    // A double click asks plotly for its own autorange, which would separate the two tick sets.
    F.yFactor = 1;
    const out = {};
    Object.entries(F.yAxes || {}).forEach(([name, fit]) => {
      if (!fit) return;
      out[name + ".range"] = fit.range.slice();
      out[name + ".autorange"] = false;
      out[name + ".dtick"] = fit.step;
      if (name === "yaxis") out[name + ".minor.dtick"] = minorStep(fit.step);
    });
    return out;
  }
  function refineY(F) {
    // A y zoom scales both axes together, so dividing both steps by one factor keeps the ticks paired.
    if (!F.yAxes || !F.yAxes.yaxis) return {};
    const range = F.plot._fullLayout.yaxis.range, span = Math.abs(range[1] - range[0]);
    let factor = 1;
    while (span / (F.yAxes.yaxis.step / factor) < 4 && factor < 15625) factor *= 5;
    F.yFactor = factor;
    const out = {};
    Object.entries(F.yAxes).forEach(([name, fit]) => {
      if (!fit) return;
      out[name + ".dtick"] = fit.step / factor;
      if (name === "yaxis") out[name + ".minor.dtick"] = minorStep(fit.step / factor);
    });
    return out;
  }

  function gridShapes(yCoord, yFactor) {
    // A heatmap hides plotly's grid, so a map draws its grid above the cells. Grey shows on every colour scale.
    const minorLine = { color: "rgba(128,128,128,.2)", width: 0.5 }, majorLine = { color: "rgba(128,128,128,.38)", width: 0.6 };
    const path = (xref, yref, d, line) => ({ type: "path", xref, yref, path: d, line, layer: "above" });
    const onMajor = (v, major) => Math.abs(v / major - Math.round(v / major)) < 1e-6;
    const [major, minor] = timeGrid();
    const [a, b] = secondsShown();
    let minorPath = "", majorPath = "";
    for (let k = Math.ceil(a / minor); k * minor <= b; k++) {
      const seconds = k * minor, x = toX(seconds), piece = "M" + x + ",0L" + x + ",1";
      if (onMajor(seconds, major)) majorPath += piece; else minorPath += piece;
    }
    const out = [];
    if (minorPath) out.push(path("x", "paper", minorPath, minorLine));
    if (majorPath) out.push(path("x", "paper", majorPath, majorLine));
    const grid = yCoord && valueGrid(Array.from(yCoord));
    if (grid) {
      grid.major /= yFactor || 1;
      grid.minor /= yFactor || 1;
      let yMinor = "", yMajor = "";
      for (let k = Math.ceil(grid.lo / grid.minor); k * grid.minor <= grid.hi; k++) {
        const y = k * grid.minor, piece = "M0," + y + "L1," + y;
        if (onMajor(y, grid.major)) yMajor += piece; else yMinor += piece;
      }
      if (yMinor) out.push(path("paper", "y", yMinor, minorLine));
      if (yMajor) out.push(path("paper", "y", yMajor, majorLine));
    }
    return out;
  }
  function baseLayout(height, showXTitle, onMap) {
    const t = timeTicks();
    const bottom = Object.assign(axisBase(), t.bottom, { title: showXTitle ? { text: X_TITLES[S.xMode], standoff: 4 } : undefined, range: currentRange(), showgrid: !onMap });
    bottom.minor = Object.assign({}, axisBase().minor, t.bottom.minor, { showgrid: !onMap });
    const top = Object.assign(axisBase(), t.top, { overlaying: "x", anchor: "y", side: "top", showgrid: false, autorange: false, title: { text: t.title, standoff: 2, font: { size: 10 } }, tickfont: { size: 10 } });
    top.minor = Object.assign({}, t.top.minor, { showgrid: false });
    return {
      autosize: true, height, margin: PLOT_MARGIN, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: font(), hovermode: false, dragmode: "zoom", showlegend: false,
      xaxis: bottom, xaxis2: top,
    };
  }
  function fullRange() { return [toX(0), toX((G.T - 1) * G.dt + G.dt)]; }
  let viewRange = null;
  function currentRange() { return viewRange ? [toX(viewRange[0]), toX(viewRange[1])] : fullRange(); }
  const PLOT_CONFIG = { displaylogo: false, scrollZoom: false, responsive: true, doubleClick: "autosize",
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"], toImageButtonOptions: { format: "svg" } };

  function shapes(onMap, yCoord, yFactor) {
    // On a map a tint would change the colour the reader decodes, so an event is an outline there.
    const out = onMap ? gridShapes(yCoord, yFactor) : [];
    const rect = (s0, s1, fill, above) => out.push({ type: "rect", xref: "x", yref: "paper", x0: toX(s0), x1: toX(s1), y0: 0, y1: 1, fillcolor: fill, line: { width: 0 }, layer: above ? "above" : "below" });
    const outline = (s0, s1, colour) => out.push({ type: "rect", xref: "x", yref: "paper", x0: toX(s0), x1: toX(s1), y0: 0, y1: 1, fillcolor: "rgba(0,0,0,0)", line: { width: 1.5, color: colour }, layer: "above" });
    const rawS = n => n * G.raw_dt;
    const event = (s0, s1, tint, line) => { if (onMap) outline(s0, s1, cssVar(line)); else rect(s0, s1, cssVar(tint), false); };
    if (S.overlays.warmup) rect(0, G.warmup * G.dt - G.dt / 2, cssVar("--warmup"), onMap);
    if (S.overlays.contractions) for (const [a, p, b] of D.events.contractions) event(rawS(a), rawS(b), "--contraction", "--contraction-line");
    if (S.overlays.decelerations) for (const [a, p, b] of D.events.decelerations) event(rawS(a), rawS(b), "--deceleration", "--deceleration-line");
    if (S.overlays.gaps && !onMap && D.arrays.up_valid) {
      const v = arr("up_valid").v;
      let start = -1;
      for (let i = 0; i <= v.length; i++) {
        const bad = i < v.length && v[i] < 0.5;
        if (bad && start < 0) start = i;
        if (!bad && start >= 0) { rect(start * G.raw_dt, i * G.raw_dt, cssVar("--gap"), false); start = -1; }
      }
    }
    if (S.overlays.unscored) {
      const scored = arr("score_contributing").v;
      let start = -1;
      for (let i = 0; i <= nAnchors; i++) {
        const unscored = i < nAnchors && AX.anchor_valid[i] && !(scored[i] > 0.5);
        if (unscored && start < 0) start = i;
        if (!unscored && start >= 0) { rect((AX.anchors[start] - 0.5) * G.dt, (AX.anchors[i - 1] + 0.5) * G.dt, cssVar("--unscored"), onMap); start = -1; }
      }
    }
    if (pinned !== null) {
      const t = toX(AX.anchors[pinned] * G.dt);
      out.push({ type: "line", xref: "x", yref: "paper", x0: t, x1: t, y0: 0, y1: 1, line: { color: cssVar("--text-primary"), width: 1.5 }, layer: "above" });
    }
    return out;
  }
  function refreshShapes() {
    // Overlays and the pin change no data, so the reader's y zoom survives them.
    Object.values(FIGS).forEach(F => Plotly.relayout(F.plot, { shapes: shapes(F.type === "map", F.map && F.map.yCoord, F.yFactor) }));
  }

  // ---- Figures -----------------------------------------------------------------------------------
  const FIGS = {};
  let syncing = false;
  function makeList(host, entries, store, side, onchange, type) {
    host.innerHTML = "";
    const search = document.createElement("input");
    search.type = "search"; search.placeholder = "Filter"; search.setAttribute("aria-label", "Filter list");
    const list = document.createElement("div"); list.className = "list";
    const groups = {};
    entries.forEach(e => { (groups[e.g] = groups[e.g] || []).push(e); });
    const items = [];
    Object.keys(groups).forEach((g, gi) => {
      const d = document.createElement("details");
      d.open = gi < 8;
      const s = document.createElement("summary"); s.textContent = g; d.appendChild(s);
      groups[g].forEach(e => {
        const label = document.createElement("label");
        label.className = "item" + (e.p && e.p.length ? " param" : "");
        const input = document.createElement("input");
        input.type = type; input.name = host.id + "-radio"; input.value = e.id;
        input.checked = type === "radio" ? store.value === e.id : store.value.includes(e.id);
        input.addEventListener("change", () => onchange(e.id, input.checked));
        label.appendChild(input);
        label.appendChild(document.createTextNode(e.l));
        label.title = e.l + (e.u ? " (" + e.u + ")" : "");
        d.appendChild(label);
        items.push({ label, e, d });
      });
      list.appendChild(d);
    });
    search.addEventListener("input", () => {
      const q = search.value.toLowerCase();
      items.forEach(it => { it.label.style.display = !q || it.e.l.toLowerCase().includes(q) ? "" : "none"; });
      list.querySelectorAll("details").forEach(d => { d.open = true; });
    });
    host.appendChild(search);
    host.appendChild(list);
  }
  function paramRow(host, names, store, onchange) {
    host.innerHTML = "";
    const seen = new Set();
    names.forEach(p => {
      if (seen.has(p)) return;
      seen.add(p);
      const info = D.params[p];
      const label = document.createElement("label");
      label.textContent = info.label;
      const select = document.createElement("select");
      info.options.forEach((o, i) => { const opt = document.createElement("option"); opt.value = i; opt.textContent = o; select.appendChild(opt); });
      select.value = paramOf(store, p);
      select.addEventListener("change", () => { store[p] = +select.value; onchange(); });
      label.appendChild(select);
      host.appendChild(label);
    });
  }

  function traceFigure(key, title) {
    const fig = S.figs[key];
    const el = document.createElement("section");
    el.className = "fig fig-" + (key[0] === "t" ? "trace" : "map");
    el.innerHTML =
      '<div class="fig-side left"><div class="side-title"><span>' + title + ' left</span><span class="axis-tag">left y axis</span></div><div class="params" data-side="L"></div>' +
      '<div class="params"><label>Y scale<select data-scale="L"><option value="auto">Fit</option><option value="robust">Robust 1 to 99%</option><option value="z">Z score each trace</option><option value="ctg">CTG paper</option></select></label></div><div class="list-host" data-side="L" id="' + key + '-L" style="display:flex;flex-direction:column;gap:4px;flex:1;min-height:0"></div></div>' +
      '<div class="fig-main"><div class="fig-head"><div class="legend"></div><span class="corr"></span><select class="preset" aria-label="Preset"><option value="">Preset</option></select><select class="smooth" aria-label="Smoothing"><option value="0">Raw</option><option value="8">Smooth 8 s</option><option value="20">Smooth 20 s</option><option value="60">Smooth 60 s</option></select></div><div class="fig-plot"><div class="plot"></div><div class="crosshair"></div></div></div>' +
      '<div class="fig-side right"><div class="side-title"><span>' + title + ' right</span><span class="axis-tag">right y axis</span></div><div class="params" data-side="R"></div>' +
      '<div class="params"><label>Y scale<select data-scale="R"><option value="auto">Fit</option><option value="robust">Robust 1 to 99%</option><option value="z">Z score each trace</option><option value="ctg">CTG paper</option></select></label></div><div class="list-host" data-side="R" id="' + key + '-R" style="display:flex;flex-direction:column;gap:4px;flex:1;min-height:0"></div></div>';
    const plot = el.querySelector(".plot");
    const F = { key, type: "trace", el, plot, height: TRACE_PLOT_PX, cache: [] };
    FIGS[key] = F;
    PRESETS.forEach((p, i) => { if (p.left.some(has) && p.right.some(has)) { const o = document.createElement("option"); o.value = i; o.textContent = p.name; el.querySelector(".preset").appendChild(o); } });
    el.querySelector(".preset").addEventListener("change", ev => {
      const p = PRESETS[+ev.target.value];
      if (!p) return;
      fig.left = p.left.filter(has); fig.right = p.right.filter(has);
      ev.target.value = "";
      rebuildLists(); renderTrace(F); saveState();
    });
    el.querySelector(".smooth").value = fig.smooth;
    el.querySelector(".smooth").addEventListener("change", ev => { fig.smooth = +ev.target.value; renderTrace(F); saveState(); });
    ["L", "R"].forEach(side => {
      const sel = el.querySelector('select[data-scale="' + side + '"]');
      sel.value = fig.scale[side];
      sel.addEventListener("change", () => { fig.scale[side] = sel.value; renderTrace(F); saveState(); });
    });
    function rebuildLists() {
      ["L", "R"].forEach(side => {
        const prop = side === "L" ? "left" : "right";
        makeList(el.querySelector('.list-host[data-side="' + side + '"]'), D.traces, { value: fig[prop] }, side, (id, on) => {
          fig[prop] = on ? fig[prop].concat([id]) : fig[prop].filter(v => v !== id);
          sync(); renderTrace(F); saveState();
        }, "checkbox");
        sync();
      });
    }
    function sync() {
      ["L", "R"].forEach(side => {
        const prop = side === "L" ? "left" : "right";
        const names = [];
        fig[prop].forEach(id => traceById[id].p.forEach(p => names.push(p)));
        paramRow(el.querySelector('.params[data-side="' + side + '"]'), names, fig.params[side], () => { renderTrace(F); saveState(); });
      });
    }
    F.rebuild = rebuildLists;
    rebuildLists();
    return F;
  }

  function bridged(y, spec, grid) {
    // Drawing only: a raw trace with isolated invalid samples would read as a dashed line, so a
    // gap of at most one second is joined by a straight segment. Readouts keep the true gaps.
    if (spec.c !== "raw") return y;
    const limit = Math.round(1 / (grid.dx || G.dt));
    let out = y;
    for (let i = 1; i < y.length; i++) {
      if (Number.isFinite(y[i]) || !Number.isFinite(y[i - 1])) continue;
      let end = i;
      while (end < y.length && !Number.isFinite(y[end])) end++;
      if (end < y.length && end - i <= limit) {
        if (out === y) out = Float32Array.from(y);
        const step = (y[end] - y[i - 1]) / (end - i + 1);
        for (let j = i; j < end; j++) out[j] = y[i - 1] + step * (j - i + 1);
      }
      i = end;
    }
    return out;
  }

  function renderTrace(F) {
    const fig = S.figs[F.key];
    const live = fig.left.concat(fig.right);
    releaseColours(fig, live);
    const data = [], cache = [], legend = F.el.querySelector(".legend");
    legend.innerHTML = "";
    const axes = { L: { units: [], ctg: null, vals: [] }, R: { units: [], ctg: null, vals: [] } };
    [["L", fig.left], ["R", fig.right]].forEach(([side, ids]) => {
      ids.forEach(id => {
        const spec = traceById[id];
        const ev = evalTrace(spec, fig.params[side]);
        let y = smoothed(ev.y, spec, fig.smooth, ev);
        if (fig.scale[side] === "z") y = zscored(y);
        const colour = seriesColour(assignColour(fig, id));
        const xs = ev.xs ? Array.from(ev.xs, toX) : undefined;
        const base = { x0: ev.xs ? undefined : toX(ev.x0), dx: ev.xs ? undefined : toX(ev.dx) - toX(0) };
        const name = labelOf(spec, fig.params[side]);
        data.push(Object.assign({
          type: "scatter", mode: "lines", y: bridged(y, spec, ev), x: xs, name, connectgaps: false, hoverinfo: "skip",
          line: { color: colour, width: spec.c === "raw" ? 1.6 : 2.4, dash: "solid", simplify: true },
          yaxis: side === "L" ? "y" : "y2", xaxis: "x",
        }, base));
        cache.push({ spec, side, ev, y, name, colour });
        axes[side].units.push(fig.scale[side] === "z" ? "z score" : spec.u);
        if (spec.ctg) axes[side].ctg = spec.ctg;
        axes[side].vals.push(y);
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.innerHTML = '<span class="key" style="border-color:' + colour + '"></span>' + escapeHtml(name) + ' <b class="side-mark">' + side + "</b>";
        legend.appendChild(chip);
      });
    });
    const layout = baseLayout(F.height, true);
    layout.shapes = shapes(false);
    const used = ["L", "R"].filter(side => axes[side].vals.length);
    const fits = alignAxes(used.map(side => targetRange(fig.scale[side], axes[side].vals, axes[side].ctg)));
    const fitOf = {};
    used.forEach((side, i) => { fitOf[side] = fits[i]; });
    // With traces only on the right, the left axis carries the grid and prints no numbers of its own.
    const leftFit = fitOf.L || fitOf.R || { range: [0, 1], step: 0.2, tick0: 0 };
    const axisTitle = side => ({ text: [...new Set(axes[side].units)].join(", "), standoff: 2 });
    layout.yaxis = Object.assign(axisBase(), tickProps(leftFit), { title: axisTitle("L"), showticklabels: !!fitOf.L });
    layout.yaxis.minor = Object.assign({}, axisBase().minor, { tick0: leftFit.tick0, dtick: minorStep(leftFit.step) });
    layout.yaxis2 = fitOf.R
      ? Object.assign(axisBase(), tickProps(fitOf.R), { title: axisTitle("R"), overlaying: "y", side: "right", showgrid: false, minor: { showgrid: false } })
      : { overlaying: "y", side: "right", visible: false, showgrid: false, range: [0, 1], autorange: false };
    F.yAxes = { yaxis: leftFit, yaxis2: fitOf.R || null };
    F.yFactor = 1;
    data.push(...anchorTraces());
    F.cache = cache;
    updateCorrelation(F);
    Plotly.react(F.plot, data, layout, PLOT_CONFIG);
  }
  function labelOf(spec, store) {
    if (!spec.p.length) return spec.l;
    const bits = spec.p.map(p => D.params[p].options[paramOf(store, p)]);
    return spec.l.replace(/, channel$|, dimension$/, "") + " (" + bits.join(", ") + ")";
  }
  function escapeHtml(t) { return String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

  function updateCorrelation(F) {
    const out = F.el.querySelector(".corr");
    const L = F.cache.filter(c => c.side === "L"), R = F.cache.filter(c => c.side === "R");
    if (L.length !== 1 || R.length !== 1) { out.textContent = ""; return; }
    const lo = viewRange ? viewRange[0] : 0, hi = viewRange ? viewRange[1] : (G.T - 1) * G.dt;
    const a = binOnSteps(L[0]), b = binOnSteps(R[0]);
    let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (let t = 0; t < G.T; t++) {
      const s = t * G.dt;
      if (s < lo || s > hi || !Number.isFinite(a[t]) || !Number.isFinite(b[t])) continue;
      n++; sx += a[t]; sy += b[t]; sxx += a[t] * a[t]; syy += b[t] * b[t]; sxy += a[t] * b[t];
    }
    const denominator = Math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    out.textContent = n > 10 && denominator > 0 ? "r = " + ((n * sxy - sx * sy) / denominator).toFixed(2) + ", visible, n = " + n : "";
    out.title = "Pearson correlation of the left and right trace on the stored-step grid, over the visible time window.";
  }
  function binOnSteps(item) {
    const out = new Float64Array(G.T).fill(NaN);
    const sum = new Float64Array(G.T), count = new Float64Array(G.T);
    const y = item.y;
    for (let i = 0; i < y.length; i++) {
      if (!Number.isFinite(y[i])) continue;
      const s = item.ev.xs ? item.ev.xs[i] : item.ev.x0 + i * item.ev.dx;
      const t = Math.floor(s / G.dt + 1e-6);
      if (t >= 0 && t < G.T) { sum[t] += y[i]; count[t]++; }
    }
    for (let t = 0; t < G.T; t++) if (count[t]) out[t] = sum[t] / count[t];
    return out;
  }

  function heatFigure(key, title) {
    const fig = S.figs[key];
    const el = document.createElement("section");
    el.className = "fig fig-" + (key[0] === "t" ? "trace" : "map");
    el.innerHTML =
      '<div class="fig-side left"><div class="side-title"><span>' + title + '</span><span class="axis-tag">heatmap</span></div><div class="params" data-side="M"></div>' +
      '<div class="list-host" data-side="M" id="' + key + '-M" style="display:flex;flex-direction:column;gap:4px;flex:1;min-height:0"></div></div>' +
      '<div class="fig-main"><div class="fig-head"><div class="legend"></div>' +
      '<select class="cmap" aria-label="Colour scale"></select><select class="cmode" aria-label="Colour range"><option value="auto">Range: segment</option><option value="robust">Range: 1 to 99%</option><option value="shared">Range: run-wide</option></select>' +
      '<select class="cscale" aria-label="Colour transform"><option value="">Default scale</option><option value="linear">Linear</option><option value="log">Logarithmic</option></select></div>' +
      '<div class="fig-plot"><div class="plot"></div><div class="crosshair"></div></div></div>' +
      '<div class="fig-side right"><div class="side-title"><span>' + title + ' overlay</span><span class="axis-tag">right y axis</span></div><div class="params" data-side="O"></div><div class="list-host" data-side="O" id="' + key + '-O" style="display:flex;flex-direction:column;gap:4px;flex:1;min-height:0"></div></div>';
    const plot = el.querySelector(".plot");
    const F = { key, type: "map", el, plot, height: MAP_PLOT_PX, cache: [] };
    FIGS[key] = F;
    const cmap = el.querySelector(".cmap");
    Object.keys(COLORSCALES).forEach(n => { const o = document.createElement("option"); o.value = n; o.textContent = n; cmap.appendChild(o); });
    const autoOption = document.createElement("option");
    autoOption.value = "auto"; autoOption.textContent = "Colour: default";
    cmap.prepend(autoOption);
    cmap.value = fig.scale || "auto";
    cmap.addEventListener("change", () => { fig.scale = cmap.value === "auto" ? null : cmap.value; renderMap(F); saveState(); });
    el.querySelector(".cmode").value = fig.cmode;
    el.querySelector(".cmode").addEventListener("change", ev => { fig.cmode = ev.target.value; renderMap(F); saveState(); });
    el.querySelector(".cscale").value = fig.log || "";
    el.querySelector(".cscale").addEventListener("change", ev => { fig.log = ev.target.value || null; renderMap(F); saveState(); });
    F.syncParams = () => {
      const spec = mapById[fig.map];
      const cmode = el.querySelector(".cmode"), shared = cmode.querySelector('[value="shared"]');
      const hasLimit = !!spec.lim && D.limits[spec.lim] !== undefined;
      shared.disabled = !hasLimit;
      shared.textContent = hasLimit ? "Range: run-wide" : "Range: run-wide (none for this map)";
      if (!hasLimit && fig.cmode === "shared") { fig.cmode = "auto"; cmode.value = "auto"; }
      const cscale = el.querySelector(".cscale"), logOption = cscale.querySelector('[value="log"]');
      logOption.disabled = spec.sym;
      if (spec.sym && fig.log === "log") { fig.log = null; cscale.value = ""; }
      paramRow(el.querySelector('.params[data-side="M"]'), spec.p, fig.params, () => { renderMap(F); saveState(); });
      const names = [];
      fig.overlay.forEach(id => traceById[id].p.forEach(p => names.push(p)));
      paramRow(el.querySelector('.params[data-side="O"]'), names, fig.overlayParams, () => { renderMap(F); saveState(); });
    };
    makeList(el.querySelector('.list-host[data-side="M"]'), D.heatmaps, { value: fig.map }, "M", id => { fig.map = id; F.syncParams(); renderMap(F); saveState(); }, "radio");
    makeList(el.querySelector('.list-host[data-side="O"]'), D.traces, { value: fig.overlay }, "O", (id, on) => {
      fig.overlay = on ? fig.overlay.concat([id]) : fig.overlay.filter(v => v !== id);
      F.syncParams(); renderMap(F); saveState();
    }, "checkbox");
    F.syncParams();
    return F;
  }

  function mapMatrix(spec, store) {
    // Returns {z:[row arrays], x0|xs, ys, n}, with rows indexed by the vertical axis.
    const param = p => paramOf(store, p);
    let z = [], xs = null, ny;
    if (spec.k === "m") {
      const a = arr(spec.a);
      const [n, m] = a.d;
      const rows = spec.rows || Array.from({ length: m }, (_, i) => i);
      ny = rows.length;
      z = rows.map(r => { const c = new Float32Array(n); for (let i = 0; i < n; i++) c[i] = a.v[i * m + r]; return c; });
      return { z, dt: G.dt, ny, rows, n };
    }
    if (spec.k === "head") {
      const a = arr(spec.a);
      const [n, heads, lags] = a.d;
      // Choice 0 is the mean over heads, and choice m + 1 is head m.
      const h = param("hm");
      const mean = arr("attn_mean");
      z = [];
      for (let l = 0; l < lags; l++) {
        const c = new Float32Array(n);
        for (let i = 0; i < n; i++) c[i] = h === 0 ? mean.v[i * lags + l] : a.v[(i * heads + h - 1) * lags + l];
        z.push(c);
      }
      return { z, dt: G.dt, ny: lags, n };
    }
    if (spec.k === "fc_tau") {
      const C = arr("mu_full").d[2];
      const tau = param("tau");
      xs = new Float64Array(nAnchors);
      for (let i = 0; i < nAnchors; i++) xs[i] = (AX.anchors[i] + 1 + tau) * G.dt;
      for (let c = 0; c < C; c++) z.push(fcField(spec.f, c, tau));
      return { z, xs, ny: C };
    }
    if (spec.k === "fc_ch") {
      const c = param("tch");
      xs = new Float64Array(nAnchors);
      for (let i = 0; i < nAnchors; i++) xs[i] = AX.anchors[i] * G.dt;
      for (let tau = 0; tau < G.H; tau++) z.push(fcField(spec.f, c, tau));
      return { z, xs, ny: G.H };
    }
    return { z: [], dt: G.dt, ny: 0 };
  }
  function G_C() { return arr("mu_full").d[2]; }

  function renderMap(F) {
    const fig = S.figs[F.key];
    const spec = mapById[fig.map];
    const m = mapMatrix(spec, fig.params);
    const cacheZ = m.z;
    // Colour.
    let useLog = fig.log ? fig.log === "log" : spec.log;
    const all = [];
    cacheZ.forEach(c => { for (const v of c) if (Number.isFinite(v)) all.push(v); });
    let zmin, zmax;
    const sym = spec.sym;
    if (fig.cmode === "shared" && spec.lim && D.limits[spec.lim] !== undefined) {
      zmax = D.limits[spec.lim]; zmin = sym ? -zmax : 0;
    } else if (fig.cmode === "robust" && all.length) {
      const q = quantiles(all, 0.01, 0.99);
      zmin = q[0]; zmax = q[1];
      if (sym) { const b = Math.max(Math.abs(q[0]), Math.abs(q[1])); zmin = -b; zmax = b; }
    } else if (all.length) {
      let lo = Infinity, hi = -Infinity;
      for (const v of all) { if (v < lo) lo = v; if (v > hi) hi = v; }
      zmin = lo; zmax = hi;
      if (sym) { const b = Math.max(Math.abs(lo), Math.abs(hi)); zmin = -b; zmax = b; }
    } else { zmin = 0; zmax = 1; }
    if (zmax <= zmin) zmax = zmin + 1;
    let shown = cacheZ, colorbar = { thickness: 10, x: 1 + COLORBAR_OFFSET_PX / Math.max(F.plot.clientWidth - PLOT_MARGIN.l - PLOT_MARGIN.r, 200), xanchor: "left", outlinewidth: 0, tickfont: { size: 10 }, title: { text: spec.u, side: "right", font: { size: 10 } } };
    if (useLog && !sym && zmax > 0) {
      const floor = Math.max(zmax * 1e-4, zmin > 0 ? zmin : 0) || zmax * 1e-4;
      const lo = Math.log10(floor), hi = Math.log10(zmax);
      shown = cacheZ.map(c => c.map(v => Number.isFinite(v) ? Math.log10(Math.max(v, floor)) : NaN));
      zmin = lo; zmax = hi;
      const ticks = [];
      for (let k = Math.ceil(lo); k <= Math.floor(hi); k++) ticks.push(k);
      colorbar.tickvals = ticks;
      colorbar.ticktext = ticks.map(k => k >= -2 && k <= 2 ? String(10 ** k) : "1e" + k);
    }
    const colorscale = fig.scale ? COLORSCALES[fig.scale] : (sym ? "RdBu" : "Viridis");
    // Axes.
    const yCoord = spec.yv ? AX[spec.yv] : spec.rows ? spec.rows : Array.from({ length: m.ny }, (_, i) => i);
    const x = m.xs ? Array.from(m.xs, toX) : undefined;
    const x0 = m.xs ? undefined : toX(0);
    const dx = m.xs ? undefined : toX(m.dt) - toX(0);
    const data = [{
      type: "heatmap", z: shown, x, x0, dx, y: yCoord, zmin, zmax, zsmooth: false, colorscale,
      hoverongaps: false, hoverinfo: "skip", colorbar,
    }];
    const layout = baseLayout(F.height, true, true);
    F.yFactor = 1;
    layout.shapes = shapes(true, yCoord, 1);
    const yGrid = valueGrid(Array.from(yCoord));
    // The range hugs the outer cells. An autorange would pad for the ridge line and leave a blank strip.
    const count = yCoord.length;
    const lowEdge = count > 1 ? yCoord[0] - (yCoord[1] - yCoord[0]) / 2 : yCoord[0] - 0.5;
    const highEdge = count > 1 ? yCoord[count - 1] + (yCoord[count - 1] - yCoord[count - 2]) / 2 : yCoord[0] + 0.5;
    const mapRange = [Math.min(lowEdge, highEdge), Math.max(lowEdge, highEdge)];
    const mapFit = yGrid ? { range: mapRange, step: yGrid.major, tick0: 0 } : null;
    layout.yaxis = Object.assign(axisBase(), { title: { text: spec.yt, standoff: 2 }, range: mapRange, autorange: false, showgrid: false }, mapFit ? tickProps(mapFit) : {});
    layout.yaxis.minor = { showgrid: false };
    // Overlays: the warm-up staircase, the map's own ridge, and the reader's traces on the right axis.
    if (spec.stair) {
      const w = AX[spec.stair];
      const rows = spec.rows || w.map((_, i) => i);
      const sx = [], sy = [];
      rows.forEach((r, i) => { sx.push(toX(w[r] * G.dt)); sy.push(yCoord[i]); });
      data.push({ type: "scatter", mode: "lines", x: sx, y: sy, line: { color: cssVar("--text-primary"), width: 1.4, shape: "hv" }, hoverinfo: "skip", showlegend: false, xaxis: "x", yaxis: "y" });
    }
    if (spec.ridge && D.arrays[spec.ridge]) {
      const h = paramOf(fig.params, "hm");
      const y = spec.k === "head" && h > 0 ? column(arr("attn_argmax_head"), h - 1) : arr(spec.ridge).v;
      data.push({ type: "scatter", mode: "lines", x0: toX(0), dx: toX(G.dt) - toX(0), y, line: { color: "#ffffff", width: 1.8 }, hoverinfo: "skip", opacity: 0.85, showlegend: false, connectgaps: false });
    }
    const legend = F.el.querySelector(".legend");
    legend.innerHTML = '<span class="chip"><b>' + escapeHtml(spec.l) + '</b></span>';
    if (spec.ridge && D.arrays[spec.ridge]) legend.innerHTML += '<span class="chip"><span class="key" style="border-color:#ffffff;box-shadow:0 0 0 1px var(--border)"></span> max lag</span>';
    const cache = [];
    const unitSet = [];
    releaseColours(fig, fig.overlay);
    fig.overlay.forEach(id => {
      const t = traceById[id];
      const ev = evalTrace(t, fig.overlayParams);
      const colour = seriesColour(assignColour(fig, id));
      const name = labelOf(t, fig.overlayParams);
      const y = ev.y;
      data.push(Object.assign({ type: "scatter", mode: "lines", y: bridged(y, t, ev), name, connectgaps: false, hoverinfo: "skip", line: { color: colour, width: 2.2, dash: "solid", simplify: true }, xaxis: "x", yaxis: "y2" },
        ev.xs ? { x: Array.from(ev.xs, toX) } : { x0: toX(ev.x0), dx: toX(ev.dx) - toX(0) }));
      cache.push({ spec: t, side: "O", ev, y, name, colour });
      unitSet.push(t.u);
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.innerHTML = '<span class="key" style="border-color:' + colour + '"></span>' + escapeHtml(name);
      legend.appendChild(chip);
    });
    const overlayFit = mapFit && cache.length ? fitToTicks(mapRange, yGrid.major, targetRange("auto", cache.map(c => c.y))) : null;
    layout.yaxis2 = Object.assign(axisBase(), { overlaying: "y", side: "right", showgrid: false, autorange: true, title: { text: [...new Set(unitSet)].join(", "), standoff: 2 }, visible: fig.overlay.length > 0 }, overlayFit ? tickProps(overlayFit) : {});
    layout.yaxis2.minor = { showgrid: false };
    F.yAxes = { yaxis: mapFit, yaxis2: overlayFit };
    data.push(...anchorTraces());
    F.cache = cache;
    F.map = { spec, m, yCoord };
    Plotly.react(F.plot, data, layout, PLOT_CONFIG);
  }

  // ---- Page wiring ---------------------------------------------------------------------------------
  const figures = document.getElementById("figures");
  const order = [["t1", "Figure A"], ["t2", "Figure B"], ["h1", "Heatmap A"], ["h2", "Heatmap B"]];
  let pinned = (() => {
    // Open on the anchor with the largest divergence, which is usually the one worth reading first.
    const k = arr("kld").v;
    let best = null, bestV = -Infinity;
    AX.anchors.forEach((a, i) => { if (AX.anchor_valid[i] && Number.isFinite(k[a]) && k[a] > bestV) { bestV = k[a]; best = i; } });
    return best;
  })();

  order.forEach(([key, title]) => {
    const F = key[0] === "t" ? traceFigure(key, title) : heatFigure(key, title);
    figures.appendChild(F.el);
  });
  function renderAll() { Object.values(FIGS).forEach(F => F.type === "trace" ? renderTrace(F) : renderMap(F)); renderInspector(); }

  function bindSync() {
    Object.values(FIGS).forEach(F => {
      F.plot.on("plotly_relayout", ev => {
        if (syncing) return;
        // A drag reports the two ends separately, and an API call reports them as one array.
        let a = ev["xaxis.range[0]"], b = ev["xaxis.range[1]"];
        if (Array.isArray(ev["xaxis.range"])) [a, b] = ev["xaxis.range"];
        const xChanged = a !== undefined && b !== undefined;
        const moved = xChanged || ev["xaxis.autorange"] === true;
        const yReset = ev["yaxis.autorange"] === true;
        const yZoomed = ev["yaxis.range[0]"] !== undefined || Array.isArray(ev["yaxis.range"]);
        if (!moved && !yReset && !yZoomed) return;
        if (moved) {
          const next = xChanged ? [fromX(+a), fromX(+b)] : null;
          viewRange = next && next[1] > next[0] ? next : null;
        }
        syncing = true;
        const yProps = yReset ? restoreY(F) : yZoomed ? refineY(F) : {};
        const updates = Object.values(FIGS).map(O => {
          const props = Object.assign(moved ? timeUpdate(O) : {}, O === F ? yProps : {});
          if (!moved && O === F && O.type === "map") props.shapes = shapes(true, O.map.yCoord, O.yFactor);
          return Object.keys(props).length ? Plotly.relayout(O.plot, props) : Promise.resolve();
        });
        Promise.all(updates).finally(() => { syncing = false; Object.values(FIGS).forEach(G_ => { if (G_.type === "trace") updateCorrelation(G_); }); });
      });
      const area = F.plot;
      area.addEventListener("mousemove", ev => pointerMove(F, ev));
      area.addEventListener("mouseleave", () => { hideCross(); });
      area.addEventListener("mousedown", ev => { F.downAt = [ev.clientX, ev.clientY]; }, true);
      area.addEventListener("click", ev => pointerClick(F, ev));
    });
  }
  function plotPixel(F, ev) {
    const xa = F.plot._fullLayout.xaxis, ya = F.plot._fullLayout.yaxis;
    const rect = F.plot.getBoundingClientRect();
    const px = ev.clientX - rect.left - xa._offset;
    const py = ev.clientY - rect.top - ya._offset;
    if (px < 0 || px > xa._length || py < 0 || py > ya._length) return null;
    return { x: xa.p2l(px), y: ya.p2l(py), px, py };
  }
  function pointerMove(F, ev) {
    const p = plotPixel(F, ev);
    if (!p) { hideCross(); return; }
    const seconds = fromX(p.x);
    Object.values(FIGS).forEach(O => {
      const xa = O.plot._fullLayout.xaxis, ya = O.plot._fullLayout.yaxis;
      const cross = O.el.querySelector(".crosshair");
      cross.style.display = "block";
      cross.style.left = (xa._offset + xa.d2p(toX(seconds))) + "px";
      cross.style.top = ya._offset + "px";
      cross.style.height = ya._length + "px";
    });
    showTooltip(F, seconds, p, ev);
  }
  function hideCross() {
    Object.values(FIGS).forEach(O => { O.el.querySelector(".crosshair").style.display = "none"; });
    document.getElementById("tooltip").hidden = true;
  }
  function valueAt(item, seconds) {
    const ev = item.ev, y = item.y;
    let i;
    if (ev.xs) {
      let lo = 0, hi = ev.xs.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ev.xs[mid] < seconds) lo = mid; else hi = mid; }
      i = Math.abs(ev.xs[lo] - seconds) <= Math.abs(ev.xs[hi] - seconds) ? lo : hi;
      if (Math.abs(ev.xs[i] - seconds) > G.dt) return NaN;
    } else i = Math.round((seconds - ev.x0) / ev.dx);
    return i >= 0 && i < y.length ? y[i] : NaN;
  }
  const fmt = v => Number.isFinite(v) ? (Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(3)) : "n/a";
  function showTooltip(F, seconds, p, ev) {
    const tip = document.getElementById("tooltip");
    const parts = ['<div class="tt-time">' + timeLabel(seconds) + "</div>"];
    if (F.type === "map") {
      const value = mapValueAt(F, seconds, p.y);
      const m = F.map;
      parts.push('<div class="tt-row"><span>' + escapeHtml(m.spec.l) + "</span><b>" + fmt(value.v) + " " + escapeHtml(m.spec.u) + "</b></div>");
      parts.push('<div class="tt-row"><span>' + escapeHtml(m.spec.yt) + "</span><span>" + escapeHtml(String(value.y)) + "</span></div>");
    }
    F.cache.forEach(item => {
      parts.push('<div class="tt-row"><span>' + '<span style="color:' + item.colour + '">▬</span> ' + escapeHtml(item.name) + "</span><b>" + fmt(valueAt(item, seconds)) + " " + escapeHtml(item.spec.u) + "</b></div>");
    });
    tip.innerHTML = parts.join("");
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(window.innerWidth - w - 12, ev.clientX + 14) + "px";
    tip.style.top = Math.min(window.innerHeight - h - 12, ev.clientY + 14) + "px";
  }
  function mapValueAt(F, seconds, y) {
    const { m, yCoord, spec } = F.map;
    let row = 0, best = Infinity;
    yCoord.forEach((v, i) => { const d = Math.abs(v - y); if (d < best) { best = d; row = i; } });
    let i;
    if (m.xs) {
      i = 0; let d = Infinity;
      for (let k = 0; k < m.xs.length; k++) { const e = Math.abs(m.xs[k] - seconds); if (e < d) { d = e; i = k; } }
    } else i = Math.round(seconds / m.dt);
    const col = m.z[row];
    const v = col && i >= 0 && i < col.length ? col[i] : NaN;
    let name = yCoord[row];
    if (spec.k === "m" && spec.yt === "kept channel") name = (spec.a === "source" || spec.a === "source_valid" ? D.params.sch : D.params.tch).options[spec.rows ? spec.rows[row] : row];
    if (spec.k === "fc_tau") name = D.params.tch.options[row];
    return { v, y: name };
  }
  function timeLabel(seconds) {
    const parts = [(seconds).toFixed(0) + " s"];
    parts.push(((G.t0 + seconds) / 60).toFixed(2) + " min relative to delivery");
    return parts.join(" · ");
  }
  function pointerClick(F, ev) {
    const p = plotPixel(F, ev);
    if (!p) return;
    // A drag that ends with a click is a zoom, not a pin.
    if (F.downAt && Math.hypot(ev.clientX - F.downAt[0], ev.clientY - F.downAt[1]) > 4) return;
    let step = Math.round(fromX(p.x) / G.dt);
    // A column of a forecast map is the step forecast, so the anchor that made it is earlier.
    if (F.type === "map" && F.map.spec.k === "fc_tau") step -= 1 + paramOf(S.figs[F.key].params, "tau");
    let best = null, d = Infinity;
    AX.anchors.forEach((a, i) => { if (AX.anchor_valid[i] && Math.abs(a - step) < d) { d = Math.abs(a - step); best = i; } });
    if (best === null) return;
    pinned = best;
    refreshShapes();
    renderInspector();
  }

  // ---- Anchor inspector -------------------------------------------------------------------------------
  function renderInspector() {
    if (pinned === null) return;
    const anchor = AX.anchors[pinned];
    const tch = +document.getElementById("insp-channel").value;
    const C = G_C(), H = G.H;
    document.getElementById("inspector-where").textContent = "Anchor at step " + anchor + ", " + (anchor * G.dt).toFixed(0) + " s from segment start, " + ((G.t0 + anchor * G.dt) / 60).toFixed(1) + " min relative to delivery.";
    const cell = (name, tau) => arr(name).v[(pinned * H + tau) * C + tch];
    const scoredAt = tau => cellScored(tch, tau);
    const phi = arr("ar_coef").v[tch];
    const marginalSigma = name => {
      // Under an AR(1) residual the marginal variance accumulates the innovations along the horizon.
      let variance = 0;
      return Array.from({ length: H }, (_, t) => { variance = Math.exp(cell(name, t)) + phi * phi * variance; return Math.sqrt(variance); });
    };
    const taus = Array.from({ length: H }, (_, i) => i);
    const xFuture = taus.map(t => (t + 1) * G.dt);
    const truthAt = s => { return s >= 0 && s < G.T ? arr("truth").v[s * C + tch] : NaN; };
    const xPast = [], yPast = [];
    for (let k = -30; k <= 0; k++) { xPast.push(k * G.dt); yPast.push(truthAt(anchor + k)); }
    const truthFuture = taus.map(t => stepScored(pinned, t) ? truthAt(anchor + 1 + t) : NaN);
    const band = (mu, lv, sign) => { const sd = marginalSigma(lv); return taus.map(t => scoredAt(t) ? cell(mu, t) + sign * 2 * sd[t] : NaN); };
    const meanOf = mu => taus.map(t => scoredAt(t) ? cell(mu, t) : NaN);
    const colB = seriesColour(1), colF = seriesColour(0), ink = cssVar("--text-primary");
    const fill = c => c.startsWith("#") ? c + "33" : c;
    const dataF = [
      { x: xFuture, y: band("mu_base", "logvar_base", -1), mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x: xFuture, y: band("mu_base", "logvar_base", 1), mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: fill(colB), name: "base, 2 marginal sigma", hoverinfo: "skip" },
      { x: xFuture, y: band("mu_full", "logvar_full", -1), mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x: xFuture, y: band("mu_full", "logvar_full", 1), mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: fill(colF), name: "full, 2 marginal sigma", hoverinfo: "skip" },
      { x: xFuture, y: meanOf("mu_base"), mode: "lines", line: { color: colB, width: 2.6 }, name: "base mean" },
      { x: xFuture, y: meanOf("mu_full"), mode: "lines", line: { color: colF, width: 2.6 }, name: "full mean" },
      { x: xPast.concat(xFuture), y: yPast.concat(truthFuture), mode: "lines", line: { color: ink, width: 2 }, name: "truth" },
    ];
    const smallLayout = (xt, yt) => ({
      autosize: true, height: 240, margin: { l: 52, r: 56, t: 40, b: 34 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: font(),
      hovermode: "x unified", showlegend: true, legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0, font: { size: 10 } },
      xaxis: Object.assign(axisBase(), { title: { text: xt, standoff: 4 } }), yaxis: Object.assign(axisBase(), { title: { text: yt, standoff: 2 } }),
    });
    const fl = smallLayout("seconds from anchor", D.params.tch.options[tch]);
    fl.shapes = [{ type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1, line: { color: cssVar("--text-muted"), width: 1.2 } }];
    Plotly.react("insp-forecast", dataF, fl, { displaylogo: false, responsive: true });

    const lags = AX.lag_s, nLag = lags.length;
    const att = arr("attn"), heads = att.d[1];
    const dataL = [];
    for (let h = 0; h < heads; h++) {
      const y = lags.map((_, l) => att.v[(anchor * heads + h) * nLag + l]);
      dataL.push({ x: lags, y, mode: "lines", name: "head " + h, line: { color: seriesColour(h), width: 1.8 } });
    }
    const m = arr("attn_mean");
    dataL.push({ x: lags, y: lags.map((_, l) => m.v[anchor * nLag + l]), mode: "lines", name: "mean", line: { color: ink, width: 2.8 } });
    const right = document.getElementById("insp-right").value;
    const lagLayout = smallLayout("lag (s, stored-coefficient time)", "attention weight");
    let y2 = null, y2title = "";
    if (right === "up" && D.arrays.up) {
      const up = arr("up_stored").v;
      y2 = lags.map(l => { const s = Math.round((anchor * G.dt - l) / G.raw_dt); return s >= 0 && s < up.length ? up[s] : NaN; });
      y2title = "UA at anchor minus lag";
    } else {
      const k = arr("kl_lag");
      y2 = lags.map((_, l) => k.v[anchor * nLag + l]);
      y2title = "KL attribution (nats)";
    }
    dataL.push({ x: lags, y: y2, mode: "lines", name: y2title, yaxis: "y2", line: { color: seriesColour(4), width: 2.4 } });
    const [lagLeft, lagRight] = alignAxes([targetRange("auto", dataL.slice(0, -1).map(t => t.y)), targetRange("auto", [y2])]);
    lagLayout.yaxis = Object.assign(lagLayout.yaxis, tickProps(lagLeft));
    lagLayout.yaxis.minor = Object.assign({}, axisBase().minor, { tick0: lagLeft.tick0, dtick: minorStep(lagLeft.step) });
    lagLayout.yaxis2 = Object.assign(axisBase(), tickProps(lagRight), { overlaying: "y", side: "right", showgrid: false, minor: { showgrid: false }, title: { text: y2title, standoff: 2 } });
    Plotly.react("insp-lag", dataL, lagLayout, { displaylogo: false, responsive: true });

    const k = arr("kld_per_dim");
    const dims = k.d[1];
    const values = Array.from({ length: dims }, (_, d) => k.v[anchor * dims + d]);
    const klLayout = smallLayout("latent dimension", "nats");
    klLayout.showlegend = false; klLayout.hovermode = "closest";
    Plotly.react("insp-kl", [{ type: "bar", x: values.map((_, d) => d), y: values, marker: { color: seriesColour(0) }, hovertemplate: "z %{x}: %{y:.3f} nats<extra></extra>" }], klLayout, { displaylogo: false, responsive: true });
  }

  // ---- Header -------------------------------------------------------------------------------------
  function header() {
    const M = D.meta;
    const tag = v => v ? '<span class="tag">' + escapeHtml(v) + "</span>" : "";
    document.getElementById("identity").innerHTML =
      '<span class="title">Sample ' + String(M.index).padStart(4, "0") + "</span>" + tag(M.class) + tag(M.subgroup) +
      '<span class="muted">GUID ' + escapeHtml(M.guid) + "</span>" +
      '<span class="muted">' + M.hours_before_delivery.toFixed(2) + " h before delivery</span>" +
      (M.labour_onset !== null ? '<span class="muted">' + (M.labour_onset / 3600).toFixed(2) + " h from labour onset</span>" : "") +
      (M.second_stage !== null ? '<span class="muted">' + (M.second_stage / 3600).toFixed(2) + " h from second stage onset</span>" : "") +
      (M.run ? '<span class="muted">' + escapeHtml(M.run) + "</span>" : "");
    const finite = name => { const v = arr(name).v; let n = 0, s = 0; for (const x of v) if (Number.isFinite(x)) { n++; s += x; } return n ? s / n : NaN; };
    const contributing = arr("score_contributing").v;
    const scoredCount = contributing.reduce((n, v) => n + (v > 0.5 ? 1 : 0), 0);
    const scoredMean = name => { const v = arr(name).v; let n = 0, s = 0; AX.anchors.forEach((a, i) => { if (contributing[i] > 0.5 && Number.isFinite(v[a])) { n++; s += v[a]; } }); return n ? s / n : NaN; };
    const tiles = [
      ["NLL base", M.metrics.nll_base_block], ["NLL full", M.metrics.nll_full_block], ["Gap", M.metrics.pred_gap],
      ["MC gap", M.metrics.mc_pred_gap], ["Mean gap", M.metrics.mean_pred_gap], ["KL", M.metrics.source_conditioned_kl_raw],
      ["Mean K_t, scored anchors", scoredMean("kld")],
      ["Scored anchors", scoredCount + " of " + nAnchors],
      ["Contractions", D.events.contractions.length], ["Decelerations", D.events.decelerations.length],
    ];
    if (D.arrays.up_valid) tiles.push(["UA valid %", 100 * finite("up_valid")]);
    if (D.arrays.fhr_valid) tiles.push(["FHR valid %", 100 * finite("fhr_valid")]);
    if (D.arrays.event_gate) tiles.push(["Mean gate", finite("event_gate")]);
    document.getElementById("tiles").innerHTML = tiles.map(([n, v]) =>
      '<div class="tile"><div class="name">' + n + '</div><div class="value">' + (typeof v === "string" ? v : v === null || v === undefined || Number.isNaN(v) ? "n/a" : (Math.abs(v) >= 1000 ? v.toFixed(0) : Number.isInteger(v) ? v : v.toFixed(2))) + "</div></div>").join("");
    document.getElementById("foot").textContent = M.lag_note + " Rows from the collection pass come from Monte Carlo draws; traces marked 'this forward' come from one forward pass.";

    const list = window.DASHBOARD_INDEX || [];
    const file = decodeURIComponent(location.pathname.split("/").pop());
    const pos = list.findIndex(e => e.file === file);
    const set = (id, target) => {
      const a = document.getElementById(id);
      if (target) { a.href = encodeURI(target.file); a.addEventListener("click", ev => { ev.preventDefault(); location.href = encodeURI(target.file) + "#view=" + encodeURIComponent(btoa(JSON.stringify(S))); }); }
      else a.setAttribute("aria-disabled", "true");
    };
    set("nav-prev", pos > 0 ? list[pos - 1] : null);
    set("nav-next", pos >= 0 && pos < list.length - 1 ? list[pos + 1] : null);
    if (pos >= 0) document.getElementById("nav-pos").textContent = (pos + 1) + " of " + list.length;
  }

  // ---- Controls ------------------------------------------------------------------------------------
  const inspChannel = document.getElementById("insp-channel");
  D.params.tch.options.forEach((o, i) => { const opt = document.createElement("option"); opt.value = i; opt.textContent = o; inspChannel.appendChild(opt); });
  inspChannel.value = paramOf({}, "tch");
  inspChannel.addEventListener("change", renderInspector);
  document.getElementById("insp-right").addEventListener("change", renderInspector);
  const xMode = document.getElementById("x-mode");
  xMode.value = S.xMode;
  xMode.addEventListener("change", () => { S.xMode = xMode.value; renderAll(); saveState(); });
  Object.keys(S.overlays).forEach(name => {
    const box = document.getElementById("ov-" + name);
    if (!box) return;
    box.checked = S.overlays[name];
    box.addEventListener("change", () => { S.overlays[name] = box.checked; refreshShapes(); saveState(); });
  });
  document.getElementById("reset-zoom").addEventListener("click", () => {
    viewRange = null; renderAll();
  });
  const themeButton = document.getElementById("theme-toggle");
  const paintTheme = () => { themeButton.textContent = S.theme === "dark" ? "Light" : "Dark"; };
  paintTheme();
  themeButton.addEventListener("click", () => {
    S.theme = S.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = S.theme;
    paintTheme(); renderAll(); saveState();
  });

  header();
  renderAll();
  bindSync();
  let resizeTimer = 0;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderAll, 200); });
})();
