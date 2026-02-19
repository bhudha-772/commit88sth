(function(){
  // ---- config ----
  const TICK_DISPLAY_LIMIT = 10;           // show 10 in-panel
  const OVERLAY_MAX_ROWS = 200;           // how many rows we present in modal
  const ANALYSIS_BUFFER_LIMIT = 35;       // buffer visible in UI
  const EVENTS_URL = "/events";
  const START_DERIV = "/control/start_deriv";
  const STOP_DERIV = "/control/stop_deriv";
  const START_ANALYSIS = "/control/start_analysis";
  const STOP_ANALYSIS = "/control/stop_analysis";
  const CLEAR_TICKS = "/control/clear_ticks";
  const STATUS_URL = "/control/status";

  // ---- state ----
  let ticks = [];
  let analysisBuffer = [];
  let summary = [];
  let wins = 0, losses = 0;
  let derivRunning = false;
  let analysisRunning = false;
  let pendingPrediction = null; // {prediction_id, pred_digit, confidence, reason, ts}
  const INITIAL_MARKET = ((window.HERO_INITIAL_MARKET || "") + "").trim().toUpperCase();
  const ACTIVE_MARKET_FILTER = (INITIAL_MARKET && INITIAL_MARKET !== "ANALYSIS") ? INITIAL_MARKET : null;

  // account/auth state
  let accountAuth = { connected: false, mode: "demo", token: null, app_id: 71710 };
  let balancesPollInterval = null;

  // digit stats map for confidence display
  const digitStats = {}; // digit -> {confidence:0..1, passedCount, passedRules, lastUpdatedEpoch, isCandidate}

  // ---- DOM refs (defensive: may be null if UI variant hides elements) ----
  const safe = id => document.getElementById(id) || null;

  const last10Container = safe("last10Container");
  const tickListEl = safe("tickList");
  const tickCountEl = safe("tickCount");
  const btnToggleTickList = safe("btnToggleTickList");
  const bufferListEl = safe("bufferList");
  const analysisBufferSizeEl = safe("analysisBufferSize");
  const digitsGridEl = safe("digitsGrid");
  const analysisStatusText = safe("analysisStatusText");
  const summaryBody = safe("summaryBody");
  const winsCountEl = safe("winsCount");
  const lossesCountEl = safe("lossesCount");
  const rawDebugEl = safe("rawDebug");
  const sseStateEl = safe("sseState");
  const latestAnalysisEl = safe("latestAnalysis");
  const latestTickDigitEl = safe("latestTickDigit");

  const btnStartDeriv = safe("btnStartDeriv");
  const btnStartAnalysis = safe("btnStartAnalysis");
  const btnClearTicks = safe("btnClearTicks");
  const btnClearAnalysis = safe("btnClearAnalysis");
  const btnClearSummary = safe("btnClearSummary");
  const btnClearRaw = safe("btnClearRaw");
  const btnManualPredict = safe("btnManualPredict");
  const btnSimTick = safe("btnSimTick");

  const btnExpandAnalysis = safe("btnExpandAnalysis");

  // account UI
  const accountModeEl = safe("accountMode");
  const apiTokenEl = safe("apiToken");
  const btnConnectAccount = safe("btnConnectAccount");
  const realBalanceEl = safe("realBalance");
  const demoBalanceEl = safe("demoBalance");

  // overlay refs for tick
  const tickOverlay = safe("tickOverlay");
  const overlayTickList = safe("overlayTickList");
  const overlayTickCount = safe("overlayTickCount");
  const btnCloseTickOverlay = safe("btnCloseTickOverlay");

  // overlay refs for analysis
  const analysisOverlay = safe("analysisOverlay");
  const overlayAnalysisCount = safe("overlayAnalysisCount");
  const overlayDigits = safe("overlayDigits");
  const overlayBufferList = safe("overlayBufferList");
  const btnCloseAnalysisOverlay = safe("btnCloseAnalysisOverlay");

  // build digits grid (main UI) — create cells only if digitsGridEl exists
  if (digitsGridEl){
    for (let d=0; d<10; d++){
      const cell = document.createElement("div");
      cell.className = "digit-cell";
      cell.dataset.d = d;
      cell.innerHTML = `<div class="digit-primary">${d}</div><div class="digit-confidence" data-d="${d}">—</div>`;
      digitsGridEl.appendChild(cell);
      digitStats[d] = { confidence: 0.0, passedCount: 0, lastUpdatedEpoch: 0, isCandidate: false };
    }
  } else {
    // still initialize stats so other logic won't fail
    for (let d=0; d<10; d++) digitStats[d] = { confidence: 0.0, passedCount: 0, lastUpdatedEpoch: 0, isCandidate: false };
  }

  // ---- helpers ----
  function logRaw(msg){
    const ts = new Date().toISOString().slice(11,23);
    if (rawDebugEl) {
      rawDebugEl.textContent = `${ts} ${msg}\n` + rawDebugEl.textContent;
      if (rawDebugEl.textContent.length > 24000) rawDebugEl.textContent = rawDebugEl.textContent.slice(0,24000);
    } else {
      console.log("RAW:", ts, msg);
    }
  }

  function extractLastDecimal(payload){
    if (payload == null) return null;
    if (payload.hasOwnProperty("last_decimal") && payload.last_decimal !== null && payload.last_decimal !== "" ){
      const v = payload.last_decimal;
      const n = parseInt(v);
      if (!Number.isNaN(n) && n >= 0 && n <= 9) return n;
    }
    // fallback: derive from price-like fields if server omits last_decimal
    if (!Array.isArray(payload) && typeof payload === "object"){
      for (const key of ["price_str_text", "price", "quote", "ask", "bid"]) {
        if (!Object.prototype.hasOwnProperty.call(payload, key)) continue;
        const raw = payload[key];
        if (raw === null || raw === undefined || raw === "") continue;
        let s = String(raw).trim();
        if (s.startsWith("'") || s.startsWith('"')) s = s.slice(1).trim();
        if (s.includes(".")) {
          const frac = s.split(".")[1] || "";
          for (let i = frac.length - 1; i >= 0; i--){
            const ch = frac[i];
            if (ch >= "0" && ch <= "9") return parseInt(ch, 10);
          }
        }
      }
    }
    // support legacy array rows
    if (Array.isArray(payload) && payload.length >= 5){
      const n = parseInt(payload[4]);
      if (!Number.isNaN(n) && n >= 0 && n <= 9) return n;
    }
    return null;
  }

  function getPayloadSymbol(payload){
    try {
      return ((payload && (payload.symbol || payload.market || payload.market_code || "")) + "").toUpperCase().trim();
    } catch(e){
      return "";
    }
  }

  function isAllowedByMarketFilter(payload){
    if (!ACTIVE_MARKET_FILTER) return true;
    return getPayloadSymbol(payload) === ACTIVE_MARKET_FILTER;
  }

  function getVisibleTicks(){
    if (!ACTIVE_MARKET_FILTER) return ticks;
    return ticks.filter(t => getPayloadSymbol(t) === ACTIVE_MARKET_FILTER);
  }

  function fmtTime(epochSec){
    try {
      const v = Number(epochSec) || Date.now();
      const ms = (v > 1e12) ? v : v * 1000;
      return new Date(ms).toLocaleTimeString();
    } catch(e){
      return "";
    }
  }

  // Render in-panel preview (last N ticks)
  function renderLast10(){
    if (!last10Container) return;
    last10Container.innerHTML = "";
    const visibleTicks = getVisibleTicks();
    const lastDigits = visibleTicks.slice(-TICK_DISPLAY_LIMIT).map(p => {
      const d = extractLastDecimal(p);
      return (d === null || d === undefined) ? "—" : String(d);
    });

    // pad up to TICK_DISPLAY_LIMIT
    const padCount = Math.max(0, TICK_DISPLAY_LIMIT - lastDigits.length);
    for (let i=0;i<padCount;i++){
      const el = document.createElement("div");
      el.className = "tick-cell";
      el.textContent = "—";
      last10Container.appendChild(el);
    }
    for (let v of lastDigits){
      const el = document.createElement("div");
      el.className = "tick-cell";
      el.textContent = v;
      last10Container.appendChild(el);
    }

    if (tickCountEl) tickCountEl.textContent = `${visibleTicks.length} ticks`;
    renderTickList();
  }

  // Render panel tick-list: only the last N rows, newest at top
  function renderTickList(){
    if (!tickListEl) return;
    tickListEl.innerHTML = "";
    const visibleTicks = getVisibleTicks();
    if (!visibleTicks.length){
      const p = document.createElement("div");
      p.className = "tick-row";
      p.innerHTML = `<div style="color:var(--muted)">${ACTIVE_MARKET_FILTER ? ("No ticks for " + ACTIVE_MARKET_FILTER) : "No ticks yet"}</div>`;
      tickListEl.appendChild(p);
      return;
    }

    const rows = visibleTicks.slice(-TICK_DISPLAY_LIMIT).reverse(); // newest first (max N)
    for (let row of rows){
      const time = fmtTime(row.epoch || Math.floor(Date.now()/1000));
      const sym = row.symbol || "";
      const d = extractLastDecimal(row);
      const dstr = (d === null || d === undefined) ? "—" : String(d);
      const wrapper = document.createElement("div");
      wrapper.className = "tick-row";
      wrapper.dataset.epoch = row.epoch || Math.floor(Date.now()/1000);
      wrapper.dataset.d = (dstr === "—") ? "" : dstr;
      wrapper.innerHTML = `
        <div class="tick-time">${time}</div>
        <div class="tick-symbol">${sym}</div>
        <div class="tick-digit">${dstr}</div>
        <div style="color:var(--muted);font-size:13px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
          ${row.price !== undefined && row.price !== null ? `price ${row.price}` : ""}
        </div>
      `;
      tickListEl.appendChild(wrapper);
    }

    // newest at top
    tickListEl.scrollTop = 0;
  }

  // Overlay render: show up to OVERLAY_MAX_ROWS newest-first
  function renderTickOverlay(){
    if (!overlayTickList || !overlayTickCount) return;
    overlayTickList.innerHTML = "";
    const visibleTicks = getVisibleTicks();
    const rows = visibleTicks.slice(-OVERLAY_MAX_ROWS).reverse();
    for (let row of rows){
      const time = fmtTime(row.epoch || Math.floor(Date.now()/1000));
      const sym = row.symbol || "";
      const d = extractLastDecimal(row);
      const dstr = (d === null || d === undefined) ? "—" : String(d);
      const wrapper = document.createElement("div");
      wrapper.className = "tick-row";
      wrapper.style.borderBottom = "1px solid rgba(2,6,23,0.04)";
      wrapper.style.padding = "8px 6px";
      wrapper.dataset.epoch = row.epoch || Math.floor(Date.now()/1000);
      wrapper.dataset.d = (dstr === "—") ? "" : dstr;
      wrapper.innerHTML = `
        <div class="tick-time">${time}</div>
        <div class="tick-symbol">${sym}</div>
        <div class="tick-digit">${dstr}</div>
        <div style="color:var(--muted);font-size:13px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
          ${row.price !== undefined && row.price !== null ? `price ${row.price}` : ""}
        </div>
      `;
      overlayTickList.appendChild(wrapper);
    }
    overlayTickCount.textContent = `${visibleTicks.length} ticks`;
    overlayTickList.parentElement && (overlayTickList.parentElement.scrollTop = 0);
  }

  // Render analysis overlay: show digits grid and full buffer
  function renderAnalysisOverlay(){
    if (!overlayDigits || !overlayBufferList || !overlayAnalysisCount) return;
    overlayDigits.innerHTML = "";
    // digits (bigger display)
    const digitsWrap = document.createElement("div");
    digitsWrap.style.display = "grid";
    digitsWrap.style.gridTemplateColumns = "repeat(10,1fr)";
    digitsWrap.style.gap = "8px";
    digitsWrap.style.marginBottom = "12px";
    for (let d=0; d<10; d++){
      const el = document.createElement("div");
      el.className = "digit-cell";
      el.style.height = "48px";
      el.innerHTML = `<div class="digit-primary">${d}</div><div class="digit-confidence" data-d="${d}">${(digitStats[d] && digitStats[d].confidence) ? (Number(digitStats[d].confidence).toFixed(3)) : "—"}</div>`;
      if (analysisBuffer.length && analysisBuffer[analysisBuffer.length-1] === d && analysisRunning){
        el.classList.add("current");
      }
      if (digitStats[d] && digitStats[d].isCandidate){
        el.style.boxShadow = "0 12px 36px rgba(6,182,212,0.12)";
      }
      digitsWrap.appendChild(el);
    }
    overlayDigits.appendChild(digitsWrap);

    // buffer list (vertical)
    overlayBufferList.innerHTML = "";
    const buf = analysisBuffer.slice(-200); // show last 200 in overlay
    const bufWrap = document.createElement("div");
    bufWrap.style.display = "flex";
    bufWrap.style.flexDirection = "column";
    bufWrap.style.gap = "6px";
    bufWrap.style.padding = "6px";
    bufWrap.style.maxHeight = "56vh";
    bufWrap.style.overflow = "auto";
    bufWrap.style.border = "1px solid rgba(2,6,23,0.04)";
    bufWrap.style.borderRadius = "8px";
    bufWrap.style.background = "#fff";
    for (let i = buf.length - 1; i >= 0; i--){ // newest first
      const v = buf[i];
      const row = document.createElement("div");
      row.style.display = "flex";
      row.style.alignItems = "center";
      row.style.gap = "8px";
      row.style.padding = "8px";
      row.style.borderBottom = "1px solid rgba(2,6,23,0.03)";
      const idx = document.createElement("div");
      idx.style.color = "var(--muted)";
      idx.style.width = "70px";
      idx.textContent = `#${i}`;
      const val = document.createElement("div");
      val.style.fontWeight = "700";
      val.textContent = (v === null || v === undefined) ? "—" : String(v);
      row.appendChild(idx);
      row.appendChild(val);
      bufWrap.appendChild(row);
    }
    overlayBufferList.appendChild(bufWrap);

    overlayAnalysisCount.textContent = `buffer: ${analysisBuffer.length}`;
    overlayAnalysisCount.parentElement && (overlayAnalysisCount.parentElement.scrollTop = 0);
  }

  function renderAnalysisBuffer(){
    if (analysisBufferSizeEl) analysisBufferSizeEl.textContent = analysisBuffer.length;
    if (!bufferListEl) return;
    bufferListEl.innerHTML = "";
    const last = analysisBuffer.slice(-ANALYSIS_BUFFER_LIMIT);
    for (let v of last){
      const el = document.createElement("div");
      el.className = "buffer-item";
      el.textContent = (v === null || v === undefined) ? "—" : String(v);
      bufferListEl.appendChild(el);
    }
  }

  function updateDigitsCursor(){
    // clear caret and "current" classes
    if (!digitsGridEl) return;
    const oldCursors = digitsGridEl.querySelectorAll(".digit-cursor");
    oldCursors.forEach(c => c.remove());
    const cells = digitsGridEl.querySelectorAll(".digit-cell");
    cells.forEach(c => { c.classList.remove("current"); c.style.boxShadow = ""; });
    const last = analysisBuffer.length ? analysisBuffer[analysisBuffer.length-1] : null;

    if (last === null || last === undefined){
      if (latestTickDigitEl) latestTickDigitEl.textContent = "—";
      return;
    }
    if (latestTickDigitEl) latestTickDigitEl.textContent = String(last);

    if (!analysisRunning) return;

    const node = digitsGridEl.querySelector(`.digit-cell[data-d='${last}']`);
    if (node) {
      node.classList.add("current");
      const caret = document.createElement("div");
      caret.className = "digit-cursor";
      node.appendChild(caret);
    }
  }

  function addSummaryRow(item){
    if (!summaryBody) return;
    const tr = document.createElement("tr");
    const time = new Date((item.ts || Date.now()) * 1000);
    const timestr = time.toLocaleTimeString();
    tr.innerHTML = `
      <td>${timestr}</td>
      <td>${item.predicted_digit ?? "—"}</td>
      <td>${item.actual ?? "—"}</td>
      <td class="${item.win ? "summary-win" : "summary-loss"}">${item.win ? "WIN" : "LOSS"}</td>
      <td>${(item.confidence !== undefined) ? (Number(item.confidence).toFixed(3)) : "—"}</td>
      <td style="max-width:200px;white-space:normal">${item.reason ?? ""}</td>
    `;
    if (item.win) tr.style.background = "linear-gradient(90deg, rgba(16,185,129,0.04), transparent)";
    else tr.style.background = "linear-gradient(90deg, rgba(239,68,68,0.04), transparent)";
    summaryBody.insertBefore(tr, summaryBody.firstChild);
    if (winsCountEl) winsCountEl.textContent = wins;
    if (lossesCountEl) lossesCountEl.textContent = losses;
  }

  function clearSummary(){
    summary = [];
    wins = 0; losses = 0;
    if (summaryBody) summaryBody.innerHTML = "";
    if (winsCountEl) winsCountEl.textContent = 0;
    if (lossesCountEl) lossesCountEl.textContent = 0;
  }

  function updateDigitsConfidences(){
    if (!digitsGridEl) return;
    for (let d=0; d<10; d++){
      const confEl = document.querySelector(`.digit-confidence[data-d='${d}']`);
      const stats = digitStats[d] || {confidence:0};
      if (confEl){
        if (stats && typeof stats.confidence === "number" && stats.confidence > 0){
          confEl.textContent = `${(stats.confidence*100).toFixed(1)}%`;
        } else {
          confEl.textContent = "—";
        }
      }
      const cell = digitsGridEl.querySelector(`.digit-cell[data-d='${d}']`);
      if (cell){
        if (stats && stats.isCandidate){
          cell.style.boxShadow = "0 12px 36px rgba(6,182,212,0.12)";
        } else {
          cell.style.boxShadow = "";
        }
      }
    }
  }

  function evaluateTopCandidate(){
    const arr = [];
    for (let d=0; d<10; d++){
      const s = digitStats[d] || {confidence:0};
      arr.push({digit:d, confidence: s.confidence || 0});
    }
    arr.sort((a,b) => b.confidence - a.confidence);
    const top = arr[0] || {digit:0, confidence:0};
    const second = arr[1] || {digit:1, confidence:0};

    for (let d=0; d<10; d++) (digitStats[d] || (digitStats[d]={})).isCandidate = false;

    const bufLen = analysisBuffer.length;
    const threshold = 0.70;
    const closeDelta = 0.03;

    if (bufLen < ANALYSIS_BUFFER_LIMIT){
      if (analysisStatusText) analysisStatusText.textContent = `Collecting data (${bufLen}/${ANALYSIS_BUFFER_LIMIT})...`;
      updateDigitsConfidences();
      return;
    }

    if (Math.abs(top.confidence - second.confidence) <= closeDelta && top.confidence > 0.35 && second.confidence > 0.35){
      if (analysisStatusText) analysisStatusText.textContent = `Waiting for digits to restabilize — top ${top.digit} ${(top.confidence*100).toFixed(1)}% vs ${second.digit} ${(second.confidence*100).toFixed(1)}%`;
      updateDigitsConfidences();
      return;
    }

    if (top.confidence >= threshold){
      digitStats[top.digit].isCandidate = true;
      if (analysisStatusText) analysisStatusText.textContent = `Suitable candidate: ${top.digit} (confidence ${(top.confidence*100).toFixed(1)}%) — ready to consider a prediction`;
      updateDigitsConfidences();
      return;
    }

    if (top.confidence < 0.50){
      if (analysisStatusText) analysisStatusText.textContent = `Market noisy — no confident digit yet (top ${top.digit}: ${(top.confidence*100).toFixed(1)}%)`;
      updateDigitsConfidences();
      return;
    }

    if (analysisStatusText) analysisStatusText.textContent = `Top candidate: ${top.digit} ${(top.confidence*100).toFixed(1)}% — keep observing`;
    updateDigitsConfidences();
  }

  function updateDigitStatsFromPayload(payload){
    let updated = false;
    try {
      if (!payload) return;
      if (Array.isArray(payload.scores) && payload.scores.length){
        for (const s of payload.scores){
          const d = Number(s.digit);
          if (Number.isNaN(d)) continue;
          digitStats[d] = digitStats[d] || {};
          digitStats[d].confidence = typeof s.confidence === "number" ? s.confidence : parseFloat(s.confidence) || 0;
          digitStats[d].passedCount = s.passed_count || s.passedCount || s.passed || 0;
          if (s.passed_rules) digitStats[d].passed_rules = s.passed_rules;
          digitStats[d].lastUpdatedEpoch = payload.epoch || Math.floor(Date.now()/1000);
          updated = true;
        }
      } else if (payload.digit_confidences && typeof payload.digit_confidences === "object"){
        for (const k of Object.keys(payload.digit_confidences)){
          const d = Number(k);
          if (Number.isNaN(d)) continue;
          const conf = Number(payload.digit_confidences[k]) || 0;
          digitStats[d] = digitStats[d] || {};
          digitStats[d].confidence = conf;
          digitStats[d].lastUpdatedEpoch = payload.epoch || Math.floor(Date.now()/1000);
          updated = true;
        }
      } else if (payload.scores_map && typeof payload.scores_map === "object"){
        for (const [k,v] of Object.entries(payload.scores_map)){
          const d = Number(k);
          if (Number.isNaN(d)) continue;
          digitStats[d] = digitStats[d] || {};
          digitStats[d].confidence = Number(v.confidence) || Number(v) || 0;
          digitStats[d].lastUpdatedEpoch = payload.epoch || Math.floor(Date.now()/1000);
          updated = true;
        }
      }
    } catch(e){
      logRaw("digit stats parse error: " + (e.message||e));
    }
    if (updated){
      updateDigitsConfidences();
      evaluateTopCandidate();
    }
  }

  function renderPendingVisuals(){
    if (!digitsGridEl) return;
    const cells = digitsGridEl.querySelectorAll(".digit-cell");
    cells.forEach(c => c.style.boxShadow = "");
    if (!pendingPrediction) return;
    const node = digitsGridEl.querySelector(`.digit-cell[data-d='${pendingPrediction.pred_digit}']`);
    if (node) node.style.boxShadow = "0 12px 36px rgba(6,182,212,0.12)";
  }

  function highlightTickForResult(pred_digit, result){
    if (!last10Container && !tickListEl) return;
    const last10Cells = last10Container ? Array.from(last10Container.children) : [];
    last10Cells.forEach(c => { c.classList.remove("win","loss"); });

    const rows = tickListEl ? Array.from(tickListEl.querySelectorAll(".tick-row")) : [];
    rows.forEach(r => r.classList.remove("win","loss"));

    if (result === "WIN"){
      for (let i = ticks.length - 1; i >= 0; i--){
        const t = ticks[i];
        const d = extractLastDecimal(t);
        if (d === Number(pred_digit)){
          const recent = ticks.slice(-TICK_DISPLAY_LIMIT).map(p => extractLastDecimal(p));
          const idxInLast = recent.lastIndexOf(d);
          if (idxInLast >= 0 && last10Container){
            const pad = Math.max(0, TICK_DISPLAY_LIMIT - recent.length);
            const cellPos = pad + idxInLast;
            const el = last10Container.children[cellPos];
            if (el) el.classList.add("win");
          }
          for (let r of rows){
            const rd = r.dataset.d || "";
            if (Number(rd) === Number(pred_digit)){
              r.classList.add("win");
              break;
            }
          }
          return;
        }
      }
      const newestCell = last10Container ? last10Container.lastElementChild : null;
      if (newestCell) newestCell.classList.add("win");
      if (rows.length) rows[0].classList.add("win");
    } else {
      const newestCell = last10Container ? last10Container.lastElementChild : null;
      if (newestCell) newestCell.classList.add("loss");
      if (rows.length) rows[0].classList.add("loss");
    }
  }

  // ---- SSE & events ----
  let evtSource = null;
  function connectSSE(){
    try {
      if (evtSource){ try { evtSource.close(); } catch(e){} evtSource = null; }
      evtSource = new EventSource(EVENTS_URL);
      if (sseStateEl) sseStateEl.textContent = "connecting";
      evtSource.onopen = function(){ if (sseStateEl) sseStateEl.textContent = "open"; logRaw("SSE open"); };
      evtSource.onerror = function(e){ if (sseStateEl) sseStateEl.textContent = "error"; logRaw("SSE error"); };
      evtSource.onmessage = function(ev){
        try {
          // Accept either { payload: {...}} or direct payload object
          const parsed = JSON.parse(ev.data);
          if (parsed && parsed.payload) handleServerPayload(parsed.payload);
          else handleServerPayload(parsed);
        } catch(e){
          logRaw("onmessage parse error: "+(e.message||e));
        }
      };
      evtSource.addEventListener("recent", function(ev){
        try {
          const j = JSON.parse(ev.data);
          if (j && Array.isArray(j.recent)){
            const rows = j.recent;
            ticks = [];
            const tempBuffer = [];
            for (let row of rows){
              if (!Array.isArray(row)) continue;
              const sym = (row[6] || "").toString().toUpperCase();
              if (sym === "ANALYSIS") continue;
              const payload = {
                epoch: Number(row[0]) || Math.floor(Date.now()/1000),
                price: row[3] || null,
                last_decimal: (row[4] === "" ? null : row[4]),
                last_unit: (row[5] === "" ? null : row[5]),
                symbol: row[6] || ""
              };
              if (!isAllowedByMarketFilter(payload)) continue;
              ticks.push(payload);
              const d = extractLastDecimal(payload);
              if (d !== null && d !== undefined) tempBuffer.push(d);
            }
            analysisBuffer = analysisBuffer.concat(tempBuffer).slice(-2000);
            renderLast10();
            renderAnalysisBuffer();
            updateDigitsCursor();
          }
        } catch(e){ logRaw("recent parse error: "+(e.message||e)); }
      });
      evtSource.addEventListener("analysis", function(ev){
        try {
          const payload = JSON.parse(ev.data);
          handleAnalysisEvent(payload);
        } catch(e){ logRaw("analysis parse error: "+(e.message||e)); }
      });
      logRaw("SSE attached");
    } catch(err){
      logRaw("SSE attach error: " + (err && err.message ? err.message : err));
    }
  }

  function handleServerPayload(payload){
    if (!payload) return;
    const sym = ((payload.symbol || "") + "").toUpperCase();
    if (sym === "ANALYSIS"){ logRaw("Ignoring payload with symbol=ANALYSIS"); return; }
    if (!isAllowedByMarketFilter(payload)) return;
    ticks.push(payload);
    const d = extractLastDecimal(payload);
    if (d !== null && d !== undefined){
      analysisBuffer.push(d);
      if (analysisBuffer.length > 2000) analysisBuffer = analysisBuffer.slice(-2000);
    }
    renderLast10();
    renderAnalysisBuffer();
    updateDigitsCursor();

    if (tickListEl) tickListEl.scrollTop = 0;
    if (tickOverlay && tickOverlay.classList.contains("visible")) {
      renderTickOverlay();
      overlayTickList && (overlayTickList.parentElement && (overlayTickList.parentElement.scrollTop = 0));
    }
    if (analysisOverlay && analysisOverlay.classList.contains("visible")) {
      renderAnalysisOverlay();
      overlayAnalysisCount && (overlayAnalysisCount.parentElement && (overlayAnalysisCount.parentElement.scrollTop = 0));
    }
  }

  function handleAnalysisEvent(payload){
    logRaw("ANALYSIS: " + JSON.stringify(payload));
    if (latestAnalysisEl) latestAnalysisEl.textContent = (payload.analysis_event || payload.analysisEvent || "analysis");
    const ev = (payload.analysis_event || payload.analysisEvent || "").toString();

    if (ev === "digit_score_update" || payload.scores || payload.digit_confidences || payload.scores_map){
      updateDigitStatsFromPayload(payload);
      return;
    }

    if (ev === "suitable_prediction_found" || ev === "suitable_candidate"){
      const pred = payload.prediction_digit ?? payload.pred_digit ?? payload.pred;
      const conf = payload.confidence ?? payload.conf;
      const passed_rules = payload.passed_rules || payload.passed || null;
      const symbol = payload.symbol || "R_100";
      if (pendingPrediction){
        logRaw("Received suitable_prediction_found but pendingPrediction exists — ignoring duplicate");
        return;
      }
      pendingPrediction = {
        prediction_id: payload.prediction_id || ("pred_" + Date.now()),
        pred_digit: Number(pred),
        confidence: Number(conf) || null,
        reason: Array.isArray(passed_rules) ? passed_rules.join(",") : (passed_rules || "auto"),
        ts: payload.epoch || Math.floor(Date.now()/1000),
        symbol: symbol
      };
      if (digitStats[pendingPrediction.pred_digit]) digitStats[pendingPrediction.pred_digit].isCandidate = true;
      renderPendingVisuals();
      if (analysisStatusText) analysisStatusText.textContent = `Agent decided to predict ${pendingPrediction.pred_digit} (conf ${(pendingPrediction.confidence||0)}) — awaiting next ${payload.observe_limit || payload.n_test || 3} ticks for result`;
      return;
    }

    if (ev === "prediction_posted"){
      const pid = payload.prediction_id || ("pred_"+Date.now());
      pendingPrediction = {
        prediction_id: pid,
        pred_digit: Number(payload.prediction_digit),
        confidence: Number(payload.confidence) || null,
        reason: payload.reason || "",
        ts: payload.epoch || Math.floor(Date.now()/1000),
        symbol: payload.symbol || "R_100",
        prediction_mode: payload.prediction_mode || payload.mode || "differ"
      };
      accountModeEl && (accountAuth.mode = accountModeEl.value || accountAuth.mode);
      if (digitStats[pendingPrediction.pred_digit]) digitStats[pendingPrediction.pred_digit].isCandidate = true;
      renderPendingVisuals();
      if (analysisStatusText) analysisStatusText.textContent = `Prediction posted: ${pendingPrediction.pred_digit} — awaiting settlement`;
      return;
    }

    if (ev === "prediction_result" || ev === "settled"){
      const pred = payload.prediction_digit ?? payload.pred_digit ?? payload.pred;
      const conf = payload.confidence ?? null;
      const prediction_mode = payload.prediction_mode || payload.prediction_mode || payload.mode || "differ";
      let rawResult = payload.result ?? payload.status ?? payload.win;
      let isWin = false;
      if (typeof rawResult === "string") isWin = rawResult.toUpperCase() === "WIN";
      else if (typeof rawResult === "boolean") isWin = !!rawResult;
      else {
        const observed = Array.isArray(payload.observed_ticks) ? payload.observed_ticks : (payload.observed || []);
        if (prediction_mode.toLowerCase() === "differ") {
          isWin = !(observed.some(x => Number(x) === Number(pred)));
        } else {
          isWin = observed.some(x => Number(x) === Number(pred));
        }
      }

      const obj = {
        ts: payload.epoch || Math.floor(Date.now()/1000),
        prediction_id: payload.prediction_id || ("pred_"+Date.now()),
        predicted_digit: Number(pred),
        actual: (payload.observed_ticks && payload.observed_ticks.length ? payload.observed_ticks[payload.observed_ticks.length-1] : null),
        win: isWin,
        confidence: conf,
        reason: payload.reason || payload.passed_rules || payload.evidence || "",
        prediction_mode: prediction_mode
      };
      if (obj.win) wins++; else losses++;
      summary.unshift(obj);
      addSummaryRow(obj);

      highlightTickForResult(obj.predicted_digit, (obj.win ? "WIN" : "LOSS"));

      pendingPrediction = null;
      for (let d=0; d<10; d++) if (digitStats[d]) digitStats[d].isCandidate = false;
      renderPendingVisuals();
      if (analysisStatusText) analysisStatusText.textContent = `Prediction ended: ${obj.predicted_digit} -> ${obj.win ? "WIN" : "LOSS"} (${obj.reason || "no reason"}) • continuing analysis`;
      updateDigitsConfidences();
      return;
    }

    if (ev === "learning_update"){
      if (analysisStatusText) analysisStatusText.textContent = `Learning update: ${payload.pattern || ""} → ${payload.result || ""}`;
      return;
    }
    if (ev === "stopped"){
      if (analysisStatusText) analysisStatusText.textContent = payload.message || "analysis stopped";
      return;
    }
    if (ev === "network_issue"){
      if (analysisStatusText) analysisStatusText.textContent = payload.message || "network issue";
      return;
    }
    if (payload.message && analysisStatusText) analysisStatusText.textContent = payload.message;
  }

  // ---- Controls / network ----
  async function postJSON(url, body = {}){
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(body)
      });
      if (!res.ok){
        const txt = await res.text().catch(()=>"");
        logRaw(`POST ${url} failed ${res.status} ${txt}`);
        return {ok:false, status: res.status, text: txt};
      }
      const j = await res.json().catch(()=>null);
      return {ok:true, status:200, json: j};
    } catch(e){ logRaw(`POST ${url} exception: ${e.message||e}`); return {ok:false, status:0, error: e.message||e}; }
  }

  async function getJSON(url){
    try {
      const res = await fetch(url, {method:"GET"});
      if (!res.ok) {
        const txt = await res.text().catch(()=> "");
        logRaw(`GET ${url} failed ${res.status} ${txt}`);
        return {ok:false, status: res.status, text: txt};
      }
      const j = await res.json().catch(()=>null);
      return {ok:true, json:j};
    } catch(e){ logRaw(`GET ${url} exception: ${e.message||e}`); return {ok:false, error: e.message||e}; }
  }

  // Connect account handler (click) - Moved here so it has access to local refs
  async function connectAccountFixed(){
    const token = (apiTokenEl && (apiTokenEl.value || "").trim()) || "";
    let mode = "demo";
    try {
      if (accountModeEl){
        const tag = (accountModeEl.tagName || "").toLowerCase();
        const atype = (accountModeEl.type || "").toLowerCase();
        if (atype === "checkbox") {
          mode = accountModeEl.checked ? "real" : "demo";
        } else if (atype === "radio") {
          if (accountModeEl.checked && accountModeEl.value) mode = (accountModeEl.value||"demo").toLowerCase();
          else {
            const checked = document.querySelector('input[name="'+(accountModeEl.name||'accountMode')+'"]:checked');
            if (checked && checked.value) mode = (checked.value||"demo").toLowerCase();
          }
        } else if (tag === "select" || (accountModeEl.value !== undefined)) {
          mode = (accountModeEl.value || "demo").toLowerCase();
        } else {
          const checked = document.querySelector('input[name="accountMode"]:checked');
          if (checked && checked.value) mode = (checked.value||"demo").toLowerCase();
        }
      } else {
        const checked = document.querySelector('input[name="accountMode"]:checked');
        if (checked && checked.value) mode = (checked.value||"demo").toLowerCase();
      }
    } catch(e){
      logRaw("mode detect error: " + (e && e.message ? e.message : e));
      mode = accountAuth && accountAuth.mode ? accountAuth.mode : "demo";
    }

    if (mode !== "real") mode = "demo"; else mode = "real";

    if (!token){
      alert("Please paste your API token into the field.");
      try { apiTokenEl && apiTokenEl.focus(); } catch(e){}
      return;
    }

    try { if (btnConnectAccount) btnConnectAccount.disabled = true; } catch(e){}
    try { if (btnConnectAccount) btnConnectAccount.textContent = "Connecting…"; } catch(e){}
    logRaw(`Attempting to connect (${mode})`);

    const body = { app_id: 71710, token: token, mode: mode };
    const res = await postJSON("/control/connect_account", body);
    if (!res.ok){
      try { if (btnConnectAccount) btnConnectAccount.disabled = false; } catch(e){}
      try { if (btnConnectAccount) btnConnectAccount.textContent = "Connect account"; } catch(e){}
      logRaw("connect_account failed: " + (res.text || JSON.stringify(res)));
      alert("Connect failed — check token or server. See Raw Debug for details.");
      return;
    }

    accountAuth.connected = true;
    accountAuth.mode = mode;
    accountAuth.token = token;
    try {
      if (btnConnectAccount) {
        btnConnectAccount.textContent = "Account authenticated";
        btnConnectAccount.classList.remove("btn-glow");
        btnConnectAccount.classList.add("btn-success");
      }
      if (apiTokenEl) apiTokenEl.disabled = true;
      if (accountModeEl) accountModeEl.disabled = true;
    } catch(e){}

    logRaw(`Connected to Deriv (${mode}) — fetching balances`);
    await fetchBalancesBoth();

    if (balancesPollInterval) clearInterval(balancesPollInterval);
    balancesPollInterval = setInterval(fetchBalancesBoth, 5000);
  }

  // GET balances for specified mode
  async function fetchBalances(mode){
    const url = `/control/get_balances?mode=${encodeURIComponent(mode)}`;
    const res = await getJSON(url);
    if (!res.ok || !res.json || !res.json.ok){
      logRaw(`fetchBalances ${mode} failed: ${res.text || (res.json && JSON.stringify(res.json)) || res.error}`);
      return null;
    }
    return res.json.balance || null;
  }

  async function fetchBalancesBoth(){
    try {
      const rReal = await fetchBalances("real");
      const rDemo = await fetchBalances("demo");

      if (rReal && rReal.balance !== undefined && realBalanceEl){
        realBalanceEl.textContent = `${rReal.balance} ${rReal.currency || ""}`;
      } else if (realBalanceEl) realBalanceEl.textContent = "—";

      if (rDemo && rDemo.balance !== undefined && demoBalanceEl){
        demoBalanceEl.textContent = `${rDemo.balance} ${rDemo.currency || ""}`;
      } else if (demoBalanceEl) demoBalanceEl.textContent = "—";
    } catch(e){
      logRaw("fetchBalancesBoth error: " + (e.message||e));
    }
  }

  async function toggleDeriv(){
    if (!btnStartDeriv) return;
    if (!derivRunning){
      btnStartDeriv.disabled = true;
      const r = await postJSON(START_DERIV);
      btnStartDeriv.disabled = false;
      if (r.ok) {
        derivRunning = true;
        btnStartDeriv.textContent = "Stop Deriv";
        btnStartDeriv.classList.add("btn-danger");
        btnStartDeriv.classList.remove("btn-glow");
        const el = safe("derivStatus"); if (el) el.textContent = "Deriv: running";
      } else { alert("Start deriv failed — check logs"); }
    } else {
      btnStartDeriv.disabled = true;
      const r = await postJSON(STOP_DERIV);
      btnStartDeriv.disabled = false;
      derivRunning = false;
      btnStartDeriv.textContent = "Start Deriv";
      btnStartDeriv.classList.remove("btn-danger");
      btnStartDeriv.classList.add("btn-glow");
      const el = safe("derivStatus"); if (el) el.textContent = "Deriv: stopped";
    }
  }

  async function toggleAnalysis(){
    if (!btnStartAnalysis) return;
    if (!analysisRunning){
      btnStartAnalysis.disabled = true;
      const r = await postJSON(START_ANALYSIS);
      btnStartAnalysis.disabled = false;
      if (r.ok) {
        analysisRunning = true;
        btnStartAnalysis.textContent = "Stop Analysis";
        btnStartAnalysis.classList.add("btn-danger");
        btnStartAnalysis.classList.remove("btn-glow");
        const el = safe("analysisStatus"); if (el) el.textContent = "Analysis: running";

        try {
          const recentDigits = getVisibleTicks().map(p => extractLastDecimal(p)).filter(d => d !== null && d !== undefined);
          if (!analysisBuffer.length && recentDigits.length) {
            analysisBuffer = recentDigits.slice(-2000);
          }
          renderAnalysisBuffer();
          updateDigitsCursor();
        } catch(e){}
      } else { alert("Start analysis failed — check logs"); }
    } else {
      btnStartAnalysis.disabled = true;
      const r = await postJSON(STOP_ANALYSIS);
      btnStartAnalysis.disabled = false;
      analysisRunning = false;
      btnStartAnalysis.textContent = "Start Analysis";
      btnStartAnalysis.classList.remove("btn-danger");
      btnStartAnalysis.classList.add("btn-glow");
      const el = safe("analysisStatus"); if (el) el.textContent = "Analysis: stopped";
    }
  }

  async function clearTicks(){
    try { if (btnClearTicks) btnClearTicks.disabled = true; } catch(e){}
    const r = await postJSON(CLEAR_TICKS);
    try { if (btnClearTicks) btnClearTicks.disabled = false; } catch(e){}
    ticks = [];
    renderLast10();
    renderAnalysisBuffer();
    updateDigitsCursor();
    logRaw("Client cleared ticks (UI)");
  }

  function clearAnalysis(){
    try { if (btnClearAnalysis) btnClearAnalysis.disabled = true; } catch(e){}
    pendingPrediction = null;
    analysisBuffer = [];
    renderAnalysisBuffer();
    updateDigitsCursor();
    if (analysisStatusText) analysisStatusText.textContent = "Cleared by user";
    try { if (btnClearAnalysis) btnClearAnalysis.disabled = false; } catch(e){}
    logRaw("Client cleared analysis buffer");
  }

  function clearSummaryUI(){
    clearSummary();
    logRaw("Client cleared summary");
  }

  function clearRaw(){
    if (rawDebugEl) rawDebugEl.textContent = "";
  }

  // simulate helpers
  function simulateTick(){
    const d = Math.floor(Math.random()*10);
    const payload = { epoch: Math.floor(Date.now()/1000), price: null, last_decimal: d, symbol: ACTIVE_MARKET_FILTER || "R_100" };
    handleServerPayload(payload);
  }

  async function manualPredictTest(){
    if (pendingPrediction){
      alert("Prediction pending — cannot manual-predict until previous result observed.");
      return;
    }
    const fakeDigit = Math.floor(Math.random()*10);
    const fakePayload = {
      analysis_event: "prediction_posted",
      prediction_id: "manual_test_"+Date.now(),
      prediction_digit: fakeDigit,
      confidence: (0.6 + Math.random()*0.35).toFixed(3),
      sim_winrate: (0.5 + Math.random()*0.4).toFixed(3),
      reason: "manual-test",
      epoch: Math.floor(Date.now()/1000),
      symbol: "R_100"
    };
    handleAnalysisEvent(fakePayload);
    setTimeout(() => {
      const actual = Math.floor(Math.random()*10);
      const settled = {
        analysis_event: "prediction_result",
        prediction_id: fakePayload.prediction_id,
        prediction_digit: fakePayload.prediction_digit,
        observed_ticks: [actual],
        result: (actual === fakePayload.prediction_digit) ? "WIN" : "LOSS",
        confidence: fakePayload.confidence,
        reason: "manual settle",
        epoch: Math.floor(Date.now()/1000),
        symbol: "R_100"
      };
      handleAnalysisEvent(settled);
    }, 1500);
  }

  // Overlay open/close (Tick)
  function openTickOverlay(){
    if (!tickOverlay) return;
    renderTickOverlay();
    tickOverlay.classList.add("visible");
    tickOverlay.setAttribute("aria-hidden", "false");
    if (btnCloseTickOverlay) btnCloseTickOverlay.focus();
  }
  function closeTickOverlay(){
    if (!tickOverlay) return;
    tickOverlay.classList.remove("visible");
    tickOverlay.setAttribute("aria-hidden", "true");
  }

  // Overlay open/close (Analysis)
  function openAnalysisOverlay(){
    if (!analysisOverlay) return;
    renderAnalysisOverlay();
    analysisOverlay.classList.add("visible");
    analysisOverlay.setAttribute("aria-hidden", "false");
    if (btnCloseAnalysisOverlay) btnCloseAnalysisOverlay.focus();
  }
  function closeAnalysisOverlay(){
    if (!analysisOverlay) return;
    analysisOverlay.classList.remove("visible");
    analysisOverlay.setAttribute("aria-hidden", "true");
  }

  // close overlay via backdrop click / close button / Esc
  if (tickOverlay){
    tickOverlay.addEventListener("click", function(e){
      if (e.target === tickOverlay) closeTickOverlay();
    });
  }
  if (btnCloseTickOverlay) btnCloseTickOverlay.addEventListener("click", closeTickOverlay);

  if (analysisOverlay){
    analysisOverlay.addEventListener("click", function(e){
      if (e.target === analysisOverlay) closeAnalysisOverlay();
    });
  }
  if (btnCloseAnalysisOverlay) btnCloseAnalysisOverlay.addEventListener("click", closeAnalysisOverlay);

  document.addEventListener("keydown", function(e){
    if (e.key === "Escape") {
      if (tickOverlay && tickOverlay.classList.contains("visible")) closeTickOverlay();
      if (analysisOverlay && analysisOverlay.classList.contains("visible")) closeAnalysisOverlay();
    }
  });

  // wire toggleTickList to open tick overlay
  function toggleTickList(){
    openTickOverlay();
  }

  // attach listeners (guarded)
  if (btnStartDeriv) btnStartDeriv.addEventListener("click", toggleDeriv);
  if (btnStartAnalysis) btnStartAnalysis.addEventListener("click", toggleAnalysis);
  if (btnClearTicks) btnClearTicks.addEventListener("click", clearTicks);
  if (btnClearAnalysis) btnClearAnalysis.addEventListener("click", clearAnalysis);
  if (btnClearSummary) btnClearSummary.addEventListener("click", clearSummaryUI);
  if (btnClearRaw) btnClearRaw.addEventListener("click", clearRaw);
  if (btnSimTick) btnSimTick.addEventListener("click", simulateTick);
  if (btnManualPredict) btnManualPredict.addEventListener("click", manualPredictTest);
  if (btnToggleTickList) btnToggleTickList.addEventListener("click", toggleTickList);
  if (btnExpandAnalysis) btnExpandAnalysis.addEventListener("click", openAnalysisOverlay);

  if (btnConnectAccount) btnConnectAccount.addEventListener("click", connectAccountFixed);

  // status poll
  async function pollStatus(){
    try {
      const r = await fetch(STATUS_URL);
      if (!r.ok) return;
      const j = await r.json();
      if (j) {
        derivRunning = !!j.deriv_running;
        analysisRunning = !!j.analysis_running;
        const ds = safe("derivStatus"); if (ds) ds.textContent = derivRunning ? "Deriv: running" : "Deriv: stopped";
        const as = safe("analysisStatus"); if (as) as.textContent = analysisRunning ? "Analysis: running" : "Analysis: stopped";
        if (btnStartDeriv) {
          btnStartDeriv.textContent = derivRunning ? "Stop Deriv" : "Start Deriv";
          btnStartDeriv.classList.toggle("btn-danger", derivRunning);
          btnStartDeriv.classList.toggle("btn-glow", !derivRunning);
        }
        if (btnStartAnalysis) {
          btnStartAnalysis.textContent = analysisRunning ? "Stop Analysis" : "Start Analysis";
          btnStartAnalysis.classList.toggle("btn-danger", analysisRunning);
          btnStartAnalysis.classList.toggle("btn-glow", !analysisRunning);
        }
      }
    } catch(e){}
    finally { setTimeout(pollStatus, 2000); }
  }

  // Start SSE and status poll on page load
  if (ACTIVE_MARKET_FILTER){
    logRaw(`Market filter active: ${ACTIVE_MARKET_FILTER}`);
  }
  connectSSE();
  pollStatus();

  // initial render
  renderLast10();
  renderAnalysisBuffer();
  updateDigitsCursor();

  // expose for debugging
  window.__herox = {
    ticks, analysisBuffer,
    simulateTick, manualPredictTest,
    clearTicks, clearAnalysis,
    openTickOverlay, closeTickOverlay, openAnalysisOverlay, closeAnalysisOverlay,
    digitStats
  };

})(); // end main IIFE
