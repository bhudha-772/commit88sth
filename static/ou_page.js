(function () {
  function q(id) { return document.getElementById(id); }
  function n(v, d) { const x = Number(v); return Number.isFinite(x) ? x.toFixed(d) : "-"; }
  function ts(v) {
    try {
      const x = Number(v);
      if (!Number.isFinite(x) || x <= 0) return "-";
      return new Date(x * 1000).toLocaleTimeString();
    } catch (_) { return "-"; }
  }
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  let lastOU = null;
  let eventRows = [];

  function addEvent(text) {
    eventRows.unshift({ t: Date.now(), text: String(text || "") });
    if (eventRows.length > 80) eventRows.length = 80;
    const feed = q("eventFeed");
    if (!feed) return;
    feed.innerHTML = eventRows.map(function (r) {
      return '<div class="row">[' + new Date(r.t).toLocaleTimeString() + "] " + esc(r.text) + "</div>";
    }).join("");
  }

  async function getJSON(url) {
    const r = await fetch(url, { credentials: "same-origin" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }

  function pickBestSymbol(symbols) {
    let best = null;
    (symbols || []).forEach(function (s) {
      if (!s || !s.best) return;
      const b = s.best;
      const pass = !!b.pass_edge;
      if (!pass) return;
      const score = Number(b.ev_per_stake || -999);
      if (!best || score > Number(best.best.ev_per_stake || -999)) best = s;
    });
    return best;
  }

  function renderOU(ou) {
    lastOU = ou || null;
    const settings = (ou && ou.settings) || {};
    const symbols = (ou && ou.symbols) || [];
    const signals = (ou && ou.signals) || [];
    const bestSym = pickBestSymbol(symbols);
    const pending = settings.pending_prediction_id || "-";

    q("sEngine").textContent = settings.enabled ? "RUNNING" : "DISABLED";
    q("sEngine").className = settings.enabled ? "v ok" : "v";
    q("sBest").textContent = bestSym ? (bestSym.symbol + " | " + (bestSym.best.name || "-")) : "-";
    q("sPending").textContent = pending;
    q("sStats").textContent = String(settings.total_signals || 0) + " / " + String(settings.total_settled || 0);

    const tbody = q("analysisTable").querySelector("tbody");
    if (!symbols.length) {
      tbody.innerHTML = '<tr><td colspan="6">No symbol analysis yet. Start ticks first.</td></tr>';
    } else {
      tbody.innerHTML = symbols.map(function (s) {
        const b = s.best || {};
        return "<tr>" +
          "<td>" + esc(s.symbol || "-") + "</td>" +
          "<td>" + Number(s.samples || 0) + "</td>" +
          "<td>" + esc(b.name || "-") + "</td>" +
          "<td>" + n(b.p_hat, 4) + "</td>" +
          "<td>" + n(b.p_be, 4) + "</td>" +
          "<td>" + n(b.ev_per_stake, 4) + "</td>" +
          "</tr>";
      }).join("");
    }

    const sf = q("signalFeed");
    if (!signals.length) {
      sf.innerHTML = '<div class="row">No signals yet.</div>';
    } else {
      sf.innerHTML = signals.slice(0, 30).map(function (s) {
        return '<div class="row"><strong>' + esc(s.symbol || "-") + "</strong> " +
          esc(s.contract || "-") + " | p=" + n(s.p_hat, 4) +
          " p_be=" + n(s.p_be, 4) + " ev=" + n(s.ev_per_stake, 4) +
          " | " + esc(s.status || "-") + " | " + ts(s.ts) +
          "</div>";
      }).join("");
    }
  }

  function renderJournal(rows) {
    const body = q("settleTable").querySelector("tbody");
    const filtered = (rows || []).filter(function (r) {
      const e = (r && r.entry) || {};
      const c = String(e.contract_type || r.contract_type || "").toUpperCase();
      return c === "DIGITOVER" || c === "DIGITUNDER" || String(e.prediction_mode || "").toLowerCase() === "ou_focus";
    }).slice(0, 40);
    if (!filtered.length) {
      body.innerHTML = '<tr><td colspan="7">No OU settlements yet.</td></tr>';
      return;
    }
    body.innerHTML = filtered.map(function (r) {
      const e = (r && r.entry) || {};
      return "<tr>" +
        "<td>" + ts(r.ts || e.timestamp) + "</td>" +
        "<td>" + esc(r.market || e.symbol || "-") + "</td>" +
        "<td>" + esc(e.contract_type || "-") + "</td>" +
        "<td>" + esc(r.pred != null ? r.pred : e.prediction_digit) + "</td>" +
        "<td>" + esc(r.actual != null ? r.actual : e.actual) + "</td>" +
        "<td>" + esc((r.result || e.result || "-").toUpperCase()) + "</td>" +
        "<td>" + n((r.profit != null ? r.profit : e.profit), 4) + "</td>" +
        "</tr>";
    }).join("");
  }

  async function refresh() {
    try {
      const [ouRes, jr] = await Promise.all([
        getJSON("/control/ou_status?symbols=24&signals=60"),
        getJSON("/control/journal?limit=200"),
      ]);
      if (ouRes && ouRes.ok) renderOU(ouRes.ou || null);
      if (jr && jr.ok) renderJournal(jr.entries || []);
    } catch (e) {
      addEvent("refresh error: " + String((e && e.message) || e));
    }
  }

  function attachSSE() {
    try {
      const es = new EventSource("/events");
      es.addEventListener("analysis", function (ev) {
        try {
          const p = JSON.parse(ev.data || "{}");
          const ae = String(p.analysis_event || "").toLowerCase();
          if (ae === "ou_analysis" || ae === "ou_signal" || ae === "prediction_result") {
            addEvent(ae + " | " + (p.symbol || p.market || ""));
            if (ae === "prediction_result") refresh();
          }
        } catch (_) {}
      });
      es.onerror = function () { addEvent("SSE disconnected, retrying..."); };
    } catch (e) {
      addEvent("SSE setup failed: " + String((e && e.message) || e));
    }
  }

  (async function init() {
    try {
      // Force automatic mode for this OU workflow.
      await postJSON("/control/ou_settings", { enabled: true, auto_predict: true });
    } catch (_) {}
    attachSSE();
    refresh();
    setInterval(refresh, 2500);
  })();
})();
