(function () {
  function q(id) { return document.getElementById(id); }
  function n(v, d) { const x = Number(v); return Number.isFinite(x) ? x.toFixed(d) : "-"; }
  function ts(v) {
    try {
      const x = Number(v);
      if (!Number.isFinite(x) || x <= 0) return "-";
      return new Date(x * 1000).toLocaleString();
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

  let eventRows = [];
  let ouEnabled = false;

  function getStakeInputValue() {
    const el = q("ouStake");
    if (!el) return 1.0;
    const v = Number(el.value);
    if (!Number.isFinite(v)) return 1.0;
    return Math.max(0.35, Number(v.toFixed(2)));
  }

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

  function setOuButtonState(isRunning) {
    ouEnabled = !!isRunning;
    const btn = q("btnOuToggle");
    const pill = q("ouStatusPill");
    if (btn) {
      btn.textContent = ouEnabled ? "Stop Over/Under Analysis" : "Start Over/Under Analysis";
    }
    if (pill) {
      pill.textContent = ouEnabled ? "OU analysis: running" : "OU analysis: stopped";
      pill.className = ouEnabled ? "pill ok" : "pill";
    }
  }

  function pickBestSymbol(symbols) {
    let best = null;
    (symbols || []).forEach(function (s) {
      if (!s || !s.best) return;
      const b = s.best;
      const score = Number(b.ev_per_stake || -999);
      if (!best || score > Number(best.best.ev_per_stake || -999)) best = s;
    });
    return best;
  }

  function renderOU(ou) {
    const settings = (ou && ou.settings) || {};
    const symbols = (ou && ou.symbols) || [];
    const signals = (ou && ou.signals) || [];
    const bestSym = pickBestSymbol(symbols);
    const pending = settings.pending_prediction_id || "-";

    setOuButtonState(!!settings.enabled);
    try {
      const st = Number(settings.trade_stake);
      if (Number.isFinite(st) && q("ouStake")) q("ouStake").value = st.toFixed(2);
    } catch (_) {}
    q("sEngine").textContent = settings.enabled ? "RUNNING" : "STOPPED";
    q("sEngine").className = settings.enabled ? "v ok" : "v";
    q("sBest").textContent = bestSym ? (bestSym.symbol + " | " + (bestSym.best.name || "-")) : "-";
    q("sPending").textContent = pending;
    q("sStats").textContent = String(settings.total_signals || 0) + " / " + String(settings.total_settled || 0);

    const tbody = q("analysisTable").querySelector("tbody");
    if (!symbols.length) {
      tbody.innerHTML = '<tr><td colspan="6">No symbol analysis yet. Start ticks and OU analysis.</td></tr>';
    } else {
      tbody.innerHTML = symbols.map(function (s) {
        const b = s.best || {};
        return "<tr>" +
          "<td>" + esc(s.symbol || "-") + "</td>" +
          "<td>" + Number(s.samples || 0) + "</td>" +
          "<td>" + esc(b.name || "-") + (b.pass_edge ? "" : " (best)") + "</td>" +
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
    }).slice(0, 80);
    if (!filtered.length) {
      body.innerHTML = '<tr><td colspan="7">No OU settlements yet.</td></tr>';
      return;
    }
    body.innerHTML = filtered.map(function (r) {
      const e = (r && r.entry) || {};
      const conf = r.conf ?? r.confidence ?? e.confidence;
      return "<tr>" +
        "<td>" + ts(r.ts || e.timestamp) + "</td>" +
        "<td>" + esc(r.market || e.symbol || "-") + "</td>" +
        "<td>" + esc((e.contract_type || r.contract_type || "-").toUpperCase()) + "</td>" +
        "<td>" + esc((r.result || e.result || "-").toUpperCase()) + "</td>" +
        "<td>" + n((r.profit != null ? r.profit : e.profit), 4) + "</td>" +
        "<td>" + (conf == null ? "-" : n(conf, 4)) + "</td>" +
        "<td>" + esc(r.id || e.prediction_id || "-") + "</td>" +
        "</tr>";
    }).join("");
  }

  function normalizeBalance(raw) {
    try {
      if (!raw) return null;
      if (typeof raw === "number") return { amount: raw, currency: "" };
      if (typeof raw.balance === "number") return { amount: raw.balance, currency: raw.currency || "" };
      if (raw.balance && typeof raw.balance === "object") {
        return { amount: raw.balance.balance, currency: raw.balance.currency || "" };
      }
      return null;
    } catch (_) {
      return null;
    }
  }

  async function refreshBalances() {
    try {
      const [demo, real] = await Promise.all([
        getJSON("/control/get_balances?mode=demo"),
        getJSON("/control/get_balances?mode=real"),
      ]);
      const d = normalizeBalance((demo && demo.balance) || null);
      const r = normalizeBalance((real && real.balance) || null);
      q("sDemoBal").textContent = d ? `${Number(d.amount).toFixed(2)} ${d.currency || ""}`.trim() : "-";
      q("sRealBal").textContent = r ? `${Number(r.amount).toFixed(2)} ${r.currency || ""}`.trim() : "-";
    } catch (_) {}
  }

  async function refresh() {
    try {
      const [ouRes, jr] = await Promise.all([
        getJSON("/control/ou_status?symbols=24&signals=60"),
        getJSON("/control/journal?limit=240"),
      ]);
      if (ouRes && ouRes.ok) renderOU(ouRes.ou || null);
      if (jr && jr.ok) renderJournal(jr.entries || []);
    } catch (e) {
      addEvent("refresh error: " + String((e && e.message) || e));
    }
    refreshBalances();
  }

  async function toggleOu() {
    const btn = q("btnOuToggle");
    if (!btn) return;
    btn.disabled = true;
    try {
      if (ouEnabled) {
        await postJSON("/control/ou_stop", {});
      } else {
        await postJSON("/control/ou_start", { trade_stake: getStakeInputValue() });
      }
      await refresh();
    } catch (e) {
      addEvent("toggle error: " + String((e && e.message) || e));
    } finally {
      btn.disabled = false;
    }
  }

  function attachSSE() {
    try {
      const es = new EventSource("/events");
      es.addEventListener("analysis", function (ev) {
        try {
          const p = JSON.parse(ev.data || "{}");
          const ae = String(p.analysis_event || "").toLowerCase();
          if (ae === "ou_analysis" || ae === "ou_signal" || ae === "prediction_result" || ae === "ou_status") {
            addEvent(ae + " | " + (p.symbol || p.market || ""));
            if (ae === "prediction_result" || ae === "ou_signal" || ae === "ou_status") refresh();
          }
        } catch (_) {}
      });
      es.onerror = function () { addEvent("SSE disconnected, retrying..."); };
    } catch (e) {
      addEvent("SSE setup failed: " + String((e && e.message) || e));
    }
  }

  (async function init() {
    const btn = q("btnOuToggle");
    if (btn) btn.addEventListener("click", toggleOu);
    const stakeEl = q("ouStake");
    if (stakeEl) {
      stakeEl.addEventListener("change", async function(){
        try {
          await postJSON("/control/ou_settings", { trade_stake: getStakeInputValue() });
          await refresh();
        } catch (e) {
          addEvent("stake save error: " + String((e && e.message) || e));
        }
      });
    }
    attachSSE();
    refresh();
    setInterval(refresh, 2500);
  })();
})();
