(function () {
  const ROOT_ID = "ouFocusPanelRoot";
  if (document.getElementById(ROOT_ID)) return;

  const POLL_MS = 2500;
  const state = {
    open: false,
    loading: false,
    ou: null,
    error: "",
  };

  function esc(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function tsFmt(ts) {
    try {
      if (!ts) return "-";
      const d = new Date(Number(ts) * 1000);
      return d.toLocaleTimeString();
    } catch (e) {
      return "-";
    }
  }

  function num(v, digits) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    return n.toFixed(digits);
  }

  async function getJSON(url) {
    const r = await fetch(url, { method: "GET", credentials: "same-origin" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
      credentials: "same-origin",
    });
    if (!r.ok) {
      let txt = "";
      try {
        txt = await r.text();
      } catch (e) {}
      throw new Error("HTTP " + r.status + (txt ? " " + txt : ""));
    }
    return r.json();
  }

  function inject() {
    const root = document.createElement("div");
    root.id = ROOT_ID;
    root.innerHTML = `
      <style>
        #${ROOT_ID} .ou-panel{
          position:fixed;right:14px;top:78px;z-index:2100000;
          width:min(460px,95vw);max-height:min(82vh,760px);overflow:auto;
          border-radius:14px;border:1px solid rgba(2,6,23,.12);
          background:linear-gradient(180deg,#ffffff,#f8fafc);
          box-shadow:0 22px 70px rgba(2,6,23,.30);padding:12px;display:none
        }
        #${ROOT_ID} .ou-panel.open{display:block}
        #${ROOT_ID} .ou-h{display:flex;justify-content:space-between;align-items:center;gap:8px}
        #${ROOT_ID} .ou-h h3{margin:0;font-size:14px;color:#0f172a}
        #${ROOT_ID} .ou-close{border:0;background:transparent;cursor:pointer;font-size:18px;line-height:1}
        #${ROOT_ID} .ou-s{margin-top:8px;font-size:12px;color:#334155}
        #${ROOT_ID} .ou-card{margin-top:10px;border:1px solid rgba(2,6,23,.10);border-radius:10px;padding:9px;background:#fff}
        #${ROOT_ID} .ou-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
        #${ROOT_ID} .ou-grid label{font-size:11px;color:#475569;display:flex;flex-direction:column;gap:4px}
        #${ROOT_ID} .ou-grid input{border:1px solid rgba(2,6,23,.16);border-radius:8px;padding:6px 8px;font-size:12px}
        #${ROOT_ID} .ou-row{display:flex;align-items:center;justify-content:space-between;gap:8px}
        #${ROOT_ID} .ou-row .tog{display:flex;align-items:center;gap:6px;font-size:12px;color:#334155}
        #${ROOT_ID} .ou-save{border:1px solid rgba(2,6,23,.20);background:#0f172a;color:#fff;border-radius:8px;padding:7px 10px;font-size:12px;cursor:pointer}
        #${ROOT_ID} table{width:100%;border-collapse:collapse;font-size:11px}
        #${ROOT_ID} th,#${ROOT_ID} td{padding:6px 4px;border-bottom:1px solid rgba(2,6,23,.08);text-align:left}
        #${ROOT_ID} th{color:#475569;font-weight:700}
        #${ROOT_ID} .ou-sig{display:flex;flex-direction:column;gap:7px;max-height:220px;overflow:auto}
        #${ROOT_ID} .ou-sig-item{border:1px solid rgba(2,6,23,.10);border-radius:9px;padding:7px;background:#fff}
        #${ROOT_ID} .ou-meta{font-size:11px;color:#64748b}
      </style>
      <div class="ou-panel" id="ouPanel">
        <div class="ou-h">
          <h3>OU Focus Engine</h3>
          <button class="ou-close" id="ouCloseBtn" aria-label="Close">x</button>
        </div>
        <div class="ou-s" id="ouSummary">Loading...</div>
        <div class="ou-card">
          <div class="ou-row">
            <label class="tog"><input type="checkbox" id="ouEnabled"> Enabled</label>
            <label class="tog"><input type="checkbox" id="ouAuto"> Auto Predict</label>
            <button class="ou-save" id="ouSaveBtn">Save</button>
          </div>
          <div class="ou-grid" style="margin-top:8px">
            <label>Window size <input id="ouWindow" type="number" min="200" step="100"></label>
            <label>Min samples <input id="ouMin" type="number" min="100" step="50"></label>
            <label>Delta <input id="ouDelta" type="number" min="0" step="0.0005"></label>
            <label>Cooldown sec <input id="ouCooldown" type="number" min="1" step="1"></label>
          </div>
        </div>
        <div class="ou-card">
          <div class="ou-row"><strong style="font-size:12px">Top Symbols</strong><span class="ou-meta">p, p_be, EV</span></div>
          <div id="ouSymbols"></div>
        </div>
        <div class="ou-card">
          <div class="ou-row"><strong style="font-size:12px">Recent Signals</strong><span class="ou-meta" id="ouSignalsMeta">0</span></div>
          <div id="ouSignals" class="ou-sig"></div>
        </div>
      </div>
    `;
    document.body.appendChild(root);

    const panel = root.querySelector("#ouPanel");
    const close = root.querySelector("#ouCloseBtn");
    close.addEventListener("click", function () {
      state.open = false;
      panel.classList.remove("open");
    });
    window.openOUFocusPanel = function () {
      state.open = true;
      panel.classList.add("open");
      refreshNow();
    };
    window.closeOUFocusPanel = function () {
      state.open = false;
      panel.classList.remove("open");
    };
    const navBtn = document.getElementById("btnOpenOUPanel");
    if (navBtn) {
      navBtn.addEventListener("click", function () {
        window.openOUFocusPanel();
      });
    }

    root.querySelector("#ouSaveBtn").addEventListener("click", async function () {
      try {
        const windowV = Number(root.querySelector("#ouWindow").value);
        const minV = Number(root.querySelector("#ouMin").value);
        const deltaV = Number(root.querySelector("#ouDelta").value);
        const cooldownV = Number(root.querySelector("#ouCooldown").value);
        const body = {
          enabled: !!root.querySelector("#ouEnabled").checked,
          auto_predict: !!root.querySelector("#ouAuto").checked,
        };
        if (Number.isFinite(windowV) && windowV > 0) body.window_size = windowV;
        if (Number.isFinite(minV) && minV > 0) body.min_samples = minV;
        if (Number.isFinite(deltaV) && deltaV >= 0) body.delta = deltaV;
        if (Number.isFinite(cooldownV) && cooldownV > 0) body.predict_cooldown_sec = cooldownV;
        await postJSON("/control/ou_settings", body);
        await refreshNow();
      } catch (e) {
        state.error = String((e && e.message) || e || "save_failed");
        render();
      }
    });
  }

  function render() {
    const root = document.getElementById(ROOT_ID);
    if (!root) return;
    const summary = root.querySelector("#ouSummary");
    const symbolsNode = root.querySelector("#ouSymbols");
    const signalsNode = root.querySelector("#ouSignals");
    const signalsMeta = root.querySelector("#ouSignalsMeta");
    const enabled = root.querySelector("#ouEnabled");
    const auto = root.querySelector("#ouAuto");
    const windowEl = root.querySelector("#ouWindow");
    const minEl = root.querySelector("#ouMin");
    const deltaEl = root.querySelector("#ouDelta");
    const cooldownEl = root.querySelector("#ouCooldown");

    if (state.loading && !state.ou) {
      summary.textContent = "Loading OU engine...";
      return;
    }
    if (!state.ou || !state.ou.settings) {
      summary.textContent = state.error ? ("Error: " + state.error) : "OU engine unavailable.";
      return;
    }

    const st = state.ou.settings || {};
    enabled.checked = !!st.enabled;
    auto.checked = !!st.auto_predict;
    windowEl.value = Number(st.window_size || 0);
    minEl.value = Number(st.min_samples || 0);
    deltaEl.value = Number(st.delta || 0);
    cooldownEl.value = Number(st.predict_cooldown_sec || 0);

    summary.textContent =
      "Status: " + (st.enabled ? "running" : "disabled") +
      " | Ticks: " + Number(st.total_ticks || 0) +
      " | Analyses: " + Number(st.total_analyses || 0) +
      " | Signals: " + Number(st.total_signals || 0) +
      (state.error ? (" | Last error: " + state.error) : "");

    const symbols = Array.isArray(state.ou.symbols) ? state.ou.symbols : [];
    if (!symbols.length) {
      symbolsNode.innerHTML = '<div class="ou-meta">No analyzed symbols yet. Start Deriv ticks first.</div>';
    } else {
      const rows = symbols.slice(0, 10).map(function (row) {
        const best = row && row.best ? row.best : null;
        const contract = best ? String(best.name || "-") : "-";
        return (
          "<tr>" +
          "<td>" + esc(row.symbol || "-") + "</td>" +
          "<td>" + Number(row.samples || 0) + "</td>" +
          "<td>" + esc(contract) + "</td>" +
          "<td>" + (best ? num(best.p_hat, 3) : "-") + "</td>" +
          "<td>" + (best ? num(best.p_be, 3) : "-") + "</td>" +
          "<td>" + (best ? num(best.ev_per_stake, 4) : "-") + "</td>" +
          "</tr>"
        );
      }).join("");
      symbolsNode.innerHTML =
        "<table>" +
        "<thead><tr><th>Symbol</th><th>N</th><th>Best</th><th>p</th><th>p_be</th><th>EV</th></tr></thead>" +
        "<tbody>" + rows + "</tbody>" +
        "</table>";
    }

    const signals = Array.isArray(state.ou.signals) ? state.ou.signals : [];
    signalsMeta.textContent = String(signals.length);
    if (!signals.length) {
      signalsNode.innerHTML = '<div class="ou-meta">No signals yet.</div>';
    } else {
      signalsNode.innerHTML = signals.slice(0, 25).map(function (s) {
        return (
          '<div class="ou-sig-item">' +
          '<div style="font-size:12px;font-weight:700">' + esc(s.symbol || "-") + " | " + esc(s.contract || "-") + "</div>" +
          '<div class="ou-meta">' +
          tsFmt(s.ts) + " | barrier " + esc(s.barrier || "-") +
          " | p " + num(s.p_hat, 4) +
          " | p_be " + num(s.p_be, 4) +
          " | ev " + num(s.ev_per_stake, 4) +
          " | " + esc(s.status || "pending") +
          "</div>" +
          "</div>"
        );
      }).join("");
    }
  }

  async function refreshNow() {
    if (state.loading) return;
    state.loading = true;
    try {
      const r = await getJSON("/control/ou_status?symbols=12&signals=30");
      if (r && r.ok) {
        state.ou = r.ou || null;
        state.error = "";
      } else {
        state.error = (r && r.error) || "status_error";
      }
    } catch (e) {
      state.error = String((e && e.message) || e || "network_error");
    } finally {
      state.loading = false;
      render();
    }
  }

  inject();
  refreshNow();
  setInterval(refreshNow, POLL_MS);
})();
