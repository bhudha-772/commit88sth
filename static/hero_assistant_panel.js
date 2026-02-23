/* hero_assistant_panel.js
   Floating assistant icon + panel (notifications + chat + learn) for HeroX.
*/
(function () {
  if (window.__hero_assistant_panel_inited) return;
  window.__hero_assistant_panel_inited = true;

  const STYLE_ID = "hero-assistant-style";
  const ROOT_ID = "hero-assistant-root";
  const LAST_SEEN_NOTES_KEY = "hero_assistant_last_seen_notif_ts_v1";
  const CHAT_HISTORY_KEY = "hero_assistant_chat_history_v1";
  const LAST_SEEN_MAJOR_NEWS_KEY = "hero_assistant_last_seen_major_news_ts_v1";

  const NEWS_CATEGORIES = ["Forex", "Deriv", "Crypto", "Stocks", "Commodities", "Indices", "Macro", "Other"];
  const NEWS_WATCHLIST = {
    Forex: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"],
    Deriv: ["R_10", "R_25", "R_50", "R_75", "R_100"],
    Crypto: ["BTC", "ETH", "SOL", "BNB", "XRP"],
    Stocks: ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"],
    Commodities: ["XAUUSD", "XAGUSD", "WTI", "Brent", "Natural Gas"],
    Indices: ["US500", "NAS100", "US30", "DAX", "FTSE"],
    Macro: ["Fed", "CPI", "NFP", "Rates", "Yield curve"],
    Other: ["Risk sentiment", "Flows", "Liquidity", "Volatility", "Regulation"],
  };

  const state = {
    activeTab: "notes",
    notificationsMaxTs: 0,
    lastSeenNotifTs: 0,
    learnedItems: [],
    newsItems: [],
    tasks: [],
    panelVisible: false,
    newsCategory: "Forex",
    lastSeenMajorNewsTs: 0,
  };

  function esc(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function toLocalTime(ts) {
    const n = Number(ts || 0);
    if (!n) return "";
    return new Date(n * 1000).toLocaleString();
  }

  function loadLocalState() {
    try {
      state.lastSeenNotifTs = Number(localStorage.getItem(LAST_SEEN_NOTES_KEY) || "0") || 0;
    } catch (_) {
      state.lastSeenNotifTs = 0;
    }
    try {
      state.lastSeenMajorNewsTs = Number(localStorage.getItem(LAST_SEEN_MAJOR_NEWS_KEY) || "0") || 0;
    } catch (_) {
      state.lastSeenMajorNewsTs = 0;
    }
  }

  function saveLastSeenNotifTs() {
    try {
      localStorage.setItem(LAST_SEEN_NOTES_KEY, String(state.lastSeenNotifTs || 0));
    } catch (_) {}
  }

  function saveLastSeenMajorNewsTs() {
    try {
      localStorage.setItem(LAST_SEEN_MAJOR_NEWS_KEY, String(state.lastSeenMajorNewsTs || 0));
    } catch (_) {}
  }

  function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      #${ROOT_ID}{position:fixed;right:16px;bottom:16px;z-index:1400000;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
      #${ROOT_ID} .ha-btn{position:relative;width:56px;height:56px;border-radius:14px;border:1px solid rgba(2,6,23,0.12);background:linear-gradient(180deg,#fff,#f8fafc);box-shadow:0 12px 34px rgba(2,6,23,0.16);display:flex;align-items:center;justify-content:center;cursor:pointer}
      #${ROOT_ID} .ha-unread-dot{position:absolute;right:8px;top:8px;width:10px;height:10px;border-radius:999px;background:#2563eb;box-shadow:0 0 0 2px #fff}
      #${ROOT_ID} .ha-hidden{display:none !important}
      #${ROOT_ID} .ha-panel{position:fixed;right:12px;bottom:12px;width:min(1000px,calc(100vw - 24px));height:min(86vh,860px);display:none;flex-direction:column;border-radius:14px;overflow:hidden;border:1px solid rgba(2,6,23,0.12);background:#fff;box-shadow:0 30px 70px rgba(2,6,23,0.26)}
      #${ROOT_ID} .ha-panel.visible{display:flex}
      #${ROOT_ID} .ha-head{padding:10px 12px;border-bottom:1px solid rgba(2,6,23,0.08);display:flex;align-items:center;justify-content:space-between;background:linear-gradient(180deg,#fff,#f8fafc)}
      #${ROOT_ID} .ha-title{font-size:13px;font-weight:800;color:#0f172a;display:flex;align-items:center;gap:8px}
      #${ROOT_ID} .ha-dot{width:8px;height:8px;border-radius:999px;background:#94a3b8}
      #${ROOT_ID} .ha-dot.online{background:#16a34a}
      #${ROOT_ID} .ha-meta{font-size:11px;color:#64748b;margin-top:2px}
      #${ROOT_ID} .ha-tabs{display:flex;gap:6px;padding:8px;border-bottom:1px solid rgba(2,6,23,0.08);background:#fff}
      #${ROOT_ID} .ha-tab{position:relative;font-size:12px;padding:6px 8px;border-radius:8px;border:1px solid rgba(2,6,23,0.08);background:#fff;cursor:pointer}
      #${ROOT_ID} .ha-tab.active{background:#0f172a;color:#fff;border-color:#0f172a}
      #${ROOT_ID} .ha-tab-note-dot{position:absolute;right:4px;top:3px;width:8px;height:8px;border-radius:999px;background:#2563eb}
      #${ROOT_ID} .ha-tab-new{font-size:11px;padding:6px 8px;border-radius:8px;border:1px solid rgba(2,6,23,0.12);background:#fff;cursor:pointer}
      #${ROOT_ID} .ha-tab-clear{margin-left:auto;font-size:11px;padding:6px 8px;border-radius:8px;border:1px solid rgba(2,6,23,0.12);background:#fff;cursor:pointer}
      #${ROOT_ID} .ha-body{padding:10px;overflow:auto;flex:1;background:#fbfdff}
      #${ROOT_ID} .ha-note{font-size:12px;padding:8px;border:1px solid rgba(2,6,23,0.06);border-radius:8px;background:#fff;margin-bottom:8px}
      #${ROOT_ID} .ha-note .meta{font-size:11px;color:#64748b;margin-bottom:4px}
      #${ROOT_ID} .ha-msg{font-size:12px;padding:8px 10px;border-radius:8px;margin-bottom:8px;max-width:92%;white-space:pre-wrap}
      #${ROOT_ID} .ha-msg.user{margin-left:auto;background:#e2e8f0;color:#0f172a}
      #${ROOT_ID} .ha-msg.bot{margin-right:auto;background:#fff;border:1px solid rgba(2,6,23,0.08);color:#0f172a}
      #${ROOT_ID} .ha-typing{display:inline-flex;align-items:center;gap:4px}
      #${ROOT_ID} .ha-dots span{display:inline-block;animation:haDotPulse 1.2s infinite ease-in-out}
      #${ROOT_ID} .ha-dots span:nth-child(2){animation-delay:.15s}
      #${ROOT_ID} .ha-dots span:nth-child(3){animation-delay:.3s}
      @keyframes haDotPulse{0%,80%,100%{opacity:.2}40%{opacity:1}}
      #${ROOT_ID} .ha-guide{font-size:12px;line-height:1.45;padding:9px 10px;border-radius:8px;border:1px solid rgba(2,6,23,0.08);background:#fff9ed;color:#334155;margin-bottom:8px}
      #${ROOT_ID} .ha-foot{padding:8px;border-top:1px solid rgba(2,6,23,0.08);display:flex;gap:8px;background:#fff;align-items:flex-end}
      #${ROOT_ID} .ha-input{flex:1;border:1px solid rgba(2,6,23,0.12);border-radius:10px;padding:8px 10px;font-size:12px;min-height:40px;max-height:160px;resize:vertical;line-height:1.4}
      #${ROOT_ID} .ha-send{padding:8px 10px;border-radius:8px;border:1px solid rgba(2,6,23,0.12);background:#0f172a;color:#fff;cursor:pointer;font-size:12px}
      #${ROOT_ID} .ha-close{border:none;background:transparent;cursor:pointer;color:#64748b;font-size:14px}
      #${ROOT_ID} .ha-learn-item{border:1px solid rgba(2,6,23,0.08);border-radius:10px;background:#fff;margin-bottom:8px;overflow:hidden}
      #${ROOT_ID} .ha-learn-head{width:100%;text-align:left;border:none;background:#fff;padding:10px 12px;cursor:pointer}
      #${ROOT_ID} .ha-learn-title{font-size:12px;font-weight:700;color:#0f172a}
      #${ROOT_ID} .ha-learn-meta{font-size:11px;color:#64748b;margin-top:3px}
      #${ROOT_ID} .ha-learn-body{padding:10px 12px;border-top:1px solid rgba(2,6,23,0.08);font-size:12px;color:#0f172a}
      #${ROOT_ID} .ha-learn-k{font-size:11px;color:#64748b;margin-top:8px;margin-bottom:3px}
      #${ROOT_ID} .ha-learn-s{padding:8px;border:1px solid rgba(2,6,23,0.06);border-radius:8px;background:#f8fafc;margin-bottom:8px}
      #${ROOT_ID} .ha-learn-s h4{margin:0 0 4px 0;font-size:12px}
      #${ROOT_ID} .ha-news-item{border:1px solid rgba(2,6,23,0.08);border-radius:10px;background:#fff;margin-bottom:8px;padding:10px}
      #${ROOT_ID} .ha-news-title{font-size:12px;font-weight:700;color:#0f172a}
      #${ROOT_ID} .ha-news-meta{font-size:11px;color:#64748b;margin-top:4px}
      #${ROOT_ID} .ha-news-summary{font-size:12px;color:#0f172a;margin-top:6px;line-height:1.45}
      #${ROOT_ID} .ha-news-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-bottom:10px}
      #${ROOT_ID} .ha-news-card{border:1px solid rgba(2,6,23,0.1);border-radius:10px;background:#fff;padding:10px;cursor:pointer}
      #${ROOT_ID} .ha-news-card.active{border-color:#0f172a;box-shadow:0 0 0 1px rgba(15,23,42,.1) inset}
      #${ROOT_ID} .ha-news-card h4{margin:0;font-size:12px;color:#0f172a}
      #${ROOT_ID} .ha-news-card .meta{margin-top:4px;font-size:11px;color:#64748b}
      #${ROOT_ID} .ha-news-details-title{font-size:12px;font-weight:700;color:#0f172a;margin:0 0 8px 0}
      #${ROOT_ID} .ha-toast-wrap{position:fixed;top:14px;right:14px;z-index:1500000;display:flex;flex-direction:column;gap:8px;max-width:min(360px,88vw)}
      #${ROOT_ID} .ha-toast{background:#0f172a;color:#fff;border-radius:10px;padding:10px 12px;box-shadow:0 16px 35px rgba(2,6,23,.4)}
      #${ROOT_ID} .ha-toast h5{margin:0 0 4px 0;font-size:12px}
      #${ROOT_ID} .ha-toast p{margin:0;font-size:11px;line-height:1.35;opacity:.95}
      #${ROOT_ID} .ha-toast .meta{margin-top:6px;font-size:10px;opacity:.78}
      @media (max-width: 980px){ #${ROOT_ID} .ha-news-grid{grid-template-columns:1fr} }
      @media (max-width: 760px){
        #${ROOT_ID} .ha-panel{right:0;bottom:0;width:100vw;height:100vh;max-width:100vw;max-height:100vh;border-radius:0}
      }
    `;
    document.head.appendChild(style);
  }

  function createUI() {
    if (document.getElementById(ROOT_ID)) return document.getElementById(ROOT_ID);
    const root = document.createElement("div");
    root.id = ROOT_ID;
    root.innerHTML = `
      <button class="ha-btn" id="haBtn" title="Open Hero Assistant" aria-label="Open Hero Assistant">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M12 2L20 7v10l-8 5-8-5V7l8-5z" stroke="#0f172a" stroke-width="1.3"/>
          <circle cx="9" cy="11" r="1.2" fill="#0f172a"/>
          <circle cx="15" cy="11" r="1.2" fill="#0f172a"/>
          <path d="M8.5 15.2c1 .9 2.1 1.3 3.5 1.3s2.5-.4 3.5-1.3" stroke="#0f172a" stroke-width="1.2" stroke-linecap="round"/>
        </svg>
        <span class="ha-unread-dot ha-hidden" id="haBtnUnread"></span>
      </button>
      <div class="ha-panel" id="haPanel" aria-hidden="true">
        <div class="ha-head">
          <div>
            <div class="ha-title"><span class="ha-dot" id="haDot"></span><span>Hero Assistant</span></div>
            <div class="ha-meta" id="haMem">Memory: --</div>
          </div>
          <button class="ha-close" id="haClose" aria-label="Close">x</button>
        </div>
        <div class="ha-tabs">
          <button class="ha-tab active" id="haTabNotes">Notifications <span class="ha-tab-note-dot ha-hidden" id="haTabNoteDot"></span></button>
          <button class="ha-tab" id="haTabChat">Chat</button>
          <button class="ha-tab" id="haTabLearn">Learn</button>
          <button class="ha-tab" id="haTabNews">News</button>
          <button class="ha-tab-new" id="haNewChat">New Chat</button>
          <button class="ha-tab-clear" id="haClearNotes">Clear</button>
        </div>
        <div class="ha-body" id="haNotesView"></div>
        <div class="ha-body ha-hidden" id="haChatView">
          <div class="ha-guide">
            How this works:
            1) Chat is model-driven and conversational.
            2) Say "learn about X" to queue learning.
            3) Auto research refreshes knowledge + news every 2 minutes.
            4) Use "what strategies do you have" and "teach strategy NAME".
          </div>
        </div>
        <div class="ha-body ha-hidden" id="haLearnView"></div>
        <div class="ha-body ha-hidden" id="haNewsView"></div>
        <div class="ha-foot">
          <textarea class="ha-input" id="haInput" placeholder="Message Hero Assistant... (Enter to send, Shift+Enter for newline)"></textarea>
          <button class="ha-send" id="haSend">Send</button>
        </div>
      </div>
      <div class="ha-toast-wrap" id="haToastWrap"></div>
    `;
    document.body.appendChild(root);
    return root;
  }

  async function getJSON(url) {
    const resp = await fetch(url, { method: "GET", headers: { Accept: "application/json" } });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    return await resp.json();
  }

  async function postJSON(url, body) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body || {}),
    });
    const json = await resp.json().catch(() => null);
    if (!resp.ok || !json || !json.ok) throw new Error((json && json.error) || ("HTTP " + resp.status));
    return json;
  }

  function setOnline(isOnline) {
    const dot = document.getElementById("haDot");
    if (!dot) return;
    dot.classList.toggle("online", !!isOnline);
  }

  function setNotificationsUnread(hasUnread) {
    const tabDot = document.getElementById("haTabNoteDot");
    const btnDot = document.getElementById("haBtnUnread");
    if (tabDot) tabDot.classList.toggle("ha-hidden", !hasUnread);
    if (btnDot) btnDot.classList.toggle("ha-hidden", !hasUnread);
  }

  function markNotificationsSeen() {
    state.lastSeenNotifTs = Math.max(Number(state.lastSeenNotifTs || 0), Number(state.notificationsMaxTs || 0));
    saveLastSeenNotifTs();
    setNotificationsUnread(false);
  }

  function renderNotifications(items) {
    const view = document.getElementById("haNotesView");
    if (!view) return;
    view.innerHTML = "";
    if (!Array.isArray(items) || items.length === 0) {
      view.innerHTML = `<div class="ha-note"><div class="meta">No notifications</div>Assistant will post trade and learning updates here.</div>`;
      return;
    }
    items.forEach((n) => {
      const row = document.createElement("div");
      row.className = "ha-note";
      const meta = [toLocalTime(n.ts), n.market, n.outcome].filter(Boolean).join(" | ");
      row.innerHTML = `<div class="meta">${esc(meta)}</div><div>${esc(n.message || "")}</div>`;
      view.appendChild(row);
    });
  }

  function renderLearned(items, tasks) {
    const view = document.getElementById("haLearnView");
    if (!view) return;
    view.innerHTML = "";

    const activeLearning = (Array.isArray(tasks) ? tasks : []).filter((t) => {
      const st = String(t && t.status || "").toLowerCase();
      const kind = String(t && t.kind || "").toLowerCase();
      return (st === "queued" || st === "running") && (kind === "learn" || String(t && t.text || "").toLowerCase().indexOf("learn") >= 0);
    });
    if (activeLearning.length) {
      const box = document.createElement("div");
      box.className = "ha-note";
      const rows = activeLearning.slice(0, 4).map((t) => `- ${esc(t.topic || t.text || t.id || "learning task")} [${esc(t.status || "")}]`);
      box.innerHTML = `<div class="meta">Learning in progress</div><div>${rows.join("<br/>")}</div>`;
      view.appendChild(box);
    }

    if (!Array.isArray(items) || items.length === 0) {
      const empty = document.createElement("div");
      empty.className = "ha-note";
      empty.innerHTML = `<div class="meta">No learned topics yet</div>Use <b>learn about &lt;topic&gt;</b> in chat.`;
      view.appendChild(empty);
      return;
    }

    items.forEach((item) => {
      const wrapper = document.createElement("div");
      wrapper.className = "ha-learn-item";
      const subtopics = Array.isArray(item && item.subtopics) ? item.subtopics : [];
      const title = String(item && item.title || item && item.topic || "Untitled learning");
      const meta = `${toLocalTime(item && item.created_ts)} | ${subtopics.length} subtopics`;

      const head = document.createElement("button");
      head.type = "button";
      head.className = "ha-learn-head";
      head.innerHTML = `<div class="ha-learn-title">${esc(title)}</div><div class="ha-learn-meta">${esc(meta)}</div>`;

      const body = document.createElement("div");
      body.className = "ha-learn-body ha-hidden";
      const sub = subtopics
        .slice(0, 8)
        .map((s) => {
          const st = esc(s && s.title || "Subtopic");
          const learned = esc(s && s.learned || "");
          const takeaway = esc(s && s.takeaway || "");
          return `<div class="ha-learn-s"><h4>${st}</h4><div>${learned || "-"}</div>${takeaway ? `<div class="ha-learn-k">Takeaway: ${takeaway}</div>` : ""}</div>`;
        })
        .join("");

      body.innerHTML =
        `<div class="ha-learn-k">Summary</div><div>${esc(item && item.summary || "-")}</div>` +
        `<div class="ha-learn-k">Main Takeaway</div><div>${esc(item && item.takeaway || "-")}</div>` +
        `<div class="ha-learn-k">Subtopics</div>${sub || "<div>-</div>"}`;

      head.addEventListener("click", () => body.classList.toggle("ha-hidden"));
      wrapper.appendChild(head);
      wrapper.appendChild(body);
      view.appendChild(wrapper);
    });
  }

  function classifyNewsCategory(item) {
    const m = String(item && item.market || "").toLowerCase();
    const t = String(item && item.title || "").toLowerCase();
    const s = String(item && item.summary || "").toLowerCase();
    const x = `${m} ${t} ${s}`;
    if (x.includes("deriv") || x.includes("synthetic") || x.includes("volatility index")) return "Deriv";
    if (x.includes("crypto") || x.includes("bitcoin") || x.includes("ethereum") || x.includes("solana") || x.includes("xrp")) return "Crypto";
    if (x.includes("stock") || x.includes("equity") || x.includes("nasdaq") || x.includes("s&p") || x.includes("dow") || x.includes("aapl") || x.includes("msft")) return "Stocks";
    if (x.includes("gold") || x.includes("silver") || x.includes("oil") || x.includes("brent") || x.includes("commodity")) return "Commodities";
    if (x.includes("index") || x.includes("indices")) return "Indices";
    if (x.includes("forex") || x.includes("eurusd") || x.includes("gbpusd") || x.includes("usdjpy") || x.includes("currency") || x.includes("fx")) return "Forex";
    if (x.includes("fed") || x.includes("cpi") || x.includes("nfp") || x.includes("inflation") || x.includes("yield") || x.includes("interest rate") || x.includes("macro")) return "Macro";
    return "Other";
  }

  function groupNewsByCategory(items) {
    const groups = {};
    NEWS_CATEGORIES.forEach((k) => { groups[k] = []; });
    (Array.isArray(items) ? items : []).forEach((n) => {
      const c = classifyNewsCategory(n);
      const row = Object.assign({}, n || {}, { __cat: c });
      groups[c].push(row);
    });
    NEWS_CATEGORIES.forEach((k) => {
      groups[k].sort((a, b) => {
        const ia = Number(a && a.importance || 0);
        const ib = Number(b && b.importance || 0);
        if (ib !== ia) return ib - ia;
        const ta = Number(a && (a.updated_ts || a.created_ts) || 0);
        const tb = Number(b && (b.updated_ts || b.created_ts) || 0);
        return tb - ta;
      });
    });
    return groups;
  }

  function showMajorNewsToasts(items) {
    const wrap = document.getElementById("haToastWrap");
    if (!wrap) return;
    const arr = Array.isArray(items) ? items : [];
    const fresh = arr
      .filter((n) => Number(n && n.importance || 0) >= 85)
      .filter((n) => Number((n && (n.updated_ts || n.created_ts)) || 0) > Number(state.lastSeenMajorNewsTs || 0))
      .sort((a, b) => Number((a.updated_ts || a.created_ts) || 0) - Number((b.updated_ts || b.created_ts) || 0));
    if (!fresh.length) return;

    let maxTs = Number(state.lastSeenMajorNewsTs || 0);
    fresh.slice(0, 2).forEach((n, idx) => {
      const ts = Number((n && (n.updated_ts || n.created_ts)) || 0);
      if (ts > maxTs) maxTs = ts;
      setTimeout(() => {
        const card = document.createElement("div");
        card.className = "ha-toast";
        card.innerHTML =
          `<h5>${esc((n && n.title) || "Major news")}</h5>` +
          `<p>${esc((n && n.summary) || "")}</p>` +
          `<div class="meta">${esc(`${n && n.market ? n.market : "Market"} | Importance ${Number(n && n.importance || 0)}`)}</div>`;
        wrap.appendChild(card);
        setTimeout(() => {
          if (card && card.parentNode) card.parentNode.removeChild(card);
        }, 15000);
      }, idx * 350);
    });
    state.lastSeenMajorNewsTs = maxTs;
    saveLastSeenMajorNewsTs();
  }

  function renderNews(items) {
    const view = document.getElementById("haNewsView");
    if (!view) return;
    view.innerHTML = "";
    const arr = Array.isArray(items) ? items : [];
    if (!arr.length) {
      view.innerHTML = `<div class="ha-note"><div class="meta">No news yet</div>Auto research will keep this list updated every 2 minutes.</div>`;
      return;
    }

    const groups = groupNewsByCategory(arr);
    const categoriesWithData = NEWS_CATEGORIES.filter((c) => (groups[c] || []).length > 0);
    if (!categoriesWithData.includes(state.newsCategory)) state.newsCategory = categoriesWithData[0] || "Forex";

    const grid = document.createElement("div");
    grid.className = "ha-news-grid";
    NEWS_CATEGORIES.forEach((cat) => {
      const rows = groups[cat] || [];
      const top = rows[0];
      const card = document.createElement("div");
      card.className = "ha-news-card" + (state.newsCategory === cat ? " active" : "");
      const wl = (NEWS_WATCHLIST[cat] || []).slice(0, 5).join(", ");
      card.innerHTML =
        `<h4>${esc(cat)}</h4>` +
        `<div class="meta">${rows.length} updates | Watchlist: ${esc(wl)}</div>` +
        `<div class="meta">${esc(top ? (top.title || "Latest update available") : "No current update")}</div>`;
      card.addEventListener("click", () => {
        state.newsCategory = cat;
        renderNews(arr);
      });
      grid.appendChild(card);
    });
    view.appendChild(grid);

    const details = document.createElement("div");
    details.innerHTML = `<div class="ha-news-details-title">${esc(state.newsCategory)} - Detailed Updates</div>`;
    const selected = groups[state.newsCategory] || [];
    if (!selected.length) {
      const empty = document.createElement("div");
      empty.className = "ha-note";
      empty.innerHTML = `No current updates in ${esc(state.newsCategory)}.`;
      details.appendChild(empty);
    } else {
      selected.slice(0, 20).forEach((n) => {
        const row = document.createElement("div");
        row.className = "ha-news-item";
        const importance = Number(n && n.importance || 0);
        const meta = [
          `Importance ${importance}`,
          n && n.market,
          n && n.source,
          toLocalTime(n && (n.updated_ts || n.created_ts)),
        ].filter(Boolean).join(" | ");
        row.innerHTML =
          `<div class="ha-news-title">${esc((n && n.title) || "Untitled")}</div>` +
          `<div class="ha-news-meta">${esc(meta)}</div>` +
          `<div class="ha-news-summary">${esc((n && n.summary) || "-")}</div>`;
        details.appendChild(row);
      });
    }
    view.appendChild(details);
  }

  function addChatMessage(text, who) {
    const view = document.getElementById("haChatView");
    if (!view) return;
    const div = document.createElement("div");
    div.className = "ha-msg " + (who === "user" ? "user" : "bot");
    div.textContent = text;
    view.appendChild(div);
    view.scrollTop = view.scrollHeight;
    try {
      const cur = JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || "[]");
      cur.push({ role: who === "user" ? "user" : "assistant", text: String(text || ""), ts: Date.now() });
      while (cur.length > 120) cur.shift();
      localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(cur));
    } catch (_) {}
  }

  function loadChatHistory() {
    try {
      const cur = JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || "[]");
      if (!Array.isArray(cur) || cur.length === 0) return;
      const view = document.getElementById("haChatView");
      if (!view) return;
      cur.forEach((m) => {
        if (!m || typeof m.text !== "string") return;
        const div = document.createElement("div");
        div.className = "ha-msg " + (m.role === "user" ? "user" : "bot");
        div.textContent = m.text;
        view.appendChild(div);
      });
      view.scrollTop = view.scrollHeight;
    } catch (_) {}
  }

  function clearChatHistory() {
    try {
      localStorage.removeItem(CHAT_HISTORY_KEY);
    } catch (_) {}
    const view = document.getElementById("haChatView");
    if (!view) return;
    const guide = view.querySelector(".ha-guide");
    view.innerHTML = "";
    if (guide) view.appendChild(guide);
  }

  function updateUnreadFromNotifications(items) {
    const list = Array.isArray(items) ? items : [];
    state.notificationsMaxTs = list.reduce((m, n) => {
      const ts = Number(n && n.ts || 0);
      return ts > m ? ts : m;
    }, 0);
    const hasUnread = Number(state.notificationsMaxTs || 0) > Number(state.lastSeenNotifTs || 0);
    if (state.panelVisible && state.activeTab === "notes") {
      markNotificationsSeen();
      return;
    }
    setNotificationsUnread(hasUnread);
  }

  async function refreshHealthAndData() {
    const reqs = await Promise.allSettled([
      getJSON("/assistant/health"),
      getJSON("/assistant/notifications?limit=40"),
      getJSON("/assistant/learned?limit=40"),
      getJSON("/assistant/news?limit=40"),
      getJSON("/assistant/tasks"),
    ]);

    const h = reqs[0].status === "fulfilled" ? reqs[0].value : null;
    const n = reqs[1].status === "fulfilled" ? reqs[1].value : null;
    const l = reqs[2].status === "fulfilled" ? reqs[2].value : null;
    const news = reqs[3].status === "fulfilled" ? reqs[3].value : null;
    const t = reqs[4].status === "fulfilled" ? reqs[4].value : null;

    if (h && h.ok) {
      setOnline(true);
      const st = h.stats || {};
      const mem = document.getElementById("haMem");
      if (mem) {
        const used = Number(st.memory_pct_used || 0).toFixed(1);
        const left = Number(st.memory_pct_remaining || 0).toFixed(1);
        const usedN = Number(st.memory_used || 0);
        const capN = Number(st.memory_capacity || 0);
        const learnedN = Number(st.learned_items_count || 0);
        const newsN = Number(st.news_items_count || 0);
        const rs = st.research || {};
        const llm = st.llm || {};
        const llmTxt = llm.ok ? `LLM ${llm.provider || ""} ok` : `LLM err: ${llm.error || "unknown"}`;
        mem.textContent = `Memory ${usedN}/${capN} (${used}% used | ${left}% left) | Learned ${learnedN} | News ${newsN} | Research ${rs.last_status || "idle"} | ${llmTxt}`;
      }
    } else {
      setOnline(false);
      const mem = document.getElementById("haMem");
      if (mem) mem.textContent = "Memory: unavailable";
    }

    const notes = (n && n.notifications) || [];
    renderNotifications(notes);
    updateUnreadFromNotifications(notes);

    state.learnedItems = (l && l.learned) || [];
    state.newsItems = (news && news.news) || [];
    state.tasks = (t && t.tasks) || [];
    renderLearned(state.learnedItems, state.tasks);
    renderNews(state.newsItems);
    showMajorNewsToasts(state.newsItems);
  }

  function wireUI() {
    const btn = document.getElementById("haBtn");
    const panel = document.getElementById("haPanel");
    const close = document.getElementById("haClose");
    const tabNotes = document.getElementById("haTabNotes");
    const tabChat = document.getElementById("haTabChat");
    const tabLearn = document.getElementById("haTabLearn");
    const tabNews = document.getElementById("haTabNews");
    const newChat = document.getElementById("haNewChat");
    const clearNotes = document.getElementById("haClearNotes");
    const notes = document.getElementById("haNotesView");
    const chat = document.getElementById("haChatView");
    const learn = document.getElementById("haLearnView");
    const news = document.getElementById("haNewsView");
    const input = document.getElementById("haInput");
    const send = document.getElementById("haSend");

    if (!btn || !panel || !close || !tabNotes || !tabChat || !tabLearn || !tabNews || !newChat || !clearNotes || !notes || !chat || !learn || !news || !input || !send) return;

    function setActiveTab(next) {
      state.activeTab = next;
      const isNotes = next === "notes";
      const isChat = next === "chat";
      const isLearn = next === "learn";
      const isNews = next === "news";

      tabNotes.classList.toggle("active", isNotes);
      tabChat.classList.toggle("active", isChat);
      tabLearn.classList.toggle("active", isLearn);
      tabNews.classList.toggle("active", isNews);
      notes.classList.toggle("ha-hidden", !isNotes);
      chat.classList.toggle("ha-hidden", !isChat);
      learn.classList.toggle("ha-hidden", !isLearn);
      news.classList.toggle("ha-hidden", !isNews);
      clearNotes.classList.toggle("ha-hidden", !isNotes);

      if (isNotes) markNotificationsSeen();
    }

    function openPanel() {
      panel.classList.add("visible");
      panel.setAttribute("aria-hidden", "false");
      state.panelVisible = true;
      setActiveTab("notes");
      refreshHealthAndData();
    }

    function closePanel() {
      panel.classList.remove("visible");
      panel.setAttribute("aria-hidden", "true");
      state.panelVisible = false;
    }

    btn.addEventListener("click", () => {
      if (panel.classList.contains("visible")) closePanel();
      else openPanel();
    });
    close.addEventListener("click", closePanel);
    tabNotes.addEventListener("click", () => setActiveTab("notes"));
    tabChat.addEventListener("click", () => setActiveTab("chat"));
    tabLearn.addEventListener("click", () => {
      setActiveTab("learn");
      renderLearned(state.learnedItems, state.tasks);
    });
    tabNews.addEventListener("click", () => {
      setActiveTab("news");
      renderNews(state.newsItems);
    });
    newChat.addEventListener("click", () => {
      clearChatHistory();
      setActiveTab("chat");
      addChatMessage("New chat started.", "bot");
    });
    clearNotes.addEventListener("click", async () => {
      try {
        await postJSON("/assistant/notifications/clear", {});
        renderNotifications([]);
        state.notificationsMaxTs = 0;
        markNotificationsSeen();
      } catch (e) {
        addChatMessage("Failed to clear notifications: " + String(e.message || e), "bot");
      }
    });

    async function sendMessage() {
      const msg = (input.value || "").trim();
      if (!msg) return;
      addChatMessage(msg, "user");
      input.value = "";
      input.style.height = "";
      send.disabled = true;
      input.disabled = true;
      const view = document.getElementById("haChatView");
      let typingEl = null;
      if (view) {
        typingEl = document.createElement("div");
        typingEl.className = "ha-msg bot ha-typing";
        typingEl.innerHTML = `Thinking <span class="ha-dots"><span>.</span><span>.</span><span>.</span></span>`;
        view.appendChild(typingEl);
        view.scrollTop = view.scrollHeight;
      }
      try {
        const r = await postJSON("/assistant/chat", { message: msg });
        if (typingEl && typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
        addChatMessage((r && r.reply) || "No response.", "bot");
        if (r && Array.isArray(r.notifications)) {
          renderNotifications(r.notifications);
          updateUnreadFromNotifications(r.notifications);
        }
        await refreshHealthAndData();
      } catch (e) {
        if (typingEl && typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
        addChatMessage("Assistant unavailable: " + String(e.message || e), "bot");
      } finally {
        send.disabled = false;
        input.disabled = false;
        input.focus();
      }
    }

    send.addEventListener("click", sendMessage);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = Math.min(160, input.scrollHeight) + "px";
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && panel.classList.contains("visible")) closePanel();
    });
  }

  loadLocalState();
  ensureStyles();
  createUI();
  loadChatHistory();
  wireUI();
  refreshHealthAndData();
  setInterval(refreshHealthAndData, 12000);
})();
