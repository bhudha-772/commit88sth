// auth_modal.js
// Lightweight token summary banner + helpers to prompt user to authenticate accounts.
// Non-destructive: reads/writes only localStorage keys hero_token_*/hero_balance_*
// and navigates to /static/auth.html when user elects to authenticate.

(function(){
  const RETURN_PARAM = 'return';
  // NOTE: serve the auth page from /static so browsers don't 404 when requesting /auth.html
  const AUTH_PAGE = '/static/auth.html';

  // Polling defaults (ms). Default 1000ms. Can be customized by setting localStorage.hero_balance_poll_ms
  const DEFAULT_POLL_MS = 1000;
  const MIN_POLL_MS = 250;

  // internal poll handle
  let __hero_balance_poll_handle = null;
  let __hero_balance_poll_running = false;
  let __hero_balance_poll_inflight = false; // prevents overlapping poll requests

  function createElement(html){
    const div = document.createElement('div');
    div.innerHTML = html.trim();
    return div.firstChild;
  }

  function showToast(msg, type='info', timeout=1800){
    try{
      const id = 'hero-auth-toast';
      let el = document.getElementById(id);
      if(!el){
        el = document.createElement('div');
        el.id = id;
        el.style.position = 'fixed';
        el.style.left = '50%';
        el.style.transform = 'translateX(-50%)';
        el.style.top = '12px';
        el.style.zIndex = 999999;
        el.style.padding = '8px 12px';
        el.style.borderRadius = '8px';
        el.style.color = '#fff';
        el.style.fontWeight = '600';
        el.style.fontSize = '13px';
        document.body.appendChild(el);
      }
      el.style.background = (type === 'ok' ? '#10b981' : (type === 'warn' ? '#f59e0b' : '#2563eb') );
      el.textContent = msg;
      el.style.opacity = '1';
      setTimeout(()=>{ try{ el.style.transition = 'opacity .35s'; el.style.opacity = '0'; setTimeout(()=>el.remove(), 400); }catch(e){} }, timeout);
    }catch(e){}
  }

    async function fetchServerTokenStatus(){
    try {
      const r = await fetch('/control/get_tokens', { method:'GET', headers: { 'Accept': 'application/json' } });
      if(!r.ok) return null;
      const j = await r.json();
      return j;
    } catch(e){
      return null;
    }
  }


  // apply saved balances (if present in localStorage) to dashboard balance cards
  function applySavedBalancesToCards(){
    try{
      // parse saved objects (strings) if present
      const demoRaw = localStorage.getItem('hero_balance_demo');
      const realRaw = localStorage.getItem('hero_balance_real');

      if(demoRaw){
        try { const parsed = JSON.parse(demoRaw); applyBalanceToDOM('demo', parsed); } catch(e){}
      }
      if(realRaw){
        try { const parsed = JSON.parse(realRaw); applyBalanceToDOM('real', parsed); } catch(e){}
      }

      // update banner
      updateAuthSummary();
    }catch(e){}
  }

  function applyBalanceToDOM(mode, obj){
    try{
      // Try multiple selectors that your UI uses (be tolerant)
      const amountId = mode === 'real' ? 'real-amount' : 'demo-amount';
      const metaId = mode === 'real' ? 'real-meta' : 'demo-meta';
      const statusId = mode === 'real' ? 'hero-real-status-small' : 'hero-demo-status-small';

      const amountEl = document.getElementById(amountId);
      const metaEl = document.getElementById(metaId);
      const statusEl = document.getElementById(statusId);

      // If object shape is nested (balance / balances / amount), pick typical fields
      let normalized = null;
      if(!obj) normalized = null;
      else if(typeof obj === 'number') normalized = { amount: obj, currency: '' };
      else if(obj.balance !== undefined && typeof obj.balance !== 'object') normalized = { amount: obj.balance, currency: obj.currency || '' };
      else if(obj.amount !== undefined) normalized = { amount: obj.amount, currency: obj.currency || '' };
      else if(obj.balance && typeof obj.balance === 'object' && obj.balance.balance !== undefined) normalized = { amount: obj.balance.balance, currency: obj.balance.currency || '', loginid: obj.balance.loginid || obj.balance.login_id || '' };
      else normalized = { amount: obj.amount || obj.balance || '', currency: obj.currency || '', loginid: obj.loginid || obj.login_id || '' };

      if(normalized && amountEl){
        amountEl.textContent = normalized.amount;
      }
      if(normalized && metaEl){
        const metaText = (normalized.currency ? normalized.currency : '') + (normalized.loginid ? ' • ' + normalized.loginid : '');
        metaEl.textContent = metaText;
      }
      if(statusEl){
        statusEl.textContent = mode === 'real' ? 'Real authenticated' : 'Demo authenticated';
      }
    }catch(e){}
  }

  

  // Update the banner tokens/balance display
  function updateAuthSummary(){
    try{
      const demoToken = !!localStorage.getItem('hero_token_demo');
      const realToken = !!localStorage.getItem('hero_token_real');
      const demoBalRaw = localStorage.getItem('hero_balance_demo');
      const realBalRaw = localStorage.getItem('hero_balance_real');
      const demoEl = document.getElementById('hero-auth-demo-status');
      const realEl = document.getElementById('hero-auth-real-status');

      if(demoEl){
        if(demoToken){
          let text = 'token ✓';
          try { const p = demoBalRaw ? JSON.parse(demoBalRaw) : null; if(p){ text += ' • ' + (p.balance || p.amount || (p.balance && p.balance.balance) || '—'); } } catch(e){}
          demoEl.textContent = text;
          demoEl.style.color = '#10b981';
        } else {
          demoEl.textContent = 'no token';
          demoEl.style.color = '#ef4444';
        }
      }

      if(realEl){
        if(realToken){
          let text = 'token ✓';
          try { const p = realBalRaw ? JSON.parse(realBalRaw) : null; if(p){ text += ' • ' + (p.balance || p.amount || (p.balance && p.balance.balance) || '—'); } } catch(e){}
          realEl.textContent = text;
          realEl.style.color = '#10b981';
        } else {
          realEl.textContent = 'no token';
          realEl.style.color = '#ef4444';
        }
      }
    }catch(e){}
  }

  // Safe navigation helper: try a quick fetch to the auth page so we can surface a friendly toast instead of a browser alert/404.
  async function openAuthPageSafely(){
    try {
      const cur = window.location.pathname + window.location.search;
      const url = AUTH_PAGE + '?' + encodeURIComponent(RETURN_PARAM) + '=' + encodeURIComponent(cur);
      // attempt a lightweight HEAD/GET to confirm availability (some servers may not allow HEAD).
      try {
        const resp = await fetch(AUTH_PAGE, { method: 'GET', cache: 'no-store' });
        if(resp && resp.ok){
          window.location.href = url;
          return;
        } else {
          showToast('Auth page returned ' + (resp && resp.status ? resp.status : 'unknown') + '. Opening anyway...', 'warn', 2200);
          window.location.href = url;
          return;
        }
      } catch(e){
        // network error or CORS; still navigate but show toast to explain
        showToast('Unable to pre-check auth page; opening...', 'warn', 1800);
        window.location.href = url;
        return;
      }
    }catch(e){
      // fallback naive navigation
      try { window.location.href = AUTH_PAGE; } catch(err){}
    }
  }

  // Create a disconnect button next to authenticated button (single unified button)
  function createDisconnectButton(){
    try {
      // avoid duplicate
      if(document.getElementById('hero-auth-disconnect')) return;
      const wrapper = document.getElementById('hero-auth-wrapper') || (()=>{ createAuthBorderlessButton(); return document.getElementById('hero-auth-wrapper'); })();
      if(!wrapper) return;
      const disc = document.createElement('button');
      disc.id = 'hero-auth-disconnect';
      disc.textContent = 'Disconnect';
      disc.style.border = '1px solid rgba(2,6,23,0.08)';
      disc.style.background = '#fff';
      disc.style.color = '#ef4444';
      disc.style.fontWeight = '600';
      disc.style.padding = '6px 8px';
      disc.style.borderRadius = '8px';
      disc.style.cursor = 'pointer';
      disc.addEventListener('click', ()=>{
        // visible confirmation
        if(!confirm('Disconnect and clear saved tokens & balances from this browser?')) return;
        // clear both tokens by default; UI can decide to only remove one if needed
        disconnectAllTokens();
        updateAuthSummary();
      });
      wrapper.appendChild(disc);
    } catch(e) {}
  }

  // remove disconnect button
  function removeDisconnectButton(){
    try{
      const el = document.getElementById('hero-auth-disconnect');
      if(el) el.remove();
    }catch(e){}
  }

  // When authenticated (on return), replace/modify the button to show 'Authenticated' and make it non-clickable.
  // This now also adds a disconnect button next to it.
  function markAuthButtonAsAuthenticated(mode){
    try {
      // ensure wrapper/button exists (create if needed)
      if(!document.getElementById('hero-auth-wrapper')){
        createAuthBorderlessButton();
      }
      const btn = document.getElementById('hero-auth-borderless');
      if(!btn) return;
      // change appearance to show it's authenticated but keep it actionable only for opening auth page if long-press
      btn.textContent = 'Authenticated';
      // visible bordered style to indicate state (per your request)
      btn.style.border = '1px solid rgba(2,6,23,0.08)';
      btn.style.background = '#fff';
      btn.style.textDecoration = 'none';
      btn.style.color = '#0f172a';
      btn.style.fontWeight = '700';
      btn.style.padding = '6px 10px';
      btn.style.borderRadius = '8px';
      // disable pointer interactions for primary button (disconnect provided)
      btn.style.pointerEvents = 'none';
      btn.setAttribute('aria-disabled', 'true');
      // add a title explaining state
      btn.title = (mode ? (mode + ' authenticated') : 'Authenticated');

      // add disconnect button visible nearby
      createDisconnectButton();

      // start balance polling for current tokens
      startBalancePolling();
      updateAuthSummary();
    } catch(e){}
  }

  // ---- disconnect helpers ----
  function disconnectAllTokens(){
    // Clear tokens and balances from localStorage
    try {
      localStorage.removeItem('hero_token_demo');
      localStorage.removeItem('hero_token_real');
      localStorage.removeItem('hero_balance_demo');
      localStorage.removeItem('hero_balance_real');
      // update UI
      // revert the borderless button to default text
      const btn = document.getElementById('hero-auth-borderless');
      if(btn){
        btn.textContent = 'authenticate accounts...';
        btn.style.border = '0';
        btn.style.background = 'transparent';
        btn.style.textDecoration = 'underline';
        btn.style.color = '#2563eb';
        btn.style.pointerEvents = 'auto';
        btn.title = '';
      }
      // remove status small badges if present
      try{ const rs = document.getElementById('hero-real-status-small'); if(rs) rs.textContent = '—'; }catch(e){}
      try{ const ds = document.getElementById('hero-demo-status-small'); if(ds) ds.textContent = '—'; }catch(e){}
      // remove disconnect button
      removeDisconnectButton();
      // stop polling
      stopBalancePolling();
      showToast('Disconnected — tokens & balances cleared', 'warn', 1800);
      // optionally inform server (non-blocking)
      try{
        fetch('/control/disconnect_account', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({mode: 'all'}) }).catch(()=>{});
      }catch(e){}
      updateAuthSummary();
    } catch(e){
      console.error('disconnectAllTokens err', e);
    }
  }

  // per-mode disconnect (if you want to remove only demo or only real)
  function disconnectToken(mode){
    try{
      if(mode==='real') localStorage.removeItem('hero_token_real');
      if(mode==='demo') localStorage.removeItem('hero_token_demo');
      if(mode==='real') localStorage.removeItem('hero_balance_real');
      if(mode==='demo') localStorage.removeItem('hero_balance_demo');
      // update small status if present
      try{ const el = document.getElementById(mode==='real' ? 'hero-real-status-small' : 'hero-demo-status-small'); if(el) el.textContent = '—'; }catch(e){}
      showToast((mode==='real' ? 'Real' : 'Demo') + ' disconnected', 'warn', 1400);
      // if no tokens left, revert main button
      const hasDemo = !!localStorage.getItem('hero_token_demo');
      const hasReal = !!localStorage.getItem('hero_token_real');
      if(!hasDemo && !hasReal) {
        const btn = document.getElementById('hero-auth-borderless');
        if(btn){
          btn.textContent = 'authenticate accounts...';
          btn.style.border = '0';
          btn.style.background = 'transparent';
          btn.style.textDecoration = 'underline';
          btn.style.color = '#2563eb';
          btn.style.pointerEvents = 'auto';
          btn.title = '';
        }
        removeDisconnectButton();
        stopBalancePolling();
      }
      updateAuthSummary();
      // optional server inform
      try{
        fetch('/control/disconnect_account', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({mode}) }).catch(()=>{});
      }catch(e){}
    }catch(e){}
  }

  // Expose disconnect to window
  window.__hero_disconnect_account = disconnectToken;
  window.__hero_disconnect_all = disconnectAllTokens;

  // ---- balance polling helpers ----
  async function fetchBalancesForMode(mode){
    // fetch /control/get_balances?mode=
    try{
      const url = '/control/get_balances?mode=' + encodeURIComponent(mode);
      const r = await fetch(url, { method: 'GET', headers: { 'Accept': 'application/json' } });
      if(!r.ok){
        // on fail, don't throw; just log
        console.warn('fetchBalancesForMode failed', mode, r.status);
        return null;
      }
      const j = await r.json();
      if(j && (j.balance || j.balances || j.amount)){
        const obj = j.balance || j.balances || j;
        try { localStorage.setItem('hero_balance_' + mode, JSON.stringify(obj)); } catch(e){}
        applyBalanceToDOM(mode, obj);
        updateAuthSummary();
        return obj;
      } else {
        // some endpoints return simple object; still save raw
        try { localStorage.setItem('hero_balance_' + mode, JSON.stringify(j)); } catch(e){}
        applyBalanceToDOM(mode, j);
        updateAuthSummary();
        return j;
      }
    }catch(e){
      console.warn('fetchBalancesForMode error', e);
      return null;
    }
  }

    async function pollBalancesOnce(){
    // If tokens exist, fetch balances for the relevant modes.
    try{
      // Prevent overlapping polls — if a previous poll is still running, skip this iteration
      if (__hero_balance_poll_inflight) return;
      __hero_balance_poll_inflight = true;

      const hasDemo = !!localStorage.getItem('hero_token_demo');
      const hasReal = !!localStorage.getItem('hero_token_real');

      // fetch them in parallel but wait for both to avoid piling up
      const promises = [];
      if(hasDemo) promises.push(fetchBalancesForMode('demo'));
      if(hasReal) promises.push(fetchBalancesForMode('real'));
      if(promises.length === 0) return;
      await Promise.all(promises);
      updateAuthSummary();
    }catch(e){
      // ignore errors (they'll be logged inside fetchBalancesForMode)
    } finally {
      __hero_balance_poll_inflight = false;
    }
  }


    function showAuthPrompt(){
    // if modal exists already, update its contents instead of re-creating
    let wrapper = document.getElementById('hero-auth-prompt');
    if(wrapper){
      // just refresh the displayed token lines from server
    } else {
      wrapper = document.createElement('div');
      wrapper.id = 'hero-auth-prompt';
      wrapper.style.position = 'fixed';
      wrapper.style.left = '0';
      wrapper.style.top = '0';
      wrapper.style.right = '0';
      wrapper.style.bottom = '0';
      wrapper.style.display = 'flex';
      wrapper.style.alignItems = 'center';
      wrapper.style.justifyContent = 'center';
      wrapper.style.background = 'rgba(10,15,25,0.45)';
      wrapper.style.zIndex = '200000';

      const card = document.createElement('div');
      card.style.width = '520px';
      card.style.maxWidth = '94%';
      card.style.background = '#fff';
      card.style.borderRadius = '10px';
      card.style.padding = '18px';
      card.style.boxShadow = '0 24px 80px rgba(2,6,23,0.3)';
      card.id = 'hero-auth-card';
      wrapper.appendChild(card);
      document.body.appendChild(wrapper);

      // base card layout
      const inner = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <div style="font-weight:800;font-size:16px">Authenticate accounts</div>
          <button id="hero-auth-close" style="border:0;background:transparent;font-weight:700;cursor:pointer">✕</button>
        </div>
        <div id="hero-auth-body" style="color:#374151;margin-bottom:12px">Loading token status…</div>
        <div style="display:flex;gap:10px;justify-content:flex-end">
          <button id="hero-auth-notnow" style="padding:8px 12px;border-radius:8px;border:1px solid rgba(0,0,0,0.06);background:transparent;cursor:pointer">Not now</button>
          <button id="hero-auth-yes" style="padding:8px 12px;border-radius:8px;border:0;background:#2563eb;color:#fff;cursor:pointer">Open Authenticate Page</button>
        </div>
      `;
      card.innerHTML = inner;

      // handlers
      document.getElementById('hero-auth-yes').addEventListener('click', ()=>{
        const cur = window.location.pathname + window.location.search;
        const url = AUTH_PAGE + '?' + encodeURIComponent(RETURN_PARAM) + '=' + encodeURIComponent(cur);
        window.location.href = url;
      });
      document.getElementById('hero-auth-notnow').addEventListener('click', ()=>{
        hideAuthPrompt();
        createAuthBorderlessButton();
      });
      document.getElementById('hero-auth-close').addEventListener('click', ()=>{
        hideAuthPrompt();
        createAuthBorderlessButton();
      });
    }

    // Fill modal body with current token status: localStorage tokens and server tokens
    (async function updateAuthModalBody(){
      try {
        const bodyEl = document.getElementById('hero-auth-body');
        if(!bodyEl) return;
        const localDemo = !!localStorage.getItem('hero_token_demo');
        const localReal = !!localStorage.getItem('hero_token_real');
        const server = await fetchServerTokenStatus();

        let html = '';
        if(server && server.ok){
          const demoPresent = server.demo_present;
          const realPresent = server.real_present;
          const demoMask = server.demo_mask || '—';
          const realMask = server.real_mask || '—';
          html += `<div style="margin-bottom:8px"><strong>Server tokens:</strong></div>`;
          html += `<div style="display:flex;gap:12px;align-items:center"><div>Demo:</div><div style="font-weight:700">${demoPresent ? ('Token ✓ ' + demoMask) : '<span style="color:#ef4444">No token</span>'}</div></div>`;
          html += `<div style="display:flex;gap:12px;align-items:center;margin-top:6px"><div>Real:</div><div style="font-weight:700">${realPresent ? ('Token ✓ ' + realMask) : '<span style="color:#ef4444">No token</span>'}</div></div>`;
          html += `<hr style="margin:12px 0">`;
          html += `<div style="margin-bottom:8px"><strong>Client tokens (browser storage):</strong></div>`;
          html += `<div>Demo: ${localDemo ? '<span style="color:#10b981">present</span>' : '<span style="color:#ef4444">none</span>'}</div>`;
          html += `<div>Real: ${localReal ? '<span style="color:#10b981">present</span>' : '<span style="color:#ef4444">none</span>'}</div>`;
        } else {
          // server unreachable -> show only local state
          html += `<div style="margin-bottom:8px"><strong>Server tokens: (unreachable)</strong></div>`;
          html += `<div>Demo (local): ${localDemo ? '<span style="color:#10b981">present</span>' : '<span style="color:#ef4444">none</span>'}</div>`;
          html += `<div>Real (local): ${localReal ? '<span style="color:#10b981">present</span>' : '<span style="color:#ef4444">none</span>'}</div>`;
        }

        // if no tokens anywhere, include directive
        const anyServer = (server && (server.demo_present || server.real_present));
        if(!anyServer && !localDemo && !localReal){
          html += `<div style="margin-top:12px;color:#b91c1c;font-weight:700">No tokens in the system — please authenticate to provide tokens for demo and/or real accounts.</div>`;
        } else {
          html += `<div style="margin-top:12px;color:#047857">Tokens present — you can open Authenticate Page to manage or replace tokens.</div>`;
        }

        bodyEl.innerHTML = html;
      } catch(e){
        // ignore
      }
    })();
  }

  function startBalancePolling(){
    try{
      if(__hero_balance_poll_running) return;
      // poll interval can be overridden in localStorage.hero_balance_poll_ms
      let ms = DEFAULT_POLL_MS;
      try{
        const v = parseInt(localStorage.getItem('hero_balance_poll_ms'));
        if(v && Number.isFinite(v) && v >= MIN_POLL_MS) ms = v;
      }catch(e){}
      __hero_balance_poll_running = true;
      // do an immediate poll
      pollBalancesOnce().catch(()=>{});
      __hero_balance_poll_handle = setInterval(()=> {
        // don't re-enter if previous poll still running (simple guard)
        pollBalancesOnce().catch(()=>{});
      }, ms);
      window.__hero_start_balance_polling = startBalancePolling;
      window.__hero_stop_balance_polling = stopBalancePolling;
      console.debug('Balance polling started, interval=' + ms + 'ms');
    }catch(e){}
  }

  function stopBalancePolling(){
    try{
      if(__hero_balance_poll_handle){
        clearInterval(__hero_balance_poll_handle);
        __hero_balance_poll_handle = null;
      }
      __hero_balance_poll_running = false;
      window.__hero_start_balance_polling = startBalancePolling;
      window.__hero_stop_balance_polling = stopBalancePolling;
      console.debug('Balance polling stopped');
    }catch(e){}
  }

  function logDebug(msg){
    // Lightweight console debug; keep no-op in prod if console unavailable
    try{ console.debug('[hero-auth] ' + msg); }catch(e){}
  }

  // If we detect a query param ?auth_done=1 we show a small toast and apply saved balances.
  function checkAuthReturnFlag(){
    try{
      const qs = new URLSearchParams(window.location.search || '');
      if(qs.get('auth_done') === '1'){
        // optionally which mode
        const mode = qs.get('mode');
        showToast('Authenticated' + (mode ? (' ('+mode+')') : '') + ' — returning to dashboard', 'ok', 2000);
        // apply saved balances to cards
        setTimeout(()=>applySavedBalancesToCards(), 250);

        // mark the header button visually as authenticated and non-clickable
        try { setTimeout(()=> { markAuthButtonAsAuthenticated(mode || ''); }, 300); } catch(e){}

        // start polling if tokens exist
        setTimeout(()=>startBalancePolling(), 500);

        // --- show masked tokens confirmation (server-prefers masked, fallback to client) ---
        try {
          (async function showCapturedTokens(){
            try {
              let demoMask = null, realMask = null;
              // Prefer server-provided masked tokens if available
              try {
                const resp = await fetch('/control/get_server_tokens', { method: 'GET', headers: { 'Accept': 'application/json' } });
                if (resp && resp.ok) {
                  const j = await resp.json().catch(()=>null);
                  if (j && j.tokens) {
                    demoMask = (j.tokens.demo && j.tokens.demo.masked) || null;
                    realMask = (j.tokens.real && j.tokens.real.masked) || null;
                  }
                }
              } catch(e){ /* ignore network/server errors */ }

              // Fallback: mask client-side localStorage tokens
              try {
                if (!demoMask) {
                  const raw = localStorage.getItem('hero_token_demo');
                  if (raw) demoMask = (String(raw).length > 8) ? (String(raw).slice(0,4) + '...' + String(raw).slice(-4)) : raw;
                }
                if (!realMask) {
                  const raw = localStorage.getItem('hero_token_real');
                  if (raw) realMask = (String(raw).length > 8) ? (String(raw).slice(0,4) + '...' + String(raw).slice(-4)) : raw;
                }
              } catch(e){}

              if (demoMask || realMask) {
                const parts = [];
                parts.push('Demo: ' + (demoMask || '—'));
                parts.push('Real: ' + (realMask || '—'));
                showToast('Captured tokens — ' + parts.join(' • '), 'ok', 6000);
              }
            } catch(e){}
          })();
        } catch(e){}


        // cleanup url (remove query param) without reloading
        try {
          qs.delete('auth_done');
          qs.delete('mode');
          const base = window.location.pathname + (qs.toString() ? ('?' + qs.toString()) : '');
          history.replaceState({}, '', base);
        } catch(e){}
      } else {
        // If tokens already exist in storage, show a non-clickable "Authenticated" button immediately
        const hasDemoToken = !!localStorage.getItem('hero_token_demo');
        const hasRealToken = !!localStorage.getItem('hero_token_real');
        if(hasDemoToken || hasRealToken){
          // create button and mark as authenticated/non-clickable
          createAuthBorderlessButton();
          try { setTimeout(()=> { markAuthButtonAsAuthenticated(hasRealToken ? 'real' : (hasDemoToken ? 'demo' : '')); }, 200); } catch(e){}
          // start polling
          try { startBalancePolling(); } catch(e){}
          updateAuthSummary();
        }
      }
    }catch(e){}
  }

  // When dashboard loads, show non-intrusive banner (instead of forcing modal).
  document.addEventListener('DOMContentLoaded', function(){
    try{
      // apply balances immediately if present
      applySavedBalancesToCards();

      // create the small summary banner
      createAuthSummaryBanner();
      updateAuthSummary();

      // if user already has both tokens stored, mark as authenticated and start polling
      const hasDemoToken = !!localStorage.getItem('hero_token_demo');
      const hasRealToken = !!localStorage.getItem('hero_token_real');
      if(hasDemoToken || hasRealToken){
        try { setTimeout(()=> { markAuthButtonAsAuthenticated(hasRealToken ? 'real' : 'demo'); }, 200); } catch(e){}
        checkAuthReturnFlag();
        return;
      }

        // new: consult server tokens + localStorage to decide flow
  (async function decideAuthUI(){
    try {
      const server = await fetchServerTokenStatus();
      const serverHasDemo = server && server.demo_present;
      const serverHasReal = server && server.real_present;
      const localDemo = !!localStorage.getItem('hero_token_demo');
      const localReal = !!localStorage.getItem('hero_token_real');

      if(serverHasDemo || serverHasReal || localDemo || localReal){
        // tokens exist somewhere -> no intrusive prompt. Show borderless authenticated button and update small status badges
        createAuthBorderlessButton();
        try { setTimeout(()=> { markAuthButtonAsAuthenticated(serverHasReal ? 'real' : (serverHasDemo ? 'demo' : '')); }, 200); } catch(e){}
        // update small status badges if server provided masks
        if(server && server.ok){
          try {
            const demoEl = document.getElementById('hero-demo-status-small') || document.getElementById('demo-status');
            const realEl = document.getElementById('hero-real-status-small') || document.getElementById('real-status');
            if(demoEl) demoEl.textContent = server.demo_present ? ('Token ✓ ' + (server.demo_mask||'')) : (localDemo ? 'Token (local)' : 'No token');
            if(realEl) realEl.textContent = server.real_present ? ('Token ✓ ' + (server.real_mask||'')) : (localReal ? 'Token (local)' : 'No token');
          } catch(e){}
        }
        // start polling balances if tokens exist anywhere
        try { startBalancePolling(); } catch(e){}
      } else {
        // no tokens anywhere -> show friendly prompt
        setTimeout(()=>{ showAuthPrompt(); checkAuthReturnFlag(); }, 700);
      }
    } catch(e){
      // fallback: show prompt if server can't be reached
      setTimeout(()=>{ showAuthPrompt(); checkAuthReturnFlag(); }, 700);
    }
  })();
    }catch(e){}
  });

  // Expose function for other scripts to apply saved balances
  window.__hero_apply_saved_balances = applySavedBalancesToCards;
  // Expose prompt opener for other scripts (if they wish)
  window.showAuthPrompt = function(){ createAuthSummaryBanner(); openAuthPageSafely(); };

  // Expose polling controls & debug helpers
  window.__hero_start_balance_polling = startBalancePolling;
  window.__hero_stop_balance_polling = stopBalancePolling;
  window.__hero_balance_poll_running = () => __hero_balance_poll_running;

  // Keep banner up-to-date across tabs
  window.addEventListener('storage', function(e){
    if(!e) return;
    if(e.key && (e.key.startsWith('hero_token_') || e.key.startsWith('hero_balance_'))){
      updateAuthSummary();
      try { applySavedBalancesToCards(); } catch(e){}
    }
  });

})();

// ---------------------- Balance UI refresher & debug helpers ----------------------
// Add this after your existing polling helpers so it can reuse functions already defined.

(function(){
  // debug toggle: set to true to see console debug logs from balance polling
  const BALANCE_DEBUG = false;

  // safe helper to print debug lines
  function _dbg(...args){
    if(BALANCE_DEBUG) console.debug('[hero-balance]', ...args);
  }

  // Force-apply localStorage balances to DOM (idempotent)
  function forceApplyLocalBalances(){
    try {
      const demoRaw = localStorage.getItem('hero_balance_demo');
      const realRaw = localStorage.getItem('hero_balance_real');
      if(demoRaw){
        try { const parsed = JSON.parse(demoRaw); applyBalanceToDOM('demo', parsed); _dbg('applied demo from localStorage', parsed); } catch(e){ _dbg('failed parse demo local', e); }
      }
      if(realRaw){
        try { const parsed = JSON.parse(realRaw); applyBalanceToDOM('real', parsed); _dbg('applied real from localStorage', parsed); } catch(e){ _dbg('failed parse real local', e); }
      }
    }catch(e){ _dbg('forceApplyLocalBalances err', e); }
  }

  // Enhanced startBalancePolling that also forces local refresh after each poll
  const originalStart = window.__hero_start_balance_polling || null;
  function enhancedStartBalancePolling(){
    try {
      // if already running, re-use
      if (typeof window.__hero_balance_poll_running === 'function' ? window.__hero_balance_poll_running() : window.__hero_balance_poll_running) {
        _dbg('balance poll already running');
        return;
      }
      // call original start if available (keeps existing behaviour)
      if (typeof originalStart === 'function') {
        originalStart();
      } else {
        // fallback: call defined startBalancePolling function if in same closure
        try { startBalancePolling(); } catch(e){ _dbg('no original startBalancePolling', e); }
      }

      // ensure immediate UI sync and then start periodic local apply
      forceApplyLocalBalances();
      // store handle so we can stop it later
      if (!window.__hero_local_balance_refresher) {
        const ms = (parseInt(localStorage.getItem('hero_balance_poll_ms')) || 1000);
        window.__hero_local_balance_refresher = setInterval(forceApplyLocalBalances, Math.max(250, ms));
        _dbg('local balance refresher started, interval', Math.max(250, ms));
      }
    } catch(e){ _dbg('enhancedStartBalancePolling err', e); }
  }

  function enhancedStopBalancePolling(){
    try {
      // stop original polling if possible
      if (typeof window.__hero_stop_balance_polling === 'function') {
        try { window.__hero_stop_balance_polling(); } catch(e){ _dbg('stop underlying poll failed', e); }
      } else {
        try { stopBalancePolling(); } catch(e){ _dbg('stopBalancePolling not present', e); }
      }
      // stop local refresher
      if (window.__hero_local_balance_refresher) {
        clearInterval(window.__hero_local_balance_refresher);
        window.__hero_local_balance_refresher = null;
        _dbg('local balance refresher stopped');
      }
    } catch(e){ _dbg('enhancedStopBalancePolling err', e); }
  }

  // Replace the window helpers with enhanced variants so external code uses them
  window.__hero_start_balance_polling = enhancedStartBalancePolling;
  window.__hero_stop_balance_polling = enhancedStopBalancePolling;

  // Also expose a debug helper to inspect server immediately
  window.__hero_debug_fetch_balances_once = async function(){
    try {
      const modes = ['demo','real'];
      for(const m of modes){
        try {
          const r = await fetch('/control/get_balances?mode=' + encodeURIComponent(m));
          const j = await r.json().catch(()=>null);
          console.log('[hero-debug] server get_balances', m, j);
        } catch(e){ console.warn('[hero-debug] fail get_balances', m, e); }
      }
      // also show local saved values
      console.log('[hero-debug] local demo', localStorage.getItem('hero_balance_demo'));
      console.log('[hero-debug] local real', localStorage.getItem('hero_balance_real'));
      // and apply UI from localStorage now
      forceApplyLocalBalances();
    } catch(e){ console.warn('[hero-debug] debug fetch err', e); }
  };

  // If tokens/balances already present, start the enhanced polling automatically
  try {
    const hasDemo = !!localStorage.getItem('hero_token_demo') || !!localStorage.getItem('hero_balance_demo');
    const hasReal = !!localStorage.getItem('hero_token_real') || !!localStorage.getItem('hero_balance_real');
    if (hasDemo || hasReal) {
      // small timeout so the rest of the page finishes init
      setTimeout(()=>{ try{ enhancedStartBalancePolling(); }catch(e){} }, 400);
    }
  } catch(e){}
})();

// --- Handle cross-window auth returns via postMessage (from auth page) ---
(function(){
  // helper to display captured token masks (same logic used previously)
  async function _displayCapturedTokensOnce(mode) {
    try {
      let demoMask = null, realMask = null;
      try {
        const resp = await fetch('/control/get_server_tokens', { method: 'GET', headers: { 'Accept': 'application/json' } });
        if (resp && resp.ok) {
          const j = await resp.json().catch(()=>null);
          if (j && j.tokens) {
            demoMask = (j.tokens.demo && j.tokens.demo.masked) || null;
            realMask = (j.tokens.real && j.tokens.real.masked) || null;
          }
        }
      } catch(e){ /* ignore */ }

      try {
        if (!demoMask) {
          const raw = localStorage.getItem('hero_token_demo');
          if (raw) demoMask = (String(raw).length > 8) ? (String(raw).slice(0,4) + '...' + String(raw).slice(-4)) : raw;
        }
        if (!realMask) {
          const raw = localStorage.getItem('hero_token_real');
          if (raw) realMask = (String(raw).length > 8) ? (String(raw).slice(0,4) + '...' + String(raw).slice(-4)) : raw;
        }
      } catch(e){}

      if (demoMask || realMask) {
        const parts = [];
        parts.push('Demo: ' + (demoMask || '—'));
        parts.push('Real: ' + (realMask || '—'));
        showToast('Captured tokens — ' + parts.join(' • '), 'ok', 6000);
      }
    } catch(e){}
  }

  // Event handler: listen for messages from auth tab
  window.addEventListener('message', function(ev) {
    try {
      if (!ev || !ev.data) return;
      // Only accept same-origin messages for safety
      if (ev.origin && ev.origin !== window.location.origin) return;

      const d = ev.data;
      if (d && d.type === 'auth_return') {
        // read mode if provided
        const mode = d.mode || (d.url ? (new URL(d.url, window.location.origin).searchParams.get('mode')) : '');
        showToast('Authenticated' + (mode ? (' ('+mode+')') : '') + ' — returning to dashboard', 'ok', 2000);
        try { setTimeout(()=>applySavedBalancesToCards(), 250); } catch(e){}
        try { setTimeout(()=> { markAuthButtonAsAuthenticated(mode || ''); }, 300); } catch(e){}
        try { setTimeout(()=>startBalancePolling(), 500); } catch(e){}
        // show masked tokens
        _displayCapturedTokensOnce(mode).catch(()=>{});
        // also try to remove auth prompt UI if present
        try { hideAuthPrompt && hideAuthPrompt(); } catch(e){}
      }
    } catch(e){}
  }, false);

  // Also keep the existing URL query param handler working (when the page actually returned with ?auth_done)
  // If we still have the old checkAuthReturnFlag() flow, keep it — it will run on load and show the same toast + captured tokens.
})();
