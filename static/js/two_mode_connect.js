/* two_mode_connect.js — refined: preserves raw debug, persists auth, hides legacy clutter, disables toggle */
(function(){
  const APP_ID = 71710;
  const DEDUPE_MS = 3000;
  const log = (...a)=>{ try{ console.log('two_mode_connect:', ...a); }catch(e){} };
  function $id(id){ return document.getElementById(id); }
  function q(sel){ try { return Array.from(document.querySelectorAll(sel)); } catch(e){ return []; } }

  /* --- CSS: hide toggle, style cards & panel --- */
  (function ensureCSS(){
    if(document.getElementById('two-mode-connect-css')) return;
    const css = `
/* hide the toggle and other legacy control areas that interfere */
#modeToggle, #modeText, #accountMode, .legacy-toggle { display: none !important; }

/* hide old balance placeholders (but NOT #rawDebug) */
.balance-placeholder, #balance-panel, #old-balance, .old-balance { display: none !important; }

/* two-mode panel styling */
#hero-two-mode { z-index: 99999; padding:12px; background:#f7f8fb; border-bottom:1px solid #e6e8ef; font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial; }
#hero-two-mode .h-row { display:flex; gap:12px; align-items:center; margin-bottom:8px; flex-wrap:wrap }
#hero-two-mode input { padding:8px; border:1px solid #cfd8e3; border-radius:6px; width:340px; max-width:58vw; }
#hero-two-mode button { padding:8px 12px; border-radius:6px; cursor:pointer; border:1px solid rgba(2,6,23,0.06); background:#fff; }
#hero-two-mode .status { margin-left:8px; font-weight:600; color:#0b66c3; }

/* balance cards */
#hero-balance-container { position: fixed; right: 12px; top: 12px; display:flex; gap:12px; z-index:2147483647; }
.balance-card { min-width:160px; padding:12px; border-radius:10px; background:linear-gradient(180deg,#fff,#fbfbff); box-shadow:0 8px 18px rgba(12,20,30,0.12); border:1px solid rgba(0,0,0,0.06); }
.balance-card h4{ margin:0 0 6px 0; font-size:13px; color:#253858; }
.balance-card .amount{ font-size:18px; font-weight:700; color:#0b66c3; margin-top:6px; }
.balance-card .meta{ font-size:12px; color:#6b7280; margin-top:6px; }
.balance-card .status { font-size:12px; font-weight:600; color:#0b66c3; margin-top:8px; }
@media (max-width:720px){ .balance-card{ max-width:44vw; min-width:120px; padding:10px; } #hero-two-mode input{ max-width:55vw; } }
`;
    const s = document.createElement('style'); s.id='two-mode-connect-css'; s.textContent = css;
    (document.head || document.documentElement).appendChild(s);
  })();

  /* --- Balance UI management (two cards) --- */
  function ensureBalanceUI(){
    if($id('hero-balance-container')) return;
    const container = document.createElement('div'); container.id='hero-balance-container';
    const demoCard = document.createElement('div'); demoCard.className='balance-card'; demoCard.id='hero-balance-demo';
    demoCard.innerHTML = '<h4>Demo Account</h4><div class="amount" id="demo-amount">-</div><div class="meta" id="demo-meta">Not connected</div><div class="status" id="hero-demo-status-small">Not connected</div>';
    const realCard = document.createElement('div'); realCard.className='balance-card'; realCard.id='hero-balance-real';
    realCard.innerHTML = '<h4>Real Account</h4><div class="amount" id="real-amount">-</div><div class="meta" id="real-meta">Not connected</div><div class="status" id="hero-real-status-small">Not connected</div>';
    container.appendChild(demoCard); container.appendChild(realCard);
    document.body.appendChild(container);
  }

  /* --- normalize a server balance object into {amount,currency,loginid} --- */
  function normalize(b){
    try {
      if(!b) return null;
      if(typeof b === 'number') return { amount: b, currency:'', loginid:'' };
      if(b.balance && typeof b.balance === 'object'){
        let inner = b.balance;
        if(inner.balance && typeof inner.balance === 'object') inner = inner.balance;
        const amount = inner.balance !== undefined ? inner.balance : (typeof inner === 'number' ? inner : undefined);
        return { amount, currency: inner.currency||'', loginid: inner.loginid||inner.login_id||'' };
      }
      if(typeof b.balance === 'number') return { amount: b.balance, currency:b.currency||'', loginid:b.loginid||'' };
      if(b.amount !== undefined) return { amount: b.amount, currency:b.currency||'', loginid:b.loginid||'' };
      return null;
    } catch(e){ return null; }
  }

  /* --- set card UI --- */
  function setCard(mode, raw){
    ensureBalanceUI();
    const n = normalize(raw);
    const amountEl = mode==='real' ? $id('real-amount') : $id('demo-amount');
    const metaEl = mode==='real' ? $id('real-meta') : $id('demo-meta');
    const statusEl = mode==='real' ? $id('hero-real-status-small') : $id('hero-demo-status-small');
    if(n && n.amount !== undefined){
      amountEl.textContent = n.amount;
      metaEl.textContent = (n.currency? n.currency : '') + (n.loginid ? ' • ' + n.loginid : '');
      if(statusEl) statusEl.textContent = (mode==='real' ? 'Real authenticated' : 'Demo authenticated');
    } else {
      if(statusEl) statusEl.textContent = 'Connected (no balance)';
    }
  }

  /* --- Migrate useful visible balance text from legacy elements into cards, but keep rawDebug visible --- */
  function migrateLegacy(){
    const selectors = ['#balance-panel', '.balance-placeholder', '#old-balance', '.old-balance'];
    selectors.forEach(sel=>{
      q(sel).forEach(el=>{
        try {
          const txt = (el.textContent||'').trim();
          if(!txt) { el.classList.add('legacy-hidden'); return; }
          // try JSON parse
          let parsed = null;
          try { parsed = JSON.parse(txt); } catch(e){}
          if(parsed && parsed.balance) { setCard('demo', parsed.balance); }
          else {
            // basic parse: look for number and login tokens
            const m = txt.match(/([0-9]{1,3}(?:[0-9,]*)(?:\.[0-9]+)?)/);
            const login = (txt.match(/(CR|VRT|CRS|VRTC?)[0-9A-Z]*/i) || [null])[0];
            const currency = (txt.match(/\b(USD|EUR|GBP|KES)\b/)||[''])[0];
            if(m) setCard('demo', { balance: parseFloat(m[1].replace(/,/g,'')), currency: currency||'', loginid: login||'' });
          }
        } catch(e){}
        el.classList.add('legacy-hidden');
      });
    });
  }

  /* --- SSE attach with simple dedupe to avoid duplicates --- */
  let lastSSEHash = null;
  function attachSSE(){
    if(window._two_mode_connect_sse) return;
    if(typeof EventSource === 'undefined'){ log('EventSource missing'); return; }
    try {
      const ev = new EventSource('/events');
      window._two_mode_connect_sse = ev;
      ev.addEventListener('message', function(evt){
        try {
          const data = JSON.parse(evt.data);
          // dedupe by stringified balance + mode
          const mode = data.mode || (data.analysis_event === 'account_connected' ? data.mode : null) || null;
          const balance = data.balance || data;
          const h = (mode||'') + '|' + JSON.stringify(balance);
          if(h === lastSSEHash) return;
          lastSSEHash = h;
          if(data.analysis_event === 'account_connected'){
            setCard(data.mode || 'demo', data.balance || data);
            // persist that this mode is connected (server confirmed)
            if(data.mode === 'real') {
              localStorage.setItem('hero_connected_real','1');
            } else {
              localStorage.setItem('hero_connected_demo','1');
            }
          } else if(data.msg_type === 'balance' && data.balance){
            // attempt best-effort: update demo card (server should include mode ideally)
            setCard('demo', data.balance);
          }
        } catch(e){ console.error('SSE parse', e); }
      });
      ev.addEventListener('open', ()=>log('SSE open'));
      ev.addEventListener('error', (err)=>log('SSE error', err));
      log('SSE attached');
    } catch(e){ console.error('attachSSE error', e); }
  }

  /* --- sendConnect with dedupe + persist token if success --- */
  const lastSent = {}; // { mode: { token, ts } }
  async function sendConnect(token, mode, buttonEl, opts){
    opts = opts || {};
    const now = Date.now();
    if(!mode) mode='demo';
    if(lastSent[mode] && lastSent[mode].token === token && (now - lastSent[mode].ts) < DEDUPE_MS){
      log('sendConnect: duplicate ignored for', mode);
      return;
    }
    lastSent[mode] = { token, ts: now };
    if(buttonEl && !opts.silent) { try{ buttonEl.disabled = true; }catch(e){} }
    // reflect immediate UI
    const smallStatus = mode==='real' ? $id('hero-real-status-small') : $id('hero-demo-status-small');
    if(smallStatus) smallStatus.textContent = 'Connecting...';
    log('sendConnect: payload', {token: token||'', app_id: APP_ID, mode});
    try {
      const res = await fetch('/control/connect_account', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ token: token||'', app_id: APP_ID, mode })
      });
      const txt = await res.text();
      log('sendConnect: response', res.status, txt);
      try {
        const j = JSON.parse(txt);
        if(j && j.balance) setCard(mode, j.balance);
      } catch(e){}
      // if responded OK (2xx), persist token & connected flag to localStorage
      if(res.ok){
        if(mode==='real'){
          if(token) localStorage.setItem('hero_token_real', token);
          localStorage.setItem('hero_connected_real','1');
          if($id('hero-real-status')) $id('hero-real-status').textContent = 'Real authenticated';
        } else {
          if(token) localStorage.setItem('hero_token_demo', token);
          localStorage.setItem('hero_connected_demo','1');
          if($id('hero-demo-status')) $id('hero-demo-status').textContent = 'Demo authenticated';
        }
      } else {
        // on server error, clear connected flag
        if(mode==='real'){ localStorage.removeItem('hero_connected_real'); }
        else { localStorage.removeItem('hero_connected_demo'); }
      }
    } catch(err){
      console.error('sendConnect error', err);
      if(mode==='real'){ localStorage.removeItem('hero_connected_real'); }
      else { localStorage.removeItem('hero_connected_demo'); }
    } finally {
      if(buttonEl && !opts.silent) { try{ buttonEl.disabled = false; }catch(e){} }
    }
  }

  /* --- attach handlers to two-button UI, restore tokens from localStorage --- */
  function attachHandlersAndRestore(){
    ensureBalanceUI();
    // migrate legacy elements (but preserve #rawDebug)
    migrateLegacy();

    const demoInput = $id('hero-demo-token') || $id('apiToken') || $id('apiTokenDemo');
    const realInput = $id('hero-real-token') || $id('apiTokenReal') || $id('apiToken_real');
    const demoBtn = $id('hero-demo-connect') || $id('btnConnectAccount');
    const realBtn = $id('hero-real-connect') || $id('btnConnectReal');

    // hide unwanted toggle if present (visual hide only)
    ['modeToggle','modeText','accountMode'].forEach(id=>{ const e=$id(id); if(e) e.classList.add('legacy-toggle'); });

    if(demoBtn && !demoBtn._two_bound){
      demoBtn._two_bound = true;
      demoBtn.addEventListener('click', function(ev){
        try{ ev && ev.preventDefault && ev.preventDefault(); ev && ev.stopImmediatePropagation && ev.stopImmediatePropagation(); }catch(e){}
        const t = demoInput ? demoInput.value : '';
        sendConnect(t, 'demo', demoBtn);
      }, false);
    }
    if(realBtn && !realBtn._two_bound){
      realBtn._two_bound = true;
      realBtn.addEventListener('click', function(ev){
        try{ ev && ev.preventDefault && ev.preventDefault(); ev && ev.stopImmediatePropagation && ev.stopImmediatePropagation(); }catch(e){}
        const t = realInput ? realInput.value : '';
        sendConnect(t, 'real', realBtn);
      }, false);
    }

    // Restore stored tokens and auto-connect silently (so reload keeps auth)
    try {
      const storedDemo = localStorage.getItem('hero_token_demo');
      const storedReal = localStorage.getItem('hero_token_real');
      if(demoInput && storedDemo) { demoInput.value = storedDemo; if(!localStorage.getItem('hero_connected_demo')) sendConnect(storedDemo, 'demo', null, {silent:true}); else sendConnect(storedDemo,'demo',null,{silent:true}); }
      if(realInput && storedReal) { realInput.value = storedReal; if(!localStorage.getItem('hero_connected_real')) sendConnect(storedReal, 'real', null, {silent:true}); else sendConnect(storedReal,'real',null,{silent:true}); }
      // set UI statuses if flags exist
      if(localStorage.getItem('hero_connected_demo') && $id('hero-demo-status')) $id('hero-demo-status').textContent = 'Demo authenticated';
      if(localStorage.getItem('hero_connected_real') && $id('hero-real-status')) $id('hero-real-status').textContent = 'Real authenticated';
    } catch(e){ console.error('restore tokens error', e); }
  }

  /* --- init --- */
  document.addEventListener('DOMContentLoaded', function(){
    try{
      ensureBalanceUI();
      attachHandlersAndRestore();
      attachSSE();
      log('two_mode_connect refined ready');
    } catch(e){ console.error('init error', e); }
  });
})();
