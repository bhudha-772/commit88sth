/*
  compact autotrade panel (autotrade_panel.js)
  - compact inline journal that fits dashboard/analysis panel
  - small book icon opens modal for full journal if needed
  - preserves autotrade logic, SSE handling, export, clear
  - stores runtime state in localStorage
*/
(function(){
  const TRADE_ENDPOINT = '/control/place_trade';
  const APP_ID = 71710;
  const DEFAULTS = {
    base_stake: 1.0,
    min_confidence: 0.5,
    recovery_mode: 'recover_with_profit',
    desired_profit: 1.0,
    payout_estimate: 0.8,
    max_stake: 100.0,
    max_consecutive_losses: 6,
    auto_resume_on_load: true
  };
  const MAX_JOURNAL = 1000;
  const log = (...a)=> { try{ console.log('AUTOTRADE:', ...a); }catch(e){} };
  const $id = id => document.getElementById(id);

  function lsGet(k, fallback){ try{ const v = localStorage.getItem('hero_autotrade_'+k); return v ? JSON.parse(v) : fallback; }catch(e){ return fallback; } }
  function lsSet(k, v){ try{ localStorage.setItem('hero_autotrade_'+k, JSON.stringify(v)); }catch(e){} }

  // runtime state
  let runtime = {
    enabled: false,
    selected_mode: 'demo',
    activeTrade: null,
    waitingForSettlement: false,
    consecutive_losses: 0,
    loss_sum: 0.0,
    last_loss_confidence: null,
    payout_last: null,
    signalQueue: [],
    journal: lsGet('journal', []),
    settings: lsGet('settings', DEFAULTS)
  };

  /* ==== UI Creation: compact panel + small book icon + modal (optional) ==== */
  function ensureUI(){
    if($id('hero-autotrade-compact')) return;

    const container = $id('hero-two-mode') || document.body;
    const panel = document.createElement('div');
    panel.id = 'hero-autotrade-compact';
    panel.style.padding = '6px 10px';
    panel.style.borderTop = '1px solid rgba(0,0,0,0.04)';
    panel.style.display = 'flex';
    panel.style.alignItems = 'center';
    panel.style.gap = '8px';
    panel.style.flexWrap = 'wrap';
    panel.style.fontSize = '13px';

    panel.innerHTML = `
      <label style="font-weight:600;margin-right:6px">Autotrade</label>
      <button id="auto-toggle" style="padding:6px 10px;border-radius:6px;">Off</button>
      <select id="auto-account" style="margin-left:6px">
        <option value="demo">Demo</option><option value="real">Real</option>
      </select>
      <input id="auto-base-stake" type="number" step="0.01" style="width:80px;margin-left:6px" title="Base stake" />
      <input id="auto-min-conf" type="number" step="0.01" min="0" max="1" style="width:70px" title="Min confidence" />
      <button id="auto-refresh" title="Refresh balances" style="padding:5px 8px;border-radius:6px">Refresh</button>

      <div id="auto-journal-inline" style="margin-left:10px; display:flex; gap:8px; align-items:center">
        <div id="auto-journal-summary" style="max-width:520px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; color:#444; font-size:12px"></div>
        <button id="auto-journal-btn" title="Open Journal" style="background:none;border:0;padding:6px;cursor:pointer">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M3 5.5A2.5 2.5 0 0 1 5.5 3H18a2 2 0 0 1 2 2v14a1 1 0 0 1-1.555.832L15 17.5l-3.445 2.332A1 1 0 0 1 10 19.5V5.5" stroke="#111" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M6 3v16" stroke="#111" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
    `;
    container.appendChild(panel);

    // modal (hidden)
    if(!$id('auto-journal-modal')){
      const modal = document.createElement('div');
      modal.id = 'auto-journal-modal';
      modal.style.position='fixed';
      modal.style.left='50%';
      modal.style.top='50%';
      modal.style.transform='translate(-50%,-50%)';
      modal.style.width='640px';
      modal.style.maxHeight='70vh';
      modal.style.overflow='auto';
      modal.style.background='#fff';
      modal.style.border='1px solid rgba(0,0,0,0.08)';
      modal.style.boxShadow='0 12px 40px rgba(0,0,0,0.12)';
      modal.style.borderRadius='8px';
      modal.style.padding='12px';
      modal.style.zIndex=999999;
      modal.style.display='none';
      modal.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
          <strong>Autotrade Journal</strong>
          <div style="display:flex;gap:8px;align-items:center">
            <button id="modal-export" style="padding:6px;border-radius:6px">Export CSV</button>
            <button id="modal-clear" style="padding:6px;border-radius:6px">Clear</button>
            <button id="modal-close" style="padding:6px;border-radius:6px">Close</button>
          </div>
        </div>
        <div id="auto-journal-contents" style="font-size:13px"></div>
      `;
      document.body.appendChild(modal);
    }

    // wire up buttons (after UI created)
    document.getElementById('auto-journal-btn').addEventListener('click', openModal);
    document.getElementById('auto-refresh').addEventListener('click', refreshBalances);
  }

  function openModal(){
    const m = $id('auto-journal-modal'); if(!m) return;
    renderModalJournal();
    m.style.display='block';
  }
  function closeModal(){ const m = $id('auto-journal-modal'); if(!m) return; m.style.display='none'; }

  /* ==== Journal helpers (compact inline + full modal) ==== */
  function pushJournal(entry){
    entry.id = 't' + Date.now();
    entry.ts = Date.now();
    runtime.journal.unshift(entry);
    if(runtime.journal.length > MAX_JOURNAL) runtime.journal.length = MAX_JOURNAL;
    lsSet('journal', runtime.journal);
    renderInlineSummary();
  }

  function renderInlineSummary(){
    const el = $id('auto-journal-summary');
    if(!el) return;
    if(runtime.journal.length === 0){
      el.textContent = 'No trades yet.';
      return;
    }
    // create a compact single-line summary of last 6 entries (aggregate)
    const slice = runtime.journal.slice(0,6).map(t=>{
      const dd = new Date(t.ts);
      const ts = dd.toLocaleTimeString();
      const o = (t.outcome ? (t.outcome[0].toUpperCase()+t.outcome.slice(1)) : 'pend');
      const stake = (t.stake !== undefined ? t.stake : '');
      return `${ts} ${t.symbol||''} ${o} ${stake}`;
    });
    el.textContent = slice.join(' • ');
  }

  function renderModalJournal(){
    const container = $id('auto-journal-contents');
    if(!container) return;
    container.innerHTML = '';
    if(runtime.journal.length === 0){
      container.innerHTML = '<div style="color:#666">No trades yet.</div>'; return;
    }
    // show newest first, compact rows with important fields
    runtime.journal.slice(0,500).forEach(tr=>{
      const row = document.createElement('div');
      row.style.display='flex';
      row.style.justifyContent='space-between';
      row.style.padding='6px 0';
      row.style.borderTop='1px solid #eee';
      row.innerHTML = `
        <div style="flex:1;min-width:0">
          <div style="font-size:13px"><strong>${new Date(tr.ts).toLocaleString()}</strong> &nbsp; <small style="color:#666">${tr.mode||''} ${tr.symbol||''}</small></div>
          <div style="font-size:12px;color:#333">${tr.direction ? tr.direction + ' • ' : ''} stake: ${tr.stake ?? ''} • outcome: <strong>${tr.outcome ?? 'pending'}</strong></div>
        </div>
        <div style="width:160px;text-align:right;font-size:12px;color:#444">
          ${tr.payout !== undefined ? ('payout: '+(isFinite(tr.payout)?Number(tr.payout).toFixed(2):tr.payout)) : ''}
          <div style="color:#666">${tr.profit_loss !== undefined ? ('P/L: '+(isFinite(tr.profit_loss)?Number(tr.profit_loss).toFixed(2):'')) : ''}</div>
        </div>
      `;
      container.appendChild(row);
    });

    // wire modal exports & clear
    $id('modal-export').onclick = function(){ exportJournalCSV(); };
    $id('modal-clear').onclick = function(){ if(confirm('Clear autotrade journal?')){ runtime.journal=[]; lsSet('journal', runtime.journal); renderInlineSummary(); renderModalJournal(); } };
    $id('modal-close').onclick = function(){ closeModal(); };
  }

  function exportJournalCSV(){
    const rows = [['ts','mode','symbol','direction','stake','outcome','payout','profit_loss','balance_before','balance_after','signal_id']];
    runtime.journal.slice().reverse().forEach(t=>{
      rows.push([t.ts,t.mode,t.symbol,t.direction,t.stake,t.outcome,t.payout,t.profit_loss,t.balance_before,t.balance_after,t.signal_id]);
    });
    const csv = rows.map(r=>r.map(c=>'"'+String(c||'').replace(/"/g,'""')+'"').join(',')).join('\n');
    const blob = new Blob([csv],{type:'text/csv'}); const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download='autotrade_journal.csv'; document.body.appendChild(a); a.click();
    setTimeout(()=>{ URL.revokeObjectURL(url); try{ a.remove(); }catch(e){} },2000);
  }

  /* ==== Balance card helper (keeps same shape as old setCard) ==== */
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
      if(typeof b.balance === 'number') return { amount:b.balance, currency:b.currency||'', loginid:b.loginid||'' };
      if(b.amount !== undefined) return { amount:b.amount, currency:b.currency||'', loginid:b.loginid||'' };
      return null;
    } catch(e){ return null; }
  }
  function setCard(mode, raw){
    try {
      const n = normalize(raw);
      const amountEl = mode==='real' ? $id('real-amount') : $id('demo-amount');
      const metaEl = mode==='real' ? $id('real-meta') : $id('demo-meta');
      const statusEl = mode==='real' ? $id('hero-real-status-small') : $id('hero-demo-status-small');
      if(n && n.amount !== undefined){
        if(amountEl) amountEl.textContent = n.amount;
        if(metaEl) metaEl.textContent = (n.currency? n.currency : '') + (n.loginid ? ' • ' + n.loginid : '');
        if(statusEl) statusEl.textContent = (mode==='real' ? 'Real authenticated' : 'Demo authenticated');
      } else {
        if(statusEl) statusEl.textContent = 'Connected (no balance)';
      }
    } catch(e){}
  }

  /* ==== Core autotrade helpers (placeTrade, computeRecoveryStake, queue processing) ==== */
  async function placeTrade(signal, stake, mode){
    const payload = {
      token: (mode==='real' ? (localStorage.getItem('hero_token_real')||'') : (localStorage.getItem('hero_token_demo')||'')),
      app_id: APP_ID,
      mode: mode,
      symbol: signal.symbol || signal.instrument || signal.asset || 'unknown',
      direction: signal.direction || signal.side || signal.action || 'call',
      stake: Number(stake),
      signal_id: signal.signal_id || signal.id || signal.sig_id || null
    };
    log('placing trade payload', payload);
    try {
      const res = await fetch(TRADE_ENDPOINT, { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      const txt = await res.text();
      let json = null;
      try { json = JSON.parse(txt); }catch(e){}
      log('placeTrade response', res.status, txt);
      const tradeRecord = {
        trade_id: (json && json.trade_id) || ('local-'+Date.now()),
        signal_id: payload.signal_id,
        mode, stake: Number(stake), symbol: payload.symbol, direction: payload.direction,
        outcome: 'pending', payout: null, profit_loss: null,
        balance_before: null, balance_after: null
      };
      runtime.activeTrade = tradeRecord;
      runtime.waitingForSettlement = true;
      pushJournal(Object.assign({}, tradeRecord));
      return json || { status: res.status, text: txt };
    } catch(err){
      log('placeTrade error', err);
      return { error: err.message || String(err) };
    }
  }

  function computeRecoveryStake(payout_decimal, L, G, mode, max_cap){
    const p = (payout_decimal && payout_decimal > 0) ? payout_decimal : runtime.payout_last || runtime.settings.payout_estimate || DEFAULTS.payout_estimate;
    const target = (mode === 'recover_loss') ? L : (L + (G || runtime.settings.desired_profit || DEFAULTS.desired_profit));
    if(!p || p <= 0) {
      return Math.min((runtime.settings.base_stake || DEFAULTS.base_stake), max_cap || runtime.settings.max_stake || DEFAULTS.max_stake);
    }
    let s = Math.ceil( (target / p) * 100 ) / 100.0;
    if(max_cap) s = Math.min(s, max_cap);
    return s;
  }

  async function tryProcessQueue(){
    if(!runtime.enabled) return;
    if(runtime.waitingForSettlement) return;
    if(runtime.signalQueue.length === 0) return;
    const settings = runtime.settings;
    const sig = runtime.signalQueue.shift();
    const conf = sig.confidence ?? sig.conf ?? sig.score ?? 0;
    if(conf < (settings.min_confidence || DEFAULTS.min_confidence)){
      pushJournal({ mode: runtime.selected_mode, stake: 0, symbol: sig.symbol || sig.asset, direction: sig.direction || sig.side, outcome: 'skipped_low_conf', confidence: conf, signal_id: sig.signal_id || sig.id });
      return;
    }
    if(runtime.consecutive_losses > 0 && runtime.last_loss_confidence != null){
      if(conf <= runtime.last_loss_confidence){
        pushJournal({ mode: runtime.selected_mode, stake: 0, symbol: sig.symbol||sig.asset, direction: sig.direction||sig.side, outcome: 'skipped_confidence_gate', confidence: conf, signal_id: sig.signal_id||sig.id });
        return;
      }
    }
    let stake = Number(settings.base_stake || DEFAULTS.base_stake);
    if(runtime.consecutive_losses > 0){
      const L = runtime.loss_sum || 0;
      const G = Number(settings.desired_profit || DEFAULTS.desired_profit);
      const p = runtime.payout_last || settings.payout_estimate || DEFAULTS.payout_estimate;
      stake = computeRecoveryStake(p, L, G, settings.recovery_mode, settings.max_stake || DEFAULTS.max_stake);
    }
    if(stake > (settings.max_stake || DEFAULTS.max_stake)){
      runtime.enabled = false; lsSet('enabled', runtime.enabled);
      updateToggleUI();
      alert('Autotrade paused — computed stake ' + stake + ' exceeds max allowed ' + (settings.max_stake || DEFAULTS.max_stake));
      return;
    }
    log('placing trade for signal', sig, 'stake', stake);
    const r = await placeTrade(sig, stake, runtime.selected_mode);
    // If immediate error returned, log to last journal entry
    if(r && r.error){
      const last = runtime.journal[0];
      if(last && last.outcome === 'pending'){ last.outcome = 'error'; last.error = String(r.error); lsSet('journal', runtime.journal); renderInlineSummary(); renderModalJournal(); }
    }
  }

  /* ==== SSE attach: listens for signals & trade results/ balances ==== */
  function attachSSE(){
    if(window._autotrade_sse) return;
    if(typeof EventSource === 'undefined'){ log('EventSource missing'); return; }
    try {
      const ev = new EventSource('/events');
      window._autotrade_sse = ev;
      ev.addEventListener('message', function(e){
        try {
          const d = JSON.parse(e.data);
          // SIGNAL detection (flexible)
          if(d.event === 'signal' || d.analysis_event === 'signal' || d.type === 'signal' || d.signal){
            const sig = {
              signal_id: d.signal_id || d.signal?.id || d.id || null,
              symbol: d.symbol || d.instrument || d.asset || (d.signal && d.signal.symbol) || null,
              direction: d.direction || d.action || d.signal?.direction || (d.signal && d.signal.side),
              confidence: d.confidence ?? d.conf ?? (d.signal && d.signal.confidence) ?? 0,
              raw: d
            };
            runtime.signalQueue.push(sig);
            tryProcessQueue();
            return;
          }
          // trade result / balance messages
          if(d.event === 'trade_result' || d.analysis_event === 'trade_result' || d.msg_type === 'trade_result' || d.trade_result){
            const res = d.trade_result || d;
            const outcome = (res.outcome || res.result || res.status || '').toLowerCase();
            const payout_percent = (res.payout_percent !== undefined) ? parseFloat(res.payout_percent) : (res.payout ? ((res.payout - (res.stake||res.amount||0)) / (res.stake||res.amount||1)) : null);
            const stake = res.stake || res.amount || null;
            const mode = res.mode || res.account_mode || null;
            const balance = res.balance || null;
            if(runtime.activeTrade){
              let profit_loss = null;
              let payout = null;
              if(payout_percent !== null && stake !== null){
                payout = Number(stake) + (Number(stake) * Number(payout_percent));
                profit_loss = (payout - Number(stake));
                if(res.payout !== undefined) payout = res.payout;
              } else if(res.payout !== undefined) {
                payout = res.payout;
                profit_loss = payout - (runtime.activeTrade.stake || 0);
              }
              runtime.activeTrade.outcome = (outcome.indexOf('win')>=0 || outcome.indexOf('success')>=0) ? 'win' : ((outcome.indexOf('loss')>=0 || outcome.indexOf('lose')>=0) ? 'loss' : outcome || 'resolved');
              runtime.activeTrade.payout = payout;
              runtime.activeTrade.profit_loss = profit_loss;
              runtime.activeTrade.balance_after = balance || null;
              if(payout_percent) runtime.payout_last = payout_percent;
              // update journal record that matches signal_id or trade_id
              for(let i=0;i<runtime.journal.length;i++){
                const it = runtime.journal[i];
                if((it.signal_id && runtime.activeTrade.signal_id && it.signal_id === runtime.activeTrade.signal_id) || (it.trade_id && runtime.activeTrade.trade_id && it.trade_id === runtime.activeTrade.trade_id)){
                  it.outcome = runtime.activeTrade.outcome;
                  it.payout = runtime.activeTrade.payout;
                  it.payout_percent = payout_percent;
                  it.profit_loss = runtime.activeTrade.profit_loss;
                  it.balance_after = runtime.activeTrade.balance_after;
                  lsSet('journal', runtime.journal);
                  break;
                }
              }
              if(runtime.activeTrade.outcome === 'loss'){
                runtime.consecutive_losses = (runtime.consecutive_losses || 0) + 1;
                runtime.loss_sum = (runtime.loss_sum || 0) + (runtime.activeTrade.stake || 0);
                runtime.last_loss_confidence = runtime.activeTrade.confidence || runtime.last_loss_confidence;
              } else if(runtime.activeTrade.outcome === 'win'){
                runtime.consecutive_losses = 0;
                runtime.loss_sum = 0;
                runtime.last_loss_confidence = null;
              }
              runtime.activeTrade = null;
              runtime.waitingForSettlement = false;
              renderInlineSummary(); renderModalJournal();
              setTimeout(()=>{ tryProcessQueue(); }, 200);
              return;
            } else {
              if(balance){
                const m = mode || (res.loginid && (String(res.loginid).startsWith('C') ? 'real' : 'demo')) || 'demo';
                setCard(m, { balance: balance });
              }
            }
          }
          if(d.msg_type === 'balance' && d.balance){
            const m = d.mode || d.account || 'demo';
            setCard(m, d.balance);
          }
        } catch(err){
          console.error('autotrade SSE parse error', err);
        }
      });
      ev.addEventListener('open', ()=>log('SSE open'));
      ev.addEventListener('error', (err)=>log('SSE error', err));
      log('autotrade SSE attached');
    } catch(e){ console.error('attachSSE failed', e); }
  }

  /* ==== UI wiring & persistence ==== */
  function bindUI(){
    ensureUI();
    const settings = runtime.settings;
    // insertion of small controls
    const baseEl = $id('auto-base-stake'); if(baseEl) baseEl.value = settings.base_stake;
    const confEl = $id('auto-min-conf'); if(confEl) confEl.value = settings.min_confidence;
    const acctEl = $id('auto-account'); if(acctEl) acctEl.value = runtime.selected_mode || 'demo';

    $id('auto-toggle').addEventListener('click', ()=>{ runtime.enabled = !runtime.enabled; lsSet('enabled', runtime.enabled); updateToggleUI(); if(runtime.enabled) tryProcessQueue(); });
    if(baseEl) baseEl.addEventListener('change', ()=>{ settings.base_stake = parseFloat(baseEl.value) || DEFAULTS.base_stake; lsSet('settings', settings); });
    if(confEl) confEl.addEventListener('change', ()=>{ settings.min_confidence = parseFloat(confEl.value) || DEFAULTS.min_confidence; lsSet('settings', settings); });
    if(acctEl) acctEl.addEventListener('change', ()=>{ runtime.selected_mode = acctEl.value; lsSet('selected_mode', runtime.selected_mode); });

    // modal buttons are wired in renderModalJournal
    updateToggleUI();
    renderInlineSummary();
  }

  function updateToggleUI(){
    const btn = $id('auto-toggle');
    if(!btn) return;
    btn.textContent = runtime.enabled ? 'On' : 'Off';
    btn.style.background = runtime.enabled ? '#10b981' : '';
    btn.style.color = runtime.enabled ? '#fff' : '';
  }

  function refreshBalances(){
    const demoToken = localStorage.getItem('hero_token_demo') || ($id('hero-demo-token') && $id('hero-demo-token').value) || '';
    const realToken = localStorage.getItem('hero_token_real') || ($id('hero-real-token') && $id('hero-real-token').value) || '';
    if(demoToken) postConnectSilent(demoToken,'demo');
    if(realToken) postConnectSilent(realToken,'real');
    log('autotrade refresh triggered');
  }
  async function postConnectSilent(token, mode){
    try {
      await fetch('/control/connect_account', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({token, app_id: APP_ID, mode})});
    } catch(e){ console.error('postConnectSilent error', e); }
  }

  // external API
  function addSignal(sig){ runtime.signalQueue.push(sig); tryProcessQueue(); }

  function restoreState(){
    runtime.settings = lsGet('settings', DEFAULTS);
    runtime.enabled = lsGet('enabled', runtime.settings.auto_resume_on_load === true);
    runtime.selected_mode = lsGet('selected_mode', runtime.selected_mode);
    runtime.journal = lsGet('journal', runtime.journal);
  }

  function tryStartAutotrade(){
    const hasTokenDemo = !!localStorage.getItem('hero_token_demo');
    const hasTokenReal = !!localStorage.getItem('hero_token_real');
    if(runtime.selected_mode === 'real' && !hasTokenReal){
      if(!confirm('No Real token stored. Continue and autotrade Real?')) { runtime.enabled=false; lsSet('enabled', runtime.enabled); updateToggleUI(); return; }
    }
    runtime.enabled = true;
    lsSet('enabled', runtime.enabled);
    updateToggleUI();
    tryProcessQueue();
  }

  function init(){
    restoreState();
    attachSSE();
    bindUI();
    if(runtime.enabled) tryStartAutotrade();
    log('compact autotrade panel ready');
  }

  window.HeroAutoTrade = { init, addSignal, runtime, computeRecoveryStake, setCard };
  document.addEventListener('DOMContentLoaded', init);
})();
