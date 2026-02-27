(function(){
  const TRADE_ENDPOINT = '/control/place_trade';
  const APP_ID = 71710;
  const MAX_JOURNAL = 1000;
  const SELF_CHECK_TIMEOUT_MS = 10000; // 10s
  const DEFAULTS = {
    stake: 1.00,            // single stake field (used whether user "wants to trade" or not)
    desired_profit: 0.02,   // informational only
    payout_estimate: 0.8,
    auto_resume_on_load: true
  };

  const RECOVERY_FACTORS = {
    none: 1,
    loss_only: 10,
    recover_with_profit: 11
  };

  const MAX_SAFE_STAKE = 10000.00; // safety cap to avoid runaway stakes

  const log = (...a) => { try { console.log('AUTOTRADE:', ...a); } catch(e){} };
  const $id = id => document.getElementById(id);
  const q = s => Array.from(document.querySelectorAll(s||'*'));

  function lsGet(k, fallback){ try{ const v = localStorage.getItem('hero_autotrade_'+k); return v ? JSON.parse(v) : fallback; }catch(e){ return fallback; } }
  function lsSet(k, v){ try{ localStorage.setItem('hero_autotrade_'+k, JSON.stringify(v)); }catch(e){} }

  let runtime = {
    enabled: false,
    selected_mode: 'demo',
    waitingForSettlement: false,
    journal: lsGet('journal', []),
    settings: lsGet('settings', Object.assign({}, DEFAULTS, { recovery_mode: 'none', base_stake: DEFAULTS.stake, current_stake: DEFAULTS.stake, recovery_step: 0 }))
  };

  /* Money helper: multiply money safely and return Number rounded to 2 decimals */
  function moneyMul(amount, factor){
    const n = parseFloat(amount || 0);
    const f = parseFloat(factor || 1);
    if (!isFinite(n) || !isFinite(f)) return NaN;
    // Compute and round to 2 decimals
    const res = Number((n * f).toFixed(2));
    return res;
  }

  /* ----- UI: single-line panel ----- */
  function ensureAutotradeUI(){
    let panel = $id('hero-autotrade-panel');
    if(panel) return panel;

    const container = document.getElementById('autotrade-container') || $id('hero-two-mode') || document.body;

    panel = document.createElement('div');
    panel.id = 'hero-autotrade-panel';
    panel.style.padding = '12px';
    panel.style.borderRadius = '10px';
    panel.style.background = '#ffffff';
    panel.style.boxShadow = '0 8px 24px rgba(11,22,40,0.06)';
    panel.style.display = 'flex';
    panel.style.alignItems = 'center';
    panel.style.gap = '12px';
    panel.style.flexWrap = 'wrap';
    panel.style.maxWidth = '100%';
    panel.style.fontFamily = 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial';
    panel.style.fontSize = '13px';
    panel.style.color = '#0f172a';
    panel.style.border = '1px solid rgba(15,23,42,0.04)';

    panel.innerHTML = `
      <div style="display:flex;align-items:center;gap:10px; flex: 0 0 auto;">
        <label style="font-weight:700; font-size:14px; margin-right:6px;">Autotrade</label>
        <label style="display:flex;align-items:center;gap:8px;">
          <input id="auto-checkbox" type="checkbox" />
          <span style="font-size:13px">Enable</span>
        </label>
      </div>

      <div style="display:flex;align-items:center;gap:8px; flex: 0 0 auto;">
        <span style="font-size:13px; margin-right:6px;">Account</span>
        <select id="auto-account" style="padding:6px;border-radius:8px;border:1px solid rgba(15,23,42,0.06);background:#fbfdff;">
          <option value="demo">Demo</option>
          <option value="real">Real</option>
        </select>
        <div id="auto-token-status" style="font-size:12px;margin-left:8px;color:#6b7280">token: —</div>
      </div>

      <div style="display:flex;align-items:center;gap:8px;">
        <label style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:13px">Stake</span>
          <input id="auto-stake" type="number" step="0.01" min="0.35" style="width:110px;padding:6px;border-radius:8px;border:1px solid rgba(15,23,42,0.06);" />
        </label>
        <div id="stake-note" style="font-size:12px;color:#ef4444;display:none;margin-left:8px;align-items:center;">
          <span id="stake-note-text">You can't trade with less than $0.35</span>
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:8px;margin-left:auto;">
        <div style="display:flex;flex-direction:column;align-items:flex-end;">
          <div style="font-size:12px;color:#64748b">Available balance</div>
          <div id="auto-balance-display" style="font-weight:700">—</div>
        </div>
      </div>

      <div style="width:1px;height:48px;background:linear-gradient(180deg, rgba(15,23,42,0.04), rgba(15,23,42,0.02)); margin: 0 8px;"></div>

      <div style="display:flex;align-items:center;gap:8px;">
        <label style="font-size:13px; display:flex;align-items:center;gap:8px;">
          Recovery
          <select id="auto-recovery-mode" style="padding:6px;border-radius:8px;border:1px solid rgba(15,23,42,0.06);background:#fbfdff;">
            <option value="none">Don't recover</option>
            <option value="loss_only">Recover (loss only — ×10)</option>
            <option value="recover_with_profit">Recover (with profit — ×11)</option>
          </select>
        </label>
        <div id="recovery-controls" style="display:flex;align-items:center;gap:8px;">
          <div style="font-size:12px;color:#64748b">Step</div>
          <div id="recovery-step" style="font-weight:700">0</div>
          <button id="recovery-reset" title="Reset recovery sequence" style="padding:6px 8px;border-radius:8px;border:1px solid rgba(15,23,42,0.06);background:#fff;cursor:pointer">Reset</button>
        </div>
      </div>

      <div style="width:1px;height:48px;background:linear-gradient(180deg, rgba(15,23,42,0.04), rgba(15,23,42,0.02)); margin: 0 8px;"></div>

      <div style="display:flex;align-items:center;gap:8px;">
        <button id="auto-refresh" title="Refresh balances" style="padding:8px 10px;border-radius:8px;border:1px solid rgba(15,23,42,0.06);background:#f8fafc;cursor:pointer">Refresh</button>
        <button id="auto-export" title="Export trade journal" style="padding:8px 10px;border-radius:8px;border:1px solid rgba(15,23,42,0.06);background:#f8fafc;cursor:pointer">Export</button>
        <button id="auto-clear-journal" title="Clear Journal" style="padding:8px 10px;border-radius:8px;border:1px solid rgba(15,23,42,0.06);background:#fff5f5;color:#991b1b;cursor:pointer">Clear</button>
      </div>
    `;

    container.appendChild(panel);
  }

  /* ----- journal helpers (lightweight) ----- */
  function pushJournal(entry){
    try {
      entry.id = entry.id || ('t' + Date.now());
      entry.ts = entry.ts || Date.now();
      runtime.journal.unshift(entry);
      if(runtime.journal.length > MAX_JOURNAL) runtime.journal.length = MAX_JOURNAL;
      lsSet('journal', runtime.journal);
    } catch(e){
      console.warn('pushJournal error', e);
    }
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

  function clearJournal(){
    if(!confirm('Clear autotrade journal?')) return;
    runtime.journal = [];
    lsSet('journal', runtime.journal);
  }

  /* ----- debug helpers ----- */
  function appendRawDebugLine(text){
    try{
      const selectors = ['#rawDebug', '#hero-raw-debug', '#analysisLog', '#analysis-panel .panel-body', '.analysis-log', '#hero-analysis-panel'];
      let el = null;
      for(const s of selectors){ el = document.querySelector(s); if(el) break; }
      if(!el){
        console.log('RAWDBG:', text);
        return;
      }
      const line = document.createElement('div');
      line.style.fontSize = '12px';
      line.style.padding = '4px 0';
      line.style.borderBottom = '1px dashed rgba(0,0,0,0.04)';
      line.textContent = (new Date()).toLocaleTimeString() + ' ' + text;
      if(el.prepend) el.prepend(line); else el.insertBefore(line, el.firstChild);
    }catch(e){ console.log('appendRawDebugLine error', e); }
  }

  /* ----- balance helpers ----- */
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
  function fmtAmount(a){
    if(a === null || typeof a === 'undefined') return '—';
    if(typeof a === 'number') return a.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
    if(!isNaN(Number(a))) return Number(a).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
    return String(a);
  }
  function setCard(mode, raw){
    try {
      const n = normalize(raw);
      const amountEl = document.getElementById(mode==='real' ? 'real-amount' : 'demo-amount')
                     || document.getElementById(mode==='real' ? 'realBalanceHeader' : 'demoBalanceHeader')
                     || document.querySelector('#auto-balance-display')
                     || null;
      const metaEl = document.getElementById(mode==='real' ? 'real-meta' : 'demo-meta') || null;
      const statusEl = document.getElementById(mode==='real' ? 'hero-real-status-small' : 'hero-demo-status-small')
                    || document.getElementById(mode==='real' ? 'real-status' : 'demo-status') || null;

      if(n && n.amount !== undefined && amountEl){
        amountEl.textContent = fmtAmount(n.amount);
      } else if(amountEl){
        try { amountEl.textContent = (raw && raw.amount) ? fmtAmount(raw.amount) : (raw && raw.balance ? JSON.stringify(raw).slice(0,40) : '—'); } catch(e){ amountEl.textContent = '—'; }
      }
      if(n && metaEl){
        metaEl.textContent = (n.currency? n.currency : '') + (n.loginid ? ' • ' + n.loginid : '');
      }
      if(statusEl){
        if(n && n.amount !== undefined) statusEl.textContent = (mode==='real' ? 'Real authenticated' : 'Demo authenticated');
        else statusEl.textContent = 'Connected';
      }

      try {
        const panelDisp = document.getElementById('auto-balance-display');
        if(panelDisp){
          const sel = (document.getElementById('auto-account') || {}).value || runtime.selected_mode;
          if(sel === mode){
            panelDisp.textContent = (n && n.amount !== undefined) ? fmtAmount(n.amount) : '—';
          }
        }
      } catch(e){}
    } catch(e){ log('setCard err', e); }
  }
  function getAvailableBalance(mode){
    try {
      const raw = localStorage.getItem('hero_balance_' + mode);
      if(!raw) return null;
      const obj = JSON.parse(raw);
      const n = normalize(obj);
      return n && n.amount !== undefined ? Number(n.amount) : null;
    } catch(e){ return null; }
  }

  /* ----- token presence UI ----- */
  function updateTokenStatus(){
    try {
      const demo = !!localStorage.getItem('hero_token_demo');
      const real = !!localStorage.getItem('hero_token_real');
      const sel = (document.getElementById('auto-account') || {}).value || runtime.selected_mode;
      const el = document.getElementById('auto-token-status');
      if(!el) return;
      if(sel === 'real'){
        el.textContent = real ? 'token: yes' : 'token: no';
        el.style.color = real ? '#10b981' : '#ef4444';
      } else {
        el.textContent = demo ? 'token: yes' : 'token: no';
        el.style.color = demo ? '#10b981' : '#ef4444';
      }
    } catch(e){}
  }

  /* ----- queue processing (NO-OP in UI) ----- */
  async function tryProcessQueue(){
    // Autotrade UI does NOT place trades — the differ_trade_check daemon is solely responsible for trade placement.
    log('tryProcessQueue: NO-OP in UI. Placement is handled by differ_trade_check daemon.');
    return;
  }

  /* ----- SSE attach (prediction_posted => UI-only) ----- */
  function attachAutotradeSSE(){
    if(window._autotrade_sse) return;
    if(typeof EventSource === 'undefined'){ log('EventSource missing - autotrade SSE disabled'); return; }
    try {
      const ev = new EventSource('/events');
      window._autotrade_sse = ev;

      function processPayload(payload){
        try {
          const aevt = payload.analysis_event || payload.event || null;

          // Prediction posted / toast -> UI-only
          if(aevt === 'prediction_posted' || aevt === 'prediction_toast'){
            const sig = {
              prediction_id: payload.prediction_id || payload.signal_id || payload.id || ('pred-'+Date.now()),
              symbol: payload.symbol || payload.market || payload.instrument || null,
              confidence: payload.confidence ?? payload.conf ?? payload.score ?? 0,
              prediction_digit: (payload.prediction_digit !== undefined) ? payload.prediction_digit : payload.predicted_digit,
              stake: payload.stake || payload.amount || null,
              timestamp: payload.ts || payload.epoch || Date.now(),
              raw: payload
            };
            appendRawDebugLine('[prediction_posted] (UI-only) received ' + (sig.prediction_id || '') + ' ' + (sig.symbol||''));
            log('autotrade UI: received prediction (ignored for placement) — passively visible only', sig);
          }

          // Final result / settled prediction - record in journal (these are broadcast by hero_service when settled)
          if(aevt === 'prediction_result' || aevt === 'prediction_settled' || aevt === 'contract_settled'){
            try {
              const pid = payload.prediction_id || payload.pred_id || payload.id || ('r'+Date.now());
              const market = (payload.symbol || payload.market || '').toUpperCase();
              const pred_digit = (payload.prediction_digit !== undefined) ? payload.prediction_digit : payload.predicted;
              const actual = payload.actual ?? payload.result_digit ?? (payload.final_contract && payload.final_contract.result_digit) ?? null;
              const profit = payload.profit ?? (payload.final_contract && payload.final_contract.profit) ?? null;
              const pct = payload.profit_percentage ?? (payload.final_contract && payload.final_contract.profit_percentage) ?? null;
              const result = (payload.result || payload.status || payload.outcome) ? String(payload.result || payload.status || payload.outcome).toUpperCase() : (profit !== null ? (Number(profit) > 0 ? 'WIN' : 'LOSS') : 'SETTLED');

              const entry = {
                timestamp: payload.epoch || payload.ts || Date.now(),
                prediction_id: pid,
                symbol: market,
                prediction_digit: pred_digit,
                actual: actual,
                result: result,
                profit: (profit !== null ? Number(profit) : null),
                profit_percentage: (pct !== null ? Number(pct) : null),
                state: result,
                __raw: payload
              };
              pushJournal(entry);
              appendRawDebugLine('[prediction_result] recorded ' + pid + ' ' + market + ' -> ' + entry.result);

              // recovery logic: adjust current stake on loss, reset on win
              try {
                handleRecoveryOnResult(entry);
              } catch(e) {
                console.warn('handleRecoveryOnResult error', e);
              }

              // update balance if provided
              if(payload.balance){
                const m = payload.mode || runtime.selected_mode || 'demo';
                setCard(m, payload.balance);
                updateTokenStatus();
              }
            } catch(e){
              console.warn('prediction_result processing error', e);
            }
          }

          // handle balance messages
          if(payload && payload.msg_type === 'balance' && payload.balance){
            const m = payload.mode || payload.account || 'demo';
            setCard(m, payload.balance);
            updateTokenStatus();
          }
        } catch(e){
          console.error('processPayload error', e);
        }
      }

      function handleSSEEvent(e){
        try {
          const parsed = e.data ? JSON.parse(e.data) : {};
          const d = parsed || {};
          const payload = d.payload || d;
          processPayload(payload);
        } catch(err){ console.error('autotrade SSE parse error', err); }
      }

      ev.addEventListener('message', handleSSEEvent);
      ev.addEventListener('analysis', handleSSEEvent);
      ev.addEventListener('prediction_posted', handleSSEEvent);
      ev.addEventListener('prediction_result', handleSSEEvent);
      ev.addEventListener('open', ()=>log('autotrade SSE open'));
      ev.addEventListener('error', (err)=>log('autotrade SSE error', err));
      log('autotrade SSE attached');
    } catch(e){
      console.error('attachAutotradeSSE failed', e);
    }
  }

  /* ----- Recovery handling ----- */
  function getRecoveryMode(){ return runtime.settings.recovery_mode || 'none'; }
  function getBaseStake(){ return parseFloat(runtime.settings.base_stake || runtime.settings.stake || DEFAULTS.stake); }
  function getCurrentStake(){ return parseFloat(runtime.settings.current_stake || getBaseStake()); }
  function setCurrentStake(val, persist=true){
    val = Number(Number(val).toFixed(2));
    if (!isFinite(val)) return;
    if (val > MAX_SAFE_STAKE) {
      showToast(`Refusing to set stake > ${MAX_SAFE_STAKE} (safety)`, 'warn', 3000);
      return;
    }
    runtime.settings.current_stake = val;
    lsSet('settings', runtime.settings);
    // push to server so analysis/trader daemons see it
    try {
      fetch('/control/set_autotrade_settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: runtime.selected_mode || 'demo', stake: Number(val) })
      }).catch(e => console.warn('set_autotrade_settings POST failed', e));
    } catch(e){
      console.warn('set_autotrade_settings error', e);
    }
    // update UI input to reflect actual current stake
    try {
      const sEl = $id('auto-stake');
      if(sEl) sEl.value = Number(val).toFixed(2);
    } catch(e){}
  }
  function resetRecoverySequence(){
    runtime.settings.recovery_step = 0;
    runtime.settings.current_stake = getBaseStake();
    lsSet('settings', runtime.settings);
    setCurrentStake(runtime.settings.current_stake);
    updateRecoveryUI();
  }
  function incrementRecoveryOnLoss(){
    runtime.settings.recovery_step = (runtime.settings.recovery_step || 0) + 1;
    const factor = RECOVERY_FACTORS[getRecoveryMode()] || 1;
    const cur = getCurrentStake();
    const next = moneyMul(cur, factor);
    runtime.settings.current_stake = next;
    lsSet('settings', runtime.settings);
    setCurrentStake(next);
    updateRecoveryUI();
  }
  function updateRecoveryUI(){
    const stepEl = $id('recovery-step');
    if(stepEl) stepEl.textContent = String(runtime.settings.recovery_step || 0);
    // ensure stake input matches current_stake
    try {
      const sEl = $id('auto-stake');
      if(sEl) sEl.value = (Number(runtime.settings.current_stake || runtime.settings.stake || 0)).toFixed(2);
    } catch(e){}
  }

  function handleRecoveryOnResult(entry){
    // Only act on LOSS / WIN. Entry.result expected to be 'WIN'|'LOSS'
    const res = (entry.result || '').toString().toUpperCase();
    if(res === 'LOSS'){
      if(getRecoveryMode() === 'none'){
        // do nothing
      } else {
        incrementRecoveryOnLoss();
        appendRawDebugLine(`[recovery] LOSS detected → step=${runtime.settings.recovery_step} stake=${getCurrentStake()}`);
      }
    } else if(res === 'WIN'){
      // on win we reset to base stake & recovery step 0
      resetRecoverySequence();
      appendRawDebugLine('[recovery] WIN detected → reset recovery sequence');
    } else {
      // Ignore other statuses
    }
  }

  /* ----- Self-check (keeps existing behavior intact) ----- */
  async function performSelfCheck(timeout_ms = SELF_CHECK_TIMEOUT_MS){
    const start = Date.now();
    const details = [];
    let ok = true;
    function note(msg, level='info'){
      details.push(msg);
      appendRawDebugLine('[autotrade-check] ' + msg);
      if(level === 'warn') ok = false;
    }

    try {
      note('performing backend reachability checks');

      // balances endpoint
      try {
        const r = await Promise.race([fetch('/control/get_balances?mode=' + encodeURIComponent(runtime.selected_mode)), new Promise((_,rj)=>setTimeout(()=>rj(new Error('timeout')),5000))]);
        if(r && r.ok) note('backend: get_balances reachable ('+runtime.selected_mode+')');
        else note('backend: get_balances returned status ' + (r? r.status : '(no response)'), 'warn');
      } catch(e){ note('backend get_balances not reachable: ' + (e && e.message ? e.message : String(e)), 'warn'); }

      // trade endpoint probe - non-destructive (only tests reachability)
      try {
        const probePayload = { test_probe: true, token: (localStorage.getItem(runtime.selected_mode==='real'?'hero_token_real':'hero_token_demo')||''), app_id: APP_ID, mode: runtime.selected_mode };
        let okTradeProbe = false;
        try {
          const resp = await fetch(TRADE_ENDPOINT, { method:'POST', headers:{'Content-Type':'application/json','X-Hero-AutoTest':'1'}, body: JSON.stringify(probePayload) });
          if(resp) { okTradeProbe = true; note('trade endpoint: probe request sent, status ' + resp.status); }
        } catch(e){ note('trade endpoint probe failed: ' + (e && e.message ? e.message : String(e)), 'warn'); }
        if(!okTradeProbe) note('trade endpoint may be unreachable', 'warn');
      } catch(e){ note('trade endpoint probe error: ' + (e && e.message ? e.message : String(e)), 'warn'); }

      // token presence
      const tok = localStorage.getItem(runtime.selected_mode==='real' ? 'hero_token_real' : 'hero_token_demo') || '';
      if(!tok) note('connect: no token stored for ' + runtime.selected_mode);
      else note('connect: token present for ' + runtime.selected_mode);

    } catch(e){
      note('self-check unexpected error: ' + (e && e.message ? e.message : String(e)), 'warn');
    }

    const elapsed = Date.now() - start;
    return { ok, details, elapsed_ms: elapsed };
  }

  /* ----- UI wiring and persistence ----- */
  function bindAutotradeUI(){
    ensureAutotradeUI();
    const settings = runtime.settings;

    // populate stake input from settings (current_stake displayed)
    const currentStake = Number(settings.current_stake || settings.stake || DEFAULTS.stake);
    $id('auto-stake').value = Number(currentStake).toFixed(2);

    // populate recovery select
    $id('auto-recovery-mode').value = settings.recovery_mode || 'none';
    $id('recovery-step').textContent = String(settings.recovery_step || 0);

    $id('auto-account').value = runtime.selected_mode || 'demo';

    // handlers
    $id('auto-account').addEventListener('change', async ()=>{
      runtime.selected_mode = $id('auto-account').value;
      lsSet('selected_mode', runtime.selected_mode);
      updateBalanceDisplay();
      updateTokenStatus();
      // push current autotrade stake to server (mode + stake)
      try {
        await fetch('/control/set_autotrade_settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode: runtime.selected_mode, stake: Number(runtime.settings.current_stake || runtime.settings.stake) })
        });
      } catch(e){
        console.warn('set_autotrade_settings POST failed', e);
      }
    });

    $id('auto-refresh').addEventListener('click', ()=>{ refreshBalances(); updateBalanceDisplay(); updateTokenStatus(); });
    $id('auto-export').addEventListener('click', ()=>{ exportJournalCSV(); });
    $id('auto-clear-journal').addEventListener('click', ()=>{ if(confirm('Clear autotrade journal?')){ runtime.journal = []; lsSet('journal', runtime.journal); }});

    $id('auto-stake').addEventListener('change', async ()=>{
      let v = parseFloat($id('auto-stake').value) || DEFAULTS.stake;
      if(v < 0.35){
        showStakeNote(true);
        v = Math.max(0.35, v);
      } else showStakeNote(false);
      // store base stake and reset current stake (changing base stake resets recovery)
      runtime.settings.base_stake = Math.round(v * 100)/100;
      runtime.settings.stake = runtime.settings.base_stake;
      runtime.settings.current_stake = runtime.settings.base_stake;
      runtime.settings.recovery_step = 0;
      $id('auto-stake').value = Number(runtime.settings.stake).toFixed(2);
      lsSet('settings', runtime.settings);

      // push stake to server so differ_trade_check daemon can use it when processing predictions
      try {
        await fetch('/control/set_autotrade_settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode: runtime.selected_mode || 'demo', stake: Number(runtime.settings.current_stake) })
        });
      } catch(e){
        console.warn('set_autotrade_settings POST failed', e);
      }

      updateRecoveryUI();
    });

    // recovery select/listener
    $id('auto-recovery-mode').addEventListener('change', ()=>{
      const mode = $id('auto-recovery-mode').value || 'none';
      runtime.settings.recovery_mode = mode;
      // reset sequence when changing mode to avoid accidental huge stakes
      runtime.settings.recovery_step = 0;
      runtime.settings.current_stake = runtime.settings.base_stake || runtime.settings.stake || DEFAULTS.stake;
      lsSet('settings', runtime.settings);
      setCurrentStake(runtime.settings.current_stake);
      updateRecoveryUI();
    });

    $id('recovery-reset').addEventListener('click', ()=>{
      if(!confirm('Reset recovery sequence and restore base stake?')) return;
      resetRecoverySequence();
    });

    // checkbox behavior: run self-check before enabling
    const checkbox = $id('auto-checkbox');
    checkbox.checked = !!(runtime.enabled);
    checkbox.addEventListener('change', async ()=>{
      const checked = checkbox.checked;
      if(checked){
        showToast('Autotrade checking...', 'info', 3000);
        try {
          appendRawDebugLine('[autotrade] user enabled autotrade — running self-check');
          const result = await Promise.race([
            performSelfCheck(),
            new Promise((_,rej)=>setTimeout(()=>rej(new Error('self-check timeout')), SELF_CHECK_TIMEOUT_MS))
          ]).catch(err => ({ ok:false, error: err && err.message ? err.message : String(err) }));

          if(result && result.ok){
            showToast('Autotrade ON — ready', 'ok', 2000);
            appendRawDebugLine('[autotrade] self-check OK');
            runtime.enabled = true;
            lsSet('enabled', runtime.enabled);
            updateToggleUI();
          } else {
            showToast('Autotrade ON — check warnings (see debug)', 'warn', 3000);
            appendRawDebugLine('[autotrade] self-check reported warnings: ' + (result && result.details ? JSON.stringify(result.details) : (result && result.error ? result.error : 'unknown')));
            runtime.enabled = true;
            lsSet('enabled', runtime.enabled);
            updateToggleUI();
          }
        } catch(err){
          showToast('Autotrade check failed', 'warn', 3000);
          appendRawDebugLine('[autotrade] self-check thrown error: ' + (err && err.message ? err.message : String(err)));
          checkbox.checked = false;
        }
      } else {
        runtime.enabled = false;
        lsSet('enabled', runtime.enabled);
        updateToggleUI();
        appendRawDebugLine('[autotrade] disabled by user');
      }
    });

    // update balance display periodically / on load
    updateBalanceDisplay();
    updateTokenStatus();

    // hook possible Start Analysis buttons on the page to show immediate toast + notify server
    hookStartAnalysisButtons();

    // keep token status in sync across tabs
    window.addEventListener('storage', (e)=>{
      if(e.key && (e.key.startsWith('hero_token_') || e.key.startsWith('hero_balance_'))){
        updateTokenStatus();
        updateBalanceDisplay();
      }
    });

    // push initial autotrade settings to server on bind so daemon sees the value
    try {
      fetch('/control/set_autotrade_settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: runtime.selected_mode || 'demo', stake: Number(runtime.settings.current_stake || runtime.settings.stake || DEFAULTS.stake) })
      }).catch(e => { console.warn('initial set_autotrade_settings POST failed', e); });
    } catch(e){
      console.warn('initial set_autotrade_settings error', e);
    }
  }

  function updateToggleUI(){
    const cb = $id('auto-checkbox');
    if(cb) cb.checked = !!runtime.enabled;
  }

  function refreshBalances(){
    try {
      fetch('/control/get_balances?mode=demo').then(()=>{}).catch(()=>{});
      fetch('/control/get_balances?mode=real').then(()=>{}).catch(()=>{});
    } catch(e){}
    log('autotrade refresh triggered');
    setTimeout(updateBalanceDisplay, 900);
  }

  function updateBalanceDisplay(){
    try {
      const sel = (document.getElementById('auto-account') || {}).value || runtime.selected_mode || 'demo';
      const disp = document.getElementById('auto-balance-display');
      if(!disp) return;
      const avail = getAvailableBalance(sel);
      disp.textContent = avail !== null ? fmtAmount(avail) : '—';
      try { setCard(sel, JSON.parse(localStorage.getItem('hero_balance_' + sel) || 'null')); } catch(e){}
      updateTokenStatus();
    } catch(e){}
  }

  // addSignal is intentionally a NO-OP for placement - UI should not enqueue/trigger trades
  function addSignal(sig){
    log('addSignal called but UI will not place trades; dropping signal', sig && (sig.signal_id || sig.id || sig.prediction_id));
    // if you want to display incoming signals in the journal or debug area, uncomment:
    // pushJournal({ mode: runtime.selected_mode, symbol: sig.symbol, prediction_digit: sig.prediction_digit || sig.pred, outcome: 'PRODUCED', stake: sig.stake || null, signal_id: sig.signal_id || sig.id, ts: Date.now() });
  }

  function restoreState(){
    runtime.settings = lsGet('settings', Object.assign({}, DEFAULTS, { recovery_mode: 'none', base_stake: DEFAULTS.stake, current_stake: DEFAULTS.stake, recovery_step: 0 }));
    runtime.enabled = lsGet('enabled', false);
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
    attachAutotradeSSE();
    bindAutotradeUI();
    if(runtime.enabled) {
      tryStartAutotrade();
    }
    log('autotrade_panel with recovery ready');
  }

  // ---- helper: inline stake-note control ----
  function showStakeNote(show){
    const note = $id('stake-note');
    if(!note) return;
    note.style.display = show ? 'flex' : 'none';
  }

  // ---- Start Analysis UI helper: show immediate toast + POST to server ----
  async function notifyServerStartAnalysis(){
    try {
      const autotradeEnabled = !!(document.getElementById('auto-checkbox') && document.getElementById('auto-checkbox').checked);
      const account = (document.getElementById('auto-account') || {}).value || (runtime && runtime.selected_mode) || 'demo';
      if (autotradeEnabled) {
        showToast(`Autotrade ON — will trade using ${account} account`, 'ok', 3000);
      } else {
        showToast('Autotrade OFF — predictions will NOT use account funds', 'info', 3000);
      }
      // Intentionally no POST /control/start_analysis here.
      // Start/stop analysis is controlled by explicit page buttons.
    } catch(e){ console.error('notifyServerStartAnalysis outer error', e); }
  }

  function hookStartAnalysisButtons(){
    try {
      const selectors = ['#start-analysis','button[data-action="start-analysis"]','button.start-analysis','[data-start-analysis]'];
      let elems = Array.from(document.querySelectorAll(selectors.join(',')));
      if(elems.length === 0){
        elems = Array.from(document.querySelectorAll('button,input[type=button],a')).filter(el=>{
          const txt = ((el.textContent || el.value || '') + '').trim().toLowerCase();
          const title = (el.getAttribute('title') || '').toLowerCase();
          return txt.includes('start analysis') || txt.includes('start differs analysis') || title.includes('start analysis') || title.includes('start differs analysis') || el.dataset.action === 'start-analysis';
        });
      }
      elems.forEach(el=>{
        if(el.__auto_start_hooked) return;
        el.addEventListener('click', ()=>{
          try { notifyServerStartAnalysis(); }catch(e){ console.error('start hook notify error', e); }
        }, { passive:true });
        el.__auto_start_hooked = true;
      });
      if(elems.length === 0) console.debug('hookStartAnalysisButtons: no Start Analysis button found');
    } catch(e){ console.error('hookStartAnalysisButtons error', e); }
  }

  // minimal toast helper
  function showToast(msg, level='info', duration=2000){
    try{
      const id = 'hero-autotrade-toast';
      let el = document.getElementById(id);
      if(!el){
        el = document.createElement('div');
        el.id = id;
        el.style.position = 'fixed';
        el.style.left = '50%';
        el.style.transform = 'translateX(-50%)';
        el.style.top = '12px';
        el.style.zIndex = 300000;
        el.style.padding = '10px 16px';
        el.style.borderRadius = '10px';
        el.style.boxShadow = '0 10px 36px rgba(2,6,23,0.12)';
        el.style.fontSize = '13px';
        el.style.fontWeight = '600';
        el.style.color = '#fff';
        el.style.maxWidth = '80%';
        el.style.textAlign = 'center';
        document.body.appendChild(el);
      }
      el.style.background = (level==='ok' ? '#10b981' : (level==='warn' ? '#f59e0b' : '#2563eb'));
      el.textContent = msg;
      el.style.opacity = '1';
      el.style.display = '';
      setTimeout(()=>{ try{ el.style.transition='opacity .45s'; el.style.opacity='0'; setTimeout(()=>{ try{ el.remove(); }catch(e){} },420); }catch(e){} }, duration);
    }catch(e){ console.log('toast err', e); }
  }

  // expose API & init on DOMContentLoaded
  window.HeroAutoTrade = { init, addSignal, runtime };

  // ---------- Recovery system ----------
/*
 runtime.recovery = {
   enabled: true|false,
   mode: 'none'|'loss_only'|'profit',   // 'profit' uses multiplier 11, 'loss_only' uses 10
   base_stake: Number,  // canonical user base stake in dollars (never cents)
   level: 0             // 0 = no recovery (use base), 1 = 1st recovery step, etc.
 };
*/
runtime.recovery = runtime.recovery || { enabled: false, mode: 'none', base_stake: Number(runtime.settings.stake || DEFAULTS.stake), level: 0 };

// guard: ensure base_stake looks sane (auto-correct cents->dollars if necessary)
function sanitizeBaseStake(stakeCandidate){
  let n = Number(stakeCandidate);
  if (!isFinite(n)) return DEFAULTS.stake;
  // If user expects <1 (like 0.35) but value is >>1 and looks 100x the base, divide by 100
  // Heuristic: if candidate >= 10 and candidate/DEFAULTS.stake between ~80 and ~120 => cents->dollars bug
  if (DEFAULTS.stake && DEFAULTS.stake < 1 && n >= 10) {
    const ratio = n / DEFAULTS.stake;
    if (ratio > 80 && ratio < 120) {
      console.warn('[autotrade] sanitizeBaseStake: detected cents->dollars; auto-correcting by /100', n);
      n = Number((n / 100).toFixed(2));
    }
  }
  return Number(n.toFixed(2));
}

runtime.recovery.base_stake = sanitizeBaseStake(runtime.recovery.base_stake);

// safe rounding to 2 decimals
function moneyRound(v){ return Number(Number(v).toFixed(2)); }

// compute stake from base and recovery level (deterministic)
function computeStakeForLevel(level, mode){
  const base = Number(runtime.recovery.base_stake || runtime.settings.stake || DEFAULTS.stake);
  const multiplier = (mode === 'profit') ? 11 : 10;
  if (!level || level <= 0) return moneyRound(base);
  // pow can grow large; cap level so it doesn't explode
  const MAX_LEVEL = 8; // configurable
  const capped = Math.min(level, MAX_LEVEL);
  const val = base * Math.pow(multiplier, capped);
  // If val is astronomically large, cap it
  const MAX_SAFE = 10000; // max stake allowed (tweak to your risk)
  return moneyRound(Math.min(val, MAX_SAFE));
}

// externally callable helper to get the current stake to use (always dollars)
function currentRecoveryStake(){
  try {
    const stake = computeStakeForLevel(runtime.recovery.level, runtime.recovery.mode);
    return stake;
  } catch(e){
    console.error('currentRecoveryStake error', e);
    return Number(DEFAULTS.stake);
  }
}

// call when a trade loses
function onTradeLoss(){
  // only act if recovery enabled
  if (!runtime.recovery.enabled || runtime.recovery.mode === 'none') {
    // keep level at 0; no change
    runtime.recovery.level = 0;
    return currentRecoveryStake();
  }
  runtime.recovery.level = (runtime.recovery.level || 0) + 1;
  const next = currentRecoveryStake();
  console.log('[autotrade] onTradeLoss -> recovery level', runtime.recovery.level, 'next stake', next);
  // persist optionally
  lsSet('recovery', runtime.recovery);
  return next;
}

// call when a trade wins (reset recovery)
function onTradeWin(){
  runtime.recovery.level = 0;
  lsSet('recovery', runtime.recovery);
  console.log('[autotrade] onTradeWin -> reset recovery');
  return currentRecoveryStake();
}

// helper to set user base stake safely (sanitizes cents->dollars problems)
function setRecoveryBaseStake(v){
  const n = sanitizeBaseStake(v);
  runtime.recovery.base_stake = n;
  runtime.settings.stake = n; // keep settings in sync
  lsSet('settings', runtime.settings);
  lsSet('recovery', runtime.recovery);
  // ensure server knows numeric stake
  fetch('/control/set_autotrade_settings', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ mode: runtime.selected_mode || 'demo', stake: Number(n) })
  }).catch(e => console.warn('set_autotrade_settings POST failed', e));
  return n;
}

// Example usage: when a settled SSE arrives, call these depending on result:
// if LOSS -> const stake = onTradeLoss(); place trade with stake
// if WIN  -> onTradeWin(); place trade with currentRecoveryStake() for next trade if desired

// persist default at startup if missing
if(!lsGet('recovery', null)) lsSet('recovery', runtime.recovery);


  document.addEventListener('DOMContentLoaded', init);
})();
