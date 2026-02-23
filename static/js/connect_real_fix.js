/* Safe client-side hotfix: ensure Real connect sends correct payload and updates UI on response/SSE */
(function(){
  function $id(id){ return document.getElementById(id); }
  function log(){ try{ console.log.apply(console, arguments); }catch(e){} }

  // send payload to backend as JSON
  async function sendConnectPayload(payload){
    log('connect_real_fix: sending payload', payload);
    try {
      const resp = await fetch('/control/connect_account', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const text = await resp.text();
      log('connect_real_fix: response', resp.status, text);
      // attempt to parse JSON if server returned JSON
      try { const json = JSON.parse(text); handleConnectResponse(json); } catch(e){ log('connect_real_fix: non-json response'); }
    } catch (err) {
      console.error('connect_real_fix: fetch error', err);
    }
  }

  // Update UI when connect response or SSE balance arrives
  function handleConnectResponse(json){
    try {
      if (!json) return;
      // if server returns balance or ok, update UI
      if (json.balance || (json.ok && json.balance)) {
        const b = json.balance && json.balance.balance ? json.balance.balance : (json.balance || null);
        if (b) updateBalanceDOM(b);
      }
      // if server returns ok:true or account_connected events handled by SSE, we also update mode text
      const modeTextEl = $id('modeText');
      if (modeTextEl) {
        modeTextEl.textContent = 'Real';
      }
    } catch(e) { console.error('connect_real_fix: handleConnectResponse error', e); }
  }

  // Simple function to display balance info (safe: appends if element missing)
  function updateBalanceDOM(balanceObj){
    try {
      // balanceObj may be nested like { balance: 9840.55, currency:'USD', loginid: 'VRTC...' }
      var amount = balanceObj.balance ?? balanceObj;
      var currency = balanceObj.currency ?? (balanceObj.currency_code || '');
      var loginid = balanceObj.loginid ?? (balanceObj.login_id || '');
      // prefer existing elements if present
      var el = $id('balanceAmount') || $id('rawDebug') || null;
      if (el) {
        el.textContent = 'Balance: ' + (amount !== undefined ? amount : '(unknown)') + (currency ? ' ' + currency : '') + (loginid ? ' (acct ' + loginid + ')' : '');
        el.classList.add('balance-updated');
      } else {
        // fallback: create a small status bar
        var bar = document.getElementById('hero-balance-bar');
        if (!bar) {
          bar = document.createElement('div');
          bar.id = 'hero-balance-bar';
          Object.assign(bar.style, {position:'fixed', left:'12px', top:'12px', zIndex:2147483647, background:'#fff', padding:'6px 10px', border:'1px solid #222', borderRadius:'6px'});
          document.body.appendChild(bar);
        }
        bar.textContent = 'Balance: ' + (amount !== undefined ? amount : '(unknown)') + (currency ? ' ' + currency : '') + (loginid ? ' (acct ' + loginid + ')' : '');
      }
    } catch(e) {
      console.error('connect_real_fix: updateBalanceDOM error', e);
    }
  }

  // Hook SSE to update balance when server sends SSE messages
  function hookSSE(){
    try {
      if (typeof EventSource === 'undefined') {
        log('connect_real_fix: EventSource not available in this browser');
        return;
      }
      // the app uses /events for SSE based on your logs; adapt if different
      if (window._connect_real_fix_sse) return; // already attached
      var ev = new EventSource('/events');
      window._connect_real_fix_sse = ev;
      ev.addEventListener('message', function(e){
        try {
          var d = JSON.parse(e.data);
          log('connect_real_fix: SSE message', d);
          // if analysis_event === account_connected or msg_type === balance update UI
          var payload = d;
          if (d.analysis_event === 'account_connected' && d.balance) {
            // sometimes nested
            var b = d.balance.balance || d.balance;
            updateBalanceDOM(b);
          } else if (d.msg_type === 'balance' && d.balance) {
            var b2 = d.balance.balance || d.balance;
            updateBalanceDOM(b2);
          } else if (d.balance && d.balance.balance) {
            updateBalanceDOM(d.balance.balance);
          }
        } catch(err){ console.error('connect_real_fix: SSE parse error', err); }
      });
      ev.addEventListener('open', function(){ log('connect_real_fix: SSE open'); });
      ev.addEventListener('error', function(err){ log('connect_real_fix: SSE error', err); });
      log('connect_real_fix: SSE attached to /events');
    } catch(e){ console.error('connect_real_fix: hookSSE error', e); }
  }

  // Attach the clickable handler for Real connect that forces sending the real token
  function attachRealHandler(){
    try {
      var btnReal = $id('btnConnectReal');
      if (!btnReal) {
        log('connect_real_fix: btnConnectReal not found');
        return;
      }
      // avoid double-binding
      if (btnReal._connect_real_fix_bound) return;
      btnReal._connect_real_fix_bound = true;

      btnReal.addEventListener('click', function(ev){
        try { ev && ev.preventDefault && ev.preventDefault(); } catch(e){}
        // copy real token to main token (if necessary)
        var apiTokenReal = $id('apiTokenReal');
        var apiToken = $id('apiToken');
        if (apiTokenReal && apiToken) apiToken.value = apiTokenReal.value;
        // set hidden mode to 'real' (if present)
        var acctMode = $id('accountMode');
        if (acctMode) acctMode.value = 'real';
        // build payload & send
        var tokenVal = (apiTokenReal && apiTokenReal.value) ? apiTokenReal.value : (apiToken && apiToken.value ? apiToken.value : '');
        var payload = { token: tokenVal, app_id: 71710, mode: 'real' };
        sendConnectPayload(payload);
        // ensure SSE attached so balances update in UI
        hookSSE();
      }, false);

      log('connect_real_fix: attached handler to btnConnectReal');
    } catch(e){ console.error('connect_real_fix: attachRealHandler error', e); }
  }

  document.addEventListener('DOMContentLoaded', function(){
    attachRealHandler();
    hookSSE();
    log('connect_real_fix: script ready');
  });
})();
