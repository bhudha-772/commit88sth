/* force_connect_mode.js — idempotent override to ensure demo/real behave correctly */
(function(){
  function $id(id){ return document.getElementById(id); }
  function log(){ try{ console.log.apply(console, arguments); }catch(e){} }

  function updateModeUI(mode){
    var modeText = $id('modeText');
    if(modeText) modeText.textContent = mode === 'real' ? 'Real authenticated' : 'Demo authenticated';
    var acctMode = $id('accountMode');
    if(acctMode) acctMode.value = mode;
  }

  function showBalance(balanceObj){
    try {
      var amount = balanceObj && (balanceObj.balance ?? balanceObj) ?? '(unknown)';
      var currency = balanceObj && (balanceObj.currency ?? '') || '';
      var loginid = balanceObj && (balanceObj.loginid ?? balanceObj.login_id) || '';
      var bar = $id('hero-balance-bar');
      if (!bar) {
        bar = document.createElement('div');
        bar.id = 'hero-balance-bar';
        Object.assign(bar.style, {position:'fixed', left:'12px', top:'12px', zIndex:2147483647, background:'#fff', padding:'8px 10px', border:'1px solid #222', borderRadius:'6px'});
        document.body.appendChild(bar);
      }
      bar.textContent = 'Balance: ' + amount + (currency ? ' ' + currency : '') + (loginid ? ' (' + loginid + ')' : '');
    } catch(e){ console.error('force_connect_mode: showBalance error', e); }
  }

  async function doConnect(payload){
    log('force_connect_mode: sending', payload);
    try {
      const r = await fetch('/control/connect_account', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      const text = await r.text();
      log('force_connect_mode: response', r.status, text);
      try {
        const j = JSON.parse(text);
        // If server sends nested balance structure (as seen in your logs), try to extract it:
        if (j && (j.balance || j.balance?.balance || j.balance?.balance?.balance)) {
          // normalized attempt
          let b = j.balance;
          if (b && b.balance && b.balance.balance) b = b.balance;
          if (b && b.balance) showBalance(b.balance || b);
          else if (j.balance.balance) showBalance(j.balance.balance);
          else showBalance(j.balance);
        } else if (j && j.ok && j.balance) showBalance(j.balance);
      } catch(e){ /* not JSON */ }
      // update UI mode text as optimistic
      if (payload.mode) updateModeUI(payload.mode);
    } catch(err){
      console.error('force_connect_mode: fetch failed', err);
    }
  }

  function attachHandlers(){
    try {
      var btnDemo = $id('btnConnectAccount');
      var btnReal = $id('btnConnectReal');
      var apiToken = $id('apiToken');
      var apiTokenReal = $id('apiTokenReal');

      if (btnDemo) {
        btnDemo.type = 'button';
        btnDemo.addEventListener('click', function(ev){
          try{ ev.preventDefault(); }catch(e){}
          var token = apiToken && apiToken.value ? apiToken.value : '';
          var payload = { token: token, app_id: 71710, mode: 'demo' };
          updateModeUI('demo');
          doConnect(payload);
        }, {passive:false});
        log('force_connect_mode: demo handler attached');
      } else {
        log('force_connect_mode: btnConnectAccount NOT found');
      }

      if (btnReal) {
        btnReal.type = 'button';
        btnReal.addEventListener('click', function(ev){
          try{ ev.preventDefault(); }catch(e){}
          // copy real into the main token field (so legacy code sees it)
          var tokenVal = (apiTokenReal && apiTokenReal.value) ? apiTokenReal.value : (apiToken && apiToken.value ? apiToken.value : '');
          if (apiToken) apiToken.value = tokenVal;
          var payload = { token: tokenVal, app_id: 71710, mode: 'real' };
          updateModeUI('real');
          doConnect(payload);
        }, {passive:false});
        log('force_connect_mode: real handler attached');
      } else {
        log('force_connect_mode: btnConnectReal NOT found');
      }
    } catch(e){ console.error('force_connect_mode: attachHandlers error', e); }
  }

  function hookSSE(){
    try {
      if (window._force_connect_mode_sse) return;
      if (typeof EventSource === 'undefined') return;
      var ev = new EventSource('/events');
      window._force_connect_mode_sse = ev;
      ev.addEventListener('message', function(evt){
        try {
          var data = JSON.parse(evt.data);
          log('force_connect_mode: SSE', data);
          // prefer analysis_event account_connected or msg_type balance paths
          if (data.analysis_event === 'account_connected') {
            // server gives nested balance in your logs: data.balance.balance.balance etc
            var b = data.balance;
            if (b && b.balance) {
              // b.balance may itself be object with .balance property
              var inner = b.balance.balance ? b.balance : b.balance;
              showBalance(inner.balance || inner);
            } else if (data.msg_type === 'balance' && data.balance) {
              showBalance(data.balance.balance || data.balance);
            }
            if (data.mode) updateModeUI(data.mode);
          } else if (data.msg_type === 'balance' && data.balance) {
            showBalance(data.balance.balance || data.balance);
          }
        } catch(e){ console.error('force_connect_mode: SSE parse', e); }
      });
      ev.addEventListener('open', function(){ log('force_connect_mode: SSE open'); });
      ev.addEventListener('error', function(e){ log('force_connect_mode: SSE error', e); });
      log('force_connect_mode: SSE hooked');
    } catch(e){ console.error('force_connect_mode: hookSSE failed', e); }
  }

  document.addEventListener('DOMContentLoaded', function(){
    attachHandlers();
    hookSSE();
    log('force_connect_mode: ready');
  });
})();
