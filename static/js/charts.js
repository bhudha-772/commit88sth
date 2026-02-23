// charts.js
(() => {
  if (window.__herox_charts_inited) return;
  window.__herox_charts_inited = true;

  const MAX_POINTS = 60; // how many points to keep per mini-chart
  const chartsGrid = document.getElementById('charts-grid');
  const emptyState = chartsGrid.querySelector('.charts-empty');
  const modal = document.getElementById('chart-modal');
  const modalTitle = document.getElementById('chart-modal-title');
  const modalCanvas = document.getElementById('chart-modal-canvas');
  const modalClose = document.getElementById('chart-modal-close');

  // store per-market data
  const markets = {}; // market -> {data: [{t, v}], miniChart: Chart, modalChart: Chart|null, cardEl }
  function ensureMarket(m) {
    if (!markets[m]) {
      markets[m] = { data: [], miniChart: null, modalChart: null, cardEl: null };
      createMiniCard(m);
    }
    return markets[m];
  }

  function createMiniCard(market) {
    if (emptyState) emptyState.style.display = 'none';

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.dataset.market = market;

    card.innerHTML = `
      <div class="chart-title">
        <div class="market">${market}</div>
        <div class="meta" data-meta>—</div>
      </div>
      <canvas class="chart-mini"></canvas>
      <div class="chart-footer">
        <div class="left">latest: <span data-latest>—</span></div>
        <div class="right" data-count>0 pts</div>
      </div>
    `;
    chartsGrid.prepend(card);
    const canvas = card.querySelector('canvas');
    const latestEl = card.querySelector('[data-latest]');
    const countEl = card.querySelector('[data-count]');
    const metaEl = card.querySelector('[data-meta]');

    // Chart.js mini chart
    const ctx = canvas.getContext('2d');
    const miniChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: market,
          data: [],
          tension: 0.25,
          borderWidth: 1,
          pointRadius: 0,
          fill: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        scales: {
          x: { display: false },
          y: { display: true, ticks: { maxTicksLimit: 3, callback: (v) => v } }
        },
        plugins: { legend: { display: false }, tooltip: { enabled: false } }
      }
    });

    // store refs
    const m = ensureMarket(market);
    m.cardEl = card;
    m.miniChart = miniChart;
    m.latestEl = latestEl;
    m.countEl = countEl;
    m.metaEl = metaEl;

    // click to expand
    card.addEventListener('click', () => openModal(market));
  }

  function updateMarketWithTick(market, ts, price) {
    try {
      const m = ensureMarket(market);
      const point = { t: ts || Date.now(), v: Number(price) };
      m.data.push(point);
      if (m.data.length > MAX_POINTS) m.data.shift();

      // update mini chart dataset
      const labels = m.data.map(p => new Date(p.t).toLocaleTimeString());
      const data = m.data.map(p => p.v);
      m.miniChart.data.labels = labels;
      m.miniChart.data.datasets[0].data = data;
      m.miniChart.update();

      // update meta/latest/count
      if (m.latestEl) m.latestEl.textContent = (typeof price === 'number' ? price.toFixed(2) : String(price));
      if (m.countEl) m.countEl.textContent = `${m.data.length} pts`;

    } catch (e) { console.warn('updateMarket error', e); }
  }

  // Expand modal: build/refresh large chart
  function openModal(market) {
    const m = ensureMarket(market);
    modal.setAttribute('aria-hidden', 'false');
    modalTitle.textContent = `${market} — Expanded`;
    // build modal chart if missing
    if (!m.modalChart) {
      // destroy any pre-existing modal Chart instance
      try { if (m.modalChart && m.modalChart.destroy) m.modalChart.destroy(); } catch(e){}

      const ctx = modalCanvas.getContext('2d');
      m.modalChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: m.data.map(p => new Date(p.t).toLocaleString()),
          datasets: [{
            label: market,
            data: m.data.map(p => p.v),
            tension: 0.2,
            borderWidth: 2,
            pointRadius: 2,
            fill: true
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 200 },
          scales: {
            x: { display: true, title: { display: true, text: 'Time' } },
            y: { display: true, title: { display: true, text: 'Price / last_decimal' } }
          },
          plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } }
        }
      });
    } else {
      // refresh existing modal chart
      m.modalChart.data.labels = m.data.map(p => new Date(p.t).toLocaleString());
      m.modalChart.data.datasets[0].data = m.data.map(p => p.v);
      m.modalChart.update();
    }
  }
  modalClose.addEventListener('click', () => modal.setAttribute('aria-hidden', 'true'));
  modal.addEventListener('click', (ev) => {
    if (ev.target === modal) modal.setAttribute('aria-hidden', 'true');
  });

  // SSE connection. Use EventSource to /events
  const sseUrl = (location.origin + '/events').replace('http:', 'http:'); // same host
  let es;
  function startSSE() {
    try {
      es = new EventSource('/events');
    } catch (e) {
      console.error('SSE start failed', e);
      return;
    }

    // generic messages (data: { "payload": ... })
    es.addEventListener('message', (evt) => {
      if (!evt || !evt.data) return;
      try {
        const parsed = JSON.parse(evt.data);
        if (parsed && parsed.payload) {
          handlePayload(parsed.payload);
        } else if (parsed && parsed.recent) {
          // recent snapshot array
          parsed.recent.forEach(r => {
            // older workers may send arrays of row-like ticks; best-effort polled below
            try {
              if (Array.isArray(r) && r.length >= 7) {
                const sym = r[6];
                const last_decimal = r[4];
                const price = last_decimal || r[3] || null;
                if (sym) handleTick(sym, Date.now(), price);
              }
            } catch(e){}
          });
        }
      } catch (e) {}
    });

    // named 'analysis' events (broadcast_analysis in server)
    es.addEventListener('analysis', (evt) => {
      if (!evt || !evt.data) return;
      try {
        const payload = JSON.parse(evt.data);
        // If analysis includes price/last_decimal
        handlePayload(payload);
      } catch (e) {}
    });

    // also listen for 'open' and 'recent' events if server sends them
    es.addEventListener('recent', (evt) => {
      if (!evt || !evt.data) return;
      try {
        const parsed = JSON.parse(evt.data);
        if (parsed && parsed.recent && Array.isArray(parsed.recent)) {
          parsed.recent.forEach(row => {
            if (Array.isArray(row) && row.length >= 7) {
              const s = row[6];
              const last_decimal = row[4];
              const price = (last_decimal !== "" ? Number(last_decimal) : (row[3] !== "" ? Number(row[3]) : null));
              if (s) handleTick(s, Date.now(), price);
            }
          });
        }
      } catch(e){}
    });

    es.onerror = (e) => {
      console.warn('SSE error', e);
      // attempt reconnect after small delay
      try { es.close(); } catch(_) {}
      setTimeout(startSSE, 2000);
    };
  }

  function handlePayload(payload) {
    if (!payload) return;
    // lots of analysis payloads can exist; try to find symbol and price/last_decimal
    // common shapes: { symbol, market, price, last_decimal, last_unit, ... }
    const sym = (payload.symbol || payload.market || payload.symbol_code || payload.symbol_code || '').toString().toUpperCase() || null;
    let price = null;
    if (payload.last_decimal !== undefined && payload.last_decimal !== null && payload.last_decimal !== '') {
      price = Number(payload.last_decimal);
    } else if (payload.price !== undefined && payload.price !== null) {
      // price might be number or string with decimal; use last digit optionally
      price = Number(payload.price);
    } else if (payload.buffer_snapshot && Array.isArray(payload.buffer_snapshot) && payload.buffer_snapshot.length) {
      // some analysis payloads include buffer_snapshot of last decimals
      const last = payload.buffer_snapshot[payload.buffer_snapshot.length - 1];
      price = (typeof last === 'number' ? last : (last && last.last_decimal ? Number(last.last_decimal) : last));
    }

    const epoch = payload.epoch || payload.ts || payload.timestamp || Date.now();

    if (sym && (price !== null && price !== undefined && !Number.isNaN(price))) {
      handleTick(sym, epoch, price);
    }
  }

  function handleTick(symbol, epoch, price) {
    if (!symbol) return;
    // normalize symbol string
    const s = String(symbol).toUpperCase();
    ensureMarket(s);
    updateMarketWithTick(s, epoch, price);
  }

  // start SSE
  startSSE();

})();
