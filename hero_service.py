#!/usr/bin/env python3
"""
hero_service.py — HTTP + SSE server for HeroX dashboard
Maintains recent ticks, SSE broadcaster, (prediction handling REMOVED),
and Deriv account connect/get_balances endpoints (minimal).
"""
from __future__ import annotations

import os
import time
import json
import math
import queue
import threading
import subprocess
import asyncio
import socket
import traceback
import sys
import shlex
import importlib
from collections import Counter, deque, defaultdict
from datetime import datetime, timezone
from typing import Deque, Any, List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from flask import (
    Flask,
    render_template,
    request,
    Response,
    jsonify,
    abort,
    send_from_directory,
    current_app,
)

from flask_compress import Compress
# --- callback HTTP helper import (requests preferred; fallback to urllib) ---
try:
    import requests
except Exception:
    requests = None
import urllib.request as _urllib_request
import urllib.error as _urllib_error
import urllib.parse as _urllib_parse
import ssl as _ssl
# --- end callback import ---

import worker_manager  # keep existing behavior; this module must exist



# in-memory tokens per-mode (ephemeral; cleared when hero_service restarts)
# keys: "demo" and "real"
GLOBAL_DERIV_TOKENS = {"demo": None, "real": None}

# server-side timestamp of last manual analysis stop (to avoid immediate restart races)
GLOBAL_ANALYSIS_LAST_STOP_TS = 0.0
GLOBAL_ANALYSIS_PROC = None
GLOBAL_ANALYSIS_PROC_LOCK = threading.Lock()

# server-side autotrade settings (mode -> settings)
# Example shape: { "demo": {"stake": 1.0}, "real": {"stake": 0.35}, "default_mode": "demo" }
GLOBAL_AUTOTRADE_SETTINGS = {"demo": {"stake": None}, "real": {"stake": None}, "default_mode": (os.environ.get("DIFFER_MODE") or "demo")}


# ---------- Config ----------
LOG_DIR = os.path.expanduser("~/.hero_logs")
os.makedirs(LOG_DIR, exist_ok=True)

RECENT_TICKS_MAX = 500
SSE_PING_INTERVAL = 15  # seconds
TICK_STALE_THRESHOLD = 8.0  # seconds with no ticks = "stalled"
MONITOR_INTERVAL = 3.0
ADMIN_TOKEN = os.environ.get("HERO_ADMIN_TOKEN", "").strip() or None
# Strictly enforced application id for Deriv calls (server-side enforced)
ENFORCED_APP_ID = 71710

app = Flask(__name__, template_folder="templates", static_folder="static")

# initialize flask-compress now that app exists
try:
    Compress(app)
except Exception as _e:
    try:
        app.logger.warning("Compress init failed: %s", _e)
    except Exception:
        print(f"Compress init failed: {_e}", file=sys.stderr)

# --- register higher/lower blueprint (do this AFTER app is created) ---
try:
    # import here to avoid NameError and reduce circular-import risk
    import higher_lower_trade
    app.register_blueprint(higher_lower_trade.hl_bp)
    app.logger.info("Registered higher_lower_trade blueprint")
except Exception as e:
    # _err is defined later; keep this early path self-contained.
    app.logger.exception("Failed registering higher_lower_trade blueprint: %s", e)


# ---------- Token persistence (persist demo/real tokens so server restart doesn't lose them) ----------
TOKEN_FILE = os.path.join(LOG_DIR, "hero_tokens.json")
ANALYSIS_PID_FILE = os.path.join(LOG_DIR, "differs_agent.pid")

def _mask_token(t):
    if not t:
        return None
    t = str(t)
    if len(t) <= 8:
        return t[:2] + "..." + t[-2:]
    return t[:4] + "..." + t[-4:]


def _save_tokens():
    try:
        payload = {"demo": GLOBAL_DERIV_TOKENS.get("demo"), "real": GLOBAL_DERIV_TOKENS.get("real")}
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        _err(f"_save_tokens: {e}")


def _load_tokens():
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                GLOBAL_DERIV_TOKENS["demo"] = data.get("demo") or GLOBAL_DERIV_TOKENS.get("demo")
                GLOBAL_DERIV_TOKENS["real"] = data.get("real") or GLOBAL_DERIV_TOKENS.get("real")
                _log("Loaded persisted tokens (demo_present=%s real_present=%s)" % (bool(GLOBAL_DERIV_TOKENS["demo"]), bool(GLOBAL_DERIV_TOKENS["real"])))
    except Exception as e:
        _err(f"_load_tokens: {e}")


# Persistent file paths
JOURNAL_FILE = os.path.join(LOG_DIR, "differ_journal.jsonl")

# ---------- Session-backed journal (in-memory) ----------
SESSION_JOURNAL: List[dict] = []  # newest-first (index 0 is newest)
SESSION_JOURNAL_MAX = int(os.environ.get("HERO_SESSION_MAX", "200"))  # memory cap

# Executor for background tasks (journal writes, callbacks)
_executor = ThreadPoolExecutor(max_workers=4)

# Protect journal trimming/reads to avoid concurrent read/overwrite races
_JOURNAL_LOCK = threading.Lock()

JOURNAL_MAX_ENTRIES = int(os.environ.get("HERO_JOURNAL_MAX", "200"))

# ---------- In-memory storage & SSE broadcaster ----------
_recent_ticks: Deque[List[Any]] = deque(maxlen=RECENT_TICKS_MAX)
_recent_lock = threading.Lock()
_sse_clients: List[queue.Queue] = []
_sse_lock = threading.Lock()
_last_tick_time = 0.0
_last_tick_lock = threading.Lock()
_monitor_state = {"deriv_prev": False, "analysis_prev": False, "network_alerted": False}
# per-market monotonic sequence (server authoritative ordering)
_market_seq: Dict[str, int] = defaultdict(int)

# queue of serialized payloads from producers (must exist before broadcaster starts)
_sse_out_q = queue.Queue(maxsize=10000)
_SSE_BATCH_INTERVAL = float(os.environ.get("HERO_SSE_BATCH_INTERVAL", "0.25"))  # sec
_SSE_BATCH_MAX = int(os.environ.get("HERO_SSE_BATCH_MAX", "200"))

# ---------- Logging helpers ----------
def _log(msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with open(os.path.join(LOG_DIR, "hero_service.out"), "a", encoding="utf-8") as f:
            f.write(f"{ts} {msg}\n")
    except Exception:
        pass


def _err(msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with open(os.path.join(LOG_DIR, "hero_service.err"), "a", encoding="utf-8") as f:
            f.write(f"{ts} ERROR: {msg}\n")
    except Exception:
        pass


def _log_callback(msg: str):
    _log(f"callback: {msg}")


def _err_callback(msg: str):
    _err(f"callback: {msg}")


def _check_admin_token(req):
    if not ADMIN_TOKEN:
        return
    token = req.args.get("token") or req.headers.get("X-ADMIN-TOKEN")
    if token != ADMIN_TOKEN:
        abort(403, description="forbidden")


# --- robust import for HL trade daemon (non-fatal) ---
try:
    from higher_lower_trade_daemon import run_trade_daemon as run_trade_daemon  # type: ignore[reportMissingImports]
except Exception as _e_hl:
    try:
        # fallback to differ implementation if HL module missing
        from differ_trade_check import run_trade_daemon as run_trade_daemon
    except Exception as _e_diff:
        run_trade_daemon = None
        # both imports failed — hero_service will proceed but will not auto-start HL daemon
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "No HL run_trade_daemon found (higher_lower_trade_daemon failed: %s; differ fallback failed: %s). HL daemon will not auto-start.",
            _e_hl, _e_diff
        )


_trade_daemon_started = False
_trade_daemon_lock = threading.Lock()


def start_trade_daemon_once() -> bool:
    global _trade_daemon_started
    with _trade_daemon_lock:
        if _trade_daemon_started:
            return True
        fn = run_trade_daemon
        if not callable(fn):
            _log("start_trade_daemon_once: run_trade_daemon unavailable; skipping")
            return False
        try:
            t = threading.Thread(target=fn, name="hl_trade_daemon", daemon=True)
            t.start()
            _trade_daemon_started = True
            _log("start_trade_daemon_once: started daemon thread")
            return True
        except Exception as e:
            _err(f"start_trade_daemon_once failed: {e}")
            return False


# --- optional: integrate higher_lower_trade in-process (no extra files) ---
try:
    # Importing higher_lower_trade will instantiate its background client (hd)
    import higher_lower_trade as hl  # noqa: E402
    _log("higher_lower_trade module imported (background client should start)")

    # Best-effort: wait a short time for its background client to become ready
    try:
        hl._wait_ready(timeout=6.0)
        _log("higher_lower_trade: _wait_ready returned (or timed out)")
    except Exception as e:
        _log(f"higher_lower_trade: _wait_ready call failed/ignored: {e}")

    # Register wrapper routes on hero_service's Flask app so HL endpoints are available
    # under /trader/* to avoid unintended path collisions.
    @app.route("/trader/simulate", methods=["POST"])
    def hl_simulate_proxy():
        try:
            # delegate into higher_lower_trade.simulate() — it uses flask.request and jsonify
            return hl.simulate()
        except Exception as e:
            _err(f"hl_simulate_proxy error: {e}")
            return jsonify({"ok": False, "error": "hl_simulate_proxy_exception", "detail": str(e)}), 500

    @app.route("/trader/trade", methods=["POST"])
    def hl_trade_proxy():
        try:
            return hl.trade()
        except Exception as e:
            _err(f"hl_trade_proxy error: {e}")
            return jsonify({"ok": False, "error": "hl_trade_proxy_exception", "detail": str(e)}), 500

    _log("higher_lower_trade routes registered at /trader/simulate and /trader/trade")

except Exception as e:
    _err(f"higher_lower_trade integration import failed (optional) — HL not auto-started: {e}")


# --- optional: integrate persistent assistant module ---
assistant_engine = None
try:
    from hero_assistant import assistant_blueprint as _assistant_bp, assistant_engine as _assistant_engine  # type: ignore

    assistant_engine = _assistant_engine
    try:
        app.register_blueprint(_assistant_bp)
        _log("hero_assistant blueprint registered")
    except Exception as e:
        _err(f"hero_assistant import succeeded but registration failed: {e}")
except Exception as e:
    _log(f"hero_assistant not available or import failed: {e}")


# ---------- SSE payload sanitization ----------
def _truncate_str(s: str, max_len: int = 2000) -> str:
    if not isinstance(s, str):
        return s
    if len(s) <= max_len:
        return s
    return s[:max_len] + "...[truncated]"


def _sanitize_for_sse(obj, max_str: int = 2000, max_depth: int = 6, _depth: int = 0):
    """Return a sanitized shallow copy of obj suitable for SSE transport."""
    try:
        if _depth >= max_depth:
            return "[too deep]"
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # drop very noisy keys
                if k in ("__raw", "raw", "big_payload", "debug_dump"):
                    continue
                # shallowly sanitize strings
                if isinstance(v, str):
                    out[k] = _truncate_str(v, max_str)
                elif isinstance(v, (int, float, bool, type(None))):
                    out[k] = v
                elif isinstance(v, dict):
                    out[k] = _sanitize_for_sse(v, max_str, max_depth, _depth + 1)
                elif isinstance(v, list):
                    # limit list length
                    if len(v) > 50:
                        out[k] = [_sanitize_for_sse(x, max_str, max_depth, _depth + 1) for x in v[:50]]
                        out[k].append(f"...({len(v)-50} more items)")
                    else:
                        out[k] = [_sanitize_for_sse(x, max_str, max_depth, _depth + 1) for x in v]
                else:
                    try:
                        out[k] = str(v)
                    except Exception:
                        out[k] = "[unserializable]"
            return out
        if isinstance(obj, list):
            if len(obj) > 200:
                return [_sanitize_for_sse(x, max_str, max_depth, _depth + 1) for x in obj[:200]] + [f"...({len(obj)-200} more)"]
            return [_sanitize_for_sse(x, max_str, max_depth, _depth + 1) for x in obj]
        # primitive
        if isinstance(obj, str):
            return _truncate_str(obj, max_str)
        return obj
    except Exception:
        return "[sanitize_error]"


# ---------- Indicator helpers (server-side analysis enrichment) ----------
def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_digit(v):
    try:
        n = int(float(v))
    except Exception:
        return None
    if 0 <= n <= 9:
        return n
    return None


def _truthy(v: Any) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _clean_price_text(v):
    if v is None:
        return ""
    try:
        s = str(v).strip()
        # Some workers send price_str_text with leading quote, e.g. "'894.9290"
        if s.startswith("'") or s.startswith('"'):
            s = s[1:]
        return s.strip()
    except Exception:
        return ""


def _extract_last_decimal_from_price_text(price_text):
    s = _clean_price_text(price_text)
    if not s:
        return None
    try:
        if "." in s:
            frac = s.split(".", 1)[1]
            for ch in reversed(frac):
                if ch.isdigit():
                    return int(ch)
    except Exception:
        return None
    return None


def _extract_last_unit_from_price_text(price_text):
    s = _clean_price_text(price_text)
    if not s:
        return None
    try:
        whole = s.split(".", 1)[0] if "." in s else s
        for ch in reversed(whole):
            if ch.isdigit():
                return int(ch)
    except Exception:
        return None
    return None


def _derive_tick_digits(last_decimal, last_unit, price, price_str_text):
    ld = _safe_digit(last_decimal)
    lu = _safe_digit(last_unit)

    if ld is None:
        ld = _extract_last_decimal_from_price_text(price_str_text)
    if ld is None:
        ld = _extract_last_decimal_from_price_text(price)

    if lu is None:
        lu = _extract_last_unit_from_price_text(price_str_text)
    if lu is None:
        lu = _extract_last_unit_from_price_text(price)

    return ld, lu


def _round_num(v, places: int = 8):
    n = _safe_float(v)
    if n is None:
        return None
    try:
        return round(n, places)
    except Exception:
        return n


def _ema_series(values: List[float], period: int) -> List[float]:
    out: List[float] = []
    if not values:
        return out
    k = 2.0 / (float(period) + 1.0)
    prev = None
    for v in values:
        if prev is None:
            prev = float(v)
        else:
            prev = (float(v) - prev) * k + prev
        out.append(prev)
    return out


def _rsi_last(values: List[float], period: int = 14):
    if not values or len(values) < 2:
        return None
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(values)):
        diff = float(values[i]) - float(values[i - 1])
        gains.append(max(0.0, diff))
        losses.append(max(0.0, -diff))

    if len(gains) < period:
        return None

    avg_gain = sum(gains[:period]) / float(period)
    avg_loss = sum(losses[:period]) / float(period)
    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / float(period)
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / float(period)

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _stddev(values: List[float]):
    if not values:
        return None
    n = float(len(values))
    mean = sum(values) / n
    var = sum((x - mean) * (x - mean) for x in values) / n
    return math.sqrt(max(var, 0.0))


def _latest_tick_symbol() -> Optional[str]:
    try:
        with _recent_lock:
            for r in reversed(_recent_ticks):
                if not isinstance(r, (list, tuple)) or len(r) < 7:
                    continue
                sym = str(r[6] or "").upper()
                if sym and sym != "ANALYSIS":
                    return sym
    except Exception:
        return None
    return None


def _recent_prices_for_symbol(symbol: Optional[str], limit: int = 400) -> List[float]:
    out: List[float] = []
    sym_u = str(symbol or "").upper()
    try:
        with _recent_lock:
            for r in reversed(_recent_ticks):
                if not isinstance(r, (list, tuple)) or len(r) < 7:
                    continue
                row_sym = str(r[6] or "").upper()
                if sym_u and row_sym != sym_u:
                    continue
                # prefer "close" column (row[3]) then fallback row[1]
                p = _safe_float(r[3] if len(r) > 3 else None)
                if p is None:
                    p = _safe_float(r[1] if len(r) > 1 else None)
                if p is None:
                    continue
                out.append(p)
                if len(out) >= limit:
                    break
    except Exception:
        return []
    out.reverse()
    return out


def _compute_indicator_snapshot_from_prices(prices: List[float]):
    if not prices:
        return None

    ema9 = _ema_series(prices, 9)
    ema50 = _ema_series(prices, 50)
    ema12 = _ema_series(prices, 12)
    ema26 = _ema_series(prices, 26)

    macd_line = [a - b for a, b in zip(ema12, ema26)]
    macd_signal = _ema_series(macd_line, 9)
    macd_hist = (macd_line[-1] - macd_signal[-1]) if macd_line and macd_signal else None

    rsi14 = _rsi_last(prices, 14)

    if len(prices) > 1:
        diffs = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        atr_window = 14
        tail = diffs[-atr_window:] if len(diffs) >= atr_window else diffs
        atr_approx = (sum(tail) / float(len(tail))) if tail else None
    else:
        atr_approx = None

    bb_tail = prices[-20:] if len(prices) >= 20 else prices
    bb_mean = (sum(bb_tail) / float(len(bb_tail))) if bb_tail else None
    bb_sd = _stddev(bb_tail) if bb_tail else None
    bb_upper = (bb_mean + (2.0 * bb_sd)) if (bb_mean is not None and bb_sd is not None) else None
    bb_lower = (bb_mean - (2.0 * bb_sd)) if (bb_mean is not None and bb_sd is not None) else None

    return {
        "price": _round_num(prices[-1]),
        "ema9": _round_num(ema9[-1] if ema9 else None),
        "ema50": _round_num(ema50[-1] if ema50 else None),
        "macdHist": _round_num(macd_hist),
        "rsi14": _round_num(rsi14),
        "atrApprox": _round_num(atr_approx),
        "bbUpper": _round_num(bb_upper),
        "bbLower": _round_num(bb_lower),
        "count": len(prices),
    }


def _enrich_analysis_payload_with_indicators(payload: dict) -> dict:
    try:
        out = dict(payload or {})
        symbol = (
            out.get("symbol")
            or out.get("market")
            or out.get("market_code")
            or out.get("symbol_code")
            or _latest_tick_symbol()
        )
        prices = _recent_prices_for_symbol(symbol, limit=400)
        indicators = _compute_indicator_snapshot_from_prices(prices)
        if indicators:
            out["indicators"] = indicators
            if symbol and not out.get("symbol"):
                out["symbol"] = str(symbol).upper()
            reason = out.get("reason")
            if reason is None:
                out["reason"] = {"indicators": indicators}
            elif isinstance(reason, dict) and "indicators" not in reason:
                reason["indicators"] = indicators
                out["reason"] = reason
        return out
    except Exception:
        return dict(payload or {})


def _enrich_tick_payload(payload: dict) -> dict:
    """
    Keep tick payloads compact but ensure reason/indicators are present for UI debug panels.
    """
    try:
        out = dict(payload or {})
        enriched = _enrich_analysis_payload_with_indicators(out)
        indicators = enriched.get("indicators")
        reason = out.get("reason")
        if indicators:
            out["indicators"] = indicators
            if reason is None:
                out["reason"] = {"source": "server_tick", "indicators": indicators}
            elif isinstance(reason, dict) and "indicators" not in reason:
                reason["indicators"] = indicators
                out["reason"] = reason
        elif reason is None:
            # Keep reason non-null for clients that render/debug this field directly.
            out["reason"] = {"source": "server_tick", "indicators": None}
        return out
    except Exception:
        return dict(payload or {})


# ---------- SSE broadcast ----------
def _enqueue_sse(item: dict, event: Optional[str] = None):
    try:
        # sanitize heavily to avoid huge blobs and deeply nested structures
        try:
            sanitized = _sanitize_for_sse(item, max_str=2000)
        except Exception:
            sanitized = {"_sanitized": True}
        payload_obj = {"event": event, "payload": sanitized}
        payload = json.dumps(payload_obj, default=str)
    except Exception:
        # fallback to minimal payload if JSON fails
        try:
            payload = json.dumps({"event": event, "payload": {"_error": "json_serialize_failed"}}, default=str)
        except Exception:
            payload = '{"event": null, "payload": {}}'
    try:
        _sse_out_q.put_nowait(payload)
    except Exception:
        # queue full -> drop oldest to keep moving (avoid blocking)
        try:
            _sse_out_q.get_nowait()
            _sse_out_q.put_nowait(payload)
        except Exception:
            _err("enqueue_sse: failed to enqueue payload (queue may be full)")


def _broadcast_analysis(payload: dict):
    # sanitize + enrich: remove raw fields and include server-computed indicators
    try:
        sanitized = dict(payload or {})
        sanitized = _enrich_analysis_payload_with_indicators(sanitized)
        sanitized = {k: v for k, v in sanitized.items() if k != "__raw"}
        try:
            if assistant_engine is not None and hasattr(assistant_engine, "ingest_analysis"):
                assistant_engine.ingest_analysis(sanitized)
        except Exception:
            pass
        _enqueue_sse(sanitized, event="analysis")
    except Exception as e:
        _err(f"_broadcast_analysis failed: {e}")


def _broadcast_tick(payload: dict):
    if (payload.get("symbol") or "").upper() == "ANALYSIS":
        _broadcast_analysis(payload)
        return
    try:
        if assistant_engine is not None and hasattr(assistant_engine, "ingest_tick"):
            assistant_engine.ingest_tick(payload)
    except Exception:
        pass
    _enqueue_sse(payload, event=None)


# ---------- broadcast helper for tokens status (missing earlier) ----------
def _broadcast_tokens_status():
    """Broadcast which modes have server-side tokens (masked)"""
    try:
        tokens = {
            m: {"present": bool(GLOBAL_DERIV_TOKENS.get(m)), "masked": _mask_token(GLOBAL_DERIV_TOKENS.get(m))}
            for m in ("demo", "real")
        }
        _broadcast_analysis({"analysis_event": "server_tokens", "tokens": tokens})
    except Exception as e:
        _err(f"_broadcast_tokens_status failed: {e}")


# ---------- Journal helpers (atomic-ish) ----------
def write_journal_async(entry: dict, max_entries: int = JOURNAL_MAX_ENTRIES):
    """ Append entry to JOURNAL_FILE and trim to max_entries in a background thread. """

    def _write():
        try:
            # Ensure log dir exists
            try:
                os.makedirs(LOG_DIR, exist_ok=True)
            except Exception:
                pass

            ln = json.dumps(entry, default=str)

            # Append atomically (best-effort)
            try:
                with open(JOURNAL_FILE, "a", encoding="utf-8") as jf:
                    jf.write(ln + "\n")
                    jf.flush()
                    try:
                        os.fsync(jf.fileno())
                    except Exception:
                        pass
            except Exception as e:
                _err(f"journal async write append failed: {e}")
                return

            # Trim file in background (protected by lock)
            try:
                with _JOURNAL_LOCK:
                    try:
                        with open(JOURNAL_FILE, "r", encoding="utf-8") as jf:
                            lines = jf.readlines()
                    except Exception:
                        lines = []
                    if len(lines) > max_entries:
                        tail = lines[-max_entries:]
                        try:
                            with open(JOURNAL_FILE, "w", encoding="utf-8") as jf:
                                jf.writelines(tail)
                        except Exception as e:
                            _err(f"journal trim write failed: {e}")
            except Exception as e:
                _err(f"journal trim failed: {e}")

            _log(f"journal write OK: {entry.get('prediction_id') or entry.get('id') or 'entry'}")
        except Exception as e:
            _err(f"journal async write failed: {e}")

    try:
        _executor.submit(_write)
    except Exception as e:
        _err(f"failed to submit journal write: {e}")


def _append_session_entry(entry: dict, max_entries: int = SESSION_JOURNAL_MAX):
    """ Insert entry (dict) at the head of SESSION_JOURNAL (newest-first) and trim. """
    try:
        pid = entry.get("prediction_id") or entry.get("pred_id") or ""  # dedupe by id if present
        global SESSION_JOURNAL
        if pid:
            SESSION_JOURNAL = [
                e
                for e in SESSION_JOURNAL
                if (e.get("prediction_id") or e.get("pred_id") or "") != pid
            ]
        SESSION_JOURNAL.insert(0, entry)
        if len(SESSION_JOURNAL) > max_entries:
            SESSION_JOURNAL = SESSION_JOURNAL[:max_entries]
    except Exception as e:
        _err(f"_append_session_entry failed: {e}")

# ---------- Helper: normalize result to only WIN or LOSS ----------
def _coerce_result_to_win_loss(result_value, profit=None, profit_pct=None):
    """
    Return 'WIN' if profit/profit_pct > 0 else 'LOSS'.
    If result_value is already 'WIN' or 'LOSS' (any case), return it.
    Otherwise infer from profit/profit_pct; default = 'LOSS'.
    """
    try:
        if isinstance(result_value, str):
            rv = result_value.strip().upper()
            if rv in ("WIN", "LOSS"):
                return rv
        # prefer explicit profit
        if profit is not None:
            try:
                p = float(profit)
                return "WIN" if p > 0 else "LOSS"
            except Exception:
                pass
        if profit_pct is not None:
            try:
                pp = float(profit_pct)
                return "WIN" if pp > 0 else "LOSS"
            except Exception:
                pass
    except Exception:
        pass
    return "LOSS"


# ---------- Helpers: settled-detection and handling ----------
def _is_final_analysis(analysis_payload: dict) -> bool:
    """Heuristics that decide if an analysis payload represents a final/settled result."""
    try:
        ae = str(analysis_payload.get("analysis_event") or analysis_payload.get("event") or "").lower()
        result_val = str(analysis_payload.get("result") or analysis_payload.get("outcome") or analysis_payload.get("status") or "").upper()
        if ae in ("prediction_result", "prediction_settled", "contract_settled", "settled"):
            return True
        if "final_contract" in analysis_payload and isinstance(analysis_payload.get("final_contract"), dict):
            return True
        if "profit" in analysis_payload and analysis_payload.get("profit") is not None:
            return True
        if result_val in ("WIN", "LOSS", "DRAW", "SETTLED"):
            return True
    except Exception:
        pass
    return False


def _handle_settled_analysis(analysis_payload: dict):
    """
    Centralised handler for settled / final analysis payloads:
    - normalizes minimal entry
    - appends to in-memory session journal
    - persists to file via write_journal_async
    - broadcasts a named 'prediction_result' SSE event (plus analysis event for compatibility)
    - best-effort: refresh account balances and broadcast balance_update
    """
    try:
        ap = dict(analysis_payload or {})
        # canonical timestamp
        ts = int(ap.get("epoch") or ap.get("ts") or ap.get("timestamp") or time.time())
        pid = ap.get("prediction_id") or ap.get("pred_id") or ap.get("id") or f"pred_{int(time.time()*1000)}"
        market = (ap.get("symbol") or ap.get("market") or "").upper()
        direction = str(ap.get("direction") or ap.get("signal") or ap.get("trade_type") or "").strip().lower()
        contract_type = str(ap.get("contract_type") or "").strip().upper()
        if not direction:
            if contract_type == "CALL":
                direction = "higher"
            elif contract_type == "PUT":
                direction = "lower"
        if not contract_type:
            if direction == "higher":
                contract_type = "CALL"
            elif direction == "lower":
                contract_type = "PUT"
        # prediction digit normalization
        pred_digit = None
        try:
            pd = _first_present(ap, ["prediction_digit", "predicted", "pred", "digit"])
            if pd is not None:
                pred_digit = int(pd)
        except Exception:
            pred_digit = None

        # result/profit: extract reliably from final_contract if present
        result = ap.get("result") or ap.get("outcome") or ap.get("status") or None
        profit = ap.get("profit")
        try:
            if profit is None and isinstance(ap.get("final_contract"), dict):
                profit = ap["final_contract"].get("profit")
        except Exception:
            profit = profit
        profit_pct = ap.get("profit_percentage") or (ap.get("final_contract", {}) or {}).get("profit_percentage")

        # Coerce result strictly to WIN or LOSS (helper prefers profit fields)
        result = _coerce_result_to_win_loss(result, profit=profit, profit_pct=profit_pct)

        # Determine actual digit (try final_contract.result_digit then observed ticks)
        actual_digit = None
        try:
            fc = ap.get("final_contract") or {}
            if isinstance(fc, dict):
                if fc.get("result_digit") is not None:
                    actual_digit = int(fc.get("result_digit"))
                elif fc.get("actual") is not None:
                    actual_digit = int(fc.get("actual"))
                else:
                    obs = fc.get("tick_stream") or fc.get("audit_details", {}).get("all_ticks") or ap.get("observed_ticks") or []
                    if isinstance(obs, (list, tuple)) and len(obs) > 0:
                        last = obs[-1]
                        try:
                            if isinstance(last, dict):
                                if "tick" in last and isinstance(last["tick"], dict):
                                    cand = last["tick"].get("last_digit") or last["tick"].get("quote")
                                else:
                                    cand = last.get("last_digit") or last.get("quote") or last.get("tick")
                                if cand is not None:
                                    actual_digit = int(str(cand)[-1:])
                            else:
                                actual_digit = int(str(last)[-1:])
                        except Exception:
                            actual_digit = None
        except Exception:
            actual_digit = None

        # journal entry saved to disk (canonical shape)
        entry = {
            "timestamp": int(ts),
            "prediction_id": pid,
            "symbol": market,
            "direction": direction,
            "trade_type": direction,
            "contract_type": contract_type,
            "prediction_digit": pred_digit,
            "confidence": ap.get("confidence"),
            "result": result,                       # guaranteed WIN|LOSS
            "profit": (float(profit) if profit is not None else None),
            "profit_percentage": (float(profit_pct) if profit_pct is not None else None),
            "state": result,
            "actual": actual_digit,
            "reason": ap.get("reason"),
            "indicators": ap.get("indicators"),
            "__raw": ap,
        }

        # session-friendly compact entry (frontend expects newest-first)
        ui_entry = {
            "ts": entry["timestamp"],
            "market": entry["symbol"],
            "direction": entry.get("direction"),
            "trade_type": entry.get("trade_type"),
            "contract_type": entry.get("contract_type"),
            "pred": entry.get("prediction_digit"),
            "actual": (ap.get("final_contract") or {}).get("result_digit") if isinstance(ap.get("final_contract"), dict) else None,
            "result": entry.get("result"),
            "profit": entry.get("profit"),
            "pct": entry.get("profit_percentage"),
            "conf": entry.get("confidence"),
            "id": entry.get("prediction_id"),
            "entry": entry,
        }


        # Append to in-memory session (newest-first)
        try:
            _append_session_entry(ui_entry)
        except Exception as _e:
            _err(f"_handle_settled_analysis _append_session_entry failed: {_e}")

        # Persist to disk (async)
        try:
            write_journal_async(entry, max_entries=JOURNAL_MAX_ENTRIES)
            _log(f"journal: queued write (id={pid})")
        except Exception as _e:
            _err(f"_handle_settled_analysis write_journal_async failed: {_e}")

        # Broadcast structured 'analysis' event (already used by UI)
        try:
            broadcast_payload = {
                "analysis_event": "prediction_result",
                "prediction_id": pid,
                "symbol": market,
                "direction": direction,
                "trade_type": direction,
                "contract_type": contract_type,
                "prediction_digit": pred_digit,
                "result": entry.get("result"),
                "profit": entry.get("profit"),
                "profit_percentage": entry.get("profit_percentage"),
                "confidence": ap.get("confidence"),
                "reason": ap.get("reason"),
                "indicators": ap.get("indicators"),
                "ui_entry": ui_entry,
                "ts": int(time.time()),
            }
            _broadcast_analysis(broadcast_payload)
        except Exception:
            pass

        # Also emit a named SSE event 'prediction_result' (some clients listen for it)
        try:
            _enqueue_sse({
                "prediction_id": pid,
                "symbol": market,
                "direction": direction,
                "trade_type": direction,
                "contract_type": contract_type,
                "prediction_digit": pred_digit,
                "result": entry.get("result"),
                "profit": entry.get("profit"),
                "profit_percentage": entry.get("profit_percentage"),
                "confidence": ap.get("confidence"),
                "reason": ap.get("reason"),
                "indicators": ap.get("indicators"),
                "ui_entry": ui_entry,
                "ts": int(time.time()),
            }, event="prediction_result")
        except Exception:
            pass

        # --- BEST-EFFORT: refresh account balances for demo/real and broadcast balance_update ---
        try:
            for m in ("demo", "real"):
                mgr = _accounts.get(m)
                if mgr and mgr.authorized:
                    try:
                        bal_res = mgr.get_balance(timeout=6.0)
                        if isinstance(bal_res, dict) and bal_res.get("ok"):
                            try:
                                _broadcast_analysis({
                                    "analysis_event": "balance_update",
                                    "mode": m,
                                    "balance": bal_res.get("balance"),
                                    "raw": bal_res.get("raw")
                                })
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

    except Exception as e:
        _err(f"_handle_settled_analysis failed: {e}")



# ---------- SSE broadcaster loop (daemon thread) ----------
def _sse_broadcaster_loop():
    """Robust SSE broadcaster: drains _sse_out_q in small batches and pushes to client queues.

    This implementation deliberately catches and logs all errors so the thread does not die
    on unexpected input or transient client queue issues. It will remove misbehaving client
    queues to keep the broadcaster healthy.
    """
    _log("SSE broadcaster loop started")
    while True:
        try:
            # Wait for at least one message (blocks up to batch interval)
            payload = _sse_out_q.get(timeout=_SSE_BATCH_INTERVAL)
            batch = [payload]
            start = time.time()

            # Drain additional messages for the remainder of the batch interval up to max batch
            while (time.time() - start) < _SSE_BATCH_INTERVAL and len(batch) < _SSE_BATCH_MAX:
                try:
                    batch.append(_sse_out_q.get_nowait())
                except queue.Empty:
                    break

            # Broadcast the batch to every connected client queue inside a lock
            with _sse_lock:
                # iterate over a snapshot list to be safe against concurrent modifications
                for q in list(_sse_clients):
                    try:
                        # push each payload item non-blocking; if client's queue is full, drop the client
                        for p in batch:
                            try:
                                q.put_nowait(p)
                            except queue.Full:
                                _err("SSE broadcaster: client queue full -> removing client to avoid blocking")
                                try:
                                    _sse_clients.remove(q)
                                except Exception:
                                    pass
                                break
                    except Exception as e_client:
                        _err(f"SSE broadcaster: error while sending to client queue: {e_client}")
                        try:
                            _sse_clients.remove(q)
                        except Exception:
                            pass

        except queue.Empty:
            # Nothing arrived in the timeout window — continue waiting quietly
            continue
        except Exception as e:
            try:
                _err(f"_sse_broadcaster_loop exception: {e}\n{traceback.format_exc()}")
            except Exception:
                pass
            time.sleep(0.1)
            continue


# start broadcaster thread at startup (daemon)
try:
    t = threading.Thread(target=_sse_broadcaster_loop, name="hero_sse_broadcaster", daemon=True)
    t.start()
except Exception as e:
    _err(f"failed to start SSE broadcaster: {e}")

# load persisted tokens (so server restart keeps them)
try:
    _load_tokens()
    # Broadcast current token presence to connected UIs (masked only)
    try:
        _broadcast_tokens_status()
    except Exception as _e:
        _err(f"broadcast tokens status after load failed: {_e}")
except Exception as e:
    _err(f"failed to load persisted tokens: {e}")


# --- preload recent journal into session (non-fatal) ---
# By default we DO NOT preload persisted journal into the in-memory session so
# the dashboard starts with an empty journal on server restart (per user request).
# To opt back in set environment variable HERO_JOURNAL_PRELOAD=1
try:
    if os.environ.get("HERO_JOURNAL_PRELOAD", "") == "1":
        if os.path.exists(JOURNAL_FILE):
            try:
                with open(JOURNAL_FILE, "r", encoding="utf-8") as jf:
                    lines = [ln.strip() for ln in jf if ln.strip()]
            except Exception:
                lines = []
            for ln in lines[-200:]:
                try:
                    jo = json.loads(ln)
                except Exception:
                    continue
                try:
                    hist = {
                        "ts": jo.get("timestamp") or jo.get("ts") or datetime.now(timezone.utc).isoformat(),
                        "symbol": (jo.get("symbol") or jo.get("market") or "").upper(),
                        "result": jo.get("result", "UNKNOWN"),
                        "entry": jo,
                    }
                except Exception:
                    hist = {"ts": datetime.now(timezone.utc).isoformat(), "symbol": "", "result": jo.get("result", "UNKNOWN")}
                try:
                    SESSION_JOURNAL.append(hist)
                except Exception:
                    pass
except Exception as _e:
    try:
        _err(f"preload journal into session failed: {_e}")
    except Exception:
        pass


# ---------- Callback helper ----------
def _notify_callback(callback_url: str, payload: dict, token: Optional[str] = None, timeout: float = 6.0):
    """Schedule a best-effort POST JSON payload to callback_url in background."""
    if not callback_url:
        return
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-CALLBACK-TOKEN"] = token

    def _do_post():
        try:
            if requests:
                try:
                    r = requests.post(callback_url, json=payload, timeout=timeout, headers=headers)
                    _log_callback(f"_notify_callback POST {callback_url} -> {getattr(r, 'status_code', getattr(r, 'status', 'unknown'))}")
                except Exception as e:
                    _err_callback(f"_notify_callback requests failed to {callback_url}: {e}")
            else:
                data = json.dumps(payload, default=str).encode("utf-8")
                req = _urllib_request.Request(callback_url, data=data, headers=headers, method="POST")
                try:
                    ctx = _ssl.create_default_context()
                    with _urllib_request.urlopen(req, timeout=timeout, context=ctx) as resp:
                        code = getattr(resp, "status", None) or getattr(resp, "getcode", lambda: None)()
                        _log_callback(f"_notify_callback urllib -> {code}")
                except Exception as e:
                    _err_callback(f"_notify_callback urllib failed to {callback_url}: {e}")
        except Exception as e:
            _err_callback(f"_notify_callback final failure for {callback_url}: {e}")

    try:
        _executor.submit(_do_post)
    except Exception as e:
        _err_callback(f"_notify_callback submit failed: {e}")


# ---------- Helper: tmux utils ----------
def session_exists(name: str) -> bool:
    try:
        r = subprocess.run(["tmux", "ls"], capture_output=True, text=True, timeout=2)
        out = (r.stdout or "") + (r.stderr or "")
        return name in out
    except Exception:
        return False


def _spawn_tmux_session(sess_name: str, cmd_shell: str) -> None:
    """ Try to spawn a detached tmux session with the given shell command. """
    try:
        proc = subprocess.run(
            ["tmux", "new-session", "-d", "-s", sess_name, "bash", "-lc", cmd_shell],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
        )
        if proc.returncode == 0:
            _log(f"_spawn_tmux_session: tmux started session {sess_name}")
            return
        else:
            stderr = (proc.stderr or "").lower()
            stdout = (proc.stdout or "").lower()
            if "duplicate session" in stderr or "duplicate session" in stdout or "failed to create session" in stderr:
                _log(f"_spawn_tmux_session: session {sess_name} already exists (tmux reported duplicate). Treating as started.")
                return
            _err(f"_spawn_tmux_session: tmux returned code {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    except FileNotFoundError:
        _err(f"_spawn_tmux_session: tmux binary not found; falling back to subprocess.Popen for {sess_name}")
    except Exception as e:
        _err(f"_spawn_tmux_session tmux attempt error for {sess_name}: {e}")

    # Fallback: start via subprocess.Popen in background, redirect stdout/stderr to files (non-blocking)
    try:
        out_path = os.path.expanduser(f"~/.hero_logs/{sess_name}_fallback.out")
        err_path = os.path.expanduser(f"~/.hero_logs/{sess_name}_fallback.err")
        with open(out_path, "a") as outf, open(err_path, "a") as errf:
            pop = subprocess.Popen(cmd_shell, shell=True, stdout=outf, stderr=errf, preexec_fn=None)
            _log(f"_spawn_tmux_session: fallback Popen started for {sess_name} (pid={pop.pid})")
    except Exception as e:
        _err(f"_spawn_tmux_session fallback failed for {sess_name}: {e}")


def _project_base_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()


def _analysis_script_path() -> Optional[str]:
    env_script = (os.environ.get("HERO_ANALYSIS_SCRIPT") or "").strip()
    candidates: List[str] = []
    if env_script:
        if os.path.isabs(env_script):
            candidates.append(env_script)
        else:
            candidates.append(os.path.join(_project_base_dir(), env_script))
    candidates.extend(
        [
            os.path.join(_project_base_dir(), "differs_agent.py"),
            os.path.join(os.getcwd(), "differs_agent.py"),
            os.path.expanduser("~/HeroX/differs_agent.py"),
        ]
    )
    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            continue
    return None


def _read_analysis_pid() -> int:
    try:
        if not os.path.exists(ANALYSIS_PID_FILE):
            return 0
        with open(ANALYSIS_PID_FILE, "r", encoding="utf-8") as f:
            return int((f.read() or "0").strip() or "0")
    except Exception:
        return 0


def _write_analysis_pid(pid: int) -> None:
    try:
        with open(ANALYSIS_PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(int(pid)))
    except Exception as e:
        _err(f"_write_analysis_pid failed: {e}")


def _clear_analysis_pid() -> None:
    try:
        if os.path.exists(ANALYSIS_PID_FILE):
            os.remove(ANALYSIS_PID_FILE)
    except Exception:
        pass


def _is_pid_alive(pid: int) -> bool:
    try:
        pid = int(pid or 0)
        if pid <= 0:
            return False
        if os.name == "nt":
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            out = (proc.stdout or "").lower()
            return str(pid) in out and "no tasks are running" not in out
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _analysis_running() -> bool:
    global GLOBAL_ANALYSIS_PROC
    with GLOBAL_ANALYSIS_PROC_LOCK:
        try:
            if GLOBAL_ANALYSIS_PROC is not None and GLOBAL_ANALYSIS_PROC.poll() is None:
                return True
        except Exception:
            pass
    pid = _read_analysis_pid()
    if _is_pid_alive(pid):
        return True
    return session_exists("differs_agent")


def _derive_sse_url_from_push(push_url: str) -> str:
    try:
        u = _urllib_parse.urlparse(push_url)
        if not u.scheme or not u.netloc:
            raise RuntimeError("bad_push_url")
        return _urllib_parse.urlunparse((u.scheme, u.netloc, "/events", "", "", ""))
    except Exception:
        return os.environ.get("HERO_DASHBOARD_SSE_URL") or "http://127.0.0.1:5000/events"


def _start_analysis_process(push_url: str) -> Dict[str, Any]:
    global GLOBAL_ANALYSIS_PROC
    with GLOBAL_ANALYSIS_PROC_LOCK:
        if GLOBAL_ANALYSIS_PROC is not None:
            try:
                if GLOBAL_ANALYSIS_PROC.poll() is None:
                    return {"ok": True, "already_running": True, "pid": int(GLOBAL_ANALYSIS_PROC.pid)}
            except Exception:
                pass

        pid = _read_analysis_pid()
        if _is_pid_alive(pid):
            return {"ok": True, "already_running": True, "pid": int(pid)}

        script = _analysis_script_path()
        if not script:
            return {"ok": False, "error": "differs_agent.py_not_found"}

        python_exec = getattr(sys, "executable", None) or "python"
        env = dict(os.environ)
        env["HERO_DASHBOARD_PUSH_URL"] = str(push_url)
        env["HERO_DASHBOARD_SSE_URL"] = str(env.get("HERO_DASHBOARD_SSE_URL") or _derive_sse_url_from_push(push_url))

        out_path = os.path.join(LOG_DIR, "differs_agent.out")
        err_path = os.path.join(LOG_DIR, "differs_agent.err")
        cmd = [python_exec, script]
        popen_kwargs: Dict[str, Any] = {
            "cwd": os.path.dirname(script),
            "env": env,
            "stdout": None,
            "stderr": None,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True

        try:
            with open(out_path, "a", encoding="utf-8") as outf, open(err_path, "a", encoding="utf-8") as errf:
                popen_kwargs["stdout"] = outf
                popen_kwargs["stderr"] = errf
                proc = subprocess.Popen(cmd, **popen_kwargs)
            GLOBAL_ANALYSIS_PROC = proc
            _write_analysis_pid(int(proc.pid))
            _log(f"_start_analysis_process: started pid={proc.pid} script={script}")
            return {"ok": True, "pid": int(proc.pid), "script": script, "python": python_exec}
        except Exception as e:
            _err(f"_start_analysis_process failed: {e}")
            return {"ok": False, "error": str(e)}


def _stop_analysis_process() -> Dict[str, Any]:
    global GLOBAL_ANALYSIS_PROC
    notes: List[str] = []
    stopped = False
    with GLOBAL_ANALYSIS_PROC_LOCK:
        proc = GLOBAL_ANALYSIS_PROC
        GLOBAL_ANALYSIS_PROC = None

    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=6)
                except Exception:
                    proc.kill()
                stopped = True
        except Exception as e:
            notes.append(f"proc_terminate:{e}")

    pid = _read_analysis_pid()
    if pid and _is_pid_alive(pid):
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
            else:
                os.kill(int(pid), 15)
            stopped = True
        except Exception as e:
            notes.append(f"pid_kill:{e}")

    _clear_analysis_pid()
    return {"ok": True, "stopped": bool(stopped), "notes": notes}


# ---------- Minimal Deriv websocket fallback client (authorize + balance only) ----------
class SimpleDerivClient:
    def __init__(self, app_id: int = ENFORCED_APP_ID, url: Optional[str] = None, timeout: float = 8.0):
        self.app_id = int(app_id or ENFORCED_APP_ID)
        self.url = url or f"https://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        # prefer wss
        self.url = self.url.replace("http:", "wss:")
        self._ws = None
        self._recv_task = None
        self._lock = asyncio.Lock()
        self._req_id = 1
        self._pending: Dict[int, asyncio.Future] = {}
        self.timeout = timeout
        self._closed = False

    async def _connect(self):
        if self._ws:
            return
        try:
            import websockets
            from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
        except Exception as e:
            raise RuntimeError("missing dependency 'websockets'. Install with: pip install websockets") from e
        try:
            self._ws = await websockets.connect(self.url, max_size=None)
        except Exception as e:
            raise RuntimeError(f"websocket connect failed: {e}")
        self._recv_task = asyncio.create_task(self._receiver_loop())

    async def _receiver_loop(self):
        try:
            while True:
                try:
                    msg = await self._ws.recv()
                    if isinstance(msg, bytes):
                        try:
                            msg = msg.decode("utf-8")
                        except Exception:
                            continue
                    data = json.loads(msg)
                except asyncio.CancelledError:
                    break
                except Exception:
                    break
                req_id = None
                try:
                    if isinstance(data, dict) and "req_id" in data:
                        try:
                            req_id = int(data.get("req_id"))
                        except Exception:
                            req_id = None
                except Exception:
                    req_id = None
                if req_id is not None and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        fut.set_result(data)
                else:
                    for k, fut in list(self._pending.items()):
                        if not fut.done():
                            fut.set_result(data)
                        self._pending.pop(k, None)
                        break
        finally:
            for k, fut in list(self._pending.items()):
                if not fut.done():
                    fut.set_exception(RuntimeError("websocket receiver terminated"))
            self._pending.clear()
            self._closed = True
            try:
                if self._ws:
                    await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _send_recv(self, payload: dict, timeout: Optional[float] = None):
        timeout = timeout or self.timeout
        await self._connect()
        async with self._lock:
            req_id = self._req_id
            self._req_id += 1
            payload = dict(payload)
            payload["req_id"] = req_id
            pfut: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending[req_id] = pfut
            try:
                await self._ws.send(json.dumps(payload))
            except Exception as e:
                self._pending.pop(req_id, None)
                raise RuntimeError(f"send failed: {e}")
            try:
                res = await asyncio.wait_for(pfut, timeout=timeout)
                return res
            except asyncio.TimeoutError:
                self._pending.pop(req_id, None)
                raise RuntimeError("request timed out")
            except Exception as e:
                self._pending.pop(req_id, None)
                raise

    async def authorize(self, token: str):
        tok = (token or "").strip()
        if not tok:
            raise RuntimeError("missing token")
        req = {"authorize": tok}
        try:
            res = await self._send_recv(req, timeout=self.timeout)
            return res
        except Exception as e:
            raise RuntimeError(f"authorize call failed: {e}")

    async def balance(self):
        try:
            res = await self._send_recv({"balance": 1}, timeout=self.timeout)
            return res
        except Exception as e:
            raise RuntimeError(f"balance call failed: {e}")

    async def close(self):
        self._closed = True
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass
        if self._recv_task:
            try:
                self._recv_task.cancel()
            except Exception:
                pass

# ---------- Account manager (authorize + balance, no trade placement) ----------
class AccountManager:
    def __init__(self, mode: str):
        self.mode = mode  # 'real' or 'demo'
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.api = None
        self.authorized = False
        self.last_balance = None
        self.token = None
        self.app_id = None
        self.lock = threading.Lock()
        self._start_loop_thread()

    def _start_loop_thread(self):
        def _run():
            try:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()
            except Exception as e:
                _err(f"AccountManager loop thread error ({self.mode}): {e}")

        t = threading.Thread(target=_run, name=f"acctmgr-{self.mode}", daemon=True)
        t.start()
        self.thread = t
        for _ in range(50):
            if self.loop is not None:
                break
            time.sleep(0.02)

    async def _authorize_coro(self, token: str, app_id: int):
        tok = (token or "").strip()
        if not tok:
            raise RuntimeError("missing token")
        try:
            try:
                from deriv_api import DerivAPI  # type: ignore
            except Exception as e_import:
                raise e_import
            api = DerivAPI(app_id=app_id)
            try:
                auth_resp = await api.authorize(tok)
            except Exception as e_plain:
                try:
                    auth_resp = await api.authorize({"authorize": tok})
                except Exception as e_dict:
                    raise RuntimeError(f"authorize call failed: {e_plain}")
            try:
                bal = await api.balance()
            except Exception:
                bal = None
            with self.lock:
                self.api = api
                self.authorized = True
                self.token = token
                self.app_id = app_id
                self.last_balance = bal
            return {"auth": auth_resp, "balance": bal}
        except Exception as e_primary:
            # fallback to lightweight websocket client
            try:
                client = SimpleDerivClient(app_id=app_id)
                auth_resp = await client.authorize(tok)
                try:
                    bal = await client.balance()
                except Exception:
                    bal = None
                with self.lock:
                    self.api = client
                    self.authorized = True
                    self.token = token
                    self.app_id = app_id
                    self.last_balance = bal
                return {"auth": auth_resp, "balance": bal}
            except Exception as e_fallback:
                _err(f"AccountManager _authorize_coro: deriv_api import failed or call failed: {e_primary}; fallback failed: {e_fallback}")
                raise RuntimeError(f"deriv_api import failed: {e_primary}")

    def authorize(self, token: str, app_id: int, timeout: float = 14.0):
        if not self.loop:
            raise RuntimeError("account manager loop not ready")
        fut = asyncio.run_coroutine_threadsafe(self._authorize_coro(token, app_id), self.loop)
        try:
            res = fut.result(timeout=timeout)
            _log(f"AccountManager authorize success ({self.mode})")
            return {"ok": True, "result": res}
        except Exception as e:
            _err(f"AccountManager authorize failed ({self.mode}): {e}")
            return {"ok": False, "error": str(e)}

    async def _fetch_balance_coro(self):
        with self.lock:
            if not self.authorized or not self.api:
                raise RuntimeError("not authorized")
            api = self.api
        try:
            bal = await api.balance()
        except Exception as e:
            raise RuntimeError(f"balance call failed: {e}")
        with self.lock:
            self.last_balance = bal
        return bal

    def get_balance(self, timeout: float = 8.0):
        """
        Fetch balance using the manager's async loop. This function is defensive:
        - catches exceptions from the coroutine
        - returns cached last_balance if fetch fails
        - always returns a dict with either {"ok": True, ...} or {"ok": False, "error": "..."}
        """
        if not self.loop:
            return {"ok": False, "error": "manager loop not ready"}
        if not self.authorized:
            return {"ok": False, "error": "not_authorized"}

        # call the coroutine on this manager's loop
        try:
            fut = asyncio.run_coroutine_threadsafe(self._fetch_balance_coro(), self.loop)
            bal = fut.result(timeout=timeout)
        except Exception as e:
            # fetch failed: log and return cached snapshot if available
            _err(f"AccountManager get_balance ({self.mode}) exception: {e}")
            with self.lock:
                cached = self.last_balance
            if cached is not None:
                # try to build a minimal parsed structure from cached value
                parsed = {}
                try:
                    if isinstance(cached, dict):
                        if "balance" in cached:
                            parsed["balance"] = cached.get("balance")
                            parsed["currency"] = cached.get("currency") or cached.get("currency_code") or ""
                        elif isinstance(cached.get("balance"), dict):
                            parsed["balance"] = cached["balance"].get("balance")
                            parsed["currency"] = cached["balance"].get("currency")
                        else:
                            for k, v in cached.items():
                                if k.lower() in ("balance", "account_balance"):
                                    parsed["balance"] = v
                                if k.lower() in ("currency", "curr"):
                                    parsed["currency"] = v
                    elif isinstance(cached, (str, int, float)):
                        parsed["balance"] = str(cached)
                        parsed["currency"] = ""
                except Exception:
                    parsed = {}
                return {"ok": True, "balance": parsed or None, "raw": cached, "warning": "returned_cached_balance_due_to_fetch_error"}
            return {"ok": False, "error": str(e)}

        # Parse returned raw balance into a friendly shape (tolerant)
        parsed = {}
        try:
            if isinstance(bal, dict):
                # common shapes
                if "balance" in bal and not isinstance(bal.get("balance"), dict):
                    parsed["balance"] = bal.get("balance")
                    parsed["currency"] = bal.get("currency") or bal.get("currency_code") or ""
                elif isinstance(bal.get("balance"), dict):
                    parsed["balance"] = bal["balance"].get("balance")
                    parsed["currency"] = bal["balance"].get("currency")
                else:
                    for k, v in bal.items():
                        if k.lower() in ("balance", "account_balance"):
                            parsed["balance"] = v
                        if k.lower() in ("currency", "curr"):
                            parsed["currency"] = v
            elif isinstance(bal, (str, int, float)):
                parsed["balance"] = str(bal)
                parsed["currency"] = ""
        except Exception as e:
            _err(f"AccountManager get_balance parse failed ({self.mode}): {e}")
            parsed = {}

        # store last raw balance snapshot
        try:
            with self.lock:
                self.last_balance = bal
        except Exception:
            pass

        # Broadcast a balance_update event, but swallow any errors
        try:
            _broadcast_analysis({
                "analysis_event": "balance_update",
                "mode": self.mode,
                "balance": parsed,
                "raw": bal,
            })
        except Exception as _e:
            _err(f"balance_update broadcast failed ({self.mode}): {_e}")

        return {"ok": True, "balance": parsed or {}, "raw": bal}

    async def _close_coro(self):
        with self.lock:
            api = self.api
            self.api = None
            self.authorized = False
            self.token = None
            self.app_id = None
            self.last_balance = None
        if api:
            try:
                if hasattr(api, "close"):
                    await api.close()
            except Exception:
                pass

    def disconnect(self, timeout: float = 6.0):
        if not self.loop:
            return {"ok": False, "error": "loop not ready"}
        fut = asyncio.run_coroutine_threadsafe(self._close_coro(), self.loop)
        try:
            fut.result(timeout=timeout)
        except Exception as e:
            _err(f"AccountManager disconnect error: {e}")
        return {"ok": True}

    def shutdown(self, timeout: float = 3.0):
        """
        Attempt to cancel pending tasks on this manager's event loop and stop it cleanly.
        Called on process shutdown to avoid 'Task was destroyed but it is pending!' warnings.
        """
        if not self.loop:
            return {"ok": False, "error": "loop not ready"}

        async def _cancel_all():
            # cancel all tasks on this manager's loop
            try:
                to_cancel = [t for t in asyncio.all_tasks(loop=self.loop) if not t.done()]
            except Exception:
                to_cancel = []
            for t in to_cancel:
                try:
                    t.cancel()
                except Exception:
                    pass
            if to_cancel:
                # wait a short while for cancellations to propagate
                await asyncio.gather(*to_cancel, return_exceptions=True)

        try:
            fut = asyncio.run_coroutine_threadsafe(_cancel_all(), self.loop)
            try:
                fut.result(timeout=timeout)
            except Exception:
                # ignore: we'll still stop the loop
                pass
            # then stop the loop
            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception:
                pass
            # join thread reasonably
            try:
                if self.thread and self.thread.is_alive():
                    self.thread.join(timeout)
            except Exception:
                pass
            return {"ok": True}
        except Exception as e:
            _err(f"AccountManager.shutdown error ({self.mode}): {e}")
            return {"ok": False, "error": str(e)}


_accounts: Dict[str, AccountManager] = {"real": AccountManager("real"), "demo": AccountManager("demo")}

# -------------------------
# Prediction lightweight logger (records produced predictions and notifies UI)
# -------------------------
_predictions_produced_count = 0
_predictions_by_symbol: Dict[str, int] = defaultdict(int)
_predictions_lock = threading.Lock()

def _make_prediction_id():
    import uuid, time
    return f"pred_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"


def _first_present(obj: dict, keys: List[str]) -> Any:
    for k in keys:
        if k in obj and obj.get(k) is not None:
            return obj.get(k)
    return None


def add_prediction_log(payload: dict) -> dict:
    """
    Lightweight logging for produced predictions.
    - No pending state or settlement logic.
    - Broadcasts 'prediction_posted', 'prediction_toast' and 'prediction_stats' via SSE.
    - Persists an entry to the journal and session-journal.
    Returns dict: {"ok": True, "prediction_id": pid, "produced_count": n}
    """
    global _predictions_produced_count
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "error": "invalid payload"}

        sym = (payload.get("symbol") or payload.get("market") or "").strip().upper()
        if not sym:
            return {"ok": False, "error": "missing symbol"}

        # normalize prediction digit
        raw_digit = _first_present(payload, ["prediction_digit", "predicted", "pred", "digit"])
        try:
            pred_digit = int(raw_digit) if raw_digit is not None else None
        except Exception:
            pred_digit = None

        pid = payload.get("prediction_id") or payload.get("pred_id") or _make_prediction_id()

        confidence = None
        try:
            if "confidence" in payload:
                confidence = float(payload.get("confidence"))
        except Exception:
            confidence = None

        ts_now = int(time.time())

        entry = {
            "timestamp": ts_now,
            "ts": ts_now,
            "symbol": sym,
            "prediction_digit": pred_digit,
            "confidence": confidence,
            "stake": payload.get("stake"),
            "amount": payload.get("amount"),
            "raw": payload.get("raw"),
        }

        # update counts
        with _predictions_lock:
            _predictions_produced_count += 1
            _predictions_by_symbol[sym] += 1
            produced_count = _predictions_produced_count

        # Determine server-side stake fallback (if none provided in payload)
        chosen_stake = None
        try:
            # payload may include 'mode' or 'account'; prefer that, else use default_mode
            mode_hint = (payload.get("mode") or payload.get("account") or GLOBAL_AUTOTRADE_SETTINGS.get("default_mode") or "demo")
            mode_hint = mode_hint.lower() if isinstance(mode_hint, str) else mode_hint
            if payload.get("stake") is not None:
                try:
                    chosen_stake = float(payload.get("stake"))
                except Exception:
                    chosen_stake = None
            if chosen_stake is None:
                # pick server-side configured stake for this mode if present
                mset = GLOBAL_AUTOTRADE_SETTINGS.get(mode_hint) or {}
                try:
                    if mset.get("stake") is not None:
                        chosen_stake = float(mset.get("stake"))
                except Exception:
                    chosen_stake = None
        except Exception:
            chosen_stake = payload.get("stake")

        # Broadcast the prediction event (so UI shows it and can toast)
        try:
            _broadcast_analysis({
                "analysis_event": "prediction_posted",
                "prediction_id": pid,
                "market": sym,
                "symbol": sym,
                "prediction_digit": pred_digit,
                "confidence": confidence,
                "stake": chosen_stake if chosen_stake is not None else payload.get("stake"),
                "amount": chosen_stake if chosen_stake is not None else payload.get("amount"),
                "timestamp": ts_now,
                "message": f"Prediction produced for {sym}: {pred_digit}"
            })
            _broadcast_analysis({
                "analysis_event": "prediction_toast",
                "prediction_id": pid,
                "symbol": sym,
                "market": sym,
                "digit": pred_digit,
                "stake": chosen_stake if chosen_stake is not None else payload.get("stake"),
                "amount": chosen_stake if chosen_stake is not None else payload.get("amount"),
                "confidence": confidence,
                "status": "produced",
                "message": f"Prediction: {sym} → {pred_digit}"
            })


            # publish stats so UI can show count
            _broadcast_analysis({
                "analysis_event": "prediction_stats",
                "produced_count": produced_count,
                "symbol_count": _predictions_by_symbol.get(sym, 0)
            })
        except Exception:
            pass

        # persist to session only (do NOT write PRODUCED to disk — keep settled-only journaling)
        try:
            _append_session_entry({
                "timestamp": ts_now,
                "symbol": sym,
                "result": "PRODUCED",
                "entry": entry
            })
        except Exception:
            pass


        return {"ok": True, "prediction_id": pid, "produced_count": produced_count}

    except Exception as e:
        _err(f"add_prediction_log error: {e}")
        return {"ok": False, "error": str(e)}



@app.route("/events")
def events():
    def gen(q: queue.Queue):
        try:
            yield 'event: open\ndata: {"ts": %d}\n\n' % int(time.time())
            with _recent_lock:
                snapshot_rows = [r for r in list(_recent_ticks)[-150:] if (len(r) < 7 or (str(r[6] or "").upper() != "ANALYSIS"))]
            if snapshot_rows:
                try:
                    yield f'event: recent\ndata: {json.dumps({"recent": snapshot_rows})}\n\n'
                except Exception:
                    pass
            last_ping = time.time()
            while True:
                try:
                    item = q.get(timeout=1.0)
                    parsed = json.loads(item)
                    ev_name = parsed.get("event")
                    payload = parsed.get("payload")
                    if ev_name:
                        yield f'event: {ev_name}\ndata: {json.dumps(payload)}\n\n'
                    else:
                        yield f'data: {json.dumps({"payload": payload})}\n\n'
                except queue.Empty:
                    now = time.time()
                    if now - last_ping > SSE_PING_INTERVAL:
                        yield 'event: ping\ndata: {}\n\n'
                        last_ping = now
                except GeneratorExit:
                    break
        finally:
            with _sse_lock:
                try:
                    _sse_clients.remove(q)
                except Exception:
                    pass

    q = queue.Queue(maxsize=500)
    with _sse_lock:
        _sse_clients.append(q)
    return Response(gen(q), content_type="text/event-stream")


# endpoint to post produced predictions
@app.route("/control/push_prediction", methods=["POST"])
def push_prediction():
    """
    Accept prediction from analysis agents or UI.
    This registers it immediately as 'produced' (no pending state).
    """
    try:
        obj = request.get_json(force=True)
    except Exception as e:
        _err(f"push_prediction: invalid json: {e}")
        return jsonify({"ok": False, "error": "invalid json"}), 400
    if not isinstance(obj, dict):
        return jsonify({"ok": False, "error": "expected object"}), 400

    res = add_prediction_log(obj)
    if not res.get("ok"):
        _err(f"push_prediction rejected: {res.get('error')}")
        return jsonify(res), 400
    return jsonify(res)


@app.route("/control/prediction_stats")
def control_prediction_stats():
    """ Return in-memory produced count and per-symbol counts. """
    try:
        with _predictions_lock:
            total = int(_predictions_produced_count)
            by_symbol = dict(_predictions_by_symbol)
        return jsonify({"ok": True, "produced_count": total, "by_symbol": by_symbol})
    except Exception as e:
        _err(f"control_prediction_stats failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


class OUStrategyEngine:
    """
    In-process Over/Under research engine focused on:
    - OVER 0 (DIGITOVER barrier 0)
    - OVER 1 (DIGITOVER barrier 1)
    - UNDER 9 (DIGITUNDER barrier 9)
    - UNDER 8 (DIGITUNDER barrier 8)
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.settings_file = os.path.join(LOG_DIR, "ou_strategy_settings.json")

        self.enabled = _truthy(os.environ.get("HERO_OU_ENABLED", "1"))
        self.auto_predict = _truthy(os.environ.get("HERO_OU_AUTO_PREDICT", "1"))
        self.window_size = max(200, int(os.environ.get("HERO_OU_WINDOW_SIZE", "5000")))
        self.min_samples = max(100, int(os.environ.get("HERO_OU_MIN_SAMPLES", "1200")))
        self.delta = max(0.0, float(os.environ.get("HERO_OU_DELTA", "0.002")))
        self.z_score = max(0.0, float(os.environ.get("HERO_OU_Z", "1.96")))
        self.analyze_interval_sec = max(0.2, float(os.environ.get("HERO_OU_ANALYZE_INTERVAL_SEC", "1.0")))
        self.predict_cooldown_sec = max(1.0, float(os.environ.get("HERO_OU_PREDICT_COOLDOWN_SEC", "12.0")))
        self.max_signal_history = max(20, int(os.environ.get("HERO_OU_SIGNAL_HISTORY_MAX", "300")))

        focus_env = os.environ.get(
            "HERO_OU_FOCUS_SYMBOLS",
            "1HZ10V,1HZ25V,1HZ50V,1HZ75V,1HZ100V,R_10,R_25,R_50,R_75,R_100",
        )
        self.focus_symbols = self._parse_focus_symbols(focus_env)

        # Total return multipliers (stake-inclusive). Can be adjusted at runtime.
        self.total_return = {
            "OVER_0": max(1.0, float(os.environ.get("HERO_OU_RETURN_OVER_0", "1.0989"))),
            "UNDER_9": max(1.0, float(os.environ.get("HERO_OU_RETURN_UNDER_9", "1.0989"))),
            "OVER_1": max(1.0, float(os.environ.get("HERO_OU_RETURN_OVER_1", "1.2346"))),
            "UNDER_8": max(1.0, float(os.environ.get("HERO_OU_RETURN_UNDER_8", "1.2346"))),
        }

        self.buffers: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=self.window_size))
        self.last_analysis_ts: Dict[str, float] = {}
        self.last_prediction_ts: Dict[str, float] = {}
        self.latest_by_symbol: Dict[str, Dict[str, Any]] = {}
        self.signals: Deque[Dict[str, Any]] = deque(maxlen=self.max_signal_history)

        self.total_ticks = 0
        self.total_analyses = 0
        self.total_signals = 0

        self._load_settings()
        _log(
            "OU engine initialized "
            f"(enabled={self.enabled} auto_predict={self.auto_predict} window={self.window_size} "
            f"min_samples={self.min_samples} focus_symbols={len(self.focus_symbols) if self.focus_symbols else 'ALL'})"
        )

    def _parse_focus_symbols(self, raw: Any) -> set:
        if raw is None:
            return set()
        if isinstance(raw, (list, tuple, set)):
            parts = [str(x or "").strip().upper() for x in raw]
        else:
            txt = str(raw or "").replace("|", ",")
            parts = [x.strip().upper() for x in txt.split(",")]
        return {x for x in parts if x}

    def _resize_buffers_locked(self, new_window_size: int) -> None:
        old_items = {k: list(v) for k, v in self.buffers.items()}
        self.window_size = max(200, int(new_window_size))
        self.buffers = defaultdict(lambda: deque(maxlen=self.window_size))
        for sym, arr in old_items.items():
            self.buffers[sym] = deque(arr[-self.window_size :], maxlen=self.window_size)

    def _load_settings(self) -> None:
        try:
            if not os.path.exists(self.settings_file):
                return
            with open(self.settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            self.apply_settings(data, persist=False)
            _log("OU engine settings loaded from disk")
        except Exception as e:
            _err(f"OU load settings failed: {e}")

    def _save_settings(self) -> None:
        try:
            payload = self.settings_snapshot(include_runtime=False)
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True)
        except Exception as e:
            _err(f"OU save settings failed: {e}")

    def _candidate_rows(self, counts: Counter, n: int) -> List[Dict[str, Any]]:
        if n <= 0:
            return []
        c0 = int(counts.get(0, 0))
        c1 = int(counts.get(1, 0))
        c8 = int(counts.get(8, 0))
        c9 = int(counts.get(9, 0))

        defs = [
            ("OVER_0", "DIGITOVER", "0", max(0, n - c0)),
            ("UNDER_9", "DIGITUNDER", "9", max(0, n - c9)),
            ("OVER_1", "DIGITOVER", "1", max(0, n - c0 - c1)),
            ("UNDER_8", "DIGITUNDER", "8", max(0, n - c8 - c9)),
        ]
        out: List[Dict[str, Any]] = []
        for name, ctype, barrier, wins in defs:
            p_hat = float(wins) / float(n)
            total_return = float(self.total_return.get(name) or 1.0)
            p_be = (1.0 / total_return) if total_return > 0 else 1.0
            se = math.sqrt(max(0.0, p_hat * (1.0 - p_hat) / float(n)))
            lower = max(0.0, p_hat - self.z_score * se)
            ev = (p_hat * total_return) - 1.0
            pass_edge = (
                n >= self.min_samples
                and p_hat >= (p_be + self.delta)
                and lower >= p_be
                and ev > 0.0
            )
            out.append(
                {
                    "name": name,
                    "contract_type": ctype,
                    "barrier": barrier,
                    "wins": int(wins),
                    "samples": int(n),
                    "p_hat": round(p_hat, 6),
                    "p_be": round(p_be, 6),
                    "se": round(se, 6),
                    "lower_ci95": round(lower, 6),
                    "ev_per_stake": round(ev, 6),
                    "pass_edge": bool(pass_edge),
                    "total_return": round(total_return, 6),
                }
            )
        out.sort(key=lambda x: (x.get("pass_edge"), x.get("ev_per_stake", -9.0), x.get("lower_ci95", 0.0)), reverse=True)
        return out

    def _build_symbol_snapshot(self, symbol: str, epoch: int, digits: List[int]) -> Dict[str, Any]:
        n = len(digits)
        counts = Counter(digits)
        rows = self._candidate_rows(counts, n)
        best = rows[0] if rows else None
        return {
            "ts": int(time.time()),
            "epoch": int(epoch or 0),
            "symbol": str(symbol or "").upper(),
            "samples": int(n),
            "eligible": bool(n >= self.min_samples),
            "best": best,
            "candidates": rows,
            "counts": {str(k): int(v) for k, v in sorted(counts.items())},
        }

    def _signal_row_from_snapshot(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            best = snapshot.get("best") or {}
            if not best.get("pass_edge"):
                return None
            return {
                "ts": int(time.time()),
                "epoch": int(snapshot.get("epoch") or 0),
                "symbol": str(snapshot.get("symbol") or "").upper(),
                "contract": str(best.get("name") or ""),
                "contract_type": str(best.get("contract_type") or ""),
                "barrier": str(best.get("barrier") or ""),
                "samples": int(best.get("samples") or snapshot.get("samples") or 0),
                "p_hat": float(best.get("p_hat") or 0.0),
                "p_be": float(best.get("p_be") or 0.0),
                "lower_ci95": float(best.get("lower_ci95") or 0.0),
                "ev_per_stake": float(best.get("ev_per_stake") or 0.0),
            }
        except Exception:
            return None

    def _emit_signal(self, signal_row: Dict[str, Any]) -> None:
        sym = str(signal_row.get("symbol") or "").upper()
        now = time.time()
        with self._lock:
            if (now - float(self.last_prediction_ts.get(sym) or 0.0)) < float(self.predict_cooldown_sec):
                return
            self.last_prediction_ts[sym] = now

        contract = str(signal_row.get("contract") or "")
        ctype = str(signal_row.get("contract_type") or "")
        barrier = str(signal_row.get("barrier") or "")
        direction = "over" if ctype == "DIGITOVER" else "under"
        prediction_payload = {
            "symbol": sym,
            "prediction_digit": int(barrier) if barrier.isdigit() else None,
            "prediction_mode": "ou_focus",
            "contract_type": ctype,
            "barrier": barrier,
            "ou_contract": contract,
            "ou_direction": direction,
            "confidence": float(signal_row.get("p_hat") or 0.0),
            "message": (
                f"OU signal {contract} on {sym}: p={signal_row.get('p_hat'):.4f}, "
                f"p_be={signal_row.get('p_be'):.4f}, ev={signal_row.get('ev_per_stake'):.4f}"
            ),
            "raw": {"ou_signal": signal_row},
        }
        res = add_prediction_log(prediction_payload)
        signal_row["prediction_id"] = res.get("prediction_id") if isinstance(res, dict) else None
        signal_row["status"] = "posted" if isinstance(res, dict) and res.get("ok") else "error"
        signal_row["error"] = None if signal_row["status"] == "posted" else (res.get("error") if isinstance(res, dict) else "unknown")

        with self._lock:
            self.total_signals += 1
            self.signals.appendleft(dict(signal_row))

        try:
            _broadcast_analysis({"analysis_event": "ou_signal", "signal": signal_row})
        except Exception:
            pass

    def ingest_tick(self, symbol: str, last_decimal: Any, epoch: int) -> None:
        sym = str(symbol or "").strip().upper()
        if not sym:
            return
        d = _safe_digit(last_decimal)
        if d is None:
            return

        now = time.time()
        with self._lock:
            self.total_ticks += 1
            if self.focus_symbols and sym not in self.focus_symbols:
                return

            self.buffers[sym].append(int(d))
            if not self.enabled:
                return
            last_ts = float(self.last_analysis_ts.get(sym) or 0.0)
            if (now - last_ts) < float(self.analyze_interval_sec):
                return
            self.last_analysis_ts[sym] = now
            digits = list(self.buffers[sym])

        snapshot = self._build_symbol_snapshot(sym, int(epoch or now), digits)
        signal = self._signal_row_from_snapshot(snapshot)
        with self._lock:
            self.latest_by_symbol[sym] = snapshot
            self.total_analyses += 1

        try:
            _broadcast_analysis(
                {
                    "analysis_event": "ou_analysis",
                    "symbol": sym,
                    "samples": snapshot.get("samples"),
                    "eligible": snapshot.get("eligible"),
                    "best": snapshot.get("best"),
                    "counts": snapshot.get("counts"),
                }
            )
        except Exception:
            pass

        if self.auto_predict and signal:
            self._emit_signal(signal)

    def settings_snapshot(self, include_runtime: bool = True) -> Dict[str, Any]:
        with self._lock:
            out = {
                "enabled": bool(self.enabled),
                "auto_predict": bool(self.auto_predict),
                "window_size": int(self.window_size),
                "min_samples": int(self.min_samples),
                "delta": float(self.delta),
                "z_score": float(self.z_score),
                "analyze_interval_sec": float(self.analyze_interval_sec),
                "predict_cooldown_sec": float(self.predict_cooldown_sec),
                "focus_symbols": sorted(list(self.focus_symbols)),
                "total_return": dict(self.total_return),
            }
            if include_runtime:
                out.update(
                    {
                        "total_ticks": int(self.total_ticks),
                        "total_analyses": int(self.total_analyses),
                        "total_signals": int(self.total_signals),
                    }
                )
            return out

    def status_snapshot(self, limit_symbols: int = 12, limit_signals: int = 30) -> Dict[str, Any]:
        with self._lock:
            sym_rows = sorted(
                list(self.latest_by_symbol.values()),
                key=lambda x: int(x.get("ts") or 0),
                reverse=True,
            )[: max(1, int(limit_symbols))]
            sig_rows = list(self.signals)[: max(1, int(limit_signals))]
        return {
            "engine": "ou_focus",
            "settings": self.settings_snapshot(include_runtime=True),
            "symbols": sym_rows,
            "signals": sig_rows,
        }

    def apply_settings(self, obj: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            return self.settings_snapshot(include_runtime=True)

        with self._lock:
            if "enabled" in obj:
                v = obj.get("enabled")
                self.enabled = v if isinstance(v, bool) else _truthy(v)
            if "auto_predict" in obj:
                v = obj.get("auto_predict")
                self.auto_predict = v if isinstance(v, bool) else _truthy(v)
            if "window_size" in obj:
                try:
                    new_window = max(200, int(obj.get("window_size")))
                    if new_window != self.window_size:
                        self._resize_buffers_locked(new_window)
                except Exception:
                    pass
            if "min_samples" in obj:
                try:
                    self.min_samples = max(100, int(obj.get("min_samples")))
                except Exception:
                    pass
            if "delta" in obj:
                try:
                    self.delta = max(0.0, float(obj.get("delta")))
                except Exception:
                    pass
            if "z_score" in obj:
                try:
                    self.z_score = max(0.0, float(obj.get("z_score")))
                except Exception:
                    pass
            if "analyze_interval_sec" in obj:
                try:
                    self.analyze_interval_sec = max(0.2, float(obj.get("analyze_interval_sec")))
                except Exception:
                    pass
            if "predict_cooldown_sec" in obj:
                try:
                    self.predict_cooldown_sec = max(1.0, float(obj.get("predict_cooldown_sec")))
                except Exception:
                    pass
            if "focus_symbols" in obj:
                self.focus_symbols = self._parse_focus_symbols(obj.get("focus_symbols"))
            if "total_return" in obj and isinstance(obj.get("total_return"), dict):
                for k in ("OVER_0", "UNDER_9", "OVER_1", "UNDER_8"):
                    if k in obj["total_return"]:
                        try:
                            self.total_return[k] = max(1.0, float(obj["total_return"][k]))
                        except Exception:
                            pass
            if obj.get("reset_runtime"):
                self.latest_by_symbol = {}
                self.signals = deque(maxlen=self.max_signal_history)
                self.last_analysis_ts = {}
                self.last_prediction_ts = {}
                self.total_ticks = 0
                self.total_analyses = 0
                self.total_signals = 0
                self.buffers = defaultdict(lambda: deque(maxlen=self.window_size))

        if persist:
            self._save_settings()
        snap = self.status_snapshot()
        try:
            _broadcast_analysis({"analysis_event": "ou_status", "ou": snap})
        except Exception:
            pass
        return snap


OU_ENGINE = OUStrategyEngine()


@app.route("/control/ou_status", methods=["GET"])
def control_ou_status():
    try:
        lim_s = int(request.args.get("symbols", "12"))
    except Exception:
        lim_s = 12
    try:
        lim_sig = int(request.args.get("signals", "30"))
    except Exception:
        lim_sig = 30
    try:
        return jsonify({"ok": True, "ou": OU_ENGINE.status_snapshot(limit_symbols=lim_s, limit_signals=lim_sig)})
    except Exception as e:
        _err(f"control_ou_status error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/control/ou_settings", methods=["POST"])
def control_ou_settings():
    try:
        obj = request.get_json(force=True)
    except Exception as e:
        _err(f"control_ou_settings invalid json: {e}")
        return jsonify({"ok": False, "error": "invalid json"}), 400
    if not isinstance(obj, dict):
        return jsonify({"ok": False, "error": "expected object"}), 400
    try:
        snap = OU_ENGINE.apply_settings(obj, persist=True)
        return jsonify({"ok": True, "ou": snap})
    except Exception as e:
        _err(f"control_ou_settings error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/control/ou_signals", methods=["GET"])
def control_ou_signals():
    try:
        lim_sig = int(request.args.get("limit", "50"))
    except Exception:
        lim_sig = 50
    try:
        snap = OU_ENGINE.status_snapshot(limit_symbols=1, limit_signals=lim_sig)
        return jsonify({"ok": True, "signals": snap.get("signals") or []})
    except Exception as e:
        _err(f"control_ou_signals error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500




# ---------- endpoints ----------
@app.route("/")
def index():
    return render_template("index.html", admin_required=bool(ADMIN_TOKEN))


@app.route("/control/connect_account", methods=["POST"])
def control_connect_account():
    try:
        obj = request.get_json(force=True)
    except Exception as e:
        _err(f"connect_account invalid json: {e}")
        return jsonify({"ok": False, "error": "invalid json"}), 400
    if not isinstance(obj, dict):
        return jsonify({"ok": False, "error": "expected object"}), 400
    try:
        client_app_id = int(obj.get("app_id") or 0)
    except Exception:
        client_app_id = 0
    token = obj.get("token") or obj.get("api_token") or obj.get("auth_token")
    mode = (obj.get("mode") or "demo").lower()
    if mode not in ("demo", "real"):
        return jsonify({"ok": False, "error": "invalid mode"}), 400
    if not token:
        return jsonify({"ok": False, "error": "missing token"}), 400
    app_id_used = ENFORCED_APP_ID
    mgr = _accounts.get(mode)
    if not mgr:
        return jsonify({"ok": False, "error": "server internal: no manager"}), 500

    _log(f"connect_account called mode={mode} client_app_id={client_app_id} -> enforcing app_id={app_id_used}")
    resp = mgr.authorize(token, app_id_used, timeout=14.0)
    if not resp.get("ok"):
        _err(f"connect_account failed mode={mode} err={resp.get('error')}")
        _broadcast_analysis({"analysis_event": "account_connect_failed", "mode": mode, "message": resp.get("error")})
        return jsonify({"ok": False, "error": resp.get("error")}), 500

    # STORE token server-side (in-memory) for the specified mode
    try:
        GLOBAL_DERIV_TOKENS[mode] = token
        _log(f"control_connect_account: stored token for mode={mode} (masked={_mask_token(token)})")
        _save_tokens()
    except Exception as e:
        _err(f"control_connect_account: failed to store token for mode={mode}: {e}")

    # Broadcast to UI that account is connected and which mode has a server-side token (masked)
    try:
        _broadcast_analysis(
            {
                "analysis_event": "account_connected",
                "mode": mode,
                "message": f"{mode} account connected",
                "balance": resp.get("result", {}).get("balance"),
                "server_token_masked": _mask_token(token),
            }
        )
    except Exception:
        pass

    return jsonify(
        {
            "ok": True,
            "mode": mode,
            "balance": resp.get("result", {}).get("balance"),
            "app_id_enforced": app_id_used,
            "server_token_masked": _mask_token(token),
        }
    )


@app.route("/control/get_server_tokens", methods=["GET"])
def control_get_server_tokens():
    """Return which modes have server-side tokens (masked only)."""
    try:
        tokens = {
            m: {"present": bool(GLOBAL_DERIV_TOKENS.get(m)), "masked": _mask_token(GLOBAL_DERIV_TOKENS.get(m))}
            for m in ("demo", "real")
        }
        return jsonify({"ok": True, "tokens": tokens})
    except Exception as e:
        _err(f"control_get_server_tokens error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/control/get_server_token", methods=["GET"])
def control_get_server_token():
    """
    Return the full server-side token for a mode **only** to local requests or to
    callers presenting the ADMIN_TOKEN header (if configured). Query param: ?mode=demo|real
    """
    try:
        mode = (request.args.get("mode") or "demo").lower()
        if mode not in ("demo", "real"):
            return jsonify({"ok": False, "error": "invalid mode"}), 400
        # allow only local callers by default
        if not _is_local_request(request):
            # if ADMIN_TOKEN configured, allow header-based access
            if ADMIN_TOKEN:
                hdr = request.headers.get("X-ADMIN-TOKEN")
                if hdr != ADMIN_TOKEN:
                    abort(403, description="forbidden")
            else:
                abort(403, description="forbidden")
        tok = GLOBAL_DERIV_TOKENS.get(mode)
        if not tok:
            return jsonify({"ok": True, "token": None, "present": False})
        return jsonify({"ok": True, "token": tok, "present": True})
    except Exception as e:
        _err(f"control_get_server_token error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/control/set_autotrade_settings", methods=["POST"])
def control_set_autotrade_settings():
    """
    Persist simple autotrade settings (mode, stake) server-side so
    the analysis/trader daemon or add_prediction_log can include them.
    Body: { "mode": "demo"|"real", "stake": 0.35 }
    """
    try:
        obj = request.get_json(force=True)
    except Exception as e:
        _err(f"set_autotrade_settings invalid json: {e}")
        return jsonify({"ok": False, "error": "invalid json"}), 400

    if not isinstance(obj, dict):
        return jsonify({"ok": False, "error": "expected object"}), 400

    mode = (obj.get("mode") or "demo").lower()
    if mode not in ("demo", "real"):
        return jsonify({"ok": False, "error": "invalid mode"}), 400

    stake = obj.get("stake")
    try:
        if stake is not None:
            stake = float(stake)
            if stake < 0.0:
                raise ValueError("negative")
    except Exception:
        return jsonify({"ok": False, "error": "invalid stake"}), 400

    try:
        GLOBAL_AUTOTRADE_SETTINGS.setdefault(mode, {})["stake"] = stake
        # option: set default_mode if requested
        if obj.get("make_default"):
            GLOBAL_AUTOTRADE_SETTINGS["default_mode"] = mode
        _log(f"set_autotrade_settings: mode={mode} stake={stake}")
        # broadcast to UI so connected clients see the change
        try:
            _broadcast_analysis({"analysis_event": "autotrade_settings_updated", "mode": mode, "stake": stake})
        except Exception:
            pass
    except Exception as e:
        _err(f"control_set_autotrade_settings failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True, "mode": mode, "stake": stake})

@app.route("/control/get_autotrade_settings", methods=["GET"])
def control_get_autotrade_settings():
    try:
        return jsonify({"ok": True, "settings": GLOBAL_AUTOTRADE_SETTINGS})
    except Exception as e:
        _err(f"control_get_autotrade_settings error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/control/get_balances")
def control_get_balances():
    """
    Return a friendly JSON for mode=demo|real.
    This handler is defensive and will not return HTTP 500 for transient errors
    from the AccountManager; instead it returns a well-formed JSON with a
    cached value or a friendly error message.
    """
    mode = (request.args.get("mode") or "demo").lower()
    if mode not in ("demo", "real"):
        return jsonify({"ok": False, "error": "invalid mode"}), 400

    mgr = _accounts.get(mode)
    if not mgr:
        return jsonify({"ok": False, "error": "no manager"}), 500

    # If not authorized, return quickly with authorized=false (200)
    if not mgr.authorized:
        return jsonify({"ok": True, "authorized": False, "balance": None, "raw": None})

    # Attempt to fetch live balance but never allow an uncaught exception to produce a 500
    try:
        res = mgr.get_balance(timeout=8.0)
    except Exception as e:
        _err(f"control_get_balances exception calling manager.get_balance: {e}")
        # try to return cached snapshot if possible
        try:
            with mgr.lock:
                cached = mgr.last_balance
            parsed = {}
            if cached is not None:
                if isinstance(cached, dict) and ("balance" in cached or isinstance(cached.get("balance"), (dict, int, float, str))):
                    # best-effort normalized view
                    parsed = {"balance": cached.get("balance") if isinstance(cached, dict) else str(cached), "currency": cached.get("currency") if isinstance(cached, dict) else ""}
                else:
                    parsed = {"balance": str(cached)}
            return jsonify({"ok": True, "authorized": mgr.authorized, "balance": parsed or None, "raw": cached, "warning": "fetch_failed_returned_cached"}), 200
        except Exception:
            return jsonify({"ok": False, "error": str(e)}), 500

    # mgr.get_balance returned a structured response
    if not isinstance(res, dict) or not res.get("ok"):
        # manager returned error — don't escalate to 500; return cached or an informative response
        _err(f"control_get_balances: manager returned error: {res}")
        with mgr.lock:
            cached = mgr.last_balance
        parsed = res.get("balance") if isinstance(res, dict) else None
        if not parsed and cached is not None:
            try:
                parsed = {"balance": cached.get("balance") if isinstance(cached, dict) else str(cached)}
            except Exception:
                parsed = None
        return jsonify({"ok": True, "authorized": mgr.authorized, "balance": parsed or None, "raw": cached, "warning": "manager_error"}), 200

    # success
    bal = res.get("balance") or {}
    return jsonify({"ok": True, "authorized": True, "balance": bal, "raw": res.get("raw")})

@app.route("/control/disconnect_account", methods=["POST"])
def control_disconnect_account():
    try:
        obj = request.get_json(force=True)
    except Exception:
        obj = request.form or {}
    mode = (obj.get("mode") or "demo").lower()
    if mode not in ("demo", "real"):
        return jsonify({"ok": False, "error": "invalid mode"}), 400
    mgr = _accounts.get(mode)
    if not mgr:
        return jsonify({"ok": False, "error": "no manager"}), 500
    resp = mgr.disconnect()
    _broadcast_analysis({"analysis_event": "account_disconnected", "mode": mode, "message": f"{mode} disconnected"})
    return jsonify({"ok": True, "mode": mode})


@app.route("/control/journal")
def control_journal():
    """ Return recent journal entries. """
    limit_q = request.args.get("limit", "20")
    source = request.args.get("source", "").lower()
    # if 'file' then read disk
    try:
        limit = int(limit_q)
    except Exception:
        limit = 20
    if source == "file":
        entries = []
        try:
            if os.path.exists(JOURNAL_FILE):
                with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            entries.append(obj)
                        except Exception:
                            continue
            # return most-recent-first
            entries = list(reversed(entries))[-limit:]
            entries = list(reversed(entries))
            return jsonify({"ok": True, "entries": entries})
        except Exception as e:
            _err(f"control_journal read error (file): {e}")
            return jsonify({"ok": False, "error": "journal read error"}), 500

    # Default: return session in-memory newest-first
    try:
        results = SESSION_JOURNAL[:limit]
        return jsonify({"ok": True, "entries": results})
    except Exception as e:
        _err(f"control_journal read error (session): {e}")
        return jsonify({"ok": False, "error": "journal read error"}), 500

@app.route("/control/clear_journal", methods=["POST"])
def control_clear_journal():
    """Clear the in-memory session journal and delete the persisted journal file (best-effort)."""
    try:
        with _JOURNAL_LOCK:
            global SESSION_JOURNAL
            SESSION_JOURNAL = []
            try:
                if os.path.exists(JOURNAL_FILE):
                    os.remove(JOURNAL_FILE)
            except Exception:
                pass
        _log("control_clear_journal: cleared session and persisted journal")
        # also broadcast a status so UI can react
        try:
            _broadcast_analysis({"analysis_event": "journal_cleared", "message": "journal cleared by user"})
        except Exception:
            pass
        return jsonify({"ok": True})
    except Exception as e:
        _err(f"control_clear_journal error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

# -------------------------
# New endpoint: place_trade
# -------------------------
@app.route("/control/place_trade", methods=["POST"])
def control_place_trade():
    """
    Accept a trade placement request from UI (autotrade panel).
    Normalizes stake/amount, validates min stake, emits an immediate 'prediction_posted' (and toast)
    and returns an ack containing prediction_id and echoed stake/amount.
    Default stake used for quick tests: 1.0 (if none provided).
    """
    try:
        obj = request.get_json(force=True)
    except Exception as e:
        _err(f"control_place_trade: invalid json: {e}")
        return jsonify({"ok": False, "error": "invalid json"}), 400

    if not isinstance(obj, dict):
        return jsonify({"ok": False, "error": "expected object"}), 400

    # Normalise
    try:
        stake_raw = obj.get("stake", None)
        if stake_raw is None:
            stake_raw = obj.get("amount", None)
        stake = None
        if stake_raw is not None:
            try:
                stake = float(stake_raw)
            except Exception:
                stake = None
        # Default stake for quick tests if none provided
        if stake is None:
            stake = 1.0
        # enforce minimum stake (server side)
        MIN_STAKE = 0.35
        if stake is None or stake < MIN_STAKE:
            return jsonify({"ok": False, "error": "stake_too_small", "min_stake": MIN_STAKE}), 400

        # symbol / prediction digit normalization
        symbol = (obj.get("symbol") or obj.get("market") or "").strip().upper()
        pred_digit = _first_present(obj, ["prediction_digit", "predicted", "pred", "digit"])
        try:
            if pred_digit is not None:
                pred_digit = int(pred_digit)
        except Exception:
            pred_digit = None

        # ensure we have a stable prediction id
        pid = obj.get("prediction_id") or obj.get("pred_id") or obj.get("id") or None
        if not pid:
            pid = _make_prediction_id()

        # Build a normalized payload for add_prediction_log (so it will broadcast)
        payload = {
            "prediction_id": pid,
            "symbol": symbol or "",
            "prediction_digit": pred_digit,
            "stake": stake,
            "amount": stake,
            "confidence": obj.get("confidence") or obj.get("conf") or None,
            "raw": obj,
        }

        # Log as produced prediction (this broadcasts 'prediction_posted' and 'prediction_toast')
        res = add_prediction_log(payload)
        if not res.get("ok"):
            _err(f"control_place_trade: add_prediction_log rejected: {res.get('error')}")
            return jsonify({"ok": False, "error": "rejected", "detail": res.get("error")}), 500

        return jsonify({"ok": True, "prediction_id": pid, "stake": stake, "amount": stake}), 200
    except Exception as e:
        _err(f"control_place_trade: exception: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500



@app.route('/analysis_panel.html')
def analysis_panel_html():
    return render_template('analysis_panel.html')


@app.route("/charts")
def charts_page():
    # Render the charts template (templates/charts.html)
    return render_template("charts.html")


@app.route("/details")
def details():
    with _recent_lock:
        tail = [
            r
            for r in list(_recent_ticks)[-RECENT_TICKS_MAX:]
            if (len(r) < 7 or (str(r[6] or "").upper() != "ANALYSIS"))
        ]
    return jsonify({"recent_ticks": [], "refresh_token": "", "should_close": False, "ticks_tail": tail})

# ---------- utility helpers ----------
def _is_local_request(req) -> bool:
    try:
        ra = (req.remote_addr or "").strip()
        return ra in ("127.0.0.1", "::1", "localhost")
    except Exception:
        return False


def _get_primary_ip() -> str:
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        try:
            s.close()
        except Exception:
            pass
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        try:
            if s:
                s.close()
        except Exception:
            pass
    try:
        hn_ip = socket.gethostbyname(socket.gethostname())
        if hn_ip and not hn_ip.startswith("127."):
            return hn_ip
    except Exception:
        pass
    try:
        addrs = socket.getaddrinfo(socket.gethostname(), None)
        for a in addrs:
            cand = a[4][0]
            if cand and isinstance(cand, str) and not cand.startswith("127."):
                return cand
    except Exception:
        pass
    return "127.0.0.1"


def _dashboard_push_url_from_request(req):
    env_override = os.environ.get("HERO_DASHBOARD_PUSH_URL")
    if env_override:
        return env_override
    try:
        port = (req.host or "").split(":")[1] if ":" in (req.host or "") else os.environ.get("HERO_PORT", "5000")
    except Exception:
        port = os.environ.get("HERO_PORT", "5000")
    ip = _get_primary_ip()
    if os.environ.get("HERO_FORCE_LOCALHOST", "") == "1":
        ip = "127.0.0.1"
    else:
        bind_host = os.environ.get("HERO_BIND") or ""
        if bind_host == "0.0.0.0":
            ip = "127.0.0.1"
    return f"http://{ip}:{port}/control/push_tick"


# --- Helper to normalize symbols from the request ---
def _parse_requested_symbols(req_body: dict) -> List[str]:
    syms: List[str] = []
    if not req_body:
        return syms
    if isinstance(req_body.get("symbols"), list):
        syms = [str(s).strip().upper() for s in req_body.get("symbols") if str(s).strip()]
    elif isinstance(req_body.get("symbols"), str) and req_body.get("symbols").strip():
        syms = [s.strip().upper() for s in req_body.get("symbols").split(",") if s.strip()]
    elif isinstance(req_body.get("symbols_csv"), str) and req_body.get("symbols_csv").strip():
        syms = [s.strip().upper() for s in req_body.get("symbols_csv").split(",") if s.strip()]
    elif isinstance(req_body.get("symbols_str"), str) and req_body.get("symbols_str").strip():
        syms = [s.strip().upper() for s in req_body.get("symbols_str").split(",") if s.strip()]
    return syms


@app.route("/control/get_token", methods=["GET"])
def control_get_token():
    """
    Backwards-compatible endpoint. Return whether demo/real tokens are present and masked.
    Prefer the more complete /control/get_tokens endpoint, but keep this for older clients.
    """
    try:
        demo = GLOBAL_DERIV_TOKENS.get("demo")
        real = GLOBAL_DERIV_TOKENS.get("real")
        return jsonify({
            "ok": True,
            "demo_present": bool(demo),
            "real_present": bool(real),
            "demo_mask": _mask_token(demo),
            "real_mask": _mask_token(real),
        })
    except Exception as e:
        _err(f"control_get_token error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/control/get_tokens", methods=["GET"])
def control_get_tokens():
    """
    Backwards-compatible alias for older clients that call /control/get_tokens.
    Reuses control_get_token() which returns the expected fields:
      {"ok": True, "demo_present": bool, "real_present": bool, "demo_mask": "...", "real_mask": "..."}
    """
    try:
        return control_get_token()
    except Exception as e:
        _err(f"control_get_tokens shim error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/control/start_deriv", methods=["POST"])
def control_start_deriv():
    """
    Start the deriv worker. Expects JSON body optionally containing: { "symbols": ["R_10","R_25"] }
    This handler is more tolerant: if worker_manager reports 'already running', we attempt a
    graceful stop and then try to start again (useful after hero_service restart or stale sessions).

    IMPORTANT SAFETY / FIX: calling worker_manager.start_worker can block in some setups.
    To avoid HTTP handler hanging, attempt a quick (short) wait for start_worker to return.
    If it doesn't return fast, schedule the start in the background and reply quickly with 202 accepted.
    Also build the push_url using _dashboard_push_url_from_request to avoid push_url mismatches.
    """
    try:
        body = request.get_json(silent=True) or {}
        symbols = body.get("symbols") or body.get("symbolsArray") or body.get("symbols_array") or []
        if isinstance(symbols, str):
            symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        elif isinstance(symbols, (list, tuple)):
            symbols = [str(s).strip().upper() for s in symbols if s]
        else:
            symbols = []

        # Build robust push_url using helper (this avoids many "no ticks" issues)
        push_url = _dashboard_push_url_from_request(request)

        # Attempt to start the worker but do not block the HTTP thread for long.
        def _call_start_worker():
            try:
                return worker_manager.start_worker(symbols=symbols, push_url=push_url)
            except Exception as e:
                return {"ok": False, "error": f"start_worker threw: {e}"}

        # Submit start to threadpool and wait a short time for synchronous completion
        try:
            fut = _executor.submit(_call_start_worker)
            try:
                res = fut.result(timeout=5.0)  # quick wait: if worker_manager blocks, we fallback to background
            except FuturesTimeoutError:
                # schedule continue in background (if not already executing)
                _log("control_start_deriv: start_worker call timed out after 5s — continuing in background")
                def _bg_wait_and_log():
                    try:
                        rbg = fut.result(timeout=60) if fut else None
                        _log(f"control_start_deriv: background start_worker result: {rbg}")
                    except Exception as e:
                        _err(f"control_start_deriv: background start_worker error: {e}")
                try:
                    _executor.submit(_bg_wait_and_log)
                except Exception:
                    pass
                # respond quickly to the client to avoid hanging UI
                return jsonify({"ok": True, "starting": True, "note": "start_worker running in background, UI will update via SSE"}), 202
        except Exception as e:
            _err(f"control_start_deriv: failed to submit start_worker: {e}")
            # fallback: try direct call (best-effort)
            try:
                res = worker_manager.start_worker(symbols=symbols, push_url=push_url)
            except Exception as e2:
                _err(f"control_start_deriv fallback start_worker failed: {e2}")
                res = {"ok": False, "error": str(e2)}

        # If start succeeded, return
        if res.get("ok"):
            _log("control_start_deriv: worker started ok")
            try:
                _broadcast_analysis({"analysis_event": "deriv_started", "message": "deriv worker started (control_start_deriv)"})
            except Exception:
                pass
            return jsonify(res)

        # If start failed but message indicates it's already running, attempt a safe restart
        err_msg = (res.get("error") or res.get("message") or "").lower()
        if "already" in err_msg or "running" in err_msg:
            _log("control_start_deriv: worker reported already running -> attempting stop & restart")
            try:
                stop_res = worker_manager.stop_worker(timeout=5.0)
                _log(f"control_start_deriv: stop_worker attempted -> {stop_res}")
            except Exception as e_stop:
                _err(f"control_start_deriv: stop_worker attempt failed: {e_stop}")

            # small pause to give tmux/pids a chance to clean up
            time.sleep(0.35)

            try:
                res2 = worker_manager.start_worker(symbols=symbols, push_url=push_url)
            except Exception as e2:
                _err(f"control_start_deriv: restart attempt threw: {e2}")
                res2 = {"ok": False, "error": str(e2)}

            if res2.get("ok"):
                _log("control_start_deriv: worker restarted ok after stop attempt")
                try:
                    _broadcast_analysis({"analysis_event": "deriv_restarted", "message": "deriv worker restarted (control_start_deriv)"})
                except Exception:
                    pass
                return jsonify({"ok": True, "restarted": True, **res2})

            # still failed after restart attempt — return helpful error (include both messages if available)
            combined_err = {
                "ok": False,
                "error": "start_worker reported already-running and restart attempt failed",
                "start_error": res.get("error") or res.get("message"),
                "restart_error": res2.get("error") if isinstance(res2, dict) else str(res2)
            }
            _err(f"control_start_deriv: restart failed -> {combined_err}")
            return jsonify(combined_err), 500

        # Generic failure (not 'already running')
        _err(f"control_start_deriv: start_worker returned error: {res}")
        return jsonify(res), 500

    except Exception as e:
        _err(f"control_start_deriv exception: {e}")
        return jsonify({"ok": False, "error": f"exception in start handler: {e}"}), 500


@app.route("/control/stop_deriv", methods=["POST"])
def control_stop_deriv():
    """ Stop the deriv worker (reads pidfile). Returns JSON about stop status. """
    try:
        res = worker_manager.stop_worker(timeout=5.0)
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"exception in stop handler: {e}"}), 500


@app.route('/control/start_ticks', methods=['POST'])
def start_ticks():
    """
    Start a worker but only for the provided symbols.
    Accepts JSON { "symbols": ["RDBEAR", ...], "worker_script": "optional_override.py", "extra_args": [] }
    This implementation ensures we pass symbols as a single CSV via extra_args to avoid
    worker scripts that expect --symbols "A,B,C".
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        symbols = body.get('symbols') or body.get('symbol') or []
        if isinstance(symbols, str):
            symbols_list = [symbols]
        elif isinstance(symbols, (list, tuple)):
            symbols_list = [s for s in symbols if s]
        else:
            symbols_list = []

        symbols_list = [str(s).strip().upper() for s in symbols_list if s]

        worker_script = body.get('worker_script')  # optional override
        extra_args = body.get('extra_args') or body.get('extraArgs') or ""
        extra_args_str = str(extra_args).strip() if extra_args else ""

        # build a robust push_url for the worker to call back to
        push_url = _dashboard_push_url_from_request(request)

        # prefer passing typed values: symbols as list and push_url explicitly
        # this avoids string-tokenization bugs and ensures --push-url is always passed
        res = worker_manager.start_worker(
            symbols=symbols_list or None,
            extra_args=extra_args_str or None,
            push_url=push_url,
            worker_script=worker_script
        )
        return jsonify(res)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/control/stop_ticks', methods=['POST'])
def stop_ticks():
    """
    Stop the worker started for tick streaming.
    This simply stops the PID in hero_worker.pid (same as stop_deriv) — you can make it target-specific if your architecture supports per-symbol pids.
    """
    try:
        res = worker_manager.stop_worker()
        return jsonify(res)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500



@app.route("/control/start_analysis", methods=["POST"])
def control_start_analysis():
    """
    Start differs_agent.py in the background (cross-platform, no separate manual command).
    """
    try:
        _check_admin_token(request)
    except Exception as e:
        _err(f"start_analysis admin token check failed: {e}")
        return jsonify({"ok": False, "error": "forbidden"}), 403

    try:
        global GLOBAL_ANALYSIS_LAST_STOP_TS
        if GLOBAL_ANALYSIS_LAST_STOP_TS and (time.time() - GLOBAL_ANALYSIS_LAST_STOP_TS) < 2.0:
            _log("control_start_analysis: refused start due to recent manual stop (grace window)")
            return jsonify({"ok": False, "error": "recently_stopped"}), 409
    except Exception:
        pass

    push_url = _dashboard_push_url_from_request(request)
    _log(f"start_analysis computed push_url={push_url}")

    try:
        res = _start_analysis_process(push_url)
        if not res.get("ok"):
            msg = str(res.get("error") or "analysis_start_failed")
            _err(f"start_analysis failed: {msg}")
            try:
                _broadcast_analysis({"analysis_event": "analysis_start_failed", "message": msg})
            except Exception:
                pass
            return jsonify({"ok": False, "error": msg}), 500

        _log(f"start_analysis: pid={res.get('pid')} already_running={bool(res.get('already_running'))}")
        try:
            _broadcast_analysis(
                {
                    "analysis_event": "analysis_started",
                    "message": "analysis agent started",
                    "pid": res.get("pid"),
                    "already_running": bool(res.get("already_running")),
                }
            )
        except Exception:
            pass

        return jsonify(
            {
                "ok": True,
                "session": "differs_agent",
                "push_url": push_url,
                "pid": res.get("pid"),
                "already_running": bool(res.get("already_running")),
                "script": res.get("script"),
                "python": res.get("python"),
            }
        )
    except Exception as e:
        _err(f"start_analysis failed: {e}")
        try:
            _broadcast_analysis({"analysis_event": "analysis_start_failed", "message": str(e)})
        except Exception:
            pass
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/control/stop_analysis", methods=["POST"])
def control_stop_analysis():
    _check_admin_token(request)
    sess = "differs_agent"
    errors = []
    try:
        stop_res = _stop_analysis_process()
        if isinstance(stop_res, dict):
            errors.extend(stop_res.get("notes") or [])

        # best-effort legacy cleanup for tmux/linux deployments
        try:
            subprocess.run(["tmux", "kill-session", "-t", sess], check=False)
        except Exception as e:
            errors.append(f"tmux_kill: {e}")

        gone = not _analysis_running()

        _log(f"stop_analysis requested; session {sess} attempted kill; verified_gone={gone}")
        try:
            _broadcast_analysis({"analysis_event": "analysis_stopped", "message": "analysis agent stopped by user", "verified_gone": gone})
        except Exception:
            pass

        # record manual stop time so accidental immediate restarts can be detected
        try:
            global GLOBAL_ANALYSIS_LAST_STOP_TS
            GLOBAL_ANALYSIS_LAST_STOP_TS = time.time()
        except Exception:
            pass

    except Exception as e:
        _err(f"stop_analysis failed: {e}")
        errors.append(str(e))
        return jsonify({"ok": False, "error": str(e), "notes": errors}), 500

    return jsonify({"ok": True, "session": sess, "verified_gone": gone, "notes": errors})

# NOTE: prediction endpoints removed per user request


# ---------- push_tick endpoint (prediction logic REMOVED) ----------
@app.route("/control/push_tick", methods=["POST"])
def push_tick():
    """
    Centralized push endpoint for ticks and analysis payloads.
    This implementation:
     - normalizes incoming payload
     - broadcasts analysis events when present
     - appends ticks to recent buffer
     - calls _handle_settled_analysis(...) when heuristics detect a final/settled result
    """
    global _last_tick_time
    try:
        obj = request.get_json(force=True)
    except Exception as e:
        _err(f"push_tick: invalid json: {e}")
        return jsonify({"ok": False, "error": "invalid json"}), 400
    if not isinstance(obj, dict):
        _err("push_tick: expected JSON object")
        return jsonify({"ok": False, "error": "expected object"}), 400

    # epoch is optional; server ordering is by seq
    epoch = None
    for k in ("epoch", "ts", "timestamp"):
        if k in obj:
            try:
                epoch = int(obj[k])
            except Exception:
                try:
                    epoch = int(float(obj[k]))
                except Exception:
                    epoch = None
            break
    if epoch is None:
        epoch = int(time.time())

    price_val = obj.get("price")
    price_str_text = obj.get("price_str_text")
    last_decimal, last_unit = _derive_tick_digits(
        obj.get("last_decimal"),
        obj.get("last_unit"),
        price_val,
        price_str_text,
    )

    # Normalize payload (strip prediction-related fields)
    payload = {
        "epoch": epoch,
        "symbol": obj.get("symbol"),
        "price": price_val,
        "price_str_text": price_str_text,
        "last_decimal": last_decimal,
        "last_unit": last_unit,
        "reason": obj.get("reason"),
        "__raw": obj,
    }

    # If this is an embedded analysis payload: obj.payload with analysis_event
    try:
        if isinstance(obj.get("payload"), dict) and obj["payload"].get("analysis_event"):
            analysis_payload = dict(obj["payload"])
            if "analysis_event" not in analysis_payload and "event" in analysis_payload:
                analysis_payload["analysis_event"] = analysis_payload.pop("event")
            if "buffer_len" not in analysis_payload:
                if "buffer" in analysis_payload:
                    analysis_payload["buffer_len"] = analysis_payload.pop("buffer")
                elif "buffer_len" in obj:
                    analysis_payload["buffer_len"] = obj.get("buffer_len")
            analysis_payload["epoch"] = epoch
            analysis_payload = {k: v for k, v in analysis_payload.items() if v is not None}
            if any(k.startswith("meta_") for k in obj.keys()):
                analysis_payload["__meta"] = {k: v for k, v in obj.items() if k.startswith("meta_")}

            # Broadcast analysis and handle settled if final
            _broadcast_analysis(analysis_payload)
            try:
                if _is_final_analysis(analysis_payload):
                    _handle_settled_analysis(analysis_payload)
            except Exception:
                pass

            _log(f"push_tick (embedded analysis) epoch={epoch} event={analysis_payload.get('analysis_event')}")
            return jsonify({"ok": True, "analysis": True})
    except Exception:
        pass

    # If symbol explicitly "ANALYSIS" (some workers send this), handle similarly
    sym = (payload.get("symbol") or "").upper()
    if sym == "ANALYSIS":
        try:
            analysis_payload = {k: v for k, v in obj.items() if k != "symbol" and k != "__raw"}
            if "analysis_event" not in analysis_payload and "event" in analysis_payload:
                analysis_payload["analysis_event"] = analysis_payload.pop("event")
            if "buffer_len" not in analysis_payload:
                if "buffer" in analysis_payload:
                    analysis_payload["buffer_len"] = analysis_payload.pop("buffer")
                elif "buffer_len" in obj:
                    analysis_payload["buffer_len"] = obj.get("buffer_len")
            analysis_payload["epoch"] = epoch
            analysis_payload = {k: v for k, v in analysis_payload.items() if v is not None}
            if any(k.startswith("meta_") for k in obj.keys()):
                analysis_payload["__meta"] = {k: v for k, v in obj.items() if k.startswith("meta_")}

            _broadcast_analysis(analysis_payload)
            try:
                if _is_final_analysis(analysis_payload):
                    _handle_settled_analysis(analysis_payload)
            except Exception:
                pass

            _log(f"push_tick (analysis) epoch={epoch} event={analysis_payload.get('analysis_event')}")
            return jsonify({"ok": True, "analysis": True})
        except Exception as e:
            _err(f"push_tick (analysis) failed building payload: {e}")
            return jsonify({"ok": False, "error": "analysis build error"}), 500

    # --- server authoritative seq handling for normal ticks ---
    with _recent_lock:
        cur_seq = _market_seq.get((payload.get("symbol") or "").upper(), 0) + 1
        _market_seq[(payload.get("symbol") or "").upper()] = cur_seq

    payload["seq"] = cur_seq

    # record recent tick row (legacy format preserved)
    row = [
        str(payload["epoch"]),
        str(payload["price"]) if payload["price"] is not None else "",
        str(payload.get("price_str_text") or ""),
        str(payload.get("price") or ""),
        str(payload.get("last_decimal") if payload.get("last_decimal") is not None else ""),
        str(payload.get("last_unit") if payload.get("last_unit") is not None else ""),
        str(payload.get("symbol") or ""),
    ]

    with _recent_lock:
        _recent_ticks.append(row)

    try:
        with _last_tick_lock:
            _last_tick_time = time.time()
    except Exception:
        pass

    # Run in-process Over/Under focused analysis.
    try:
        OU_ENGINE.ingest_tick(payload.get("symbol"), payload.get("last_decimal"), int(epoch or 0))
    except Exception as e:
        _err(f"OU ingest_tick failed: {e}")

    # broadcast tick (payload now includes seq and enriched reason/indicators)
    _broadcast_tick(_enrich_tick_payload(payload))

    # If a normal tick contains final/settled info, detect and handle
    try:
        incoming = payload or {}
        if isinstance(incoming, dict) and _is_final_analysis(incoming):
            try:
                _handle_settled_analysis(incoming)
            except Exception as _e:
                _err(f"push_tick: _handle_settled_analysis failed for incoming: {_e}")
    except Exception as _outer:
        _err(f"push_tick pre-journal check failed: {_outer}")

    _log(f"push_tick symbol={payload.get('symbol')} epoch={epoch} seq={cur_seq}")
    return jsonify({"ok": True})


# ---------- monitor thread ----------
def _monitor_loop():
    global _last_tick_time
    _log("monitor thread started")
    while True:
        try:
            deriv_here = session_exists("hero_worker")
            analysis_here = _analysis_running() or session_exists("differs_agent")

            if deriv_here and not _monitor_state["deriv_prev"]:
                _log("monitor: deriv session appeared")
                _broadcast_analysis({"analysis_event": "deriv_started", "message": "deriv worker started (monitor)"})
            if not deriv_here and _monitor_state["deriv_prev"]:
                _log("monitor: deriv session disappeared")
                _broadcast_analysis({"analysis_event": "deriv_stopped", "message": "deriv worker not running (monitor)"})
            if analysis_here and not _monitor_state["analysis_prev"]:
                _log("monitor: analysis session appeared")
                _broadcast_analysis({"analysis_event": "analysis_started", "message": "analysis agent started (monitor)"})
            if not analysis_here and _monitor_state["analysis_prev"]:
                _log("monitor: analysis session disappeared")
                _broadcast_analysis({"analysis_event": "analysis_stopped", "message": "analysis agent not running (monitor)"})

            _monitor_state["deriv_prev"] = deriv_here
            _monitor_state["analysis_prev"] = analysis_here

            try:
                with _last_tick_lock:
                    last = _last_tick_time
                now = time.time()
                if deriv_here:
                    if last == 0.0 or (now - last) > TICK_STALE_THRESHOLD:
                        if not _monitor_state["network_alerted"]:
                            _monitor_state["network_alerted"] = True
                            msg = f"No ticks received for {int(now - last) if last else '>'}s — poor network or worker issue"
                            _log("monitor: " + msg)
                            _broadcast_analysis({"analysis_event": "network_issue", "message": msg, "last_tick": last})
                    else:
                        if _monitor_state["network_alerted"]:
                            _monitor_state["network_alerted"] = False
                            _log("monitor: ticks resumed -> network_restored")
                            _broadcast_analysis({"analysis_event": "network_restored", "message": "Ticks flow resumed"})
                else:
                    if _monitor_state["network_alerted"]:
                        _monitor_state["network_alerted"] = False
            except Exception as e:
                _err(f"monitor tick-check error: {e}")

        except Exception as e:
            _err(f"monitor loop top-level error: {e}")
        finally:
            time.sleep(MONITOR_INTERVAL)


# --- Compatibility endpoint: worker may post to /push (older worker) ---
@app.route("/push", methods=["POST"])
def push_proxy():
    """ Backwards-compatible endpoint so workers that post to /push (instead of /control/push_tick) still work. """
    try:
        return push_tick()
    except Exception as e:
        _err(f"/push proxy error: {e}")
        return jsonify({"ok": False, "error": "push proxy error"}), 500

import atexit
import signal

def _shutdown_all_account_managers(timeout=3.0):
    try:
        _log("shutdown: stopping account managers")
    except Exception:
        pass
    for name, mgr in list(_accounts.items()):
        try:
            try:
                mgr.disconnect(timeout=2.0)
            except Exception:
                pass
            try:
                mgr.shutdown(timeout=timeout)
            except Exception:
                pass
        except Exception as e:
            _err(f"shutdown_all: manager {name} shutdown error: {e}")

def _signal_handler(signum, frame):
    try:
        _log(f"received signal {signum} - graceful shutdown")
    except Exception:
        pass
    _shutdown_all_account_managers(timeout=3.0)
    # ensure process exits after graceful shutdown
    try:
        sys.exit(0)
    except Exception:
        try:
            os._exit(0)
        except Exception:
            pass

# register handlers
try:
    atexit.register(_shutdown_all_account_managers)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except Exception:
    pass

# ---------- startup ----------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("HERO_BIND", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("HERO_PORT", "5000")))
    args = p.parse_args()
    host = args.host
    port = args.port
    _log(f"starting hero_service on {host}:{port} (admin_required={bool(ADMIN_TOKEN)})")

    try:
        # --- start trader tmux session automatically (differ_trade_check.py) ---
        try:
            sess_trader = "differ_trader"
            base_dir = _project_base_dir()
            candidate_trader = "differs_trade_check.py"  # NOTE: keep original file name; you used differs_trade_check.py earlier — ensure correct name on disk
            # Allow fallback names in case of naming mismatch
            candidate_trader_alts = ["differ_trade_check.py", "differs_trade_check.py", "differ_trade_check_differ.py"]
            chosen_trader = None
            for c in [candidate_trader] + candidate_trader_alts:
                if os.path.exists(os.path.join(base_dir, c)):
                    chosen_trader = c
                    break
            if chosen_trader is None:
                legacy_base = os.path.expanduser("~/HeroX")
                for c in [candidate_trader] + candidate_trader_alts:
                    if os.path.exists(os.path.join(legacy_base, c)):
                        base_dir = legacy_base
                        chosen_trader = c
                        break

            if chosen_trader is None:
                # fallback to the earlier name user provided (differ_trade_check.py)
                chosen_trader = "differ_trade_check.py"

            trader_args = "--daemon"  # we will run in tmux so it keeps running
            script_path = os.path.join(base_dir, chosen_trader)

            # choose python executable similar to analysis startup
            try:
                venv_python = os.path.expanduser("~/HeroX/venv/bin/python")
                if os.path.exists(venv_python) and os.access(venv_python, os.X_OK):
                    python_exec = venv_python
                else:
                    python_exec = getattr(sys, "executable", None) or "python3"
            except Exception:
                python_exec = "python3"

            # build push URL and SSE env for child so it can post to the dashboard
            push_url = f"http://127.0.0.1:{port}/control/push_tick"
            sse_url = os.environ.get("HERO_DASHBOARD_SSE_URL") or f"http://127.0.0.1:{port}/events"

            safe_push = str(push_url).replace("'", "'\"'\"'")
            safe_sse = str(sse_url).replace("'", "'\"'\"'")

            # Shell command: cd to base, export dashboard vars, exec script and redirect logs
            cmd_shell_trader = (
                f"cd {shlex.quote(base_dir)} && "
                f"export HERO_DASHBOARD_PUSH_URL='{safe_push}' && "
                f"export HERO_DASHBOARD_SSE_URL='{safe_sse}' && "
                f"sleep 0.2 && "
                f"exec {shlex.quote(python_exec)} {shlex.quote(chosen_trader)} {trader_args} "
                f"> ~/.hero_logs/differ_trade_check_differ.log 2> ~/.hero_logs/differ_trade_check_differ.err"
            )

            # If tmux session already exists, leave it (idempotent)
            if not session_exists(sess_trader):
                _spawn_tmux_session(sess_trader, cmd_shell_trader)
                _log(f"auto-started trader session {sess_trader} (script={chosen_trader})")
                try:
                    _broadcast_analysis({"analysis_event": "trader_started", "message": "trader daemon started", "session": sess_trader})
                except Exception:
                    pass
            else:
                _log(f"trader session {sess_trader} already exists (idempotent)")
        except Exception as e:
            _err(f"auto-start trader spawn failed: {e}")

        # start monitor thread (daemon)
        try:
            t = threading.Thread(target=_monitor_loop, name="herox_monitor", daemon=True)
            t.start()
        except Exception as e:
            _err(f"failed to start monitor thread: {e}")

        # Auto-start differs agent so only hero_service.py needs to be launched.
        try:
            if str(os.environ.get("HERO_AUTO_START_ANALYSIS", "1")).strip().lower() in ("1", "true", "yes", "on"):
                push_url = os.environ.get("HERO_DASHBOARD_PUSH_URL") or f"http://127.0.0.1:{port}/control/push_tick"
                res = _start_analysis_process(push_url)
                if not res.get("ok"):
                    _err(f"auto-start analysis failed: {res.get('error')}")
                else:
                    _log(f"auto-start analysis ok: pid={res.get('pid')} already_running={bool(res.get('already_running'))}")
        except Exception as e:
            _err(f"auto-start analysis exception: {e}")

        # Attempt to auto-start HL trade daemon (optional) — safe guarded
        try:
            start_trade_daemon_once()
        except Exception as e:
            _err(f"start_trade_daemon_once invocation failed: {e}")

        # Run Flask without the reloader (reloader can spawn extra processes which interfere with signals)
        app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        _log("hero_service: KeyboardInterrupt received - shutting down")
    except Exception as e:
        _err(f"hero_service main exception: {e}")
    finally:
        _log("hero_service: running graceful shutdown")
        try:
            _stop_analysis_process()
        except Exception:
            pass
        _shutdown_all_account_managers(timeout=3.0)
        _log("hero_service: shutdown finished")
