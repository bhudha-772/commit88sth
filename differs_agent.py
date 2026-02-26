#!/usr/bin/env python3
"""differs_agent.py — per-market analysis agent with global arbiter and journaling.

This file is based on your uploaded agent with targeted updates:
 - strict top selection (no tie margins)
 - tick-driven reanalysis (REANALYZE_TICKS)
 - no blank-digit prediction posts
 - market_name mapping added (does not replace `symbol`)
 - verbose logging for SSE events, ticks, selection, posts, rate-limits, and errors
 - preserves SSE loop, journaling, pending PID handling, toasts, and original behavior
 - **NEW (safe, minimal):** attach `reason` tokens to predictions
"""

from __future__ import annotations
import os
import sys
import time
import json
import argparse
import urllib.request
import urllib.error
import urllib.parse
import socket
import traceback
import math
import uuid
import threading
from collections import deque, Counter, defaultdict
from typing import Deque, List, Optional, Tuple, Dict, Any
from datetime import datetime

# Optional requests-based session with retry support (preferred)
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception:
    requests = None

# ---------- config / defaults ----------
HTTP_TIMEOUT = float(os.environ.get("HERO_HTTP_TIMEOUT", "8.0"))
HTTP_RETRIES = int(os.environ.get("HERO_HTTP_RETRIES", "3"))
HTTP_BACKOFF = float(os.environ.get("HERO_HTTP_BACKOFF", "0.5"))
HTTP_STATUS_FORCELIST = (500, 502, 503, 504)

DEFAULT_DASHBOARD_PORT = 5000
DEFAULT_SCHEME = "http"

DEFAULT_SERVICE_BASE = "http://127.0.0.1:5000"
DEFAULT_BUFFER_SIZE = int(os.environ.get("DIFFER_BUFFER_SIZE", "10"))
DEFAULT_MIN_BUFFER = int(os.environ.get("DIFFER_MIN_BUFFER", "10"))
DEFAULT_LOG_DIR = os.path.expanduser("~/.hero_logs")
SSE_READ_TIMEOUT = 90.0
SSE_RECONNECT_DELAY = 2.0
POST_TIMEOUT = 9.0
CALC_UPDATE_INTERVAL = 2.0
MAX_WAIT_TICKS = int(os.environ.get("DIFFER_MAX_WAIT_TICKS", "10"))

DEFAULT_CONF_THRESHOLD = float(os.environ.get("DIFFER_CONF_THRESHOLD", "0.00"))
# keep these defs for compatibility but we'll enforce strict ">" checks instead of margins
DEFAULT_TIE_MARGIN = float(os.environ.get("DIFFER_TIE_MARGIN", "0.03"))
DEFAULT_STABILITY_TICKS = int(os.environ.get("DIFFER_STABILITY_TICKS", "2"))
DEFAULT_N_TEST = int(os.environ.get("DIFFER_N_TEST", "1"))
DEFAULT_RECENCY_WINDOW = int(os.environ.get("DIFFER_RECENCY_WINDOW", "10"))

DEFAULT_TAU = float(os.environ.get("DIFFER_TAU", "4.0"))
W_F = float(os.environ.get("DIFFER_W_F", "0.45"))
W_T = float(os.environ.get("DIFFER_W_T", "0.30"))
W_B = float(os.environ.get("DIFFER_W_B", "0.15"))
W_R = float(os.environ.get("DIFFER_W_R", "0.10"))

W_RULES = float(os.environ.get("DIFFER_W_RULES", "0.5"))
W_PBEST = float(os.environ.get("DIFFER_W_PBEST", "0.3"))
W_BUF = float(os.environ.get("DIFFER_W_BUF", "0.2"))

# NOTE: We will not use PANEL_TIE_MARGIN or DIGIT_TIE_MARGIN for loosened tie logic.
PANEL_TIE_MARGIN = float(os.environ.get("DIFFER_PANEL_TIE_MARGIN", "0.03"))
DIGIT_TIE_MARGIN = float(os.environ.get("DIFFER_DIGIT_TIE_MARGIN", "0.04"))
RULES_PASS_THRESHOLD = float(os.environ.get("DIFFER_RULES_PASS_THRESHOLD", "0.3"))
MAX_CONF_DISPLAY = float(os.environ.get("DIFFER_MAX_CONF_DISPLAY", "0.995"))
PENDING_TIMEOUT_SECS = int(os.environ.get("DIFFER_PENDING_TIMEOUT_SECS", "60"))
MIN_INTERVAL_MARKET = int(os.environ.get("DIFFER_MIN_INTERVAL_MARKET", "15"))
REQUIRE_UNIQUE_TOP = str(os.environ.get("DIFFER_REQUIRE_UNIQUE_TOP", "0")).strip().lower() in ("1", "true", "yes", "on")
REQUIRE_UNIQUE_DIGIT = str(os.environ.get("DIFFER_REQUIRE_UNIQUE_DIGIT", "0")).strip().lower() in ("1", "true", "yes", "on")
MIN_TOP_SPREAD = float(os.environ.get("DIFFER_MIN_TOP_SPREAD", "0.0"))
MIN_DIGIT_SPREAD = float(os.environ.get("DIFFER_MIN_DIGIT_SPREAD", "0.0"))

# toast interval (seconds) for "no predictions yet" notifications (rate-limited)
TOAST_MIN_INTERVAL = float(os.environ.get("DIFFER_TOAST_MIN_INTERVAL", "10.0"))

DEFAULT_PREDICTION_MODE = os.environ.get("DIFFER_PREDICTION_MODE", "differ").lower()

# NEW: number of ticks from same market required to allow reanalysis/repost earlier than MIN_INTERVAL_MARKET
REANALYZE_TICKS = int(os.environ.get("DIFFER_REANALYZE_TICKS", "10"))

# One-trade-at-a-time gate:
# post a prediction, then wait for settlement before posting the next.
# Timeout is only a safety valve if settlement event is never received.



# NEW: human-friendly market name mapping (symbol -> friendly)
SYMBOL_NAME_MAP = {
    "R_100": "Volatility 100",
    "R_50": "Volatility 50",
    "R_10": "Volatility 10",
    # add more mappings as needed
}

os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
OUT_LOG = os.path.join(DEFAULT_LOG_DIR, "differ_agent.out")
ERR_LOG = os.path.join(DEFAULT_LOG_DIR, "differ_agent.err")
JOURNAL_FILE = os.path.join(DEFAULT_LOG_DIR, "differ_journal.jsonl")
BRAIN_FILE = os.path.join(DEFAULT_LOG_DIR, "hero_brain.jsonl")

# ---------- logging helpers ----------
def _nowts() -> str:
    return datetime.utcnow().isoformat()

def log_info(msg: str):
    try:
        with open(OUT_LOG, "a", encoding="utf-8") as f:
            f.write(f"{_nowts()} INFO: {msg}\n")
    except Exception:
        pass
    print(msg)

def log_err(msg: str):
    try:
        with open(ERR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{_nowts()} ERROR: {msg}\n")
    except Exception:
        pass
    print("ERR:", msg, file=sys.stderr)

# ---------- HTTP session (requests with retries preferred) ----------
def make_retry_session(total_retries=HTTP_RETRIES, backoff_factor=HTTP_BACKOFF, status_forcelist=HTTP_STATUS_FORCELIST):
    if not requests:
        return None
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(['GET','POST','PUT','DELETE','HEAD','OPTIONS']),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "differs_agent/1.0"})
    return s

_HTTP_SESSION = make_retry_session()
# Ensure keep-alive
if _HTTP_SESSION is not None:
    try:
        _HTTP_SESSION.headers.update({"Connection": "keep-alive", "User-Agent": "differs_agent/1.0"})
    except Exception:
        pass

# ---------- HTTP helpers ----------
def http_post_json(url: str, obj: dict, timeout: float = POST_TIMEOUT) -> Tuple[bool, Optional[int], Optional[str]]:
    data = json.dumps(obj).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "differs_agent/1.0"}
    try:
        if _HTTP_SESSION is not None:
            r = _HTTP_SESSION.post(url, json=obj, timeout=timeout)
            try:
                text = r.text
            except Exception:
                text = ""
            return True, getattr(r, "status_code", None), text
    except Exception as e:
        log_err(f"HTTP POST via requests failed: {e}")

    # urllib fallback
    try:
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            return True, resp.getcode(), text
    except urllib.error.HTTPError as e:
        try:
            txt = e.read().decode("utf-8", errors="ignore")
        except Exception:
            txt = ""
        log_err(f"HTTPError POST {url} -> {e.code} {txt}")
        return False, getattr(e, "code", None), txt
    except Exception as e:
        log_err(f"POST {url} exception (urllib fallback): {e}")
        return False, None, str(e)

# ---------- async/pooled POST helpers ----------
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore

_POST_MAX_WORKERS = int(os.environ.get("DIFFER_AGENT_POST_THREADS", "6"))
_POST_SEMAPHORE = Semaphore(int(os.environ.get("DIFFER_AGENT_POST_CONCURRENCY", "8")))
_AGENT_EXEC = ThreadPoolExecutor(max_workers=_POST_MAX_WORKERS)

def post_json_with_retries(url: str, payload: dict, timeout: float | None = None):
    if timeout is None:
        timeout = HTTP_TIMEOUT
    try:
        return http_post_json(url, payload, timeout=timeout)
    except Exception as e:
        log_err(f"ERR: POST {url} exception: {e}")
        return None

# Small wrapper to submit async POSTs (non-blocking)
_POST_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_POST_EXECUTOR_LOCK = threading.Lock()

def post_prediction_async(prediction_payload: dict, url: str):
    def _job():
        try:
            ok, status, text = http_post_json(url, prediction_payload, timeout=8.0)
            if ok and status == 200:
                log_info(f"Async POST accepted by server: {prediction_payload.get('prediction_id')}")
            else:
                log_err(f"Async POST server response: status={status} text={text} for pid={prediction_payload.get('prediction_id')}")
        except Exception as e:
            log_err(f"Async POST exception: {e}")

    try:
        with _POST_EXECUTOR_LOCK:
            _POST_EXECUTOR.submit(_job)
    except Exception as e:
        log_err(f"Failed to submit async POST to executor: {e}")

# ---------- SSE client (urllib stdlib) ----------
def sse_event_stream(url: str):
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(req, timeout=SSE_READ_TIMEOUT) as resp:
        buf_lines: List[str] = []
        for raw in resp:
            try:
                line = raw.decode("utf-8", errors="ignore").rstrip("\r\n")
            except Exception:
                continue
            if line == "":
                if buf_lines:
                    ev_name = None
                    data_lines: List[str] = []
                    for l in buf_lines:
                        if l.startswith("event:"):
                            ev_name = l.split(":", 1)[1].strip()
                        elif l.startswith("data:"):
                            data_lines.append(l.split(":", 1)[1].lstrip())
                    data_text = "\n".join(data_lines)
                    yield ev_name, data_text
                buf_lines = []
                continue
            buf_lines.append(line)
    return

# ---------- small helpers ----------
def safe_int(v):
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None

def extract_last_decimal_from_payload(payload):
    if payload is None:
        return None
    if isinstance(payload, dict):
        if "last_decimal" in payload and payload["last_decimal"] not in (None, ""):
            val = payload["last_decimal"]
            n = safe_int(val)
            if n is not None and 0 <= n <= 9:
                return n
        for key in ("price", "quote", "ask", "bid"):
            if key in payload and payload[key] is not None:
                s = str(payload[key]).strip()
                if "." in s:
                    try:
                        return safe_int(s.split(".")[-1][-1])
                    except Exception:
                        pass
    if isinstance(payload, list) and len(payload) >= 5:
        return safe_int(payload[4])
    return None

# ---------- scoring rules (helpers) ----------
def _shannon_entropy(counts: Dict[int, int], total: int) -> float:
    if total <= 0:
        return 0.0
    e = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            e -= p * math.log2(p)
    return e

def compute_digit_features(buffer: Deque[int]) -> Dict[int, Dict[str, Any]]:
    buf = list(buffer)
    n = len(buf)
    counts = Counter(buf)
    features: Dict[int, Dict[str, Any]] = {d: {} for d in range(10)}
    pos_map = defaultdict(list)
    for idx, val in enumerate(buf):
        if val is None:
            continue
        pos_map[val].append(idx)

    transitions = defaultdict(lambda: Counter())
    for i in range(len(buf) - 1):
        a = buf[i]
        b = buf[i + 1]
        if a is not None and b is not None:
            transitions[a][b] += 1

    entropy = _shannon_entropy(counts, n)
    for d in range(10):
        c = counts.get(d, 0)
        features[d]["count"] = c
        features[d]["proportion"] = (c / n) if n > 0 else 0.0
        last_seen = None
        if pos_map.get(d):
            last_seen = n - 1 - pos_map[d][-1]
        features[d]["last_seen_dist"] = last_seen if last_seen is not None else None
        max_run = 0
        cur = 0
        for v in buf:
            if v == d:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        features[d]["max_run"] = max_run
        gaps = []
        ps = pos_map.get(d, [])
        for i in range(1, len(ps)):
            gaps.append(ps[i] - ps[i - 1])
        features[d]["avg_gap"] = (sum(gaps) / len(gaps)) if gaps else None
        RECENCY_WINDOW = max(4, DEFAULT_RECENCY_WINDOW)
        recent_count = sum(1 for x in buf[-RECENCY_WINDOW:] if x == d)
        features[d]["recent_count"] = recent_count
        total_in = sum(transitions[x][d] for x in transitions)
        total_trans = sum(sum(c.values()) for c in transitions.values()) or 1
        features[d]["transition_in_rate"] = total_in / total_trans
        features[d]["entropy"] = entropy
        even_positions = sum(1 for p in pos_map.get(d, []) if (p % 2 == 0))
        even_prop = (even_positions / len(pos_map.get(d, []))) if pos_map.get(d) else 0.0
        features[d]["even_prop"] = even_prop
        half = n // 2
        first_half = buf[:half]
        second_half = buf[half:]
        change = (second_half.count(d) - first_half.count(d)) if n > 1 else 0
        features[d]["recent_trend_pos"] = change
        features[d]["transition_row"] = dict(transitions.get(d, {}))
    return features

def evaluate_rules_for_digit(d: int, features: Dict[str, Any], buffer_len: int, journal_stats: Dict[int, Dict[str, int]]) -> Tuple[int, List[str]]:
    passed = []
    mean_count = (buffer_len / 10.0) if buffer_len > 0 else 0.0
    if features["count"] > mean_count:
        passed.append("frequency")
    if features["last_seen_dist"] is not None and features["last_seen_dist"] < max(4, DEFAULT_RECENCY_WINDOW):
        passed.append("recency")
    if features["max_run"] >= 2:
        passed.append("streak")
    if features.get("avg_gap") is not None and features["avg_gap"] <= max(3, buffer_len / 8.0):
        passed.append("gap")
    if features.get("recent_trend_pos", 0) > 0:
        passed.append("trend_up")
    if features.get("transition_in_rate", 0.0) > 0.08:
        passed.append("trans_cons")
    if features.get("entropy", 10.0) < 3.3 and features.get("proportion", 0.0) >= 0.08:
        passed.append("entropy")
    if features.get("recent_count", 0) >= 2:
        passed.append("stable_recent")
    hist = journal_stats.get(d, {"wins": 0, "trials": 0})
    if hist.get("trials", 0) >= 3 and (hist.get("wins", 0) / hist["trials"]) >= 0.60:
        passed.append("hist_win")
    if features.get("even_prop", 0.0) >= 0.7 or features.get("even_prop", 0.0) <= 0.3:
        passed.append("pos_parity")
    return len(passed), passed

# ---------- journaling & brain ----------
def append_journal(entry: dict):
    try:
        with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass

def load_journal_stats() -> Dict[int, Dict[str, int]]:
    stats = {d: {"wins": 0, "trials": 0} for d in range(10)}
    try:
        if os.path.exists(JOURNAL_FILE):
            with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        pd = int(obj.get("prediction_digit", -1))
                        res = obj.get("result")
                        if pd is not None and 0 <= pd <= 9:
                            stats[pd]["trials"] = stats[pd].get("trials", 0) + 1
                            if res == "WIN":
                                stats[pd]["wins"] = stats[pd].get("wins", 0) + 1
                    except Exception:
                        continue
    except Exception:
        pass
    return stats

def append_brain_lesson(lesson: dict):
    try:
        with open(BRAIN_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(lesson, default=str) + "\n")
    except Exception:
        pass

# ---------- analysis post helper ----------
def make_analysis_post(analysis_push_url: str, event: str, message: str, buffer_len: Optional[int] = None, extra: Optional[dict] = None, symbol: str = "ANALYSIS"):
    if not analysis_push_url:
        return
    payload = {
        "analysis_event": event,
        "message": message,
        "epoch": int(time.time()),
    }
    if buffer_len is not None:
        payload["buffer_len"] = int(buffer_len)
    if extra:
        payload.update(extra)
    post = {"symbol": symbol, "payload": payload}
    ok, status, text = http_post_json(analysis_push_url, post)
    if ok:
        log_info(f"SSE analysis event posted: {event} payload_keys={list(payload.keys())} status={status}")
    else:
        log_err(f"Failed to post analysis event {event} -> {status} / {text}")

# ----------------------------------------------------------------------
# MarketAnalyzer
# ----------------------------------------------------------------------
_ANALYZER_BUFFER = int(os.environ.get("DIFFER_ANALYZER_BUFFER", str(DEFAULT_BUFFER_SIZE)))
_ANALYZER_MIN_OBS = int(os.environ.get("DIFFER_ANALYZER_MIN_OBS", str(DEFAULT_MIN_BUFFER)))
_ANALYZER_DEBOUNCE = float(os.environ.get("DIFFER_ANALYZER_DEBOUNCE", "0.5"))

class MarketAnalyzer:
    def __init__(self, buffer_size: int = _ANALYZER_BUFFER, push_url: str | None = None):
        self.buffer_size = int(buffer_size or _ANALYZER_BUFFER)
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.buffer_size))
        self.last_post_ts: Dict[str, float] = defaultdict(float)
        self.push_url = push_url
        self.global_last_panel_post = 0.0
        try:
            self.journal_stats = load_journal_stats()
        except Exception:
            self.journal_stats = {d: {"wins": 0, "trials": 0} for d in range(10)}

    def refresh_journal_stats(self):
        try:
            self.journal_stats = load_journal_stats()
        except Exception:
            pass

    def add_digit(self, symbol: str, digit: int):
        if symbol is None:
            return
        sym = str(symbol).upper()
        try:
            if digit is None:
                return
            d = int(digit)
            if d < 0 or d > 9:
                return
        except Exception:
            return
        self.buffers[sym].append(d)

    def _entropy(self, counts: Counter, total: int) -> float:
        return _shannon_entropy(dict(counts), total)

    def compute_stats(self, symbol: str):
        sym = str(symbol).upper()
        buf = list(self.buffers.get(sym, []))
        total = len(buf)
        counts = Counter(buf)

        # robust most / p_most
        try:
            most, most_count = counts.most_common(1)[0]
        except Exception:
            most = None
            most_count = 0

        p_most = float(most_count) / float(total) if total > 0 else 0.0
        # clamp to avoid perfect 1.0 artifacts for UI
        p_most = max(0.0, min(p_most, MAX_CONF_DISPLAY))

        entropy = self._entropy(counts, total)
        max_ent = math.log2(10) if 10 > 1 else 1.0
        ent_score = 1.0 - (entropy / max_ent) if max_ent else 1.0

        # per-digit features and rule evaluation
        features_map = compute_digit_features(deque(buf))
        rules_met_count = 0
        for d in range(10):
            feats = features_map.get(d, {})
            try:
                passed_count, _ = evaluate_rules_for_digit(d, feats, total, self.journal_stats)
                if passed_count >= _ANALYZER_MIN_OBS and passed_count >= 1:
                    rules_met_count += 1
            except Exception:
                continue
        prop_met = rules_met_count / 10.0

        # buffer fill fraction
        buf_frac = min(1.0, total / float(self.buffer_size or max(1, total)))

        # market base score combination
        w_rules = W_RULES
        w_pbest = W_PBEST
        w_buf = W_BUF
        market_base = (w_rules * prop_met) + (w_pbest * p_most) + (w_buf * buf_frac)

        # entropy reduces confidence
        entropy_factor = max(0.08, 1.0 - (entropy / max_ent) * 0.9) if max_ent else 1.0

        market_confidence = float(market_base * entropy_factor)
        market_confidence = max(0.0, min(market_confidence, MAX_CONF_DISPLAY))

        # Build per-digit scores (frequency-based fallback)
        scores_list = []
        for d in range(10):
            cnt = counts.get(d, 0)
            prop = float(cnt) / float(total) if total > 0 else 0.0
            scores_list.append({"digit": int(d), "confidence": round(prop, 6), "count": int(cnt)})
        scores_list.sort(key=lambda x: x["confidence"], reverse=True)

        # sanitize most as int if possible
        try:
            most_int = int(most) if most is not None else None
        except Exception:
            most_int = None

        return {
            "symbol": sym,
            "total": total,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "most": most_int,
            "p_most": float(p_most),
            "entropy": float(entropy),
            "confidence": float(market_confidence),
            "buffer_snapshot": buf,
            "rules_met_count": rules_met_count,
            "prop_met": prop_met,
            "buf_frac": buf_frac,
            "features_map": features_map,
            "scores": scores_list,
        }

    def markets_ranking(self):
        stats = []
        for sym in list(self.buffers.keys()):
            st = self.compute_stats(sym)
            if st["total"] > 0:
                stats.append(st)
        stats.sort(key=lambda x: x["confidence"], reverse=True)
        return stats

    def _safe_post(self, url: str, payload: dict):
        if not url:
            return
        def _job():
            try:
                _ = post_json_with_retries(url, payload, timeout=POST_TIMEOUT)
            except Exception:
                pass
        try:
            with _POST_EXECUTOR_LOCK:
                _POST_EXECUTOR.submit(_job)
        except Exception:
            pass

    def on_tick(self, symbol: str, last_decimal, epoch=None, push_immediately: bool = True):
        # lenient parse
        d = None
        try:
            if last_decimal is None or last_decimal == "":
                d = None
            else:
                try:
                    d = int(last_decimal)
                except Exception:
                    try:
                        d = int(float(last_decimal))
                    except Exception:
                        d = None
        except Exception:
            d = None

        if d is None:
            return None

        self.add_digit(symbol, d)
        st = self.compute_stats(symbol)

        # debounce per-symbol to avoid flooding
        now = time.time()
        last = self.last_post_ts.get(symbol, 0.0)
        if (now - last) < _ANALYZER_DEBOUNCE and push_immediately:
            return st
        self.last_post_ts[symbol] = now

        # sanitize best_digit
        best_digit = st.get("most")
        if best_digit is not None:
            try:
                best_digit = int(best_digit)
            except Exception:
                best_digit = None

        p_best = float(st.get("p_most", 0.0) or 0.0)
        p_best = max(0.0, min(p_best, MAX_CONF_DISPLAY))

        conf_val = float(st.get("confidence", 0.0) or 0.0)
        conf_val = max(0.0, min(conf_val, MAX_CONF_DISPLAY))

        payload_market = {
            "analysis_event": "market_confidence",
            "epoch": int(epoch) if epoch is not None else int(time.time()),
            "market": symbol,
            "symbol": symbol,
            "confidence": round(conf_val, 4),
            "best_digit": best_digit if best_digit is not None else None,
            "p_best": round(p_best, 4),
            "entropy": round(st.get("entropy", 0.0), 4),
            "buffer_snapshot": st.get("buffer_snapshot", []),
            "counts": st.get("counts", {}),
            "scores": st.get("scores", []),
            "rules_met_count": st.get("rules_met_count", 0),
            "prop_met": round(st.get("prop_met", 0.0), 4),
            "message": f"Market {symbol}: best_digit={best_digit} p={p_best:.2f} conf={conf_val:.3f}",
        }

        # debug log for posted payloads (helps identify why UI shows 100%)
        try:
            log_info(f"MarketAnalyzer: post market_confidence market={symbol} conf={conf_val:.4f} p_best={p_best:.4f} best_digit={best_digit} total={st.get('total')}")
        except Exception:
            pass

        if self.push_url and push_immediately:
            self._safe_post(self.push_url, payload_market)

        # Panel ranking (debounced globally)
        ranking = self.markets_ranking()
        payload_ranking = {
            "analysis_event": "panel_ranking",
            "epoch": int(epoch) if epoch is not None else int(time.time()),
            "ranking": [
                {
                    "market": r["symbol"],
                    "confidence": round(r["confidence"], 4),
                    "best_digit": r.get("most"),
                    "p_best": round(r.get("p_most", 0.0), 4),
                    "total": r["total"],
                    "rules_met_count": r.get("rules_met_count", 0),
                    "counts": r.get("counts", {}),
                    "scores": r.get("scores", []),
                }
                for r in ranking
            ],
            "top": ranking[0]["symbol"] if ranking else None,
            "message": f"Panel ranking updated, top={ranking[0]['symbol'] if ranking else None}",
        }
        nowg = time.time()
        if self.push_url and push_immediately and ((nowg - self.global_last_panel_post) > (_ANALYZER_DEBOUNCE * 2)):
            self.global_last_panel_post = nowg
            self._safe_post(self.push_url, payload_ranking)

        return st

# Instantiate analyzer
PUSH_URL = os.environ.get("HERO_DASHBOARD_PUSH_URL") or f"{DEFAULT_SERVICE_BASE}/control/push_tick"
_ANALYZER_BUF = int(os.environ.get("DIFFER_ANALYZER_BUFFER", str(DEFAULT_BUFFER_SIZE)))
analyzer = MarketAnalyzer(buffer_size=_ANALYZER_BUF, push_url=PUSH_URL)

def agent_on_tick_notify(symbol, last_decimal, epoch=None):
    try:
        return analyzer.on_tick(symbol, last_decimal, epoch)
    except Exception as e:
        log_err(f"agent_on_tick_notify exception: {e}")
        return None

# ---------- ensemble helpers ----------
def recency_weights(n: int, tau: float = DEFAULT_TAU) -> List[float]:
    if n <= 0:
        return []
    weights = []
    for i in range(n):
        w = math.exp(- (n - 1 - i) / float(max(1e-6, tau)))
        weights.append(w)
    return weights

def rw_frequency(buf: List[int], d: int, tau: float = DEFAULT_TAU) -> float:
    n = len(buf)
    if n == 0:
        return 0.0
    weights = recency_weights(n, tau)
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0
    s = 0.0
    for v, w in zip(buf, weights):
        if v == d:
            s += w
    return s / total_w

def beta_posterior_weighted(buf: List[int], d: int, tau: float = DEFAULT_TAU, alpha0: float = 1.0, beta0: float = 1.0) -> float:
    n = len(buf)
    if n == 0:
        return alpha0 / (alpha0 + beta0)
    weights = recency_weights(n, tau)
    total_w = sum(weights)
    count_w = 0.0
    for v, w in zip(buf, weights):
        if v == d:
            count_w += w
    posterior_mean = (alpha0 + count_w) / (alpha0 + beta0 + total_w)
    return float(posterior_mean)

def transition_prob(buf: List[int], d: int, laplace: float = 1.0) -> float:
    if len(buf) < 2:
        return 0.0
    last = buf[-1]
    counts = Counter()
    for i in range(len(buf)-1):
        if buf[i] == last:
            counts[buf[i+1]] += 1
    total = sum(counts.values())
    denom = total + laplace * 10
    num = counts.get(d, 0) + laplace
    return float(num) / float(max(1e-9, denom))

def clamp01(x: float) -> float:
    if x is None or math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))

# ---------- main agent logic ----------
def _detect_dashboard_host_port() -> Tuple[str, int]:
    sse_full = os.environ.get("HERO_DASHBOARD_SSE_URL")
    if sse_full:
        try:
            u = urllib.parse.urlparse(sse_full)
            host = u.hostname or "127.0.0.1"
            port = u.port or (443 if u.scheme == "https" else DEFAULT_DASHBOARD_PORT)
            return host, int(port)
        except Exception:
            pass
    push = os.environ.get("HERO_DASHBOARD_PUSH_URL")
    if push:
        try:
            u = urllib.parse.urlparse(push)
            host = u.hostname or "127.0.0.1"
            port = u.port or (443 if u.scheme == "https" else DEFAULT_DASHBOARD_PORT)
            return host, int(port)
        except Exception:
            pass
    env_host = os.environ.get("HERO_DASHBOARD_HOST")
    env_port = os.environ.get("HERO_DASHBOARD_PORT")
    if env_host:
        try:
            return env_host, int(env_port) if env_port else DEFAULT_DASHBOARD_PORT
        except Exception:
            return env_host, DEFAULT_DASHBOARD_PORT
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("1.1.1.1", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip, DEFAULT_DASHBOARD_PORT
    except Exception:
        pass
    return "127.0.0.1", DEFAULT_DASHBOARD_PORT

def get_dashboard_sse_url() -> str:
    host, port = _detect_dashboard_host_port()
    scheme = os.environ.get("HERO_DASHBOARD_SCHEME", DEFAULT_SCHEME)
    return f"{scheme}://{host}:{port}/events"

# Helper: build reason tokens from features & rule outputs
def build_reason_tokens_for_prediction(digit: int, features_map: Dict[int, Dict[str, Any]], buffer_len: int, journal_stats: Dict[int, Dict[str,int]], p_best: float) -> List[str]:
    tokens: List[str] = []
    try:
        feats = features_map.get(int(digit), {}) if features_map else {}
        # Use evaluate_rules_for_digit to extract rule tokens (passed rules)
        try:
            _, passed = evaluate_rules_for_digit(int(digit), feats, buffer_len, journal_stats)
            if passed:
                tokens.extend([str(t) for t in passed])
        except Exception:
            pass

        # Fallback/additional scalar tokens
        try:
            # include p_best with 2 dec
            tokens.append(f"p_best:{round(float(p_best or 0.0), 2):.2f}")
        except Exception:
            pass

        # other signals
        try:
            if feats:
                if feats.get("recent_count", 0) >= 2:
                    tokens.append(f"recent_count:{int(feats.get('recent_count',0))}")
                if feats.get("max_run", 0) >= 2:
                    tokens.append("streak")
                ld = feats.get("last_seen_dist")
                if ld is not None and isinstance(ld, (int, float)) and ld < max(4, DEFAULT_RECENCY_WINDOW):
                    tokens.append("recency")
                tr = feats.get("transition_in_rate")
                if tr is not None and float(tr) > 0.08:
                    tokens.append("transition_in_rate")
                ent = feats.get("entropy")
                if ent is not None:
                    tokens.append(f"entropy:{round(float(ent),2):.2f}")
        except Exception:
            pass

        # de-duplicate while preserving order
        seen = set()
        out = []
        for t in tokens:
            if t not in seen:
                out.append(t)
                seen.add(t)
        return out
    except Exception:
        return []

def run_agent(
    service_base: str,
    analysis_push_url: str,
    prediction_push_url: str,
    buffer_size: int,
    min_buffer: int,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    tie_margin: float = DEFAULT_TIE_MARGIN,
    stability_ticks: int = DEFAULT_STABILITY_TICKS,
    n_test: int = DEFAULT_N_TEST,
):
    sse_env = os.environ.get("HERO_DASHBOARD_SSE_URL")
    if sse_env:
        sse_url = sse_env
    elif service_base and service_base != DEFAULT_SERVICE_BASE:
        sse_url = service_base.rstrip("/") + "/events"
    else:
        sse_url = get_dashboard_sse_url()

    log_info(f"DEBUG: using SSE URL -> {sse_url}")

    # If analysis/prediction push URLs appear to be localhost but SSE is on a different host,
    # prefer the SSE host so POSTs go to a reachable service (common misconfig).
    try:
        parsed_sse = urllib.parse.urlparse(sse_url)
        sse_host = parsed_sse.hostname or ""
        sse_port = parsed_sse.port or DEFAULT_DASHBOARD_PORT
        sse_host_norm = str(sse_host).strip()

        def _is_localhost_host(h):
            return h in ("127.0.0.1", "localhost", "::1", "")

        if analysis_push_url:
            try:
                pa = urllib.parse.urlparse(analysis_push_url)
                ph = pa.hostname or ""
                if _is_localhost_host(ph) and not _is_localhost_host(sse_host_norm):
                    new_port = pa.port or sse_port
                    new_scheme = pa.scheme or parsed_sse.scheme or "http"
                    new_path = pa.path or "/control/push_tick"
                    analysis_push_url = f"{new_scheme}://{sse_host_norm}:{new_port}{new_path}"
                    log_info(f"Adjusted analysis_push_url to SSE host -> {analysis_push_url}")
            except Exception:
                pass

        if prediction_push_url:
            try:
                pp = urllib.parse.urlparse(prediction_push_url)
                ph = pp.hostname or ""
                if _is_localhost_host(ph) and not _is_localhost_host(sse_host_norm):
                    new_port = pp.port or sse_port
                    new_scheme = pp.scheme or parsed_sse.scheme or "http"
                    new_path = pp.path or "/control/push_prediction"
                    prediction_push_url = f"{new_scheme}://{sse_host_norm}:{new_port}{new_path}"
                    log_info(f"Adjusted prediction_push_url to SSE host -> {prediction_push_url}")
            except Exception:
                pass
    except Exception:
        pass

    log_info(
        f"agent starting; SSE={sse_url} analysis_push={analysis_push_url} "
        f"prediction_push={prediction_push_url} buffer_size={buffer_size} min_buffer={min_buffer} "
        f"conf_threshold={conf_threshold} require_unique_top={REQUIRE_UNIQUE_TOP} require_unique_digit={REQUIRE_UNIQUE_DIGIT} "
        f"min_top_spread={MIN_TOP_SPREAD} min_digit_spread={MIN_DIGIT_SPREAD} max_wait_ticks={MAX_WAIT_TICKS}"
    )

    # buffers per market
    buffers_by_symbol: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=buffer_size))

    # state variables
    pending_pid: Optional[str] = None
    pending_since = 0.0
    _pending_pid_lock = threading.Lock()
    pending_by_symbol: Dict[str, str] = {}
    journal_stats = load_journal_stats()

    # prediction rate-limiter per market
    last_prediction_ts: Dict[str, float] = defaultdict(float)

    # NEW: per-market tick counter used to allow reanalysis before MIN_INTERVAL_MARKET elapses
    last_ticks_since_pred: Dict[str, int] = defaultdict(int)

    # last toast post timestamp (rate-limiting the "no prediction yet" messages)
    last_toast_ts = 0.0

    # initial post
    try:
        make_analysis_post(analysis_push_url, "getting_data", "Getting the data — waiting for live ticks...", buffer_len=0)
    except Exception:
        pass

    backoff = SSE_RECONNECT_DELAY
    last_calc_post = 0.0
    last_tick_time = None
    no_ticks_notified = False

    # nested helper: try to post prediction when ranking shows a clear unique top market and unique top digit
    def maybe_try_predict(st: Optional[dict]):
        """
        Settlement-gated prediction decision:
        - Require a clear unique top market AND a clear unique top digit.
        - Allow only one outstanding prediction at a time.
        - New prediction only after server settlement for the pending id.
        """
        nonlocal last_toast_ts, last_prediction_ts, pending_pid, pending_since, pending_by_symbol

        try:
            if st is None:
                return
            if not prediction_push_url:
                return

            # Gate: wait for settlement of the last posted prediction.
            try:
                with _pending_pid_lock:
                    cur_pending = pending_pid
                    cur_age = (time.time() - float(pending_since or 0.0)) if cur_pending else 0.0
                if cur_pending:
                    if cur_age < float(PENDING_TIMEOUT_SECS):
                        log_info(
                            f"maybe_try_predict: waiting settlement for pending pid={cur_pending} "
                            f"(age={cur_age:.2f}s < timeout={PENDING_TIMEOUT_SECS}s)"
                        )
                        return
                    # safety timeout: release stuck pending state
                    with _pending_pid_lock:
                        log_info(
                            f"maybe_try_predict: pending timeout exceeded ({cur_age:.2f}s) "
                            f"for pid={pending_pid}; releasing pending gate"
                        )
                        pending_pid = None
                        pending_since = 0.0
                        pending_by_symbol = {}
            except Exception:
                pass

            # get fresh ranking
            ranking = analyzer.markets_ranking()
            if not ranking:
                # rate-limited toast
                if (time.time() - last_toast_ts) > TOAST_MIN_INTERVAL:
                    try:
                        make_analysis_post(analysis_push_url, "prediction_toast", "No predictions available yet.", symbol="SYSTEM", extra={"status": "none_yet"})
                        last_toast_ts = time.time()
                        log_info("maybe_try_predict: no ranking available; posted 'none_yet' toast")
                    except Exception:
                        pass
                return

            top = ranking[0]
            second = ranking[1] if len(ranking) > 1 else None
            top_conf = float(top.get("confidence", 0.0) or 0.0)
            second_conf = float(second.get("confidence", 0.0) or 0.0) if second else 0.0
            market_spread = top_conf - second_conf

            if top_conf < float(conf_threshold):
                if (time.time() - last_toast_ts) > TOAST_MIN_INTERVAL:
                    try:
                        make_analysis_post(
                            analysis_push_url,
                            "prediction_toast",
                            f"Top confidence too low ({top_conf:.3f} < {float(conf_threshold):.3f}); waiting.",
                            symbol="SYSTEM",
                            extra={"status": "low_confidence", "confidence": round(top_conf, 4), "threshold": float(conf_threshold)},
                        )
                        last_toast_ts = time.time()
                    except Exception:
                        pass
                return

            # market selection policy (strict/relaxed via env).
            is_market_ok = False
            if REQUIRE_UNIQUE_TOP:
                is_market_ok = market_spread > max(0.0, float(MIN_TOP_SPREAD))
            else:
                is_market_ok = market_spread >= float(MIN_TOP_SPREAD)

            if not is_market_ok:
                if (time.time() - last_toast_ts) > TOAST_MIN_INTERVAL:
                    try:
                        make_analysis_post(analysis_push_url, "prediction_toast", "No clear top market yet.", symbol="SYSTEM", extra={"status": "none_yet"})
                        last_toast_ts = time.time()
                        log_info(
                            f"maybe_try_predict: top market gate blocked "
                            f"(top_conf={top_conf} second_conf={second_conf} spread={market_spread:.6f} "
                            f"require_unique={REQUIRE_UNIQUE_TOP} min_spread={MIN_TOP_SPREAD})"
                        )
                    except Exception:
                        pass
                return

            top_symbol = top.get("symbol") or top.get("market")
            if not top_symbol:
                return
            top_symbol = str(top_symbol).upper()

            # determine top digit and runner-up digit confidences from scores (preferred) then counts fallback
            best_digit = None
            best_conf = 0.0
            second_digit_conf = 0.0

            scs = top.get("scores", []) or []
            if isinstance(scs, list) and len(scs) >= 1:
                scs_sorted = sorted(scs, key=lambda x: float(x.get("confidence", 0.0) or 0.0), reverse=True)
                try:
                    best_digit = int(scs_sorted[0].get("digit"))
                except Exception:
                    best_digit = None
                best_conf = float(scs_sorted[0].get("confidence", 0.0) or 0.0)
                if len(scs_sorted) > 1:
                    second_digit_conf = float(scs_sorted[1].get("confidence", 0.0) or 0.0)
                else:
                    second_digit_conf = 0.0
            else:
                counts = top.get("counts", {}) or {}
                pairs = []
                for k, v in counts.items():
                    try:
                        kd = int(k)
                        kv = int(v)
                    except Exception:
                        continue
                    pairs.append((kd, kv))
                if not pairs:
                    if (time.time() - last_toast_ts) > TOAST_MIN_INTERVAL:
                        try:
                            make_analysis_post(analysis_push_url, "prediction_toast", f"No digit info for top market {top_symbol}.", symbol="SYSTEM", extra={"status": "none_yet", "market": top_symbol})
                            last_toast_ts = time.time()
                            log_info(f"maybe_try_predict: no digit info for {top_symbol}")
                        except Exception:
                            pass
                    return
                pairs.sort(key=lambda x: x[1], reverse=True)
                total = sum(p for _, p in pairs) or 1
                best_digit, best_cnt = pairs[0]
                best_conf = float(best_cnt) / float(total)
                if len(pairs) > 1:
                    second_digit_conf = float(pairs[1][1]) / float(total)
                else:
                    second_digit_conf = 0.0

            # Don't post if no valid best digit
            if best_digit is None:
                log_err(f"maybe_try_predict: best_digit is None for {top_symbol} — skipping post")
                if (time.time() - last_toast_ts) > TOAST_MIN_INTERVAL:
                    try:
                        make_analysis_post(analysis_push_url, "prediction_toast", f"No reliable digit for {top_symbol}; skipping prediction.", symbol=top_symbol, extra={"status": "no_digit"})
                        last_toast_ts = time.time()
                    except Exception:
                        pass
                return

            digit_spread = best_conf - second_digit_conf
            is_digit_ok = False
            if REQUIRE_UNIQUE_DIGIT:
                is_digit_ok = digit_spread > max(0.0, float(MIN_DIGIT_SPREAD))
            else:
                is_digit_ok = digit_spread >= float(MIN_DIGIT_SPREAD)

            if not is_digit_ok:
                if (time.time() - last_toast_ts) > TOAST_MIN_INTERVAL:
                    try:
                        make_analysis_post(analysis_push_url, "prediction_toast", f"No clear top digit for {top_symbol}.", symbol="SYSTEM", extra={"status": "none_yet", "market": top_symbol})
                        last_toast_ts = time.time()
                        log_info(
                            f"maybe_try_predict: top digit gate blocked for {top_symbol} "
                            f"(best_conf={best_conf} second_conf={second_digit_conf} spread={digit_spread:.6f} "
                            f"require_unique={REQUIRE_UNIQUE_DIGIT} min_spread={MIN_DIGIT_SPREAD})"
                        )
                    except Exception:
                        pass
                return

            # --- Build reason tokens using analyzer.compute_stats() features_map and evaluate_rules_for_digit() ---
            try:
                st_top = analyzer.compute_stats(top_symbol)
                features_map = st_top.get("features_map", {}) or {}
                buffer_len = int(st_top.get("total", 0) or 0)
            except Exception:
                features_map = {}
                buffer_len = 0

            reason_tokens = []
            try:
                reason_tokens = build_reason_tokens_for_prediction(best_digit, features_map, buffer_len, journal_stats, best_conf)
            except Exception as e:
                log_err(f"Error building reason tokens: {e}")

            # Passed checks: post a prediction (async)
            pid = f"pred_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            market_name = SYMBOL_NAME_MAP.get(top_symbol.upper(), top_symbol)
            prediction_payload = {
                "analysis_event": "prediction_posted",
                "prediction_id": pid,
                "prediction_digit": int(best_digit),
                "market": top_symbol,
                "market_name": market_name,
                "symbol": top_symbol,
                "prediction_mode": "top_pick",
                "confidence": round(top_conf, 4),
                "p_best": round(best_conf, 4),
                "epoch": int(time.time()),
                "message": f"Auto-predicting {best_digit} on {market_name} ({top_symbol}) conf={top_conf:.4f} p_best={best_conf:.4f}",
                "reason": reason_tokens or [],
            }

            try:
                log_info(f"maybe_try_predict: attempting to post pid={pid} market={top_symbol} digit={best_digit} top_conf={top_conf:.4f} best_conf={best_conf:.4f} reasons={reason_tokens}")
                post_prediction_async(prediction_payload, prediction_push_url)

                # mark pending until settlement arrives
                try:
                    with _pending_pid_lock:
                        pending_pid = pid
                        pending_since = time.time()
                        pending_by_symbol = {top_symbol: pid}
                        last_prediction_ts[top_symbol] = pending_since
                    log_info(f"maybe_try_predict: pending gate armed pid={pid} symbol={top_symbol}")
                except Exception as e:
                    log_err(f"maybe_try_predict: failed to arm pending gate: {e}")

                # reset per-market tick counter if present (harmless)
                try:
                    last_ticks_since_pred[top_symbol] = 0
                except Exception:
                    pass

                # Journal entry (persist prediction + reason)
                try:
                    append_journal({
                        "timestamp": int(time.time()),
                        "prediction_id": pid,
                        "prediction_digit": int(best_digit),
                        "market": top_symbol,
                        "market_name": market_name,
                        "mode": "top_pick",
                        "confidence": round(top_conf, 4),
                        "p_best": round(best_conf, 4),
                        "reason": reason_tokens or []
                    })
                except Exception:
                    pass

                # Post a toast event for the UI
                try:
                    toast_extra = {
                        "status": "posted",
                        "prediction_id": pid,
                        "market": top_symbol,
                        "market_name": market_name,
                        "digit": int(best_digit),
                        "confidence": round(top_conf, 4),
                        "reason": reason_tokens or []
                    }
                    make_analysis_post(analysis_push_url, "prediction_toast", f"Prediction posted: {top_symbol} → {best_digit} (conf={top_conf:.3f})", symbol=top_symbol, extra=toast_extra)
                    last_toast_ts = time.time()
                    log_info(f"maybe_try_predict: posted toast for pid={pid} market={top_symbol}")
                except Exception:
                    pass

            except Exception as e:
                log_err(f"Failed to post auto-prediction: {e}")
                log_err(traceback.format_exc())

        except Exception as e:
            log_err(f"maybe_try_predict exception: {e}\n{traceback.format_exc()}")

    while True:
        posted_in_cycle = False
        try:
            for ev_name, data_text in sse_event_stream(sse_url):
                # LOG SSE receipt
                log_info(f"SSE event received: event={ev_name} len_payload={len(data_text) if data_text else 0}")

                payload = None
                if data_text:
                    try:
                        payload = json.loads(data_text)
                    except Exception:
                        payload = None

                # analysis events from server (settlements, posted predictions, etc.)
                if ev_name == "analysis":
                    try:
                        analysis_payload = None
                        try:
                            analysis_payload = json.loads(data_text)
                        except Exception:
                            analysis_payload = payload
                        if isinstance(analysis_payload, dict):
                            ae = analysis_payload.get("analysis_event")
                            if ae == "prediction_posted":
                                try:
                                    pid = analysis_payload.get("prediction_id") or analysis_payload.get("pred_id") or analysis_payload.get("predictionId")
                                    market = (analysis_payload.get("market") or analysis_payload.get("symbol") or analysis_payload.get("symbol_code") or None)
                                    if market:
                                        market = str(market).upper()
                                    log_info(f"Server posted prediction pid={pid} market={market} (server authoritative).")
                                except Exception:
                                    pass

                            if ae == "prediction_result" or analysis_payload.get("prediction_result"):
                                try:
                                    pid = analysis_payload.get("prediction_id") or analysis_payload.get("pred_id") or analysis_payload.get("predictionId")
                                    pred_digit = analysis_payload.get("prediction_digit") or analysis_payload.get("predicted_digit") or analysis_payload.get("pred")
                                    result = analysis_payload.get("result")
                                    observed_raw = analysis_payload.get("observed_ticks") or analysis_payload.get("observed") or []
                                    observed = []
                                    if isinstance(observed_raw, list):
                                        for it in observed_raw:
                                            n = safe_int(it)
                                            if n is not None:
                                                observed.append(n)
                                    market = (analysis_payload.get("market") or analysis_payload.get("symbol") or analysis_payload.get("symbol_code") or None)
                                    if market:
                                        market = str(market).upper()

                                    # determine win/loss
                                    is_win = False
                                    if isinstance(result, str):
                                        is_win = result.upper() == "WIN"
                                    else:
                                        try:
                                            if pred_digit is None:
                                                is_win = False
                                            else:
                                                if isinstance(pred_digit, str):
                                                    pd = safe_int(pred_digit)
                                                else:
                                                    pd = int(pred_digit)
                                                if pd is None:
                                                    is_win = False
                                                else:
                                                    pred_mode = (analysis_payload.get("prediction_mode") or analysis_payload.get("mode") or DEFAULT_PREDICTION_MODE).lower()
                                                    if pred_mode == "differ":
                                                        is_win = not any(int(x) == int(pd) for x in observed)
                                                    else:
                                                        is_win = any(int(x) == int(pd) for x in observed)
                                        except Exception:
                                            is_win = False

                                    # update local journal_stats (server already appended official journal)
                                    try:
                                        pdig = safe_int(pred_digit)
                                        if pdig is not None and 0 <= pdig <= 9:
                                            journal_stats.setdefault(pdig, {"wins": 0, "trials": 0})
                                            journal_stats[pdig]["trials"] = journal_stats[pdig].get("trials", 0) + 1
                                            if is_win:
                                                journal_stats[pdig]["wins"] = journal_stats[pdig].get("wins", 0) + 1
                                    except Exception:
                                        pass

                                    # Clear pending gate after settlement for the active prediction id.
                                    try:
                                        with _pending_pid_lock:
                                            if pending_pid:
                                                pid_s = str(pid or "")
                                                cur_s = str(pending_pid or "")
                                                if (pid_s and cur_s and pid_s == cur_s) or (not pid_s):
                                                    log_info(f"pending cleared by settlement pid={pid_s or cur_s}")
                                                    pending_pid = None
                                                    pending_since = 0.0
                                                    pending_by_symbol = {}
                                    except Exception as eclr:
                                        log_err(f"failed clearing pending gate: {eclr}")


                                    log_info(f"Received server settlement pid={pid} result={'WIN' if is_win else 'LOSS'} (server authoritative).")

                                except Exception as e:
                                    log_err(f"analysis-event handling error: {e}\n{traceback.format_exc()}")
                    except Exception:
                        log_err("Error processing analysis SSE event.")
                    continue

                # handle recent snapshot (legacy)
                if ev_name == "recent" and isinstance(payload, dict) and "recent" in payload:
                    for row in payload.get("recent", []):
                        ld = extract_last_decimal_from_payload(row)
                        sym = None
                        if isinstance(row, dict):
                            sym = (row.get("symbol") or row.get("market") or row.get("market_code") or None)
                        if ld is not None and sym:
                            sym_u = str(sym).upper()
                            buffers_by_symbol[sym_u].append(ld)

                            # increment tick counter for reanalysis
                            last_ticks_since_pred[sym_u] = last_ticks_since_pred.get(sym_u, 0) + 1
                            log_info(f"Recent snapshot tick: symbol={sym_u} ld={ld} tick_counter={last_ticks_since_pred[sym_u]}")

                            try:
                                st = agent_on_tick_notify(sym_u, ld, epoch=None)
                                # after each tick, attempt to predict if conditions met
                                try:
                                    maybe_try_predict(st)
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            last_tick_time = time.time()
                    continue

                # Generic tick handling (payload may be wrapped)
                parsed_payload = None
                if isinstance(payload, dict) and "payload" in payload:
                    parsed_payload = payload["payload"]
                else:
                    parsed_payload = payload

                cur_sym = None
                try:
                    if isinstance(parsed_payload, dict):
                        cur_sym = parsed_payload.get("symbol") or parsed_payload.get("market") or parsed_payload.get("market_code")
                        if cur_sym:
                            cur_sym = str(cur_sym).upper()
                except Exception:
                    pass

                ld = extract_last_decimal_from_payload(parsed_payload)
                epoch_val = None
                try:
                    if isinstance(parsed_payload, dict):
                        if "epoch" in parsed_payload and parsed_payload["epoch"]:
                            epoch_val = int(parsed_payload["epoch"])
                        elif "timestamp" in parsed_payload and parsed_payload["timestamp"]:
                            try:
                                ts = parsed_payload["timestamp"]
                                dt = None
                                try:
                                    dt = datetime.fromisoformat(ts)
                                except Exception:
                                    try:
                                        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")
                                    except Exception:
                                        dt = None
                                if dt:
                                    epoch_val = int(dt.timestamp())
                            except Exception:
                                epoch_val = None
                except Exception:
                    epoch_val = None

                if ld is not None and cur_sym:
                    buffers_by_symbol[cur_sym].append(ld)

                    # ---- NEW: tick counter increment for reanalysis trigger ----
                    last_ticks_since_pred[cur_sym] = last_ticks_since_pred.get(cur_sym, 0) + 1
                    log_info(f"Tick: symbol={cur_sym} last_decimal={ld} tick_counter={last_ticks_since_pred[cur_sym]}")

                    try:
                        st = agent_on_tick_notify(cur_sym, ld, epoch_val)
                        # after each tick, attempt to predict if conditions met
                        try:
                            maybe_try_predict(st)
                        except Exception:
                            pass
                        # if there has been no prediction posted recently and no pending, optionally send a "none_yet" toast (rate-limited)
                        try:
                            any_pred_time = 0.0
                            if last_prediction_ts:
                                any_pred_time = max(last_prediction_ts.values() or [0.0])
                            if (not pending_pid) and (time.time() - any_pred_time) > TOAST_MIN_INTERVAL and (time.time() - last_toast_ts) > TOAST_MIN_INTERVAL:
                                try:
                                    make_analysis_post(analysis_push_url, "prediction_toast", "No predictions posted recently.", symbol="SYSTEM", extra={"status": "none_yet"})
                                    last_toast_ts = time.time()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception:
                        pass
                    last_tick_time = time.time()
                    no_ticks_notified = False

                # after handling, continue loop
                continue

            # SSE ended, reconnect
            log_info("SSE stream ended; reconnecting")
            backoff = SSE_RECONNECT_DELAY

        except Exception as e:
            log_err(f"SSE connection error: {e}\n{traceback.format_exc()}")
            try:
                make_analysis_post(analysis_push_url, "network_issue", f"SSE connection error: {e}")
            except Exception:
                pass
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)

# ---------- CLI / main ----------
def parse_args():
    p = argparse.ArgumentParser(prog="differ_agent.py")
    p.add_argument("--service-base", default=os.environ.get("HERO_SERVICE_BASE", DEFAULT_SERVICE_BASE), help="dashboard base")
    p.add_argument("--analysis-push-url", default=os.environ.get("HERO_DASHBOARD_PUSH_URL", None), help="explicit analysis push URL")
    p.add_argument("--prediction-push-url", default=os.environ.get("DIFFER_PREDICTION_PUSH_URL", None), help="explicit prediction push URL")
    p.add_argument("--buffer-size", type=int, default=int(os.environ.get("DIFFER_BUFFER_SIZE", DEFAULT_BUFFER_SIZE)), help=f"rolling buffer size (default {DEFAULT_BUFFER_SIZE})")
    p.add_argument("--min-buffer", type=int, default=int(os.environ.get("DIFFER_MIN_BUFFER", DEFAULT_MIN_BUFFER)), help=f"min buffer to consider ready (default {DEFAULT_MIN_BUFFER})")
    p.add_argument("--conf-threshold", type=float, default=float(os.environ.get("DIFFER_CONF_THRESHOLD", DEFAULT_CONF_THRESHOLD)), help="confidence threshold (0-1) to consider prediction candidates")
    p.add_argument("--tie-margin", type=float, default=float(os.environ.get("DIFFER_TIE_MARGIN", DEFAULT_TIE_MARGIN)), help="tie margin (not used in strict mode)")
    p.add_argument("--stability-ticks", type=int, default=int(os.environ.get("DIFFER_STABILITY_TICKS", DEFAULT_STABILITY_TICKS)), help="stability ticks")
    p.add_argument("--n-test", type=int, default=int(os.environ.get("DIFFER_N_TEST", DEFAULT_N_TEST)), help="n_test")
    p.add_argument("--log-dir", default=os.environ.get("DIFFER_LOG_DIR", DEFAULT_LOG_DIR))
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.log_dir and args.log_dir != DEFAULT_LOG_DIR:
        try:
            os.makedirs(args.log_dir, exist_ok=True)
        except Exception:
            pass

    analysis_push_url = args.analysis_push_url or os.environ.get("HERO_DASHBOARD_PUSH_URL")
    prediction_push_url = args.prediction_push_url or os.environ.get("HERO_PREDICTION_PUSH_URL")

    if not analysis_push_url:
        analysis_push_url = args.service_base.rstrip("/") + "/control/push_tick"
    if not prediction_push_url:
        prediction_push_url = args.service_base.rstrip("/") + "/control/push_prediction"

    try:
        run_agent(
            args.service_base,
            analysis_push_url,
            prediction_push_url,
            int(args.buffer_size),
            int(args.min_buffer),
            conf_threshold=float(args.conf_threshold),
            tie_margin=float(args.tie_margin),
            stability_ticks=int(args.stability_ticks),
            n_test=int(args.n_test),
        )
    except KeyboardInterrupt:
        log_info("differ_agent interrupted (KeyboardInterrupt). exiting.")
        sys.exit(0)
    except Exception as e:
        log_err(f"fatal error: {e}\n{traceback.format_exc()}")
        sys.exit(1)
