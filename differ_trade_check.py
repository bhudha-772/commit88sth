#!/usr/bin/env python3
"""
differ_trade_check.py

Request a DIGITDIFF (Differs) proposal with a 1-tick duration and a randomly-picked barrier digit (0-9).
No fallback to CALL or other contract types is performed.

- Tries up to N_RANDOM_BARRIER_TRIES random barriers to find a valid DIGITDIFF proposal (configurable).
- If a valid proposal is returned, attempts a buy using the proposal id and returned ask price.
- Waits for the contract to reach a final/settled state (no longer uses intermediate mark-to-market profit).
- Logs to stdout and to ~/HeroX/deriv_trade_check_differ.log.

Requirements:
    pip3 install websockets requests

Usage:
    export DERIV_TOKEN="your_token_here"
    python3 differ_trade_check.py
"""
import os
import sys
import json
import time
import logging
import asyncio
import random
import threading
import json
import urllib.request
import urllib.error
import ssl
from typing import Optional

# --- logging setup ---
LOG_PATH = os.path.expanduser("~/HeroX/deriv_trade_check_differ.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
log = logging.getLogger("differ_trade_check")

# --- defaults & env ---
DERIV_TOKEN = os.environ.get("DERIV_TOKEN") or os.environ.get("DERIV_API_TOKEN") or ""
WS_URL = os.environ.get("DERIV_WS", "wss://ws.binaryws.com/websockets/v3?app_id=71710")
# Dashboard endpoints (hero_service will export these when spawning the trader).
# If not provided, fall back to localhost endpoints so script remains usable during dev.
HERO_DASHBOARD_PUSH_URL = os.environ.get("HERO_DASHBOARD_PUSH_URL") or "http://127.0.0.1:5000/control/push_tick"
HERO_DASHBOARD_SSE_URL = os.environ.get("HERO_DASHBOARD_SSE_URL") or "http://127.0.0.1:5000/events"

SYMBOL = os.environ.get("DIFFER_SYMBOL", "R_10")
AMOUNT = float(os.environ.get("DIFFER_AMOUNT", "1.0"))
# Use tick unit for digit contracts
DURATION = int(os.environ.get("DIFFER_DURATION", "1"))       # number of ticks (we expect 1)
DURATION_UNIT = os.environ.get("DIFFER_DURATION_UNIT", "t")  # 't' for ticks
CURRENCY = os.environ.get("DIFFER_CURRENCY", "USD")
# How many random barrier attempts before giving up
N_RANDOM_BARRIER_TRIES = int(os.environ.get("DIFFER_RANDOM_TRIES", "6"))
# timeout for websocket request responses
TIMEOUT_SECONDS = float(os.environ.get("DIFFER_TIMEOUT", "10.0"))
# small in-memory dedupe cache for prediction ids (protects against duplicate SSE events)
from collections import deque as _deque
import threading as _threading

_SEEN_PIDS_LOCK = _threading.Lock()
_SEEN_PIDS_DEQUE = _deque(maxlen=1000)
_SEEN_PIDS_SET = set()
# timeout waiting for final settlement after buy (seconds)
FINAL_POLL_TIMEOUT = float(os.environ.get("DIFFER_FINAL_POLL_TIMEOUT", "30.0"))
FINAL_POLL_INTERVAL = float(os.environ.get("DIFFER_FINAL_POLL_INTERVAL", "0.5"))

# --- minimal websockets helper (async) ---
try:
    import websockets
except Exception:
    log.error("Missing dependency 'websockets'. Install with: pip3 install websockets")
    raise

class DerivWS:
    def __init__(self, url: str, timeout: float = 10.0):
        self.url = url
        self.ws = None
        self.timeout = timeout
        self.req_id = 1
        self.pending = {}  # req_id -> asyncio.Future
        self._recv_task = None

    async def connect(self):
        self.ws = await websockets.connect(self.url, max_size=None)
        # start receiver task
        self._recv_task = asyncio.create_task(self._receiver_loop())

    async def _receiver_loop(self):
        try:
            async for msg in self.ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                req_id = None
                if isinstance(data, dict):
                    req_id = data.get("req_id")
                if req_id and req_id in self.pending:
                    fut = self.pending.pop(req_id)
                    if not fut.done():
                        fut.set_result(data)
                else:
                    # best-effort: deliver to first waiting future
                    for k, fut in list(self.pending.items()):
                        if not fut.done():
                            fut.set_result(data)
                            self.pending.pop(k, None)
                            break
        except Exception as e:
            # propagate exception to any pending futures
            for k, fut in list(self.pending.items()):
                if not fut.done():
                    fut.set_exception(e)
            self.pending.clear()

    async def send_and_wait(self, payload: dict, timeout: float | None = None):
        if self.ws is None:
            await self.connect()
        rid = self.req_id
        self.req_id += 1
        payload = dict(payload)
        payload["req_id"] = rid
        fut = asyncio.get_event_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps(payload))
        try:
            res = await asyncio.wait_for(fut, timeout=(timeout or self.timeout))
            return res
        finally:
            # cleanup in case nothing consumed it
            self.pending.pop(rid, None)

    async def close(self):
        try:
            if self.ws:
                await self.ws.close()
        except Exception:
            pass
        try:
            if self._recv_task:
                self._recv_task.cancel()
        except Exception:
            pass

# --- helpers ---

def get_trader_token(prefer_mode="demo", server_base="http://127.0.0.1:5000"):
    """
    Resolve token for trader in headless mode.
    Order:
     1) DERIV_TOKEN env
     2) server-local: /control/get_server_token?mode=demo|real (only local)
     3) GLOBAL legacy env (if any)
    Returns (mode, token) or (None, None) if none found.
    """
    # 1) env
    tok = os.environ.get("DERIV_TOKEN") or os.environ.get("DERIV_API_TOKEN")
    if tok:
        return ("env", tok)

    # 2) server-local - try demo then real
    for mode in ("demo", "real"):
        try:
            url = f"{server_base}/control/get_server_token?mode={mode}"
            req = urllib.request.Request(url, method="GET")
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=4, context=ctx) as resp:
                try:
                    body = resp.read().decode("utf-8")
                    jo = json.loads(body)
                except Exception:
                    jo = {}
            if jo.get("ok") and jo.get("token"):
                return (mode, jo.get("token"))
        except Exception:
            # ignore and continue to next mode
            pass

    # not found
    return (None, None)

def pick_random_barrier() -> str:
    return str(random.randint(0, 9))

def parse_balance_response(resp) -> float:
    """
    Accepts the raw response for {"balance":1} and tries to return a numeric balance.
    Returns None if it can't find one.
    """
    try:
        if isinstance(resp, dict):
            # some replies: { "balance": { "balance": 123.45, ... }, ... }
            b = resp.get("balance")
            if isinstance(b, dict):
                # nested structure from some responses
                if "balance" in b:
                    return float(b.get("balance"))
                # sometimes it's just the numeric value directly stored under 'balance'
            if isinstance(b, (int, float, str)):
                return float(b)
            # other shape: resp itself contains 'balance' field as nested
            if "balance" in resp and isinstance(resp["balance"], (int, float, str)):
                return float(resp["balance"])
    except Exception:
        pass
    return None

def is_final_contract_state(poc: dict) -> bool:
    """Return True if poc clearly indicates final/settled contract state."""
    if not isinstance(poc, dict):
        return False
    # explicit flags
    try:
        if bool(poc.get("is_settled")) or bool(poc.get("is_expired")):
            return True
    except Exception:
        pass
    status = str(poc.get("status") or "").lower()
    if status in ("sold", "settled", "closed", "finished", "expired"):
        return True
    # some responses include 'is_sold'
    if bool(poc.get("is_sold")):
        return True
    return False

async def poll_contract_until_finished(client: DerivWS, contract_id: int, timeout: float = FINAL_POLL_TIMEOUT, interval: float = FINAL_POLL_INTERVAL):
    """
    Poll proposal_open_contract for contract_id until it is definitely finished/settled,
    or until timeout. Returns the final contract object (dict) or None on timeout/error.
    Ignores intermediate 'profit' values while status == 'open' or validation errors indicate 'waiting'.
    """
    start = time.time()
    last_resp = None
    while True:
        if time.time() - start > timeout:
            return None
        try:
            payload = {"proposal_open_contract": 1, "contract_id": int(contract_id)}
            resp = await client.send_and_wait(payload, timeout=max(3.0, interval + 1.0))
            last_resp = resp
            poc = None
            if isinstance(resp, dict):
                if "proposal_open_contract" in resp and isinstance(resp["proposal_open_contract"], dict):
                    poc = resp["proposal_open_contract"]
                elif resp.get("contract_id") or resp.get("status") or resp.get("buy_price") or resp.get("profit") is not None:
                    poc = resp
            if poc:
                # If it's an obviously final/settled state, return it
                if is_final_contract_state(poc):
                    return poc
                # If validation_error indicates the contract hasn't even started, keep polling
                val_err = str(poc.get("validation_error") or "").strip().lower()
                if "waiting for entry" in val_err or val_err.startswith("waiting for"):
                    await asyncio.sleep(interval)
                    continue
                # If still open, continue polling until settled or timeout
        except Exception:
            # swallow transient issues and retry until timeout
            pass
        await asyncio.sleep(interval)

# --- Simple notifier back to hero dashboard (POST JSON) ---
import urllib.request as _urllib_request
import urllib.error as _urllib_error

def notify_hero_push(payload: dict):
    # Use environment value if provided (hero_service injects this env automatically).
    push_url = os.environ.get("HERO_DASHBOARD_PUSH_URL") or HERO_DASHBOARD_PUSH_URL

    # Wrap analysis payloads under "payload" so hero_service.push_tick() processes them
    send_obj = payload
    try:
        # If payload looks like an analysis event, wrap it
        if isinstance(payload, dict) and payload.get("analysis_event"):
            send_obj = {"payload": payload}
    except Exception:
        send_obj = payload

    data = json.dumps(send_obj).encode("utf-8")
    req = _urllib_request.Request(push_url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _urllib_request.urlopen(req, timeout=4.0) as resp:
            return True
    except Exception as e:
        log.warning("notify_hero_push failed: %s", e)
        return False

# Reliable small wrapper that ensures the expected final payload shape and retries a couple times
def push_final_to_hero(
    final_contract: dict,
    prediction_id: str = None,
    predicted_digit: int = None,
    result: str = None,
    profit: float = None,
    profit_percentage: float = None,
    contract_type: str = None,
    barrier: str = None,
    prediction_mode: str = None,
):
    """
    Build a canonical 'prediction_result' payload and send it to hero dashboard.
    This enforces strict canonical fields:
      - result: only 'WIN' or 'LOSS' (zero treated as LOSS)
      - includes actual digit when extractable as 'actual'/'result_digit'
      - includes profit and profit_percentage when available
    Retries a couple times on failure with small backoff.
    """
    try:
        # Normalize market/symbol
        market = ''
        symbol = ''
        try:
            if isinstance(final_contract, dict):
                market = final_contract.get('underlying') or final_contract.get('shortcode') or final_contract.get('display_name') or final_contract.get('symbol') or ''
                symbol = market
        except Exception:
            market = symbol = ''

        # Try to extract canonical profit / pct if not passed in
        try:
            if profit is None:
                # prefer top-level profit fields on final_contract
                p = None
                if isinstance(final_contract, dict):
                    for pk in ("profit", "profit_amount", "payout", "cash_profit"):
                        if pk in final_contract and final_contract.get(pk) is not None:
                            try:
                                p = float(final_contract.get(pk))
                                break
                            except Exception:
                                p = None
                profit = p
        except Exception:
            profit = profit

        try:
            if profit_percentage is None:
                pp = None
                if isinstance(final_contract, dict):
                    for ppk in ("profit_percentage", "profitPct", "profit_percent"):
                        if ppk in final_contract and final_contract.get(ppk) is not None:
                            try:
                                pp = float(final_contract.get(ppk))
                                break
                            except Exception:
                                pp = None
                profit_percentage = pp
        except Exception:
            profit_percentage = profit_percentage

        # Try to get actual/result digit
        actual_digit = None
        try:
            if isinstance(final_contract, dict):
                if "result_digit" in final_contract and final_contract.get("result_digit") is not None:
                    actual_digit = int(final_contract.get("result_digit"))
                elif "actual" in final_contract and final_contract.get("actual") is not None:
                    actual_digit = int(final_contract.get("actual"))
                else:
                    # look inside audit_details or tick_stream
                    ts = final_contract.get('tick_stream') or final_contract.get('audit_details', {}).get('all_ticks') or final_contract.get('observed_ticks') or []
                    if isinstance(ts, (list, tuple)) and len(ts) > 0:
                        last = ts[-1]
                        try:
                            # last may be a dict: { "tick": {"last_digit": X} } or similar
                            if isinstance(last, dict):
                                if "tick" in last and isinstance(last["tick"], dict):
                                    cand = last["tick"].get("last_digit") or last["tick"].get("last_decimal") or last["tick"].get("quote")
                                else:
                                    cand = last.get("last_digit") or last.get("last_decimal") or last.get("tick") or last.get("quote")
                                if cand is not None:
                                    actual_digit = int(str(cand)[-1:])
                            else:
                                actual_digit = int(str(last)[-1:])
                        except Exception:
                            actual_digit = None
        except Exception:
            actual_digit = None

        # Build canonical payload
        payload = {
            "analysis_event": "prediction_result",
            "prediction_id": prediction_id or str(final_contract.get('contract_id') or final_contract.get('id') or f"c{int(time.time()*1000)}"),
            "prediction_digit": (int(predicted_digit) if predicted_digit is not None else None),
            "market": market,
            "symbol": symbol,
            "contract_type": (str(contract_type).upper() if contract_type else None),
            "barrier": (str(barrier) if barrier is not None else None),
            "prediction_mode": (str(prediction_mode) if prediction_mode else None),
            # 'actual' and 'result_digit' are the same concept: include both for compatibility
            "actual": actual_digit,
            "result_digit": actual_digit,
            # profit fields
            "profit": (float(profit) if profit is not None else (final_contract.get('profit') if isinstance(final_contract, dict) and final_contract.get('profit') is not None else None)),
            "profit_percentage": (float(profit_percentage) if profit_percentage is not None else (final_contract.get('profit_percentage') if isinstance(final_contract, dict) and final_contract.get('profit_percentage') is not None else None)),
            "observed_ticks": final_contract.get('tick_stream') or final_contract.get('audit_details', {}).get('all_ticks') or final_contract.get('observed_ticks') or [],
            "final_contract": final_contract,
            "ts": int(time.time()),
            "epoch": int(time.time())
        }

        # Force result to only WIN or LOSS using profit preference (ZERO => LOSS per user)
        try:
            # if caller provided result, ignore it (we enforce strict mapping)
            p = payload.get('profit')
            pp = payload.get('profit_percentage')
            if p is not None:
                payload['result'] = 'WIN' if float(p) > 0 else 'LOSS'
            elif pp is not None:
                payload['result'] = 'WIN' if float(pp) > 0 else 'LOSS'
            else:
                # Last-resort inference if profit fields are missing.
                pred = payload.get('prediction_digit')
                act = payload.get('actual')
                try:
                    ctype = str(payload.get("contract_type") or "").upper()
                    bar = payload.get("barrier")
                    if ctype == "DIGITDIFF":
                        payload['result'] = 'WIN' if (pred is not None and act is not None and int(pred) != int(act)) else 'LOSS'
                    elif ctype == "DIGITOVER":
                        thresh = int(bar if bar is not None else pred)
                        payload['result'] = 'WIN' if (act is not None and int(act) > int(thresh)) else 'LOSS'
                    elif ctype == "DIGITUNDER":
                        thresh = int(bar if bar is not None else pred)
                        payload['result'] = 'WIN' if (act is not None and int(act) < int(thresh)) else 'LOSS'
                    elif (pred is not None) and (act is not None) and int(pred) == int(act):
                        payload['result'] = 'WIN'
                    else:
                        payload['result'] = 'LOSS'
                except Exception:
                    payload['result'] = 'LOSS'
        except Exception:
            payload['result'] = 'LOSS'

        # tidy: remove None values so dashboard gets compact payload
        tidy = {k: v for k, v in payload.items() if v is not None}

        # Try notify_hero_push multiple times
        retries = 3
        backoff = 0.5
        for attempt in range(1, retries + 1):
            ok = notify_hero_push(tidy)
            if ok:
                log.info("push_final_to_hero: posted prediction_result to dashboard (id=%s, result=%s)", tidy.get('prediction_id'), tidy.get('result'))
                return True
            else:
                log.warning("push_final_to_hero: attempt %d failed, retrying after %.2fs", attempt, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 4.0)
        log.error("push_final_to_hero: all attempts failed for prediction_id=%s", tidy.get('prediction_id'))
        return False
    except Exception as ex:
        log.exception("push_final_to_hero exception: %s", ex)
        return False

    """
    Build a canonical 'prediction_result' payload and send it to hero dashboard.
    Retries a couple times on failure with small backoff.
    """
    try:
        # Normalize a market/symbol extraction
        market = None
        symbol = None
        try:
            # Deriv final contract commonly has 'underlying' or 'shortcode' or 'display_name'
            market = final_contract.get('underlying') or final_contract.get('shortcode') or final_contract.get('display_name') or ''
            symbol = market
        except Exception:
            market = symbol = ''

        payload = {
            "analysis_event": "prediction_result",
            "prediction_id": prediction_id or str(final_contract.get('contract_id') or final_contract.get('id') or f"c{int(time.time()*1000)}"),
            "prediction_digit": (int(predicted_digit) if predicted_digit is not None else None),
            "market": market,
            "symbol": symbol,
            "result": (str(result).upper() if result is not None else None),
            "profit": (float(profit) if profit is not None else (final_contract.get('profit') if final_contract.get('profit') is not None else None)),
            "profit_percentage": (float(profit_percentage) if profit_percentage is not None else (final_contract.get('profit_percentage') if final_contract.get('profit_percentage') is not None else None)),
            "observed_ticks": final_contract.get('tick_stream') or final_contract.get('audit_details', {}).get('all_ticks') or [],
            "final_contract": final_contract,
            "ts": int(time.time()),
            "epoch": int(time.time())
        }

                # If result wasn't provided, infer from profit fields.
        # STRICT: only WIN or LOSS (zero => LOSS).
        if not payload.get('result'):
            try:
                p = payload.get('profit')
                pp = payload.get('profit_percentage')
                if p is not None:
                    payload['result'] = 'WIN' if float(p) > 0 else 'LOSS'
                elif pp is not None:
                    payload['result'] = 'WIN' if float(pp) > 0 else 'LOSS'
                else:
                    # no profit info: treat as LOSS (safer default per user)
                    payload['result'] = 'LOSS'
            except Exception:
                payload['result'] = 'LOSS'


        # Remove None fields to keep payload tidy (hero/journal tolerates missing but prefer present)
        tidy = {k: v for k, v in payload.items() if v is not None}

        # Try notify_hero_push multiple times
        retries = 3
        backoff = 0.5
        for attempt in range(1, retries + 1):
            ok = notify_hero_push(tidy)
            if ok:
                log.info("push_final_to_hero: posted prediction_result to dashboard (id=%s, result=%s)", tidy.get('prediction_id'), tidy.get('result'))
                return True
            else:
                log.warning("push_final_to_hero: attempt %d failed, retrying after %.2fs", attempt, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 4.0)
        log.error("push_final_to_hero: all attempts failed for prediction_id=%s", tidy.get('prediction_id'))
        return False
    except Exception as ex:
        log.exception("push_final_to_hero exception: %s", ex)
        return False

# ---------------------------------------------------------------------
# Helper: extract canonical fields from a Deriv "final contract" object
# Returns tuple: (actual_digit_or_None, profit_or_None, profit_pct_or_None, observed_ticks_list_or_None)
def _extract_final_details(final_contract: dict):
    actual = None
    profit = None
    profit_pct = None
    observed = None
    try:
        if not isinstance(final_contract, dict):
            return (None, None, None, None)

        # Profit: try common field names and coerce to float
        for pk in ("profit", "profit_amount", "payout", "cash_profit"):
            if pk in final_contract and final_contract.get(pk) is not None:
                try:
                    profit = float(final_contract.get(pk))
                    break
                except Exception:
                    profit = None

        # Profit percentage: common field names
        for ppk in ("profit_percentage", "profitPct", "profit_percent"):
            if ppk in final_contract and final_contract.get(ppk) is not None:
                try:
                    profit_pct = float(final_contract.get(ppk))
                    break
                except Exception:
                    profit_pct = None

        # Observed / tick stream
        if "tick_stream" in final_contract and isinstance(final_contract.get("tick_stream"), (list, tuple)):
            observed = final_contract.get("tick_stream")
        elif "audit_details" in final_contract and isinstance(final_contract.get("audit_details"), dict):
            observed = final_contract.get("audit_details").get("all_ticks") or final_contract.get("audit_details").get("ticks")
        elif "observed_ticks" in final_contract:
            observed = final_contract.get("observed_ticks")

        # Try direct fields for actual/result digit
        if "result_digit" in final_contract and final_contract.get("result_digit") is not None:
            try:
                actual = int(final_contract.get("result_digit"))
            except Exception:
                actual = None
        elif "actual" in final_contract and final_contract.get("actual") is not None:
            try:
                actual = int(final_contract.get("actual"))
            except Exception:
                actual = None
        else:
            # Last-resort: derive from last tick in observed/tick_stream
            ts = observed or []
            if isinstance(ts, (list, tuple)) and len(ts) > 0:
                last = ts[-1]
                try:
                    if isinstance(last, dict):
                        # common shapes: { "tick": { "last_digit": X, "quote": ... } }
                        if "tick" in last and isinstance(last["tick"], dict):
                            cand = last["tick"].get("last_digit") or last["tick"].get("last_decimal") or last["tick"].get("quote")
                        else:
                            cand = last.get("last_digit") or last.get("last_decimal") or last.get("tick") or last.get("quote")
                        if cand is not None:
                            try:
                                actual = int(cand)
                            except Exception:
                                # try last char fallback
                                try:
                                    actual = int(str(cand).strip()[-1:])
                                except Exception:
                                    actual = None
                    else:
                        # if tick is primitive/number/string, attempt parse
                        actual = int(str(last).strip()[-1:])
                except Exception:
                    actual = None

    except Exception:
        pass
    return (actual, profit, profit_pct, observed)
# ---------------------------------------------------------------------

def start_sse_listener(sse_url: str, loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[int]"):
    try:
        import requests
    except Exception:
        log.error("SSE listener requires 'requests' package. Install with: pip3 install requests")
        return

    def _run():
        backoff = 1.0
        while True:
            try:
                effective_url = sse_url or os.environ.get("HERO_DASHBOARD_SSE_URL") or HERO_DASHBOARD_SSE_URL
                log.info("SSE listener connecting to %s", effective_url)
                with requests.get(effective_url, stream=True, timeout=15) as resp:
                    if resp.status_code != 200:
                        log.warning("SSE connect returned status %s", resp.status_code)
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, 30.0)
                        continue
                    backoff = 1.0
                    buf_lines = []
                    for raw_line in resp.iter_lines(decode_unicode=True):
                        if raw_line is None:
                            continue
                        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

                        # blank line => dispatch block
                        if line.strip() == "":
                            event_name = None
                            data_lines = []
                            for l in buf_lines:
                                try:
                                    if l.startswith("event:"):
                                        event_name = l.split(":", 1)[1].strip()
                                    elif l.startswith("data:"):
                                        data_lines.append(l.split(":", 1)[1].strip())
                                except Exception:
                                    continue
                            buf_lines = []
                            if not data_lines:
                                continue

                            data_text = "\n".join(data_lines)
                            payload = None
                            try:
                                payload = json.loads(data_text)
                            except Exception:
                                payload = None

                            # Only process 'analysis' events that are primary prediction postings.
                            try:
                                if event_name == "analysis" and isinstance(payload, dict):
                                    ae = payload.get("analysis_event") or payload.get("event") or None
                                    # Only enqueue the canonical production events to avoid duplicates.
                                    if ae and ae in ("prediction_posted", "prediction_produced"):
                                        # extract canonical prediction id for dedupe
                                        pid = payload.get("prediction_id") or payload.get("pred_id") or payload.get("id") or None

                                        # dedupe check
                                        dedupged = False
                                        if pid:
                                            try:
                                                with _SEEN_PIDS_LOCK:
                                                    if pid in _SEEN_PIDS_SET:
                                                        dedupged = True
                                                    else:
                                                        _SEEN_PIDS_SET.add(pid)
                                                        _SEEN_PIDS_DEQUE.append(pid)
                                                        # ensure set matches deque capacity
                                                        if len(_SEEN_PIDS_DEQUE) == _SEEN_PIDS_DEQUE.maxlen:
                                                            # remove leftmost now/then (deque automatically evicts once append occurs past maxlen)
                                                            pass
                                            except Exception:
                                                pass

                                        if dedupged:
                                            # skip duplicate prediction event
                                            log.info("SSE: skipped duplicate prediction pid=%s", pid)
                                            continue

                                        pd = payload.get("prediction_digit") or payload.get("predicted") or payload.get("pred")
                                        market = payload.get("market") or payload.get("symbol") or payload.get("market_code") or None

                                        # coerce pd to integer if possible
                                        try:
                                            pd_int = int(pd) if pd is not None else None
                                        except Exception:
                                            pd_int = None

                                        # Build normalized signal object
                                        sig = dict(payload) if isinstance(payload, dict) else {}
                                        sig["prediction_digit"] = pd_int
                                        if market:
                                            sig["symbol"] = str(market).upper()

                                        # robust stake extraction: look at stake, amount, and nested raw element
                                        sig_stake = None
                                        try:
                                            if "stake" in payload and payload.get("stake") is not None:
                                                sig_stake = payload.get("stake")
                                            elif "amount" in payload and payload.get("amount") is not None:
                                                sig_stake = payload.get("amount")
                                            else:
                                                # look for nested raw payload
                                                raw_v = payload.get("raw") or payload.get("payload") or {}
                                                if isinstance(raw_v, dict):
                                                    if "stake" in raw_v and raw_v.get("stake") is not None:
                                                        sig_stake = raw_v.get("stake")
                                                    elif "amount" in raw_v and raw_v.get("amount") is not None:
                                                        sig_stake = raw_v.get("amount")
                                            if sig_stake is not None:
                                                try:
                                                    sig["stake"] = float(sig_stake)
                                                except Exception:
                                                    sig["stake"] = None
                                        except Exception:
                                            pass

                                        # enqueue the full signal for the daemon to process
                                        try:
                                            loop.call_soon_threadsafe(queue.put_nowait, sig)
                                        except Exception as e:
                                            log("SSE enqueue failed (sig obj): %s", e)
                            except Exception as e:
                                log.warning("SSE processing error (ignored): %s -- data_text=%s", e, (data_text[:400] + "...") if data_text and len(data_text) > 400 else data_text)
                        else:
                            buf_lines.append(line)

            except Exception as e:
                log.warning("SSE listener error: %s", e)
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 30.0)

    t = threading.Thread(target=_run, name="sse_listener", daemon=True)
    t.start()


# Helper: fetch server-side token for a given mode (demo|real)
def fetch_server_token(mode: str, hero_base: str = None, timeout: float = 3.0) -> Optional[str]:
    """
    Try to GET full token from hero_service endpoint /control/get_server_token?mode=<mode>.
    hero_base: base URL, e.g. http://127.0.0.1:5000 (if None, use env HERO_DASHBOARD_PUSH_URL base or http://127.0.0.1:5000)
    """
    try:
        import urllib.request as ureq, urllib.error as uerr, urllib.parse as uparse
        if not mode or mode not in ("demo", "real"):
            return None
        # derive base from env fallback
        if not hero_base:
            hero_base = os.environ.get("HERO_DASHBOARD_PUSH_URL") or os.environ.get("HERO_DASHBOARD_SSE_URL") or "http://127.0.0.1:5000"
            hero_base = hero_base.rstrip("/")
            if hero_base.endswith("/control/push_tick"):
                hero_base = hero_base.rsplit("/control/push_tick", 1)[0]
        url = f"{hero_base}/control/get_server_token?mode={ureq.quote(mode)}"
        req = ureq.Request(url, method="GET")
        with ureq.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            raw = resp.read().decode("utf-8")
            try:
                j = json.loads(raw)
            except Exception:
                return None
            if j.get("ok") and j.get("present"):
                return j.get("token")
    except Exception:
        return None
    return None

# --- Main flow ---
async def run_trade_check():
    if not DERIV_TOKEN:
        # try to fetch server-side demo token as a fallback
        token_for_auth = fetch_server_token("demo")
        if token_for_auth:
            log.info("Using server-provided demo token for single-run check")
        else:
            log.error("DERIV_TOKEN not set and no server token available. export DERIV_TOKEN or set it in the dashboard.")
            return 1
    else:
        token_for_auth = DERIV_TOKEN

    client = DerivWS(WS_URL, timeout=TIMEOUT_SECONDS)
    log.info(f"Starting DIGITDIFF trade check: ws={WS_URL} symbol={SYMBOL} amount={AMOUNT} duration={DURATION}{DURATION_UNIT} (ticks)")

    try:
        await client.connect()
    except Exception as e:
        log.error(f"WS connect failed: {e}")
        return 1

    # Authorize
    try:
        log.info("WS open -> sending authorize")
        auth_payload = {"authorize": token_for_auth}
        auth_res = await client.send_and_wait(auth_payload, timeout=8.0)
        log.info("Authorize response: %s", json.dumps(auth_res, indent=2))
    except Exception as e:
        log.error("Authorize failed: %s", e)
        await client.close()
        return 1

    # Balance before trade (best-effort)
    balance_before = None
    try:
        bal_res = await client.send_and_wait({"balance": 1}, timeout=6.0)
        bal_val = parse_balance_response(bal_res)
        if bal_val is not None:
            balance_before = bal_val
            log.info("Balance before trade: %s", balance_before)
        else:
            log.info("Balance response (raw): %s", json.dumps(bal_res, indent=2))
    except Exception as e:
        log.warning("Balance request failed (non-fatal): %s", e)

    # Build DIGITDIFF proposal payload template (tick duration, barrier will be set)
    def build_proposal_payload(barrier_digit: str, target_symbol: str, amount: float = None):
        amt = float(amount) if amount is not None else float(AMOUNT)
        return {
            "proposal": 1,
            "amount": amt,
            "basis": "stake",
            "contract_type": "DIGITDIFF",
            "currency": CURRENCY,
            "duration": int(DURATION),
            "duration_unit": DURATION_UNIT,
            "symbol": str(target_symbol),
            "barrier": str(barrier_digit),   # required for digit contracts
            "product_type": "basic",
        }

    proposal = None
    chosen_barrier = None
    proposal_resp_raw = None
    target_symbol = SYMBOL

    # Try up to N_RANDOM_BARRIER_TRIES random barrier digits (0..9)
    tried = set()
    for attempt in range(1, max(1, N_RANDOM_BARRIER_TRIES) + 1):
        barrier = pick_random_barrier()
        # avoid repeating same barrier in this run
        if barrier in tried and len(tried) < 10:
            continue
        tried.add(barrier)
        pl = build_proposal_payload(barrier, target_symbol)
        log.info("Attempt #%d - Sent DIGITDIFF proposal request (barrier=%s): %s", attempt, barrier, json.dumps(pl, indent=2))
        try:
            resp = await client.send_and_wait(pl, timeout=8.0)
            proposal_resp_raw = resp
            # Check for errors in response - if error, try another barrier
            if isinstance(resp, dict) and resp.get("error"):
                log.warning("Proposal response error for barrier %s: %s", barrier, json.dumps(resp.get("error"), indent=2))
                continue
            # Successful proposal usually appears under 'proposal' or top-level
            if isinstance(resp, dict):
                if "proposal" in resp and isinstance(resp["proposal"], dict):
                    proposal = resp["proposal"]
                else:
                    # If top-level includes keys like 'ask_price' or 'id', treat resp itself as proposal
                    if resp.get("ask_price") or resp.get("id") or resp.get("display_value"):
                        proposal = resp
                if proposal:
                    chosen_barrier = barrier
                    log.info("Received valid DIGITDIFF proposal for barrier %s: %s", barrier, json.dumps(proposal, indent=2))
                    break
            # else keep trying
            log.warning("Proposal response unexpected (no proposal): %s", json.dumps(resp, indent=2))
        except Exception as e:
            log.warning("Proposal request for barrier %s failed: %s", barrier, e)
            continue

    if not proposal:
        log.error("No valid DIGITDIFF proposal received after %d attempts. Aborting (no fallback). Last raw response: %s", len(tried), json.dumps(proposal_resp_raw or {}, indent=2))
        await client.close()
        return 2

    # extract ask_price
    ask_price = None
    if isinstance(proposal, dict):
        ask_price = proposal.get("ask_price") or proposal.get("display_value") or proposal.get("price") or None
        if isinstance(ask_price, str):
            try:
                ask_price = float(ask_price)
            except Exception:
                pass
    if ask_price is None:
        ask_price = float(AMOUNT)

    # get proposal id
    proposal_id = None
    if isinstance(proposal, dict):
        proposal_id = proposal.get("id") or proposal.get("proposal_id") or proposal.get("proposal")

    # Build buy payload using proposal id + price to satisfy InputValidation
    if proposal_id:
        buy_payload = {"buy": str(proposal_id), "price": ask_price}
    else:
        # older-style fallback that repeats proposal params (still DIGITDIFF)
        buy_payload = {
            "buy": 1,
            "price": ask_price,
            "proposal": 1,
            "amount": float(AMOUNT),
            "basis": "stake",
            "contract_type": "DIGITDIFF",
            "currency": CURRENCY,
            "duration": int(DURATION),
            "duration_unit": DURATION_UNIT,
            "symbol": str(target_symbol),
            "barrier": chosen_barrier,
            "product_type": "basic",
        }

    buy_result = None
    buy_error = None
    contract_id = None
    try:
        log.info("Attempting buy for DIGITDIFF proposal id=%s barrier=%s price=%s", proposal_id, chosen_barrier, ask_price)
        buy_resp = await client.send_and_wait(buy_payload, timeout=8.0)
        log.info("Buy response: %s", json.dumps(buy_resp, indent=2))
        if isinstance(buy_resp, dict):
            buy_result = buy_resp.get("buy") or buy_resp
            # buy_result usually contains 'contract_id' or 'contract_id' under buy
            if isinstance(buy_result, dict):
                contract_id = buy_result.get("contract_id") or buy_result.get("contract_id")
            # if top-level buy_resp includes a buy dict, pick that id
            if not contract_id and "buy" in buy_resp and isinstance(buy_resp["buy"], dict):
                contract_id = buy_resp["buy"].get("contract_id")
            # error detection
            if buy_resp.get("error"):
                buy_error = buy_resp.get("error")
    except Exception as e:
        buy_error = {"exception": str(e)}
        log.error("Buy failed exception: %s", e)

    # If buy didn't return contract id, try to extract later from buy_result keys
    try:
        if not contract_id and isinstance(buy_result, dict):
            contract_id = buy_result.get("contract_id") or buy_result.get("id")
    except Exception:
        pass

    # Wait for final contract settlement if we have a contract id
    final_contract = None
    if contract_id:
        log.info("Waiting for contract to finish (contract_id=%s)...", contract_id)
        final_contract = await poll_contract_until_finished(client, contract_id, timeout=FINAL_POLL_TIMEOUT, interval=FINAL_POLL_INTERVAL)
        if final_contract:
            log.info("Final contract state: %s", json.dumps(final_contract, indent=2))
            try:
                # Normalise final_contract fields and extract canonical values
                actual_digit, p_val, p_pct, observed_ticks = _extract_final_details(final_contract)

                # Ensure final_contract contains commonly expected keys so hero/journal can extract them reliably
                try:
                    if isinstance(final_contract, dict):
                        if actual_digit is not None:
                            final_contract.setdefault("result_digit", actual_digit)
                            final_contract.setdefault("actual", actual_digit)
                        if p_val is not None:
                            final_contract.setdefault("profit", p_val)
                        if p_pct is not None:
                            final_contract.setdefault("profit_percentage", p_pct)
                        if observed_ticks:
                            final_contract.setdefault("tick_stream", observed_ticks)
                        # always annotate symbol from our chosen target so dashboard has it
                        final_contract.setdefault("underlying", final_contract.get("underlying") or target_symbol)
                        final_contract.setdefault("symbol", final_contract.get("symbol") or target_symbol)
                except Exception:
                    pass

                pred_id = proposal_id or (buy_result.get("id") if isinstance(buy_result, dict) else None) or f"trade_{int(time.time()*1000)}"
                pred_digit = int(chosen_barrier) if chosen_barrier is not None else None

                # Derive final result deterministically from profit if possible
                result_str = None
                if p_val is not None:
                    try:
                        f = float(p_val)
                        result_str = "WIN" if f > 0 else ("LOSS" if f < 0 else "DRAW")
                    except Exception:
                        result_str = None

                # Fallback: derive from status string
                if not result_str:
                    st = str(final_contract.get("status") or final_contract.get("state") or "").upper()
                    if "WIN" in st:
                        result_str = "WIN"
                    elif "LOSS" in st or "LOSE" in st:
                        result_str = "LOSS"
                    elif "DRAW" in st or "TIE" in st:
                        result_str = "DRAW"
                    else:
                        # If still unknown, mark SETTLED (not POSTED)
                        result_str = "SETTLED"

                # Post normalized final payload (push_final_to_hero will tidy fields further)
                push_final_to_hero(
                    final_contract,
                    prediction_id=pred_id,
                    predicted_digit=pred_digit,
                    result=result_str,
                    profit=p_val,
                    profit_percentage=p_pct,
                )
            except Exception as ex:
                log.warning("Failed to push final to hero (single-run): %s", ex)
        else:
            log.warning("Timed out waiting for final contract state (contract_id=%s).", contract_id)
    else:
        log.warning("No contract_id found - cannot poll for final contract state.")

    # Balance after trade (best-effort)
    balance_after = None
    try:
        bal2 = await client.send_and_wait({"balance": 1}, timeout=6.0)
        bal2_val = parse_balance_response(bal2)
        if bal2_val is not None:
            balance_after = bal2_val
            log.info("Balance after trade: %s", balance_after)
        else:
            log.info("Balance (raw): %s", json.dumps(bal2, indent=2))
    except Exception as e:
        log.warning("Balance-after request failed (non-fatal): %s", e)

    # Compute profit/loss
    # 1) prefer final_contract['profit'] if final and present
    # 2) else, fall back to balance difference (if we have both before & after)
    contract_profit = None
    percent_return_contract = None
    profit_by_balance = None
    percent_by_balance = None

    if final_contract and isinstance(final_contract, dict):
        # contract profit may be numeric or string
        p = final_contract.get("profit")
        try:
            if p is not None:
                contract_profit = float(p)
        except Exception:
            contract_profit = None
        # try compute percent from buy_price if present
        buy_price = None
        try:
            buy_price = float(final_contract.get("buy_price")) if final_contract.get("buy_price") is not None else None
        except Exception:
            buy_price = None
        if contract_profit is not None and buy_price:
            try:
                percent_return_contract = (contract_profit / float(buy_price)) * 100.0
            except Exception:
                percent_return_contract = None

    if balance_before is not None and balance_after is not None:
        try:
            profit_by_balance = float(balance_after) - float(balance_before)
            # percent by initial balance (not typical, but useful)
            if float(balance_before) != 0.0:
                percent_by_balance = (profit_by_balance / float(balance_before)) * 100.0
        except Exception:
            profit_by_balance = None
            percent_by_balance = None

    # Final summary logs
    log.info("=== SUMMARY ===")
    if balance_before is not None:
        log.info("Balance before trade: %s", balance_before)
    else:
        log.info("Balance before trade: (unknown)")

    if balance_after is not None:
        log.info("Balance after trade: %s", balance_after)
    else:
        log.info("Balance after trade: (unknown)")

    log.info("Chosen barrier: %s", json.dumps(chosen_barrier))
    log.info("Proposal (raw): %s", json.dumps(proposal, indent=2))
    log.info("Buy: %s", json.dumps(buy_result if buy_result is not None else None, indent=2))
    log.info("Final contract (raw): %s", json.dumps(final_contract if final_contract is not None else {}, indent=2))
    log.info("Error (if any): %s", json.dumps(buy_error if buy_error is not None else None, indent=2))

    # Preferred: use contract_profit if present and final_contract was final
    if contract_profit is not None:
        log.info("Profit/Loss (contract): %s", ("{:+.8g}".format(contract_profit)))
        if percent_return_contract is not None:
            # round nicely
            log.info("Percent return (contract): %s%%", (round(percent_return_contract, 4)))
    elif profit_by_balance is not None:
        log.info("Profit/Loss (by balance diff): %s", ("{:+.8g}".format(profit_by_balance)))
        if percent_by_balance is not None:
            log.info("Percent return (by balance): %s%%", (round(percent_by_balance, 6)))
    else:
        log.info("Profit/Loss: (unknown)")

    await client.close()
    # exit codes: 0 success (buy_result present), 3 buy succeeded but final not found, 4 buy failed
    if buy_result:
        return 0
    else:
        return 4

# Module-level cache of clients per mode (non-blocking)
_mode_clients = {"demo": None, "real": None}
_mode_client_locks = {"demo": threading.Lock(), "real": threading.Lock()}

async def _get_client_for_mode(mode: str) -> Optional[DerivWS]:
    """
    Return an authorized DerivWS client for `mode`. If token is not present server-side,
    this will try to fetch it (and retry a few times), then authorize the client.
    """
    if mode not in ("demo", "real"):
        mode = "demo"
    existing = _mode_clients.get(mode)
    if existing:
        return existing
    # Try retrieving token (env preferred, then server)
    token = DERIV_TOKEN if DERIV_TOKEN else fetch_server_token(mode)
    tries = 0
    while not token and tries < 40:  # up to ~20s retry
        await asyncio.sleep(0.5)
        token = fetch_server_token(mode)
        tries += 1
    if not token:
        log.warning("No token available for mode %s after retries", mode)
        return None
    try:
        c = DerivWS(WS_URL, timeout=TIMEOUT_SECONDS)
        await c.connect()
        auth_res = await c.send_and_wait({"authorize": token}, timeout=8.0)
        log.info("Authorized client for mode %s: %s", mode, json.dumps(auth_res)[:200])
        _mode_clients[mode] = c
        return c
    except Exception as e:
        log.warning("Failed to create/authorize client for mode %s: %s", mode, e)
        try:
            if c:
                await c.close()
        except Exception:
            pass
        return None

# Main daemon that listens for predictions and trades
async def run_trade_daemon(sse_url: str = None):
    if not DERIV_TOKEN:
        log.info("DERIV_TOKEN not set in env; will attempt to fetch server tokens at runtime when needed")

    # Prefer an explicit SSE URL env var, then fall back to push-url + /events, then default /events.
    sse_env = os.environ.get("HERO_DASHBOARD_SSE_URL")
    push_env = os.environ.get("HERO_DASHBOARD_PUSH_URL")
    if sse_url and str(sse_url).strip():
        sse_url = str(sse_url).rstrip("/")
    elif sse_env and str(sse_env).strip():
        sse_url = str(sse_env).rstrip("/")
    elif push_env and str(push_env).strip():
        sse_url = str(push_env).rstrip("/") + "/events"
    else:
        sse_url = "http://127.0.0.1:5000/events"

    log.info("Starting differ_trade_daemon: SSE=%s", sse_url)

    # queue for incoming predicted digits
    prediction_q: "asyncio.Queue[int]" = asyncio.Queue()

    # start SSE listener thread
    start_sse_listener(sse_url, asyncio.get_event_loop(), prediction_q)

    # maintain a persistent Deriv WebSocket client per mode; reconnect loop will handle recreation
    while True:
        try:
            # Wait for next predicted digit (blocks here until an item is available)
            try:
                item = await prediction_q.get()
            except asyncio.CancelledError:
                break

            # DEBUG: log dequeued item
            log.info("Dequeued prediction item from queue: %s", repr(item))

            # item may be just an int (back-compat) or a tuple (digit, market) or a dict
            pdigit = None
            target_symbol = None
            mode = None
            signal_contract_type = "DIGITDIFF"
            signal_barrier = None
            signal_prediction_id = None
            signal_prediction_mode = None
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pdigit, target_symbol = item[0], item[1]
                elif isinstance(item, dict):
                    # payload might be full analysis object
                    pdigit = (
                        item.get("prediction_digit")
                        if item.get("prediction_digit") is not None
                        else item.get("predicted")
                        if item.get("predicted") is not None
                        else item.get("pred")
                        if item.get("pred") is not None
                        else item.get("digit")
                    )
                    target_symbol = item.get("market") or item.get("symbol") or item.get("instrument") or SYMBOL
                    mode = item.get("mode") or item.get("account") or None
                    signal_prediction_id = item.get("prediction_id") or item.get("pred_id") or item.get("id")
                    signal_prediction_mode = item.get("prediction_mode") or item.get("mode_name")
                    ctype = str(item.get("contract_type") or item.get("trade_contract_type") or "").strip().upper()
                    if ctype in ("DIGITDIFF", "DIGITOVER", "DIGITUNDER"):
                        signal_contract_type = ctype
                    if item.get("barrier") is not None:
                        signal_barrier = str(item.get("barrier"))
                else:
                    pdigit = item
                # fallback to global SYMBOL if no market provided
                if not target_symbol:
                    try:
                        target_symbol = str(SYMBOL)
                    except Exception:
                        target_symbol = SYMBOL
                target_symbol = str(target_symbol).upper()
            except Exception:
                pdigit = item
                target_symbol = str(SYMBOL)

            # choose mode: prefer provided mode, else env DIFFER_MODE, else default 'demo'
            try:
                if mode and isinstance(mode, str):
                    mode = mode.lower()
                else:
                    mode = os.environ.get("DIFFER_MODE") or "demo"
                if mode not in ("demo", "real"):
                    mode = "demo"
            except Exception:
                mode = "demo"

            # when we receive a predicted digit, try to create a DIGITDIFF proposal with that barrier and buy
            try:
                if signal_barrier is not None and str(signal_barrier).strip() != "":
                    chosen_barrier = str(int(signal_barrier) % 10)
                else:
                    chosen_barrier = str(int(pdigit) % 10)
            except Exception:
                # fallback random barrier
                chosen_barrier = pick_random_barrier()
            log.info(
                "Daemon: received prediction digit=%s contract_type=%s market=%s -> attempting trade (mode=%s)",
                chosen_barrier,
                signal_contract_type,
                target_symbol,
                mode,
            )

            # optional: small delay to avoid sending proposals too quickly
            await asyncio.sleep(0.05)


                        # extract per-signal stake/amount (fallback to env DEFAULT AMOUNT)
            stake_amount = None
            try:
                if isinstance(item, dict):
                    if item.get("stake") is not None:
                        try:
                            stake_amount = float(item.get("stake"))
                        except Exception:
                            stake_amount = None
                    elif item.get("amount") is not None:
                        try:
                            stake_amount = float(item.get("amount"))
                        except Exception:
                            stake_amount = None
            except Exception:
                stake_amount = None

            # fallback to default AMOUNT (from env or constants)
            try:
                if stake_amount is not None:
                    stake_amount = float(stake_amount)
                else:
                    stake_amount = float(os.environ.get("DIFFER_AMOUNT", AMOUNT))
                    log.info("Using default stake_amount fallback: %s", stake_amount)
            except Exception:
                stake_amount = float(AMOUNT)
                log.info("Using fallback AMOUNT constant stake_amount: %s", stake_amount)

            # Build and request proposal
            pl = {
                "proposal": 1,
                "amount": float(stake_amount),    # <-- use per-signal stake
                "basis": "stake",
                "contract_type": signal_contract_type,
                "currency": CURRENCY,
                "duration": int(DURATION),
                "duration_unit": DURATION_UNIT,
                "symbol": str(target_symbol),
                "barrier": chosen_barrier,
                "product_type": "basic",
            }


            # Ensure we have an authorized client for the requested mode
            client_for_mode = await _get_client_for_mode(mode)
            if not client_for_mode:
                log.warning("No authorized client available for mode %s - skipping this trade", mode)
                continue

            try:
                resp = await client_for_mode.send_and_wait(pl, timeout=8.0)
            except Exception as e:
                log.warning("Proposal request failed for barrier %s: %s", chosen_barrier, e)
                # try to re-establish client for this mode (close & clear)
                try:
                    await client_for_mode.close()
                except Exception:
                    pass
                _mode_clients[mode] = None
                continue

            # validate proposal
            if isinstance(resp, dict) and resp.get("error"):
                log.warning("Proposal response error for barrier %s: %s", chosen_barrier, json.dumps(resp.get("error")))
                continue
            proposal = None
            if isinstance(resp, dict):
                if "proposal" in resp and isinstance(resp["proposal"], dict):
                    proposal = resp["proposal"]
                elif resp.get("ask_price") or resp.get("id") or resp.get("display_value"):
                    proposal = resp
            if not proposal:
                log.warning("No valid proposal returned for barrier %s: %s", chosen_barrier, json.dumps(resp))
                continue

            ask_price = proposal.get("ask_price") or proposal.get("display_value") or proposal.get("price") or float(AMOUNT)
            proposal_id = proposal.get("id") or proposal.get("proposal_id") or proposal.get("proposal")

            if proposal_id:
                buy_payload = {"buy": str(proposal_id), "price": ask_price}
            else:
                buy_payload = {
                    "buy": 1,
                    "price": ask_price,
                    "proposal": 1,
                    "amount": float(stake_amount),    # <-- ensure amount included here too
                    "basis": "stake",
                    "contract_type": signal_contract_type,
                    "currency": CURRENCY,
                    "duration": int(DURATION),
                    "duration_unit": DURATION_UNIT,
                    "symbol": str(target_symbol),
                    "barrier": chosen_barrier,
                    "product_type": "basic",
                }


            # attempt buy using the mode-specific client
            try:
                buy_resp = await client_for_mode.send_and_wait(buy_payload, timeout=8.0)
                log.info("Daemon buy response: %s", json.dumps(buy_resp))
            except Exception as e:
                log.error("Daemon buy failed: %s", e)
                # clear client so next time it will re-authorize
                try:
                    await client_for_mode.close()
                except Exception:
                    pass
                _mode_clients[mode] = None
                continue

            # extract contract id if present and optionally poll until settlement
            contract_id = None
            buy_result = None
            if isinstance(buy_resp, dict):
                buy_result = buy_resp.get("buy") or buy_resp
                if isinstance(buy_result, dict):
                    contract_id = buy_result.get("contract_id")
                if not contract_id and "buy" in buy_resp and isinstance(buy_resp["buy"], dict):
                    contract_id = buy_resp["buy"].get("contract_id")

            # Notify dashboard (toast) that we took a trade
            toast = {
                "analysis_event": "prediction_toast",
                "prediction_id": signal_prediction_id or proposal_id or contract_id or f"trade_{int(time.time()*1000)}",
                "symbol": target_symbol,
                "market": target_symbol,
                "digit": int(chosen_barrier),
                "contract_type": signal_contract_type,
                "barrier": chosen_barrier,
                "prediction_mode": signal_prediction_mode,
                "message": f"Auto-trade placed type={signal_contract_type} barrier={chosen_barrier} amount={stake_amount} mode={mode}",
                "status": "posted",
                "epoch": int(time.time()),
            }
            try:
                notify_hero_push(toast)
                log.info("Daemon: posted toast to dashboard")
            except Exception as e:
                log.warning("Daemon: failed to post toast: %s", e)

            # Optionally wait for final contract settlement (non-blocking to queue next predictions, but here we poll for this contract)
            if contract_id:
                final = await poll_contract_until_finished(client_for_mode, contract_id, timeout=FINAL_POLL_TIMEOUT, interval=FINAL_POLL_INTERVAL)
                if final:
                    log.info("Daemon: Final contract: %s", json.dumps(final))
                    try:
                        # Normalize + extract final details
                        actual_digit, p_val, p_pct, observed_ticks = _extract_final_details(final)

                        # make sure final dict contains useful keys for the dashboard
                        try:
                            if isinstance(final, dict):
                                if actual_digit is not None:
                                    final.setdefault("result_digit", actual_digit)
                                    final.setdefault("actual", actual_digit)
                                if p_val is not None:
                                    final.setdefault("profit", p_val)
                                if p_pct is not None:
                                    final.setdefault("profit_percentage", p_pct)
                                if observed_ticks:
                                    final.setdefault("tick_stream", observed_ticks)
                                final.setdefault("underlying", final.get("underlying") or target_symbol)
                                final.setdefault("symbol", final.get("symbol") or target_symbol)
                        except Exception:
                            pass

                        pred_id = toast.get("prediction_id") or (proposal_id or contract_id or f"trade_{int(time.time()*1000)}")
                        pred_digit = int(chosen_barrier) if chosen_barrier is not None else None

                        # determine result robustly from profit first
                        result_str = None
                        if p_val is not None:
                            try:
                                result_str = "WIN" if float(p_val) > 0 else ("LOSS" if float(p_val) < 0 else "DRAW")
                            except Exception:
                                result_str = None

                        # fallback: status string
                        if not result_str:
                            st = str(final.get("status") or final.get("state") or "").upper()
                            if "WIN" in st:
                                result_str = "WIN"
                            elif "LOSS" in st or "LOSE" in st:
                                result_str = "LOSS"
                            elif "DRAW" in st or "TIE" in st:
                                result_str = "DRAW"
                            else:
                                result_str = "SETTLED"

                        # Post via reliable wrapper - includes retries
                        push_final_to_hero(
                            final,
                            prediction_id=pred_id,
                            predicted_digit=pred_digit,
                            result=result_str,
                            profit=p_val,
                            profit_percentage=p_pct,
                            contract_type=signal_contract_type,
                            barrier=chosen_barrier,
                            prediction_mode=signal_prediction_mode,
                        )
                    except Exception as ex:
                        log.warning("Daemon: failed to push final contract: %s", ex)

                    except Exception:
                        try:
                            notify_hero_push(final_payload)
                        except Exception:
                            pass
                else:
                    log.warning("Daemon: timed out waiting for final contract %s", contract_id)
            # loop to next prediction (no exit)
        except Exception as e:
            log.exception("Daemon top-level error: %s", e)
            # attempt to close clients & reconnect will be implicit
            try:
                for m in ("demo", "real"):
                    c = _mode_clients.get(m)
                    if c:
                        try:
                            asyncio.create_task(c.close())
                        except Exception:
                            pass
                        _mode_clients[m] = None
            except Exception:
                pass
            await asyncio.sleep(2.0)

    # unreachable normally
    return 0


# ---------- CLI / main ----------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(prog="differ_trade_check.py")
    p.add_argument("--daemon", action="store_true", help="Run continuously and listen to HERO dashboard SSE for predictions")
    p.add_argument(
    "--sse-url",
    default=os.environ.get("HERO_DASHBOARD_SSE_URL", None),
    help="Explicit SSE base (will append /events if necessary). If you pass a push URL (/control/push_tick) it will be converted to the events endpoint.",
)

    args = p.parse_args()

    if args.daemon:
        sse_url = args.sse_url
        if sse_url:
            sse_url = str(sse_url).rstrip("/")
            # tolerate being passed the push URL by converting it to the events endpoint
            if "/control/push_tick" in sse_url:
                sse_url = sse_url.replace("/control/push_tick", "/events")
            if not sse_url.endswith("/events"):
                sse_url = sse_url + "/events"
        else:
            sse_url = "http://127.0.0.1:5000/events"
        try:
            rc = asyncio.run(run_trade_daemon(sse_url))
            sys.exit(int(rc or 0))
        except KeyboardInterrupt:
            log.info("Daemon interrupted by user")
            sys.exit(1)
        except Exception as e:
            log.exception("Daemon unhandled exception: %s", e)
            sys.exit(2)
    else:
        # default single-run behaviour (unchanged)
        try:
            rc = asyncio.run(run_trade_check())
            sys.exit(int(rc or 0))
        except KeyboardInterrupt:
            log.info("Interrupted by user")
            sys.exit(1)
        except Exception as e:
            log.exception("Unhandled exception: %s", e)
            sys.exit(2)
